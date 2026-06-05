from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from arcengine import ActionInput, FrameData, FrameDataRaw, GameAction, GameState

from agents.agent import Agent
from agents.templates.continual_harness.helpers import (
    available_game_actions,
    build_action_tools,
    parse_action_response,
)
from agents.templates.continual_harness.prompts import HARNESS_SYSTEM_INSTRUCTION
from agents.templates.continual_harness.trace import (
    TraceWriter,
    default_trace_path,
    serialize_response,
)
from agents.templates.utils.vlm_backend import (
    VLM,
    AnthropicBackend,
    GeminiBackend,
    OpenAIBackend,
)


def _function_response(name: str, args: dict[str, Any] | None = None) -> Any:
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            function_call=SimpleNamespace(name=name, args=args or {})
                        )
                    ]
                )
            )
        ]
    )


def _text_response(text: str) -> Any:
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=text)]))
        ]
    )


@pytest.mark.unit
class TestContinualHarnessParsing:
    def test_simple_function_call_response(self) -> None:
        action = parse_action_response(
            _function_response("ACTION2"),
            [GameAction.ACTION1, GameAction.ACTION2],
        )

        assert action is GameAction.ACTION2

    def test_action6_function_call_coerces_coordinates(self) -> None:
        action = parse_action_response(
            _function_response("ACTION6", {"x": "99", "y": "-4"}),
            [GameAction.ACTION6],
        )

        assert action is GameAction.ACTION6
        assert action.action_data.x == 63
        assert action.action_data.y == 0

    def test_json_text_fallback(self) -> None:
        action = parse_action_response(
            _text_response('{"action": "ACTION5", "arguments": {}}'),
            [GameAction.ACTION5],
        )

        assert action is GameAction.ACTION5

    def test_unparseable_response_returns_none(self) -> None:
        # ACTION7 is not in the available set → parser returns None so the agent can retry.
        action = parse_action_response(
            _function_response("ACTION7"),
            [GameAction.ACTION1, GameAction.ACTION2],
        )

        assert action is None

    def test_available_actions_and_tools_include_action7(self) -> None:
        actions = available_game_actions([1, 7])
        tools = build_action_tools(actions)

        assert actions == [GameAction.ACTION1, GameAction.ACTION7]
        # Tool specs are now flat dicts; no Gemini-specific `function_declarations` wrapper.
        assert [tool["name"] for tool in tools] == ["ACTION1", "ACTION7"]
        # reasoning is now required on every action so Gemini must emit it before tool-calling.
        for tool in tools:
            params = tool["parameters"]
            assert "reasoning" in params["required"]
            assert params["properties"]["reasoning"]["type"] == "string"

    def test_action6_tool_requires_reasoning_x_y(self) -> None:
        tools = build_action_tools([GameAction.ACTION6])
        params = tools[0]["parameters"]
        assert set(params["required"]) == {"reasoning", "x", "y"}

    def test_function_call_reasoning_is_captured_on_action_object(self) -> None:
        action = parse_action_response(
            _function_response(
                "ACTION6",
                {"reasoning": "moving up because wall is left", "x": 5, "y": 7},
            ),
            [GameAction.ACTION6],
        )

        assert action is GameAction.ACTION6
        # Reasoning lives on action.reasoning; action_data still gets only x/y.
        assert action.reasoning == "moving up because wall is left"
        assert action.action_data.x == 5
        assert action.action_data.y == 7


@pytest.mark.unit
class TestContinualHarnessPrompts:
    def test_system_prompt_holds_game_context(self) -> None:
        assert "{game_name}" in HARNESS_SYSTEM_INSTRUCTION
        assert "take_actions" in HARNESS_SYSTEM_INSTRUCTION
        assert "run_skill" in HARNESS_SYSTEM_INSTRUCTION



class _ActionStampEnv:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def step(
        self,
        action: GameAction,
        data: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> FrameDataRaw:
        self.calls.append(
            {"action": action, "data": data or {}, "reasoning": reasoning or {}}
        )
        raw = FrameDataRaw(
            game_id="stamp-test",
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=1,
            action_input=ActionInput(),
            available_actions=[1, 2, 3, 4, 6],
        )
        raw.frame = [np.array([[1, 2], [3, 4]], dtype=np.int8)]
        return raw


class _ActionStampAgent(Agent):
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return False

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        return GameAction.ACTION1


@pytest.mark.unit
class TestAgentActionInputStamping:
    def test_do_action_request_stamps_submitted_action_on_returned_frame(
        self,
    ) -> None:
        env = _ActionStampEnv()
        agent = _ActionStampAgent(
            card_id="card",
            game_id="stamp-test",
            agent_name="agent",
            ROOT_URL="https://example.com",
            record=False,
            arc_env=env,  # type: ignore[arg-type]
        )
        action = GameAction.ACTION6
        action.set_data({"x": 12, "y": 34})
        action.reasoning = "click the visible target"

        frame = agent.do_action_request(action)
        action.reasoning = None

        assert frame.action_input.id is GameAction.ACTION6
        assert frame.action_input.data == {"x": 12, "y": 34}
        assert frame.action_input.reasoning == "click the visible target"
        assert env.calls == [
            {
                "action": GameAction.ACTION6,
                "data": {"game_id": "", "x": 12, "y": 34},
                "reasoning": {"reasoning": "click the visible target"},
            }
        ]


class _FakeClient:
    """Stand-in for google.genai.Client used by GeminiBackend tests."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.models = self

    def generate_content(self, *_: Any, **__: Any) -> Any:
        return _text_response("")


def _install_fake_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    google = ModuleType("google")
    genai = ModuleType("google.genai")
    types_mod = ModuleType("google.genai.types")

    class _Passthrough:
        def __init__(self, **_: Any) -> None:
            pass

    genai.Client = _FakeClient  # type: ignore[attr-defined]
    types_mod.HttpOptions = _Passthrough  # type: ignore[attr-defined]
    types_mod.GenerateContentConfig = _Passthrough  # type: ignore[attr-defined]
    google.genai = genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)


@pytest.mark.unit
class TestVLMBackendRouting:
    def test_auto_detects_gemini_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_gemini(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        vlm = VLM("gemini-2.5-pro", backend="auto")

        assert vlm.backend_name == "gemini"
        assert isinstance(vlm.backend, GeminiBackend)

    def test_placeholders_raise_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            OpenAIBackend("gpt-4o").get_text_query("hello")

        with pytest.raises(NotImplementedError):
            AnthropicBackend("claude-opus").get_text_query("hello")

    def test_missing_gemini_key_raises_value_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_gemini(monkeypatch)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with pytest.raises(ValueError, match="Gemini API key is missing"):
            GeminiBackend("gemini-2.5-pro")


@pytest.mark.unit
class TestTraceWriter:
    def test_round_trips_jsonl(self, tmp_path: Any) -> None:
        writer = TraceWriter(tmp_path / "run.trace.jsonl")
        writer.write({"attempt": 1, "chosen_action": "ACTION1"})
        writer.write({"attempt": 2, "chosen_action": "ACTION3"})

        raw = (tmp_path / "run.trace.jsonl").read_text()
        lines = raw.splitlines()
        assert raw.endswith("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        assert first["chosen_action"] == "ACTION1"
        assert second["chosen_action"] == "ACTION3"
        # Timestamp is auto-injected by the writer.
        assert "timestamp" in first and "timestamp" in second

    def test_default_path_uses_run_log_path_env(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RUN_DIR", raising=False)
        run_log = tmp_path / "logs" / "continualharness-x.log"
        monkeypatch.setenv("RUN_LOG_PATH", str(run_log))
        path = default_trace_path()
        assert path == run_log.with_suffix(".trace.jsonl")

    def test_default_path_uses_run_dir_with_game_id(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = tmp_path / "logs" / "run"
        monkeypatch.setenv("RUN_DIR", str(run_dir))

        path = default_trace_path("ls20")

        assert path == run_dir / "ls20" / "trace.jsonl"

    def test_serialize_response_extracts_function_calls_and_text(self) -> None:
        # Single-candidate response with one function_call and one text part.
        resp = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    finish_reason=1,
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(text="thinking...", function_call=None),
                            SimpleNamespace(
                                text=None,
                                function_call=SimpleNamespace(
                                    name="ACTION1", args={"reasoning": "go up"}
                                ),
                            ),
                        ]
                    ),
                )
            ]
        )
        out = serialize_response(resp)
        assert out["text"] == "thinking..."
        assert out["function_calls"] == [
            {"name": "ACTION1", "args": {"reasoning": "go up"}}
        ]
        assert out["finish_reason"] == 1


@pytest.mark.unit
class TestTokenUsage:
    def _make_backend(self, monkeypatch: pytest.MonkeyPatch) -> GeminiBackend:
        _install_fake_gemini(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        return GeminiBackend("gemini-2.5-pro")

    def test_gemini_extract_usage_returns_canonical_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = self._make_backend(monkeypatch)
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=1842,
                candidates_token_count=28,
                total_token_count=1870,
                thoughts_token_count=612,
                cached_content_token_count=1024,
                tool_use_prompt_token_count=None,
            )
        )

        usage = backend.extract_usage(response)

        assert usage == {
            "prompt": 1842,
            "output": 28,
            "total": 1870,
            "thoughts": 612,
            "cached": 1024,
            "tool_use": None,
        }

    def test_gemini_extract_usage_returns_none_when_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = self._make_backend(monkeypatch)
        # Response with no usage_metadata at all.
        assert backend.extract_usage(SimpleNamespace()) is None

    def test_vlm_extract_usage_delegates_to_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_gemini(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        vlm = VLM("gemini-2.5-pro", backend="auto")

        calls: list[Any] = []

        def fake_extract(resp: Any) -> dict[str, int | None] | None:
            calls.append(resp)
            return {
                "prompt": 1,
                "output": 2,
                "total": 3,
                "thoughts": None,
                "cached": None,
                "tool_use": None,
            }

        monkeypatch.setattr(vlm.backend, "extract_usage", fake_extract)
        marker = SimpleNamespace(usage_metadata=None)
        out = vlm.extract_usage(marker)

        assert calls == [marker]
        assert out == {
            "prompt": 1,
            "output": 2,
            "total": 3,
            "thoughts": None,
            "cached": None,
            "tool_use": None,
        }

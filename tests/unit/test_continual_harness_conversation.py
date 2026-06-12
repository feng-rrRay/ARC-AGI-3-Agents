"""Tests for the conversation-until-action loop in ContinualHarness._vlm_loop_inner.

Covers both layers of the change:
  * the GeminiBackend multi-turn helpers (build_user_turn / build_tool_results_turn
    / model_turn / get_query_contents), and
  * the orchestrator loop: one decision = one conversation that persists across
    tool-only turns until an action fires, with one trace record per turn sharing
    a `conversation_id`, observations cleared after turn 0, and the runaway cap.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from arcengine import ActionInput, FrameData, FrameDataRaw, GameState
from PIL import Image

from agents.templates.continual_harness.context import format_tool_record_md
from agents.templates.continual_harness.models import (
    PendingActionObservation,
    ToolCallRecord,
)
from agents.templates.continual_harness.sandbox import SandboxState
from agents.templates.continual_harness_agent import ContinualHarness
from agents.templates.utils.vlm_backend import GeminiBackend


# --- response builders (Gemini SimpleNamespace shape) -------------------------


def _fc_part(name: str, args: dict[str, Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        text=None, function_call=SimpleNamespace(name=name, args=args or {})
    )


def _response(*parts: SimpleNamespace, finish_reason: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason=finish_reason, content=SimpleNamespace(parts=list(parts))
            )
        ]
    )


# --- fake google-genai SDK (records the contents passed to generate_content) --


class _RecordingClient:
    def __init__(
        self,
        responses: list[Any] | None = None,
        token_count: int = 1,
        *_: Any,
        **__: Any,
    ) -> None:
        self.models = self
        self._responses = list(responses or [])
        self._token_count = token_count
        self.contents_calls: list[Any] = []
        self.count_tokens_calls: list[dict[str, Any]] = []

    def generate_content(self, *, model: Any, contents: Any, config: Any) -> Any:
        self.contents_calls.append(contents)
        if self._responses:
            return self._responses.pop(0)
        return _response()  # empty (no function calls)

    def count_tokens(
        self, *, model: str, contents: Any, config: Any = None
    ) -> SimpleNamespace:
        self.count_tokens_calls.append(
            {"model": model, "contents": contents, "config": config}
        )
        return SimpleNamespace(total_tokens=self._token_count)


def _install_fake_gemini(monkeypatch: pytest.MonkeyPatch, client: Any = None) -> Any:
    google = ModuleType("google")
    genai = ModuleType("google.genai")
    types_mod = ModuleType("google.genai.types")

    class _Passthrough:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _Part:
        def __init__(self, kind: str, **data: Any) -> None:
            self.kind = kind
            self.data = data

        @staticmethod
        def from_text(*, text: str) -> "_Part":
            return _Part("text", text=text)

        @staticmethod
        def from_bytes(*, data: bytes, mime_type: str) -> "_Part":
            return _Part("bytes", mime_type=mime_type, nbytes=len(data))

        @staticmethod
        def from_function_response(*, name: str, response: dict[str, Any]) -> "_Part":
            return _Part("function_response", name=name, response=response)

    class _Content:
        def __init__(self, *, role: str, parts: list[Any]) -> None:
            self.role = role
            self.parts = parts

    used_client = client if client is not None else _RecordingClient()
    genai.Client = lambda *a, **k: used_client  # type: ignore[attr-defined]
    types_mod.HttpOptions = _Passthrough  # type: ignore[attr-defined]
    types_mod.GenerateContentConfig = _Passthrough  # type: ignore[attr-defined]
    types_mod.CountTokensConfig = _Passthrough  # type: ignore[attr-defined]
    types_mod.Part = _Part  # type: ignore[attr-defined]
    types_mod.Content = _Content  # type: ignore[attr-defined]
    google.genai = genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)
    return used_client


# --- backend-level tests ------------------------------------------------------


@pytest.mark.unit
class TestGeminiConversationBackend:
    def test_build_turns_and_query_contents(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _RecordingClient()
        _install_fake_gemini(monkeypatch, client)
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        backend = GeminiBackend("gemini-x", system_instruction="sys")
        backend.set_tools(
            [{"name": "take_actions", "description": "", "parameters": {}}]
        )

        # User turn = text part + one image part.
        user = backend.build_user_turn("hello", [Image.new("RGB", (2, 2))])
        assert user.role == "user"
        assert user.parts[0].kind == "text" and user.parts[0].data["text"] == "hello"
        assert user.parts[1].kind == "bytes"

        # Tool-results turn = one function_response part per pair.
        tr = backend.build_tool_results_turn(
            [("process_memory", {"result": {"id": "m1"}})]
        )
        assert tr.role == "user"
        assert tr.parts[0].kind == "function_response"
        assert tr.parts[0].data["name"] == "process_memory"

        # model_turn returns the response's candidate content.
        resp = _response(_fc_part("take_actions", {"actions": []}))
        assert backend.model_turn(resp) is resp.candidates[0].content

        # get_query_contents forwards the full conversation to generate_content.
        out = backend.get_query_contents([user, tr], module_name="t")
        assert client.contents_calls[-1] == [user, tr]
        assert out is not None

    def test_model_turn_falls_back_to_empty_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_gemini(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        backend = GeminiBackend("gemini-x")
        empty = backend.model_turn(SimpleNamespace(candidates=[]))
        assert empty.role == "model" and empty.parts == []

    def test_safety_filtered_initial_image_turn_retries_text_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fallback_response = _response(
            _fc_part("take_actions", {"actions": [{"name": "ACTION1"}]})
        )
        client = _RecordingClient(
            [_response(finish_reason=12), fallback_response]
        )
        _install_fake_gemini(monkeypatch, client)
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        backend = GeminiBackend("gemini-x")
        backend.set_tools(
            [{"name": "take_actions", "description": "", "parameters": {}}]
        )
        user = backend.build_user_turn("hello", [Image.new("RGB", (2, 2))])

        out = backend.get_query_contents([user], module_name="t")

        assert out is fallback_response
        assert client.contents_calls == [[user], ["hello"]]

    def test_count_input_tokens_uses_sdk_with_system_and_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _RecordingClient(token_count=123)
        _install_fake_gemini(monkeypatch, client)
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        backend = GeminiBackend("gemini-3.5-flash", system_instruction="sys")
        backend.set_tools(
            [{"name": "take_actions", "description": "", "parameters": {}}]
        )
        user = backend.build_user_turn("hello", [Image.new("RGB", (2, 2))])

        tokens = backend.count_input_tokens([user], module_name="t")

        assert tokens == 123
        assert backend.context_window_tokens() == 1_048_576
        call = client.count_tokens_calls[-1]
        assert call["model"] == "gemini-3.5-flash"
        assert call["contents"] == [user]
        assert call["config"].system_instruction == "sys"
        assert call["config"].tools


# --- orchestrator loop fixtures ----------------------------------------------


class _NoopEnv:
    def __init__(self) -> None:
        self.observation_space: Any = FrameDataRaw(
            game_id="conv-test",
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=1,
            action_input=ActionInput(),
            available_actions=[1, 2, 3, 6],
        )
        self.observation_space.frame = [np.array([[0]], dtype=np.int8)]


class _ConvVLM:
    """Scripted multi-turn VLM. Pops one response per get_query_contents call."""

    def __init__(
        self,
        responses: list[Any],
        *,
        token_counts: list[int] | None = None,
        context_window: int | None = 1_048_576,
    ) -> None:
        self._responses = list(responses)
        self._token_counts = list(token_counts or [])
        self._context_window = context_window
        self.contents_lens: list[int] = []
        self.contents_calls: list[Any] = []
        self.count_tokens_calls: list[Any] = []

    def set_tools(self, tools: Any) -> None:
        pass

    def extract_usage(self, response: Any) -> None:
        return None

    def build_user_turn(self, text: str, images: Any) -> Any:
        return ("user", text, len(list(images)))

    def build_tool_results_turn(self, pairs: Any) -> Any:
        return ("tool_results", list(pairs))

    def model_turn(self, response: Any) -> Any:
        return ("model", response)

    def get_query_contents(self, contents: list[Any], module_name: str = "x") -> Any:
        self.contents_calls.append(list(contents))
        self.contents_lens.append(len(contents))
        return self._responses.pop(0)

    def count_input_tokens(
        self, contents: list[Any], module_name: str = "x"
    ) -> int:
        self.count_tokens_calls.append(list(contents))
        if self._token_counts:
            return self._token_counts.pop(0)
        return 1

    def context_window_tokens(self) -> int | None:
        return self._context_window


def _make_frame() -> FrameData:
    return FrameData(
        game_id="conv-test",
        frame=[[[0, 0], [0, 0]]],
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        win_levels=1,
        action_input=ActionInput(),
        available_actions=[1, 2, 3, 6],
    )


def _make_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ContinualHarness:
    _install_fake_gemini(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("RUN_LOG_PATH", str(tmp_path / "run.log"))
    monkeypatch.setenv("RUN_MEMORY_PATH", str(tmp_path / "memory.json"))
    monkeypatch.setenv("RUN_SKILLS_PATH", str(tmp_path / "skills.json"))
    monkeypatch.setenv("RUN_SUBAGENTS_PATH", str(tmp_path / "subagents.json"))
    monkeypatch.setenv("RUN_PROMPT_PATH", str(tmp_path / "prompt.current.md"))
    monkeypatch.setenv(
        "RUN_PROMPT_EVOLUTION_PATH", str(tmp_path / "prompt_evolution.jsonl")
    )
    return ContinualHarness(
        card_id="card",
        game_id="conv-test",
        agent_name="convagent",
        ROOT_URL="https://example.com",
        record=False,
        arc_env=_NoopEnv(),  # type: ignore[arg-type]
    )


def _trace_rows(agent: ContinualHarness) -> list[dict[str, Any]]:
    if not agent.trace.path.exists():
        return []
    return [json.loads(line) for line in agent.trace.path.read_text().splitlines() if line]


@pytest.mark.unit
class TestConversationLoop:
    def test_tool_then_action_is_one_conversation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        # Isolate the loop from the engine: take_actions "executes" 1 action.
        monkeypatch.setattr(
            agent, "_dispatch_take_actions", lambda args, source="vlm": 1
        )
        agent.vlm = _ConvVLM(  # type: ignore[assignment]
            [
                _response(_fc_part("process_memory", {"action": "add", "content": "x"})),
                _response(_fc_part("take_actions", {"actions": [{"name": "ACTION1"}]})),
            ]
        )
        frame = _make_frame()
        agent.frames = [frame, frame]
        # Seed an observation so we can prove it is cleared after turn 0.
        agent._pending_observations = [
            PendingActionObservation(
                action_counter=0, action_name="ACTION1", pre_frame_index=0,
                post_frame_index=1, valid_frame=True,
            )
        ]

        executed = agent._vlm_loop_inner(frame)

        assert executed == 1
        rows = _trace_rows(agent)
        assert len(rows) == 2  # two turns in one conversation
        assert rows[0]["conversation_id"] == rows[1]["conversation_id"]
        assert [r["conversation_turn"] for r in rows] == [0, 1]
        # Images only on turn 0; follow-up turn attaches none.
        assert rows[0]["input"]["images_attached_count"] >= 1
        assert rows[1]["input"]["images_attached_count"] == 0
        # Observation block delivered on turn 0 then cleared.
        assert "OBSERVATIONS SINCE LAST QUERY" in rows[0]["input"]["user_prompt"]
        assert agent._pending_observations == []
        # Final turn's non-action results carry forward (none on the action turn).
        assert agent._recent_tool_results == []

    def test_immediate_action_is_single_turn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        monkeypatch.setattr(
            agent, "_dispatch_take_actions", lambda args, source="vlm": 1
        )
        agent.vlm = _ConvVLM(  # type: ignore[assignment]
            [_response(_fc_part("take_actions", {"actions": [{"name": "ACTION1"}]}))]
        )
        frame = _make_frame()
        agent.frames = [frame]

        executed = agent._vlm_loop_inner(frame)

        assert executed == 1
        rows = _trace_rows(agent)
        assert len(rows) == 1
        assert rows[0]["conversation_turn"] == 0

    def test_zero_action_take_actions_gets_function_response(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        dispatch_results = [0, 1]

        def fake_dispatch(args: dict[str, Any], source: str = "vlm") -> int:
            return dispatch_results.pop(0)

        monkeypatch.setattr(agent, "_dispatch_take_actions", fake_dispatch)
        vlm = _ConvVLM(
            [
                _response(_fc_part("take_actions", {"actions": []})),
                _response(
                    _fc_part("take_actions", {"actions": [{"name": "ACTION1"}]})
                ),
            ]
        )
        agent.vlm = vlm  # type: ignore[assignment]
        frame = _make_frame()
        agent.frames = [frame]

        executed = agent._vlm_loop_inner(frame)

        assert executed == 1
        rows = _trace_rows(agent)
        assert rows[0]["tool_calls"][0]["name"] == "take_actions"
        assert rows[0]["tool_calls"][0]["result"]["executed"] == 0
        tool_results_turn = vlm.contents_calls[1][-1]
        assert tool_results_turn[0] == "tool_results"
        assert tool_results_turn[1][0][0] == "take_actions"

    def test_context_guard_breaks_and_reason_is_next_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        vlm = _ConvVLM(
            [_response(_fc_part("take_actions", {"actions": [{"name": "ACTION1"}]}))],
            token_counts=[801],
            context_window=1000,
        )
        agent.vlm = vlm  # type: ignore[assignment]
        frame = _make_frame()
        agent.frames = [frame]

        executed = agent._vlm_loop_inner(frame)

        assert executed == 0
        assert vlm.contents_calls == []
        assert "context_window_guard" in (agent._previous_no_action_reason or "")
        rows = _trace_rows(agent)
        assert rows[0]["error"].startswith("context_window_guard")
        prompt, _grids = agent._build_working_prompt(frame)
        assert "## PREVIOUS CONVERSATION STOPPED WITHOUT ACTION" in prompt
        assert "context_window_guard" in prompt

    def test_successful_action_clears_previous_no_action_reason(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        agent._previous_no_action_reason = "context_window_guard: previous"
        monkeypatch.setattr(
            agent, "_dispatch_take_actions", lambda args, source="vlm": 1
        )
        agent.vlm = _ConvVLM(  # type: ignore[assignment]
            [_response(_fc_part("take_actions", {"actions": [{"name": "ACTION1"}]}))]
        )
        frame = _make_frame()
        agent.frames = [frame]

        executed = agent._vlm_loop_inner(frame)

        assert executed == 1
        assert agent._previous_no_action_reason is None

    def test_vlm_usage_pricing_updates_cumulative_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)

        low = agent._record_vlm_usage(
            {
                "prompt": 100_000,
                "output": 1_000,
                "thoughts": 500,
                "total": 101_500,
                "cached": 20_000,
            }
        )
        high = agent._record_vlm_usage(
            {
                "prompt": 250_000,
                "output": 1_000,
                "thoughts": 0,
                "total": 251_000,
                "cached": 50_000,
            }
        )

        assert low is not None and low["tier"] == "le_200k"
        assert high is not None and high["tier"] == "gt_200k"
        assert agent.total_calls == 2
        assert agent.total_prompt_tokens == 350_000
        assert agent.total_output_tokens == 2_000
        assert agent.total_tokens == 352_500
        assert agent.total_priced_calls == 2
        assert agent.total_vlm_cost_usd == pytest.approx(high["cumulative_usd"])

    def test_vlm_usage_pricing_unsupported_model_still_counts_tokens(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        agent.model_name = "gemini-2.5-pro"

        cost = agent._record_vlm_usage(
            {"prompt": 100, "output": 20, "thoughts": 5, "total": 125, "cached": 0}
        )

        assert cost is None
        assert agent.total_calls == 1
        assert agent.total_tokens == 125
        assert agent.total_priced_calls == 0
        assert agent.total_vlm_cost_usd == 0.0

    def test_runaway_cap_forces_break(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        monkeypatch.setattr(agent, "MAX_CONVERSATION_TURNS", 3)
        # Never emits an action: only tool-only turns.
        agent.vlm = _ConvVLM(  # type: ignore[assignment]
            [
                _response(
                    _fc_part("process_memory", {"action": "add", "content": f"n{i}"})
                )
                for i in range(3)
            ]
        )
        frame = _make_frame()
        agent.frames = [frame]

        executed = agent._vlm_loop_inner(frame)

        assert executed == 0
        rows = _trace_rows(agent)
        assert len(rows) == 3  # turns 0,1,2 then forced break
        assert {r["conversation_id"] for r in rows} == {rows[0]["conversation_id"]}
        assert [r["conversation_turn"] for r in rows] == [0, 1, 2]
        assert "max_conversation_turns" in (agent._previous_no_action_reason or "")

    def test_run_skill_sees_memory_added_earlier_in_same_response(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        frame = _make_frame()
        agent.frames = [frame]
        agent._current_sandbox_state = SandboxState(
            latest_frame={"state": "NOT_FINISHED"},
            memory_entries=[],
            skill_entries=[],
        )
        skill = agent.skills.add(
            name="read_memory_ids",
            description="Return visible memory entry titles.",
            code="result = [e['title'] for e in state.memory_entries]",
            tags=[],
        )
        response = _response(
            _fc_part(
                "process_memory",
                {
                    "reasoning": "remember a fresh fact",
                    "operation": "add",
                    "title": "fresh-memory",
                    "body": "body",
                    "confidence": 4,
                },
            ),
            _fc_part(
                "run_skill",
                {"reasoning": "read memory", "id": skill.id},
            ),
        )

        actions, records, _logs, terminal, had_fcs = agent._dispatch_response(
            response
        )

        assert actions == 0
        assert terminal is False
        assert had_fcs is True
        assert [r.name for r in records] == ["process_memory", "run_skill"]
        assert records[1].result["result"] == ["fresh-memory"]  # type: ignore[index]

    def test_memory_add_without_confidence_is_error_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        frame = _make_frame()
        agent.frames = [frame]
        response = _response(
            _fc_part(
                "process_memory",
                {
                    "reasoning": "remember without committing to a belief",
                    "operation": "add",
                    "title": "uncalibrated",
                    "body": "body",
                },
            )
        )

        actions, records, _logs, _terminal, _had_fcs = agent._dispatch_response(
            response
        )

        assert actions == 0
        assert records[0].result["success"] is False  # type: ignore[index]
        assert "confidence" in records[0].result["error"]  # type: ignore[index]
        assert agent.memory.all_entries() == []


@pytest.mark.unit
class TestToolResultMarkdown:
    def test_run_skill_code_rendered_as_python_block(self) -> None:
        r = ToolCallRecord(
            name="run_skill",
            args={"id": "navigate"},
            result={
                "success": True,
                "result": {"path_len": 5},
                "stdout": "ok\n",
                "stderr": "",
                "id": "navigate",
                "name": "navigate",
                "version": 2,
                "code": "def step(state):\n    return state['x'] + 1\n",
                "actions_taken_inline": 0,
            },
            actions_taken_inline=0,
        )
        out = format_tool_record_md(r)
        assert out.startswith("### TOOL RESULT")
        assert "name: run_skill" in out
        # Code is lifted out of the JSON skeleton into a fenced python block.
        assert "```python" in out
        assert "def step(state):" in out
        assert "<code [" in out  # pointer left in the JSON skeleton

    def test_handler_failure_lives_in_result_not_error_field(self) -> None:
        # delete of a missing id: handler returns success=false + error INSIDE
        # result; the ToolCallRecord.error field stays None.
        r = ToolCallRecord(
            name="process_memory",
            args={"operation": "delete", "id": "mem_9"},
            result={
                "success": False,
                "operation": "delete",
                "id": "mem_9",
                "deleted": False,
                "error": "no entry with id=mem_9",
            },
            error=None,
        )
        out = format_tool_record_md(r)
        assert "error:\n(none)" in out  # top-level field is None...
        assert '"success": false' in out  # ...but the failure shows in result
        assert "no entry with id=mem_9" in out

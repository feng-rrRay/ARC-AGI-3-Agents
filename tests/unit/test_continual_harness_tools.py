from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from arcengine import ActionInput, FrameData, FrameDataRaw, GameAction, GameState

from agents.templates.continual_harness.models import ToolCallRecord
from agents.templates.continual_harness.tools import (
    ContinualToolRouter,
    FunctionCall,
    extract_function_calls,
    is_action_tool,
    render_tool_results,
)
from agents.templates.continual_harness_agent import ContinualHarness

# --- shared fake-Gemini SDK fixtures (same shape as test_continual_harness.py) ---


class _FakeClient:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.models = self

    def generate_content(self, *_: Any, **__: Any) -> Any:
        return SimpleNamespace(candidates=[])


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


def _fc_part(name: str, args: dict[str, Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        text=None, function_call=SimpleNamespace(name=name, args=args or {})
    )


def _response(*parts: SimpleNamespace) -> SimpleNamespace:
    """Build a response with all parts inside a single candidate/content."""
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason=1,
                content=SimpleNamespace(parts=list(parts)),
            )
        ]
    )


# --- low-level tool helpers ---------------------------------------------------


@pytest.mark.unit
class TestExtractFunctionCalls:
    def test_yields_all_calls_in_order_across_parts(self) -> None:
        resp = _response(
            _fc_part("ACTION1", {"reasoning": "a"}),
            _fc_part("get_recent_trajectory", {"reasoning": "b"}),
            _fc_part("ACTION2", {"reasoning": "c"}),
        )
        fcs = extract_function_calls(resp)
        assert [fc.name for fc in fcs] == [
            "ACTION1",
            "get_recent_trajectory",
            "ACTION2",
        ]
        assert fcs[0].args == {"reasoning": "a"}

    def test_skips_parts_with_no_function_call(self) -> None:
        resp = _response(
            SimpleNamespace(text="thinking", function_call=None),
            _fc_part("ACTION3"),
        )
        fcs = extract_function_calls(resp)
        assert [fc.name for fc in fcs] == ["ACTION3"]

    def test_empty_response_returns_empty_list(self) -> None:
        assert extract_function_calls(SimpleNamespace()) == []
        assert extract_function_calls(SimpleNamespace(candidates=[])) == []


@pytest.mark.unit
class TestIsActionTool:
    def test_true_for_game_action_names(self) -> None:
        assert is_action_tool("ACTION1") is True
        assert is_action_tool("ACTION7") is True

    def test_false_for_analysis_tools(self) -> None:
        assert is_action_tool("get_recent_trajectory") is False
        assert is_action_tool("nonexistent") is False


@pytest.mark.unit
class TestContinualToolRouter:
    def test_executes_known_tool(self) -> None:
        router = ContinualToolRouter(
            {"echo": lambda args: {"success": True, "got": args.get("x")}}
        )
        rec = router.execute(FunctionCall("echo", {"x": 7}))
        assert rec.name == "echo"
        assert rec.result == {"success": True, "got": 7}
        assert rec.error is None

    def test_unknown_tool_yields_error_record(self) -> None:
        router = ContinualToolRouter({})
        rec = router.execute(FunctionCall("nope", {}))
        assert rec.result is None
        assert rec.error is not None
        assert "unknown tool" in rec.error

    def test_handler_exception_is_captured(self) -> None:
        def boom(_args: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("explode")

        router = ContinualToolRouter({"boom": boom})
        rec = router.execute(FunctionCall("boom", {}))
        assert rec.result is None
        assert rec.error is not None
        assert "explode" in rec.error


@pytest.mark.unit
class TestRenderToolResults:
    def test_starts_with_header_then_valid_json(self) -> None:
        out = render_tool_results(
            [
                ToolCallRecord(
                    name="get_recent_trajectory",
                    args={"reasoning": "why", "limit": 3},
                    result={"count": 0, "history": "none"},
                )
            ]
        )
        assert out.startswith("## TOOL RESULTS\n")
        body = out[len("## TOOL RESULTS\n") :]
        parsed = json.loads(body)
        assert isinstance(parsed, list) and len(parsed) == 1
        assert parsed[0]["name"] == "get_recent_trajectory"
        assert parsed[0]["error"] is None


# --- orchestrator integration tests ------------------------------------------


class _NoopEnv:
    """Fake arc_env — Agent.__init__ only stores it; we never call .step() in unit tests."""

    def __init__(self) -> None:
        self.observation_space: Any = FrameDataRaw(
            game_id="orch-test",
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=1,
            action_input=ActionInput(),
            available_actions=[1, 2, 3, 6],
        )
        self.observation_space.frame = [np.array([[0]], dtype=np.int8)]


class _ScriptedVLM:
    """Minimal VLM stand-in: returns scripted responses and records calls."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[Any, str]] = []
        self.set_tools_calls: list[list[dict[str, Any]] | None] = []

    def set_tools(self, tools: list[dict[str, Any]] | None) -> None:
        self.set_tools_calls.append(list(tools) if tools else None)

    def get_query(self, payload: Any, prompt: str, module_name: str = "x") -> Any:
        self.calls.append((payload, prompt))
        return self._responses.pop(0)

    def extract_usage(self, response: Any) -> dict[str, int | None] | None:
        return None


def _make_frame(available: list[int] | None = None) -> FrameData:
    return FrameData(
        game_id="orch-test",
        frame=[[[0, 0], [0, 0]]],
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        win_levels=1,
        action_input=ActionInput(),
        available_actions=available if available is not None else [1, 2, 3, 6],
    )


def _make_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ContinualHarness:
    _install_fake_gemini(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("RUN_LOG_PATH", str(tmp_path / "run.log"))
    env = _NoopEnv()
    return ContinualHarness(
        card_id="card",
        game_id="orch-test",
        agent_name="orchagent",
        ROOT_URL="https://example.com",
        record=False,
        arc_env=env,  # type: ignore[arg-type]
    )


def _read_trajectory(agent: ContinualHarness) -> list[dict[str, Any]]:
    if not agent.trajectory.path.exists():
        return []
    return [
        json.loads(line)
        for line in agent.trajectory.path.read_text().splitlines()
        if line.strip()
    ]


@pytest.mark.unit
class TestOrchestrator:
    def test_runs_analysis_then_commits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        scripted = _ScriptedVLM(
            [
                _response(
                    _fc_part(
                        "get_recent_trajectory", {"reasoning": "check", "limit": 5}
                    )
                ),
                _response(_fc_part("ACTION3", {"reasoning": "go left"})),
            ]
        )
        agent.vlm = scripted  # type: ignore[assignment]
        frame = _make_frame([1, 2, 3])

        chosen = agent.choose_action([frame], frame)

        assert chosen is GameAction.ACTION3
        # Two VLM calls (analysis round + commit round).
        assert len(scripted.calls) == 2
        # The second prompt carries the tool-results block.
        round2_prompt = scripted.calls[1][1]
        assert "## TOOL RESULTS" in round2_prompt
        # Invariant: tool results MUST land above `# TURN:` so the action cue stays
        # the last thing the model reads. (Pre-fix bug shipped them below TURN.)
        assert round2_prompt.index("## TOOL RESULTS") < round2_prompt.index("# TURN:")
        # Trajectory has exactly one row, with the analysis tool recorded.
        rows = _read_trajectory(agent)
        assert len(rows) == 1
        assert rows[0]["chosen_action"] == "ACTION3"
        assert [c["name"] for c in rows[0]["tool_calls"]] == ["get_recent_trajectory"]
        # `game_id` is filtered from chosen_action_data so history lines aren't polluted.
        assert "game_id" not in rows[0]["chosen_action_data"]

    def test_raises_after_max_rounds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        # Three rounds in a row that never commit (analysis on rounds 1-2; round 3
        # is the forced-action round but the model returns an unavailable action,
        # so commit fails and RuntimeError fires).
        scripted = _ScriptedVLM(
            [
                _response(_fc_part("get_recent_trajectory", {"reasoning": "1"})),
                _response(_fc_part("get_recent_trajectory", {"reasoning": "2"})),
                _response(_fc_part("ACTION7", {"reasoning": "unavailable!"})),
            ]
        )
        agent.vlm = scripted  # type: ignore[assignment]
        frame = _make_frame([1, 2, 3])

        with pytest.raises(RuntimeError, match="could not select an action"):
            agent.choose_action([frame], frame)

        rows = _read_trajectory(agent)
        assert len(rows) == 1
        assert rows[0]["chosen_action"] is None
        # The final round was forced to action_only — assert via set_tools history.
        final_call_names = [t["name"] for t in scripted.set_tools_calls[-1] or []]
        assert "get_recent_trajectory" not in final_call_names

    def test_strips_analysis_tools_when_budget_exhausted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        monkeypatch.setattr(agent, "MAX_ANALYSIS_CALLS_PER_STEP", 2)
        # Round 1: 3 parallel analysis calls (budget 2 → 2 succeed, 1 refused).
        # Round 2: must be action-only since budget=0.
        scripted = _ScriptedVLM(
            [
                _response(
                    _fc_part("get_recent_trajectory", {"reasoning": "a"}),
                    _fc_part("get_recent_trajectory", {"reasoning": "b"}),
                    _fc_part("get_recent_trajectory", {"reasoning": "c"}),
                ),
                _response(_fc_part("ACTION1", {"reasoning": "commit"})),
            ]
        )
        agent.vlm = scripted  # type: ignore[assignment]
        frame = _make_frame([1, 2, 3])

        chosen = agent.choose_action([frame], frame)

        assert chosen is GameAction.ACTION1
        rows = _read_trajectory(agent)
        assert len(rows) == 1
        # 3 tool_calls were attempted: 2 succeeded, 1 refused (budget exhausted).
        tcs = rows[0]["tool_calls"]
        assert len(tcs) == 3
        errored = [tc for tc in tcs if tc.get("error")]
        assert len(errored) == 1
        assert "budget exhausted" in errored[0]["error"]
        # The second set_tools call shed the analysis tool.
        second_call_names = [t["name"] for t in scripted.set_tools_calls[1] or []]
        assert "get_recent_trajectory" not in second_call_names

    def test_forces_action_tools_on_final_round(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        monkeypatch.setattr(agent, "MAX_TOOL_ROUNDS", 2)
        scripted = _ScriptedVLM(
            [
                _response(_fc_part("get_recent_trajectory", {"reasoning": "look"})),
                _response(_fc_part("ACTION3", {"reasoning": "act"})),
            ]
        )
        agent.vlm = scripted  # type: ignore[assignment]
        frame = _make_frame([1, 2, 3])

        chosen = agent.choose_action([frame], frame)

        assert chosen is GameAction.ACTION3
        # First set_tools is the full list; second is action-only (final round).
        assert len(scripted.set_tools_calls) == 2
        first_names = [t["name"] for t in scripted.set_tools_calls[0] or []]
        second_names = [t["name"] for t in scripted.set_tools_calls[1] or []]
        assert "get_recent_trajectory" in first_names
        assert "get_recent_trajectory" not in second_names

    def test_raises_when_final_round_returns_invalid_action(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        # Force "this is the only round, also the final round".
        monkeypatch.setattr(agent, "MAX_TOOL_ROUNDS", 1)
        # ACTION7 is not in available_actions=[1] so commit fails.
        scripted = _ScriptedVLM(
            [_response(_fc_part("ACTION7", {"reasoning": "unavailable"}))]
        )
        agent.vlm = scripted  # type: ignore[assignment]
        frame = _make_frame([1])

        with pytest.raises(RuntimeError, match="could not select an action"):
            agent.choose_action([frame], frame)

        rows = _read_trajectory(agent)
        assert len(rows) == 1
        assert rows[0]["chosen_action"] is None
        # Final (only) round was action-only — invariant fires regardless of which set_tools
        # call shed the analysis tool. The LAST set_tools call must be action-only.
        names = [t["name"] for t in scripted.set_tools_calls[-1] or []]
        assert "get_recent_trajectory" not in names

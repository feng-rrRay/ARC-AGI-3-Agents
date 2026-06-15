from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from arcengine import ActionInput, FrameData, FrameDataRaw, GameAction, GameState

import agents.templates.continual_harness_agent as harness_module
from agents.templates.continual_harness.models import ToolCallRecord
from agents.templates.continual_harness.subagents import DEFAULT_SUBAGENT_ALLOWED_TOOLS
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
    monkeypatch.setenv("RUN_MEMORY_PATH", str(tmp_path / "memory.json"))
    monkeypatch.setenv("RUN_SKILLS_PATH", str(tmp_path / "skills.json"))
    monkeypatch.setenv("RUN_SUBAGENTS_PATH", str(tmp_path / "subagents.json"))
    monkeypatch.setenv("RUN_PROMPT_PATH", str(tmp_path / "prompt.current.md"))
    monkeypatch.setenv(
        "RUN_PROMPT_EVOLUTION_PATH", str(tmp_path / "prompt_evolution.jsonl")
    )
    # Bootstrap env vars are intentionally NOT deleted here so tests can opt in
    # by setting them before calling _make_agent. monkeypatch isolates env per
    # test, so leakage across tests is impossible.
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


# --- memory-related orchestrator tests ---------------------------------------


def _make_agent_with_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seed_entries: list[dict[str, Any]] | None = None,
) -> tuple[ContinualHarness, Path]:
    """Same as _make_agent but with --bootstrap-memory simulated via env var.

    Optionally pre-seeds the memory file before the agent loads it (mirrors a
    user passing a previously-written file as bootstrap).
    """
    memory_path = tmp_path / "memory.json"
    if seed_entries is not None:
        memory_path.write_text(
            json.dumps({"next_id": len(seed_entries) + 1, "entries": seed_entries})
        )
    monkeypatch.setenv("CONTINUAL_HARNESS_BOOTSTRAP_MEMORY", str(memory_path))
    return _make_agent(tmp_path, monkeypatch), memory_path


@pytest.mark.unit
class TestRunLocalMemory:
    def test_empty_overview_in_prompt_without_bootstrap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CONTINUAL_HARNESS_BOOTSTRAP_MEMORY", raising=False)
        agent = _make_agent(tmp_path, monkeypatch)
        scripted = _ScriptedVLM([_response(_fc_part("ACTION1", {"reasoning": "go"}))])
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        assert agent.memory.path == tmp_path / "memory.json"
        assert "## LONG-TERM MEMORY (0 entries)" in scripted.calls[0][1]

    def test_process_memory_tool_present_without_bootstrap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CONTINUAL_HARNESS_BOOTSTRAP_MEMORY", raising=False)
        agent = _make_agent(tmp_path, monkeypatch)
        scripted = _ScriptedVLM([_response(_fc_part("ACTION1", {"reasoning": "go"}))])
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        tool_names = [t["name"] for t in scripted.set_tools_calls[0] or []]
        assert "process_memory" in tool_names

    def test_run_local_memory_is_written_on_mutation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CONTINUAL_HARNESS_BOOTSTRAP_MEMORY", raising=False)
        agent = _make_agent(tmp_path, monkeypatch)
        scripted = _ScriptedVLM(
            [
                _response(
                    _fc_part(
                        "process_memory",
                        {
                            "reasoning": "save",
                            "operation": "add",
                            "title": "run local",
                            "body": "available without bootstrap",
                            "confidence": 3,
                        },
                    )
                ),
                _response(_fc_part("ACTION1", {"reasoning": "commit"})),
            ]
        )
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        stored = json.loads((tmp_path / "memory.json").read_text())
        assert stored["entries"][0]["title"] == "run local"


@pytest.mark.unit
class TestMemoryOn:
    def test_empty_overview_present_in_first_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, _ = _make_agent_with_memory(tmp_path, monkeypatch)
        scripted = _ScriptedVLM([_response(_fc_part("ACTION1", {"reasoning": "go"}))])
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        prompt = scripted.calls[0][1]
        assert "## LONG-TERM MEMORY (0 entries)" in prompt
        assert prompt.index("## LONG-TERM MEMORY") < prompt.index("# TURN:")

    def test_process_memory_tool_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, _ = _make_agent_with_memory(tmp_path, monkeypatch)
        scripted = _ScriptedVLM([_response(_fc_part("ACTION1", {"reasoning": "go"}))])
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        tool_names = [t["name"] for t in scripted.set_tools_calls[0] or []]
        assert "process_memory" in tool_names

    def test_add_returns_id_and_overview_updates_next_round(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, memory_path = _make_agent_with_memory(tmp_path, monkeypatch)
        scripted = _ScriptedVLM(
            [
                _response(
                    _fc_part(
                        "process_memory",
                        {
                            "reasoning": "save",
                            "operation": "add",
                            "title": "Orange block is the player",
                            "body": "After ACTION1 we observed the orange block move up; "
                            "the white cross is just a target.",
                            "tags": ["player_identity"],
                            "confidence": 4,
                        },
                    )
                ),
                _response(_fc_part("ACTION1", {"reasoning": "go"})),
            ]
        )
        agent.vlm = scripted  # type: ignore[assignment]

        chosen = agent.choose_action([_make_frame([1])], _make_frame([1]))

        assert chosen is GameAction.ACTION1
        # Round 2's prompt overview reflects the just-added entry.
        round2_prompt = scripted.calls[1][1]
        assert "## LONG-TERM MEMORY (1 entries)" in round2_prompt
        assert (
            "[mem_001][c4] Orange block is the player (player_identity)"
            in round2_prompt
        )
        # Body stays out of the auto-injected overview block (it only renders
        # title + tags). The tool-result echo below carries the body since the
        # model just supplied it as an arg — that's expected, not a leak.
        overview_start = round2_prompt.index("## LONG-TERM MEMORY")
        overview_end = round2_prompt.index("##", overview_start + 1)
        assert (
            "white cross is just a target"
            not in round2_prompt[overview_start:overview_end]
        )
        # Persistence: file on disk has the entry.
        stored = json.loads(memory_path.read_text())
        assert stored["entries"][0]["title"] == "Orange block is the player"

    def test_delete_round_trip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, _ = _make_agent_with_memory(
            tmp_path,
            monkeypatch,
            seed_entries=[
                {
                    "id": "mem_001",
                    "game_id": "g",
                    "title": "seed",
                    "body": "seed body",
                    "tags": [],
                    "created_at": "",
                    "updated_at": "",
                }
            ],
        )
        scripted = _ScriptedVLM(
            [
                _response(
                    _fc_part(
                        "process_memory",
                        {
                            "reasoning": "wrong fact",
                            "operation": "delete",
                            "id": "mem_001",
                        },
                    )
                ),
                _response(_fc_part("ACTION1", {"reasoning": "commit"})),
            ]
        )
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        round2_prompt = scripted.calls[1][1]
        assert "## LONG-TERM MEMORY (0 entries)" in round2_prompt

    def test_edit_round_trip_changes_overview_title(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, _ = _make_agent_with_memory(
            tmp_path,
            monkeypatch,
            seed_entries=[
                {
                    "id": "mem_001",
                    "game_id": "g",
                    "title": "old title",
                    "body": "old body",
                    "tags": [],
                    "created_at": "",
                    "updated_at": "",
                }
            ],
        )
        scripted = _ScriptedVLM(
            [
                _response(
                    _fc_part(
                        "process_memory",
                        {
                            "reasoning": "refine",
                            "operation": "edit",
                            "id": "mem_001",
                            "title": "new title",
                        },
                    )
                ),
                _response(_fc_part("ACTION1", {"reasoning": "commit"})),
            ]
        )
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        round2_prompt = scripted.calls[1][1]
        # Seeded entry predates the confidence field → defaults to 3 in the index.
        assert "[mem_001][c3] new title" in round2_prompt
        assert "old title" not in round2_prompt

    def test_search_returns_full_body(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, _ = _make_agent_with_memory(
            tmp_path,
            monkeypatch,
            seed_entries=[
                {
                    "id": "mem_001",
                    "game_id": "g",
                    "title": "wall mechanics",
                    "body": "Walls of color 3 block movement",
                    "tags": ["mechanics"],
                    "created_at": "",
                    "updated_at": "",
                }
            ],
        )
        scripted = _ScriptedVLM(
            [
                _response(
                    _fc_part(
                        "process_memory",
                        {
                            "reasoning": "recall",
                            "operation": "search",
                            "query": "wall",
                        },
                    )
                ),
                _response(_fc_part("ACTION1", {"reasoning": "commit"})),
            ]
        )
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        round2_prompt = scripted.calls[1][1]
        # The tool-result block (above # TURN:) carries the full body text.
        assert "Walls of color 3 block movement" in round2_prompt

    def test_unknown_operation_is_error_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, _ = _make_agent_with_memory(tmp_path, monkeypatch)
        scripted = _ScriptedVLM(
            [
                _response(
                    _fc_part(
                        "process_memory",
                        {"reasoning": "oops", "operation": "purge"},
                    )
                ),
                _response(_fc_part("ACTION1", {"reasoning": "commit"})),
            ]
        )
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        round2_prompt = scripted.calls[1][1]
        assert "unknown operation" in round2_prompt

    def test_bootstrap_file_is_loaded_at_init(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, _ = _make_agent_with_memory(
            tmp_path,
            monkeypatch,
            seed_entries=[
                {
                    "id": "mem_001",
                    "game_id": "g",
                    "title": "first preloaded",
                    "body": "b",
                    "tags": [],
                    "created_at": "",
                    "updated_at": "",
                },
                {
                    "id": "mem_002",
                    "game_id": "g",
                    "title": "second preloaded",
                    "body": "b",
                    "tags": [],
                    "created_at": "",
                    "updated_at": "",
                },
            ],
        )
        scripted = _ScriptedVLM([_response(_fc_part("ACTION1", {"reasoning": "go"}))])
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        prompt = scripted.calls[0][1]
        assert "## LONG-TERM MEMORY (2 entries)" in prompt
        # Bootstrap entries written before the confidence field default to c3.
        assert "[mem_001][c3] first preloaded" in prompt
        assert "[mem_002][c3] second preloaded" in prompt

    def test_bootstrap_file_is_written_back_on_mutation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, memory_path = _make_agent_with_memory(tmp_path, monkeypatch)
        scripted = _ScriptedVLM(
            [
                _response(
                    _fc_part(
                        "process_memory",
                        {
                            "reasoning": "save",
                            "operation": "add",
                            "title": "persisted",
                            "body": "persisted body",
                            "confidence": 5,
                        },
                    )
                ),
                _response(_fc_part("ACTION1", {"reasoning": "go"})),
            ]
        )
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        # Open a SECOND store on the same path; entry must be visible.
        from agents.templates.continual_harness.memory import MemoryStore

        reopened = MemoryStore(memory_path, game_id="orch-test")
        entries = reopened.all_entries()
        assert len(entries) == 1
        assert entries[0].title == "persisted"
        assert entries[0].confidence == 5


# --- skill / sandbox orchestrator tests --------------------------------------


@pytest.mark.unit
class TestSkillsAlwaysOn:
    def test_skill_tools_present_and_run_code_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # run_code is intentionally disabled — it must not appear in the
        # orchestrator's tool set even though RUN_CODE_TOOL still exists.
        agent = _make_agent(tmp_path, monkeypatch)
        scripted = _ScriptedVLM([_response(_fc_part("ACTION1", {"reasoning": "go"}))])
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        tool_names = [t["name"] for t in scripted.set_tools_calls[0] or []]
        assert "process_skill" in tool_names
        assert "run_skill" in tool_names
        assert "run_code" not in tool_names

    def test_skills_overview_in_first_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        scripted = _ScriptedVLM([_response(_fc_part("ACTION1", {"reasoning": "go"}))])
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        prompt = scripted.calls[0][1]
        assert "## SKILLS (0 saved)" in prompt
        assert prompt.index("## SKILLS") < prompt.index("# TURN:")

    def test_run_local_skills_written_on_mutation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        scripted = _ScriptedVLM(
            [
                _response(
                    _fc_part(
                        "process_skill",
                        {
                            "reasoning": "save",
                            "operation": "add",
                            "name": "find_player",
                            "description": "Locate player.",
                            "code": "result = (0, 0)",
                            "tags": ["geo"],
                        },
                    )
                ),
                _response(_fc_part("ACTION1", {"reasoning": "commit"})),
            ]
        )
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        stored = json.loads((tmp_path / "skills.json").read_text())
        assert stored["entries"][0]["name"] == "find_player"

    def test_bootstrap_skills_takes_precedence_over_run_skills_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bootstrap_path = tmp_path / "bootstrap.skills.json"
        monkeypatch.setenv("CONTINUAL_HARNESS_BOOTSTRAP_SKILLS", str(bootstrap_path))
        agent = _make_agent(tmp_path, monkeypatch)
        assert agent.skills.path == bootstrap_path
        # The run-local path is NOT used when bootstrap is set.
        assert not (tmp_path / "skills.json").exists()


@pytest.mark.unit
class TestSkillsHandlers:
    def test_add_updates_overview_next_round(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        scripted = _ScriptedVLM(
            [
                _response(
                    _fc_part(
                        "process_skill",
                        {
                            "reasoning": "save",
                            "operation": "add",
                            "name": "find_player",
                            "description": "Locate the player cell.",
                            "code": "result = (0, 0)",
                            "tags": ["geo"],
                        },
                    )
                ),
                _response(_fc_part("ACTION1", {"reasoning": "go"})),
            ]
        )
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        round2_prompt = scripted.calls[1][1]
        assert "## SKILLS (1 saved)" in round2_prompt
        assert "[skill_001] find_player (geo)" in round2_prompt

    def test_run_skill_executes_and_returns_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Pre-seed a skill that doubles its `x` arg.
        skills_path = tmp_path / "skills.json"
        skills_path.write_text(
            json.dumps(
                {
                    "next_id": 2,
                    "entries": [
                        {
                            "id": "skill_001",
                            "game_id": "g",
                            "name": "doubler",
                            "description": "Doubles args['x'].",
                            "code": "result = args['x'] * 2",
                            "tags": [],
                            "version": 1,
                            "created_at": "",
                            "updated_at": "",
                        }
                    ],
                }
            )
        )
        agent = _make_agent(tmp_path, monkeypatch)
        scripted = _ScriptedVLM(
            [
                _response(
                    _fc_part(
                        "run_skill",
                        {
                            "reasoning": "compute",
                            "id": "skill_001",
                            "args": {"x": 7},
                        },
                    )
                ),
                _response(_fc_part("ACTION1", {"reasoning": "commit"})),
            ]
        )
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        round2_prompt = scripted.calls[1][1]
        # The tool-result block above # TURN: carries the sandbox result.
        assert '"result": 14' in round2_prompt

    def test_run_code_is_rejected_as_unknown_tool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # run_code is currently disabled — even if the model hallucinates the
        # name, the router must surface "unknown tool" rather than executing.
        agent = _make_agent(tmp_path, monkeypatch)
        scripted = _ScriptedVLM(
            [
                _response(
                    _fc_part(
                        "run_code",
                        {
                            "reasoning": "compute",
                            "code": "result = sum(range(10))",
                        },
                    )
                ),
                _response(_fc_part("ACTION1", {"reasoning": "commit"})),
            ]
        )
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        round2_prompt = scripted.calls[1][1]
        assert "unknown tool: run_code" in round2_prompt

    def test_run_skill_unknown_id_returns_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        scripted = _ScriptedVLM(
            [
                _response(
                    _fc_part(
                        "run_skill",
                        {"reasoning": "try", "id": "skill_999"},
                    )
                ),
                _response(_fc_part("ACTION1", {"reasoning": "commit"})),
            ]
        )
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        round2_prompt = scripted.calls[1][1]
        assert "no skill with id=skill_999" in round2_prompt

    def test_skill_call_counts_against_analysis_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        monkeypatch.setattr(agent, "MAX_ANALYSIS_CALLS_PER_STEP", 2)
        # Round 1: 3 parallel get_recent_trajectory calls (budget 2 → 2 are
        # executed, the 3rd is refused with "budget exhausted").
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

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        rows = _read_trajectory(agent)
        tcs = rows[0]["tool_calls"]
        assert len(tcs) == 3
        errored = [tc for tc in tcs if tc.get("error")]
        assert len(errored) == 1
        assert "budget exhausted" in errored[0]["error"]


@pytest.mark.unit
class TestSubagentHandlers:
    def test_process_subagent_defaults_allowed_tools_when_omitted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        scripted = _ScriptedVLM(
            [
                _response(
                    _fc_part(
                        "process_subagent",
                        {
                            "reasoning": "make helper",
                            "operation": "add",
                            "name": "summarizer",
                            "description": "Summarize current state.",
                            "system_instructions": "Return a concise summary.",
                        },
                    )
                ),
                _response(_fc_part("ACTION1", {"reasoning": "commit"})),
            ]
        )
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        stored = json.loads((tmp_path / "subagents.json").read_text())
        assert stored["entries"][0]["allowed_tools"] == list(
            DEFAULT_SUBAGENT_ALLOWED_TOOLS
        )

    def test_run_subagent_return_includes_per_round_steps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "subagents.json").write_text(
            json.dumps(
                {
                    "next_id": 2,
                    "entries": [
                        {
                            "id": "subagent_001",
                            "game_id": "orch-test",
                            "name": "summarizer",
                            "description": "Summarize.",
                            "system_instructions": "Return the answer.",
                            "allowed_tools": [],
                            "tags": [],
                            "version": 1,
                            "created_at": "",
                            "updated_at": "",
                        }
                    ],
                }
            )
        )
        agent = _make_agent(tmp_path, monkeypatch)

        class _ReturningSubVLM:
            def __init__(self, *_: Any, **__: Any) -> None:
                self.tools: list[dict[str, Any]] = []

            def set_tools(self, tools: list[dict[str, Any]] | None) -> None:
                self.tools = list(tools or [])

            def get_query(
                self, payload: Any, prompt: str, module_name: str = "x"
            ) -> Any:
                return _response(
                    _fc_part(
                        "subagent_return",
                        {
                            "reasoning": "done",
                            "answer": "summary",
                            "status": "success",
                        },
                    )
                )

            def extract_usage(self, response: Any) -> dict[str, int | None] | None:
                return None

        monkeypatch.setattr(harness_module, "VLM", _ReturningSubVLM)
        scripted = _ScriptedVLM(
            [
                _response(
                    _fc_part(
                        "run_subagent",
                        {
                            "reasoning": "ask helper",
                            "id": "subagent_001",
                            "task": "summarize",
                        },
                    )
                ),
                _response(_fc_part("ACTION1", {"reasoning": "commit"})),
            ]
        )
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        rows = _read_trajectory(agent)
        result = rows[0]["tool_calls"][0]["result"]
        assert result["success"] is True
        assert result["result"]["answer"] == "summary"
        assert result["steps"][0]["inner_round"] == 1
        assert result["steps"][0]["subagent_return"]["answer"] == "summary"
        assert result["steps"][0]["tool_calls"] == []

    def test_run_subagent_usage_counts_in_scoped_totals_and_trace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "subagents.json").write_text(
            json.dumps(
                {
                    "next_id": 2,
                    "entries": [
                        {
                            "id": "subagent_001",
                            "game_id": "orch-test",
                            "name": "summarizer",
                            "description": "Summarize.",
                            "system_instructions": "Return the answer.",
                            "handler_type": "one_step",
                            "max_turns": 1,
                            "allowed_tools": [],
                            "tags": [],
                            "version": 1,
                            "created_at": "",
                            "updated_at": "",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        agent = _make_agent(tmp_path, monkeypatch)

        class _UsageSubVLM:
            def __init__(self, *_: Any, **__: Any) -> None:
                pass

            def set_tools(self, tools: list[dict[str, Any]] | None) -> None:
                return None

            def get_query(
                self, payload: Any, prompt: str, module_name: str = "x"
            ) -> Any:
                return _response(
                    _fc_part(
                        "subagent_return",
                        {
                            "reasoning": "done",
                            "answer": "summary",
                            "status": "success",
                        },
                    )
                )

            def extract_usage(self, response: Any) -> dict[str, int | None] | None:
                return {
                    "prompt": 100,
                    "output": 20,
                    "thoughts": 5,
                    "total": 125,
                    "cached": 0,
                }

        monkeypatch.setattr(harness_module, "VLM", _UsageSubVLM)
        frame = _make_frame([1])
        agent.frames = [frame]
        agent._current_latest_frame = frame
        agent._current_images = []
        agent._current_outer_round = 7

        record = agent.tool_router.execute(
            FunctionCall(
                "run_subagent",
                {
                    "reasoning": "ask helper",
                    "id": "subagent_001",
                    "task": "summarize",
                },
            )
        )

        assert record.result["success"] is True
        assert agent.total_calls == 1
        assert agent.total_tokens == 125
        assert agent.usage_by_scope["subagent"]["calls"] == 1
        assert agent.usage_by_scope["subagent"]["total_tokens"] == 125
        assert agent.usage_by_scope["subagent"]["priced_calls"] == 1

        trace_rows = [
            json.loads(line)
            for line in (tmp_path / "run.trace.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert trace_rows[-1]["tools_exposed"] == "subagent"
        assert trace_rows[-1]["usage_scope"] == "subagent"
        assert trace_rows[-1]["usage_accounted"] is True
        assert trace_rows[-1]["usage"]["total"] == 125
        assert trace_rows[-1]["usage_cost"]["current_usd"] > 0

    def test_run_subagent_images_match_prompt_rendered_grids(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "subagents.json").write_text(
            json.dumps(
                {
                    "next_id": 2,
                    "entries": [
                        {
                            "id": "subagent_001",
                            "game_id": "orch-test",
                            "name": "summarizer",
                            "description": "Summarize.",
                            "system_instructions": "Return text only.",
                            "handler_type": "one_step",
                            "max_turns": 1,
                            "allowed_tools": [],
                            "tags": [],
                            "version": 1,
                            "created_at": "",
                            "updated_at": "",
                        }
                    ],
                }
            )
        )
        agent = _make_agent(tmp_path, monkeypatch)

        class _CapturingSubVLM:
            instances: list["_CapturingSubVLM"] = []

            def __init__(self, *_: Any, **__: Any) -> None:
                self.calls: list[tuple[Any, str]] = []
                self.instances.append(self)

            def set_tools(self, tools: list[dict[str, Any]] | None) -> None:
                return None

            def get_query(
                self, payload: Any, prompt: str, module_name: str = "x"
            ) -> Any:
                self.calls.append((payload, prompt))
                return _response(
                    SimpleNamespace(text="concise summary", function_call=None)
                )

            def extract_usage(self, response: Any) -> dict[str, int | None] | None:
                return None

        monkeypatch.setattr(harness_module, "VLM", _CapturingSubVLM)
        agent._current_latest_frame = FrameData(
            game_id="orch-test",
            frame=[
                [[0, 0], [0, 0]],
                [[9, 0], [0, 0]],
                [[9, 9], [0, 0]],
                [[0, 0], [0, 0]],
            ],
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=1,
            action_input=ActionInput(),
            available_actions=[1],
        )
        agent._current_images = []
        agent._current_outer_round = 1

        record = agent.tool_router.execute(
            FunctionCall(
                "run_subagent",
                {
                    "reasoning": "ask helper",
                    "id": "subagent_001",
                    "task": "summarize",
                },
            )
        )

        assert record.result["success"] is True
        payload, prompt = _CapturingSubVLM.instances[0].calls[0]
        assert isinstance(payload, list)
        assert len(payload) == 3
        assert [img.size for img in payload] == [(2, 2), (2, 2), (2, 2)]
        assert "current grid (latest_frame.frame[-1]):" in prompt
        assert "SELECTED TRANSIENT KEYFRAMES: frame indices [1, 2]" in prompt

        trace_rows = [
            json.loads(line)
            for line in (tmp_path / "run.trace.jsonl").read_text().splitlines()
            if line.strip()
        ]
        image_rows = trace_rows[-1]["input"]["images"]
        assert trace_rows[-1]["input"]["images_attached_count"] == 3
        assert [row["label"] for row in image_rows] == [
            "current_state_frame",
            "current_frame_transient_1",
            "current_frame_transient_2",
        ]

    def test_one_step_subagent_auto_returns_text_response(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "subagents.json").write_text(
            json.dumps(
                {
                    "next_id": 2,
                    "entries": [
                        {
                            "id": "subagent_001",
                            "game_id": "orch-test",
                            "name": "summarizer",
                            "description": "Summarize.",
                            "system_instructions": "Return text only.",
                            "handler_type": "one_step",
                            "max_turns": 1,
                            "allowed_tools": [],
                            "tags": [],
                            "version": 1,
                            "created_at": "",
                            "updated_at": "",
                        }
                    ],
                }
            )
        )
        agent = _make_agent(tmp_path, monkeypatch)

        class _TextSubVLM:
            def __init__(self, *_: Any, **__: Any) -> None:
                pass

            def set_tools(self, tools: list[dict[str, Any]] | None) -> None:
                return None

            def get_query(
                self, payload: Any, prompt: str, module_name: str = "x"
            ) -> Any:
                return _response(
                    SimpleNamespace(text="concise summary", function_call=None)
                )

            def extract_usage(self, response: Any) -> dict[str, int | None] | None:
                return None

        monkeypatch.setattr(harness_module, "VLM", _TextSubVLM)
        agent._current_latest_frame = _make_frame([1])
        agent._current_images = []
        agent._current_outer_round = 1

        record = agent.tool_router.execute(
            FunctionCall(
                "run_subagent",
                {
                    "reasoning": "ask helper",
                    "id": "subagent_001",
                    "task": "summarize",
                },
            )
        )

        result = record.result
        assert result["success"] is True
        assert result["error"] is None
        assert result["result"]["answer"] == "concise summary"
        assert result["result"]["reasoning"] == "one_step auto-return"

    def test_one_step_subagent_preserves_vlm_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "subagents.json").write_text(
            json.dumps(
                {
                    "next_id": 2,
                    "entries": [
                        {
                            "id": "subagent_001",
                            "game_id": "orch-test",
                            "name": "summarizer",
                            "description": "Summarize.",
                            "system_instructions": "Return text only.",
                            "handler_type": "one_step",
                            "max_turns": 1,
                            "allowed_tools": [],
                            "tags": [],
                            "version": 1,
                            "created_at": "",
                            "updated_at": "",
                        }
                    ],
                }
            )
        )
        agent = _make_agent(tmp_path, monkeypatch)

        class _FailingSubVLM:
            def __init__(self, *_: Any, **__: Any) -> None:
                pass

            def set_tools(self, tools: list[dict[str, Any]] | None) -> None:
                return None

            def get_query(
                self, payload: Any, prompt: str, module_name: str = "x"
            ) -> Any:
                raise RuntimeError("quota exhausted")

            def extract_usage(self, response: Any) -> dict[str, int | None] | None:
                return None

        monkeypatch.setattr(harness_module, "VLM", _FailingSubVLM)
        agent._current_latest_frame = _make_frame([1])
        agent._current_images = []
        agent._current_outer_round = 1

        record = agent.tool_router.execute(
            FunctionCall(
                "run_subagent",
                {
                    "reasoning": "ask helper",
                    "id": "subagent_001",
                    "task": "summarize",
                },
            )
        )

        result = record.result
        assert result["success"] is False
        assert result["result"] is None
        assert "quota exhausted" in result["error"]
        assert result["steps"][0]["error"] == result["error"]

    def test_run_subagent_forces_failure_return_on_max_rounds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "subagents.json").write_text(
            json.dumps(
                {
                    "next_id": 2,
                    "entries": [
                        {
                            "id": "subagent_001",
                            "game_id": "orch-test",
                            "name": "looper",
                            "description": "Loops.",
                            "system_instructions": "Keep using tools.",
                            "handler_type": "looping",
                            "max_turns": 2,
                            "allowed_tools": ["get_recent_trajectory"],
                            "tags": [],
                            "version": 1,
                            "created_at": "",
                            "updated_at": "",
                        }
                    ],
                }
            )
        )
        agent = _make_agent(tmp_path, monkeypatch)

        class _LoopingSubVLM:
            def __init__(self, *_: Any, **__: Any) -> None:
                self.responses = [
                    _response(
                        _fc_part(
                            "get_recent_trajectory",
                            {"reasoning": "read 1", "limit": 1},
                        )
                    ),
                    _response(
                        _fc_part(
                            "get_recent_trajectory",
                            {"reasoning": "read 2", "limit": 2},
                        )
                    ),
                ]

            def set_tools(self, tools: list[dict[str, Any]] | None) -> None:
                return None

            def get_query(
                self, payload: Any, prompt: str, module_name: str = "x"
            ) -> Any:
                return self.responses.pop(0)

            def extract_usage(self, response: Any) -> dict[str, int | None] | None:
                return None

        monkeypatch.setattr(harness_module, "VLM", _LoopingSubVLM)
        scripted = _ScriptedVLM(
            [
                _response(
                    _fc_part(
                        "run_subagent",
                        {
                            "reasoning": "ask helper",
                            "id": "subagent_001",
                            "task": "loop until forced",
                        },
                    )
                ),
                _response(_fc_part("ACTION1", {"reasoning": "commit"})),
            ]
        )
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1])], _make_frame([1]))

        rows = _read_trajectory(agent)
        result = rows[0]["tool_calls"][0]["result"]
        assert result["success"] is False
        assert result["forced_return"] is True
        assert "exhausted max inner rounds" in result["warning"]
        assert result["result"]["status"] == "failure"
        assert len(result["steps"]) == 2
        assert result["steps"][0]["tool_calls"][0]["name"] == "get_recent_trajectory"
        assert result["steps"][0]["tool_calls"][0]["args"]["limit"] == 1
        assert result["steps"][1]["tool_calls"][0]["args"]["limit"] == 2


@pytest.mark.unit
class TestFinalRoundStripsSkillsToo:
    def test_round_max_strips_all_analysis_tools(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        monkeypatch.setattr(agent, "MAX_TOOL_ROUNDS", 2)
        scripted = _ScriptedVLM(
            [
                _response(
                    _fc_part("run_code", {"reasoning": "peek", "code": "result = 1"})
                ),
                _response(_fc_part("ACTION3", {"reasoning": "go"})),
            ]
        )
        agent.vlm = scripted  # type: ignore[assignment]

        agent.choose_action([_make_frame([1, 2, 3])], _make_frame([1, 2, 3]))

        final_names = [t["name"] for t in scripted.set_tools_calls[-1] or []]
        assert "process_skill" not in final_names
        assert "run_skill" not in final_names
        assert "run_code" not in final_names
        assert "process_memory" not in final_names
        assert "get_recent_trajectory" not in final_names


# --- Fix 1: level-change batch abort + RPC signal ----------------------------


def _patch_take_action_with_levels(
    agent: ContinualHarness,
    levels_per_step: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> list[GameAction]:
    """Replace `take_action` with a stub that walks through scripted level values.

    Each call returns a FrameData with `levels_completed=levels_per_step[i]`,
    same available_actions as the agent's current frame, and NOT_FINISHED
    state. Records calls in the returned list so the test can assert how
    many actions actually fired.
    """
    seen: list[GameAction] = []
    levels = iter(levels_per_step)

    def _stub(action: GameAction) -> FrameData:
        seen.append(action)
        lvl = next(levels)
        return FrameData(
            game_id=agent.game_id,
            frame=[[[0]]],
            state=GameState.NOT_FINISHED,
            levels_completed=lvl,
            win_levels=2,
            action_input=ActionInput(),
            available_actions=[1, 2, 3, 6],
        )

    monkeypatch.setattr(agent, "take_action", _stub)
    return seen


@pytest.mark.unit
class TestBatchAbortsOnLevelChange:
    """Fix 1: queued take_actions stops when levels_completed increments."""

    def test_remaining_steps_skipped_after_level_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        agent.frames = [_make_frame()]
        # Level flips from 0 to 1 on the 2nd action; remaining 2 actions
        # were planned for the old level and must NOT run.
        seen = _patch_take_action_with_levels(agent, [0, 1, 1, 1], monkeypatch)

        executed = agent._dispatch_take_actions(
            {
                "reasoning": "test",
                "actions": [
                    {"name": "ACTION1"},
                    {"name": "ACTION1"},
                    {"name": "ACTION3"},
                    {"name": "ACTION1"},
                ],
            },
            source="vlm",
        )

        assert executed == 2
        assert len(seen) == 2  # only the first two actions actually fired

    def test_batch_runs_to_end_when_level_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        agent.frames = [_make_frame()]
        seen = _patch_take_action_with_levels(agent, [0, 0, 0, 0], monkeypatch)

        executed = agent._dispatch_take_actions(
            {
                "reasoning": "test",
                "actions": [
                    {"name": "ACTION1"},
                    {"name": "ACTION1"},
                    {"name": "ACTION3"},
                    {"name": "ACTION1"},
                ],
            },
            source="vlm",
        )

        assert executed == 4
        assert len(seen) == 4


@pytest.mark.unit
class TestSandboxRpcSignalsLevelChange:
    """Fix 1: the sandbox RPC return value carries `level_changed` so an
    engine-driving skill can react when a mid-batch level transition fires."""

    def test_level_changed_true_when_levels_completed_increments(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        agent.frames = [_make_frame()]
        _patch_take_action_with_levels(agent, [1], monkeypatch)

        resp = agent._sandbox_rpc(
            "take_actions",
            {"reasoning": "skill drives", "actions": [{"name": "ACTION1"}]},
        )

        assert resp["ok"] is True
        assert resp["value"]["level_changed"] is True
        assert resp["value"]["terminal"] is False
        assert resp["value"]["score"] == 1

    def test_level_changed_false_when_level_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(tmp_path, monkeypatch)
        agent.frames = [_make_frame()]
        _patch_take_action_with_levels(agent, [0], monkeypatch)

        resp = agent._sandbox_rpc(
            "take_actions",
            {"reasoning": "skill drives", "actions": [{"name": "ACTION1"}]},
        )

        assert resp["ok"] is True
        assert resp["value"]["level_changed"] is False

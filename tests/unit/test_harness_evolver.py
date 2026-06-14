"""Tests for harness_evolver.py.

Covers the prompt-persistence primitives that now live in harness_evolver
(validation bounds, PromptFile read/write, PromptEvolutionStore append,
active_*_path resolution, build_evolution_prompt), the semantic evolution
triggers on HarnessEvolver (game_over / stagnation), and the JSON-driven
component passes (skills / subagents / memory).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from arcengine import FrameData, GameState

from agents.templates.continual_harness.models import ToolCallRecord, ToolEvidenceRecord
from agents.templates.continual_harness.harness_evolver import (
    PROMPT_MAX_CHARS,
    PROMPT_MIN_CHARS,
    HarnessEvolver,
    PromptEvolutionRecord,
    PromptEvolutionStore,
    PromptFile,
    SUBAGENT_EVOLUTION_PROMPT,
    active_prompt_evolution_path,
    active_prompt_path,
    build_evolution_prompt,
    validate_evolved_prompt,
)


@pytest.mark.unit
class TestValidateEvolvedPrompt:
    def test_empty_rejected(self) -> None:
        ok, err = validate_evolved_prompt("")
        assert ok is False
        assert err == "evolved prompt is empty"

    def test_whitespace_only_rejected(self) -> None:
        ok, err = validate_evolved_prompt("   \n  \t  ")
        assert ok is False
        assert err == "evolved prompt is empty"

    def test_too_short_rejected(self) -> None:
        ok, err = validate_evolved_prompt("hi")
        assert ok is False
        assert err is not None and "too short" in err

    def test_too_long_rejected(self) -> None:
        ok, err = validate_evolved_prompt("x" * (PROMPT_MAX_CHARS + 1))
        assert ok is False
        assert err is not None and "too long" in err

    def test_boundary_min_accepted(self) -> None:
        ok, err = validate_evolved_prompt("a" * PROMPT_MIN_CHARS)
        assert ok is True
        assert err is None

    def test_boundary_max_accepted(self) -> None:
        ok, err = validate_evolved_prompt("a" * PROMPT_MAX_CHARS)
        assert ok is True
        assert err is None

    def test_typical_length_accepted(self) -> None:
        # Lenient validation — no keyword check, length only.
        text = "a" * 1500
        ok, err = validate_evolved_prompt(text)
        assert ok is True
        assert err is None


@pytest.mark.unit
class TestPromptFile:
    def test_baseline_seeded_when_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "prompt.current.md"
        baseline = "x" * 300
        store = PromptFile(path, baseline=baseline)
        assert path.exists()
        assert store.read() == baseline

    def test_baseline_not_overwritten_when_file_exists(self, tmp_path: Path) -> None:
        path = tmp_path / "prompt.current.md"
        path.write_text("preexisting", encoding="utf-8")
        PromptFile(path, baseline="other baseline")
        assert path.read_text(encoding="utf-8") == "preexisting"

    def test_write_then_read_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "prompt.current.md"
        store = PromptFile(path, baseline="b" * 250)
        store.write("hello evolved")
        assert store.read() == "hello evolved"

    def test_write_is_atomic(self, tmp_path: Path) -> None:
        # tempfile.replace means we never observe a partial write.
        path = tmp_path / "prompt.current.md"
        store = PromptFile(path, baseline="b" * 250)
        store.write("final text")
        # No leftover .tmp file:
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == []


@pytest.mark.unit
class TestPromptEvolutionStore:
    def _record(
        self, generation: int = 1, accepted: bool = True
    ) -> PromptEvolutionRecord:
        return PromptEvolutionRecord(
            generation=generation,
            action_counter=generation * 25,
            accepted=accepted,
            reasoning="why",
            proposed_prompt="proposed " * 50,
            previous_prompt="previous " * 50,
            new_prompt="new " * 100,
            validation_error=None,
            usage={"prompt": 10, "output": 20, "total": 30},
            timestamp="2026-05-18T00:00:00",
        )

    def test_append_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "evolution.jsonl"
        store = PromptEvolutionStore(path)
        store.append(self._record())
        assert path.exists()

    def test_append_writes_one_jsonl_line_per_record(self, tmp_path: Path) -> None:
        path = tmp_path / "evolution.jsonl"
        store = PromptEvolutionStore(path)
        store.append(self._record(1))
        store.append(self._record(2, accepted=False))
        store.append(self._record(3))
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        for line in lines:
            parsed = json.loads(line)
            assert "generation" in parsed
            assert "accepted" in parsed

    def test_all_records_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "evolution.jsonl"
        store = PromptEvolutionStore(path)
        store.append(self._record(1))
        store.append(self._record(2, accepted=False))
        records = store.all_records()
        assert len(records) == 2
        assert records[0]["generation"] == 1
        assert records[1]["accepted"] is False

    def test_all_records_empty_when_file_missing(self, tmp_path: Path) -> None:
        store = PromptEvolutionStore(tmp_path / "missing.jsonl")
        assert store.all_records() == []


@pytest.mark.unit
class TestActivePromptPath:
    def test_per_game_under_run_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "logs" / "run-id"
        monkeypatch.setenv("RUN_DIR", str(run_dir))

        assert active_prompt_path("ls20") == run_dir / "ls20" / "prompt.current.md"
        assert active_prompt_path("ls20") != active_prompt_path("vc33")

    def test_falls_back_to_run_dir_without_game_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "logs" / "run-id"
        monkeypatch.setenv("RUN_DIR", str(run_dir))
        monkeypatch.delenv("RUN_LOG_PATH", raising=False)
        assert active_prompt_path() == run_dir / "prompt.current.md"


@pytest.mark.unit
class TestActivePromptEvolutionPath:
    def test_per_game_under_run_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "logs" / "run-id"
        monkeypatch.setenv("RUN_DIR", str(run_dir))

        assert (
            active_prompt_evolution_path("ls20")
            == run_dir / "ls20" / "prompt_evolution.jsonl"
        )
        assert active_prompt_evolution_path("ls20") != active_prompt_evolution_path(
            "vc33"
        )

    def test_falls_back_to_run_dir_without_game_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "logs" / "run-id"
        monkeypatch.setenv("RUN_DIR", str(run_dir))
        monkeypatch.delenv("RUN_LOG_PATH", raising=False)
        assert active_prompt_evolution_path() == run_dir / "prompt_evolution.jsonl"


@pytest.mark.unit
class TestBuildEvolutionPrompt:
    def test_substitutes_all_placeholders(self) -> None:
        text = build_evolution_prompt(
            system_prompt="FIXED SYSTEM PROMPT BODY",
            current_base_prompt="CURRENT BASE PROMPT BODY",
            trajectory_rows=[
                {
                    "action_counter": 1,
                    "state": "NOT_FINISHED",
                    "score": 0,
                    "chosen_action": "ACTION1",
                    "reasoning": "trying up",
                    "tool_calls": [],
                }
            ],
        )
        assert "FIXED SYSTEM PROMPT BODY" in text
        assert "CURRENT BASE PROMPT BODY" in text
        assert "ACTION1" in text
        assert "trying up" in text
        assert "IMPROVED BASE PROMPT:" in text

    def test_handles_empty_trajectory(self) -> None:
        text = build_evolution_prompt(
            system_prompt="fixed system prompt",
            current_base_prompt="x" * 250,
            trajectory_rows=[],
        )
        assert "IMPROVED BASE PROMPT:" in text

    def test_includes_recent_tool_evidence(self) -> None:
        text = build_evolution_prompt(
            system_prompt="fixed system prompt",
            current_base_prompt="x" * 250,
            trajectory_rows=[],
            tool_evidence="[conv 2.t0] tool: run_skill",
        )
        assert "## Recent Tool Evidence" in text
        assert "[conv 2.t0] tool: run_skill" in text


# ----------------------------------------------------------------------
# Stubs + builder for HarnessEvolver tests
# ----------------------------------------------------------------------


class _TrajectoryStub:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records
        self.requested: list[int] = []

    def tail(self, n: int) -> list[dict[str, object]]:
        self.requested.append(n)
        return self.records[-n:]


class _StoreStub:
    def all_entries(self) -> list[object]:
        return []


class _TraceStub:
    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []

    def write(self, entry: dict[str, object]) -> None:
        self.entries.append(entry)


class _RecordingSkillStore(_StoreStub):
    def __init__(self) -> None:
        self.added: list[dict[str, object]] = []
        self.edited: list[tuple[str, object]] = []
        self.deleted: list[str] = []

    def add(self, name, description, code, tags=None):  # type: ignore[no-untyped-def]
        self.added.append({"name": name, "code": code})
        return SimpleNamespace(id=f"skill_{len(self.added):03d}", name=name)

    def get_by_id_or_name(self, key):  # type: ignore[no-untyped-def]
        return SimpleNamespace(id=key, name=key)

    def edit(self, skill_id, *, name=None, description=None, code=None, tags=None):  # type: ignore[no-untyped-def]
        self.edited.append((skill_id, code))
        return SimpleNamespace(id=skill_id)

    def delete(self, skill_id):  # type: ignore[no-untyped-def]
        self.deleted.append(skill_id)
        return True


class _RecordingMemoryStore(_StoreStub):
    def __init__(self) -> None:
        self.added: list[dict[str, object]] = []
        self.edited: list[tuple[str, object]] = []
        self.deleted: list[str] = []

    def add(self, title, body, tags=None, confidence=None):  # type: ignore[no-untyped-def]
        if confidence not in (1, 2, 3, 4, 5):
            raise ValueError("confidence must be an integer 1-5")
        self.added.append({"title": title, "confidence": confidence})
        return SimpleNamespace(id=f"mem_{len(self.added):03d}", title=title)

    def edit(self, entry_id, *, title=None, body=None, tags=None, confidence=None):  # type: ignore[no-untyped-def]
        self.edited.append((entry_id, confidence))
        return SimpleNamespace(id=entry_id)

    def delete(self, entry_id):  # type: ignore[no-untyped-def]
        self.deleted.append(entry_id)
        return True


class _RecordingSubagentStore(_StoreStub):
    def __init__(self) -> None:
        self.added: list[dict[str, object]] = []
        self.edited: list[tuple[str, object]] = []
        self.deleted: list[str] = []

    def add(self, name, description, instructions, allowed_tools=None, tags=None):  # type: ignore[no-untyped-def]
        self.added.append({"name": name, "allowed_tools": allowed_tools})
        return SimpleNamespace(id=f"subagent_{len(self.added):03d}", name=name)

    def edit(  # type: ignore[no-untyped-def]
        self, subagent_id, *, name=None, description=None,
        instructions=None, allowed_tools=None, tags=None,
    ):
        self.edited.append((subagent_id, allowed_tools))
        return SimpleNamespace(id=subagent_id)

    def delete(self, subagent_id):  # type: ignore[no-untyped-def]
        self.deleted.append(subagent_id)
        return True


def _noop_records(n: int, *, action: str = "ACTION1") -> list[dict[str, object]]:
    return [
        {
            "action_counter": i,
            "state": "NOT_FINISHED",
            "state_after": "NOT_FINISHED",
            "score": 0,
            "score_delta": 0,
            "chosen_action": action,
            "chosen_action_data": {},
            "source": "vlm",
            "grid_delta": None,
            "grid_change": None,
        }
        for i in range(n)
    ]


def _make_evolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    game_id: str = "evo-test",
    trajectory: object | None = None,
    memory: object | None = None,
    skills: object | None = None,
    subagents: object | None = None,
) -> HarnessEvolver:
    monkeypatch.setenv("RUN_DIR", str(tmp_path))
    return HarnessEvolver(
        model_name="test-model",
        system_instruction="SYS",
        game_id=game_id,
        agent_name="agent.test",
        memory=memory or _StoreStub(),
        skills=skills or _StoreStub(),
        subagents=subagents or _StoreStub(),
        trajectory=trajectory or _TrajectoryStub([]),
        trace=_TraceStub(),
        record_usage=lambda usage: None,
        baseline_prompt="b" * 250,
    )


@pytest.mark.unit
class TestEvolverConfig:
    def test_stagnation_after_default_is_100(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CONTINUAL_HARNESS_PROMPT_EVOLVE_FREQUENCY", raising=False)
        ev = _make_evolver(tmp_path, monkeypatch)
        assert ev.stagnation_after == 100

    def test_stagnation_after_zero_disables(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CONTINUAL_HARNESS_PROMPT_EVOLVE_FREQUENCY", "0")
        ev = _make_evolver(tmp_path, monkeypatch)
        assert ev.stagnation_after == 0

    def test_pass_kill_switches_default_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for var in (
            "CONTINUAL_HARNESS_EVOLVE_SKILLS",
            "CONTINUAL_HARNESS_EVOLVE_SUBAGENTS",
            "CONTINUAL_HARNESS_EVOLVE_MEMORY",
        ):
            monkeypatch.delenv(var, raising=False)
        ev = _make_evolver(tmp_path, monkeypatch)
        assert ev._evolve_skills_on and ev._evolve_subagents_on and ev._evolve_memory_on

    def test_pass_kill_switch_disables(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CONTINUAL_HARNESS_EVOLVE_SKILLS", "0")
        ev = _make_evolver(tmp_path, monkeypatch)
        assert ev._evolve_skills_on is False

    def test_main_frequency_parser_accepts_non_negative_ints(self) -> None:
        from main import _parse_prompt_evolve_frequency

        assert _parse_prompt_evolve_frequency("0") == 0
        assert _parse_prompt_evolve_frequency("25") == 25
        assert _parse_prompt_evolve_frequency(3) == 3

    def test_main_frequency_parser_rejects_invalid_values(self) -> None:
        from main import _parse_prompt_evolve_frequency

        with pytest.raises(argparse.ArgumentTypeError):
            _parse_prompt_evolve_frequency("-1")
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_prompt_evolve_frequency("abc")


@pytest.mark.unit
class TestGameOverEvolutionTrigger:
    def _evolver(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, frequency: int = 75
    ) -> HarnessEvolver:
        ev = _make_evolver(
            tmp_path, monkeypatch, trajectory=_TrajectoryStub(_noop_records(30))
        )
        ev.stagnation_after = frequency
        ev._last_evolution_step = 0
        ev._last_game_over_evolution_step = -1
        return ev

    def _frame(self) -> FrameData:
        return FrameData(
            game_id="evo-test", frame=[[[0]]],
            state=GameState.GAME_OVER, levels_completed=0,
        )

    def _playing_frame(self) -> FrameData:
        return FrameData(
            game_id="evo-test", frame=[[[0]]],
            state=GameState.NOT_FINISHED, levels_completed=0,
        )

    def test_game_over_evolves_once_and_marks_step(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ev = self._evolver(tmp_path, monkeypatch)
        frame = self._frame()
        calls: list[tuple[FrameData, int]] = []
        ev.evolve = lambda latest, action_counter, **_: calls.append((latest, action_counter))  # type: ignore[method-assign]

        ev.evolve_on_game_over(frame, 12)

        assert calls == [(frame, 12)]
        assert ev._last_game_over_evolution_step == 12
        assert ev._last_evolution_step == 12

    def test_game_over_respects_disabled_frequency(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ev = self._evolver(tmp_path, monkeypatch, frequency=0)
        calls: list[FrameData] = []
        ev.evolve = lambda latest, action_counter, **_: calls.append(latest)  # type: ignore[method-assign]

        ev.evolve_on_game_over(self._frame(), 12)

        assert calls == []
        assert ev._last_game_over_evolution_step == -1

    def test_game_over_only_once_per_action_counter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ev = self._evolver(tmp_path, monkeypatch)
        frame = self._frame()
        calls: list[FrameData] = []
        ev.evolve = lambda latest, action_counter, **_: calls.append(latest)  # type: ignore[method-assign]

        ev.evolve_on_game_over(frame, 12)
        ev.evolve_on_game_over(frame, 12)

        assert calls == [frame]

    def test_game_over_blocks_immediate_stagnation_evolution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ev = self._evolver(tmp_path, monkeypatch, frequency=10)
        calls: list[int] = []
        ev.evolve = lambda latest, action_counter, **_: calls.append(action_counter)  # type: ignore[method-assign]

        ev.evolve_on_game_over(self._frame(), 12)
        # Only 1 action later: blocked by STAGNATION_MIN_ACTIONS_SINCE_EVOLUTION.
        ev.maybe_evolve_on_stagnation(self._playing_frame(), 13, last_progress_step=-100)

        assert calls == [12]


@pytest.mark.unit
class TestStagnationEvolutionTrigger:
    def _evolver(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        frequency: int = 100,
        trajectory: object | None = None,
    ) -> HarnessEvolver:
        ev = _make_evolver(
            tmp_path, monkeypatch,
            trajectory=trajectory or _TrajectoryStub(_noop_records(30)),
        )
        ev.stagnation_after = frequency
        ev._last_evolution_step = 0
        return ev

    def _frame(self) -> FrameData:
        return FrameData(
            game_id="evo-test", frame=[[[0]]],
            state=GameState.NOT_FINISHED, levels_completed=0,
        )

    def test_stagnation_evolves_when_no_progress_and_noop_window(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ev = self._evolver(tmp_path, monkeypatch)
        frame = self._frame()
        calls: list[tuple[FrameData, dict[str, object]]] = []

        def evolve(latest: FrameData, action_counter: int, **kwargs: object) -> None:
            calls.append((latest, kwargs))

        ev.evolve = evolve  # type: ignore[method-assign]
        ev.maybe_evolve_on_stagnation(frame, 130, last_progress_step=0)

        assert len(calls) == 1
        assert calls[0][0] == frame
        assert calls[0][1]["trigger"] == "stagnation"
        assert "no-op/invalid" in " ".join(calls[0][1]["trigger_evidence"])  # type: ignore[arg-type]
        assert ev._last_evolution_step == 130
        assert ev.trajectory.requested == [HarnessEvolver.STAGNATION_WINDOW]

    def test_stagnation_requires_progress_gap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ev = self._evolver(tmp_path, monkeypatch)
        calls: list[FrameData] = []
        ev.evolve = lambda latest, action_counter, **_: calls.append(latest)  # type: ignore[method-assign]

        # Only 50 actions since progress (< frequency 100): blocked.
        ev.maybe_evolve_on_stagnation(self._frame(), 130, last_progress_step=80)

        assert calls == []

    def test_stagnation_requires_pattern_evidence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        moving = _TrajectoryStub(
            [
                {
                    **record,
                    "chosen_action": f"ACTION{i}",
                    "grid_delta": [[0, 0, 1, 2]],
                    "grid_change": [[1, 2, 1, 0, 0, 0, 0]],
                }
                for i, record in enumerate(_noop_records(30))
            ]
        )
        ev = self._evolver(tmp_path, monkeypatch, trajectory=moving)
        calls: list[FrameData] = []
        ev.evolve = lambda latest, action_counter, **_: calls.append(latest)  # type: ignore[method-assign]

        ev.maybe_evolve_on_stagnation(self._frame(), 130, last_progress_step=0)

        assert calls == []


@pytest.mark.unit
class TestParseJsonResponse:
    def test_plain_object(self) -> None:
        assert HarnessEvolver._parse_json_response('{"a": 1}') == {"a": 1}

    def test_json_fenced(self) -> None:
        assert HarnessEvolver._parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}

    def test_embedded_in_prose(self) -> None:
        assert HarnessEvolver._parse_json_response('prefix {"a": 1} suffix') == {"a": 1}

    def test_malformed_returns_none(self) -> None:
        assert HarnessEvolver._parse_json_response("not json at all") is None


@pytest.mark.unit
class TestComponentEvolutionPasses:
    def test_evolve_injects_windowed_tool_evidence_into_all_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ev = _make_evolver(tmp_path, monkeypatch)
        ev._last_evolution_step = 10
        frame = FrameData(
            game_id="evo-test",
            frame=[[[0]]],
            state=GameState.NOT_FINISHED,
            levels_completed=0,
        )
        old = ToolEvidenceRecord(
            conversation_id=1,
            conversation_turn=0,
            round=1,
            action_counter_before=4,
            action_counter_after=8,
            tool_call=ToolCallRecord(
                name="old_tool",
                args={},
                result={"success": True, "message": "outside window"},
            ),
        )
        boundary = ToolEvidenceRecord(
            conversation_id=2,
            conversation_turn=0,
            round=2,
            action_counter_before=10,
            action_counter_after=10,
            tool_call=ToolCallRecord(
                name="run_subagent",
                args={"id": "subagent_001"},
                result={"success": False, "status": "max_rounds"},
            ),
        )
        current = ToolEvidenceRecord(
            conversation_id=2,
            conversation_turn=1,
            round=3,
            action_counter_before=12,
            action_counter_after=14,
            tool_call=ToolCallRecord(
                name="process_memory",
                args={"operation": "add", "title": "gate rule"},
                result={"success": True, "id": "mem_009"},
            ),
        )
        captured: dict[str, str] = {}

        def fake_meta_query(
            system: str, user: str, images: list[object], tag: object
        ) -> tuple[str, None, None]:
            del system, images
            captured[str(tag)] = user
            if tag == 1:
                return ("improved prompt " * 30, None, None)
            return (
                json.dumps({"analysis": "x", "add": [], "edit": [], "delete": []}),
                None,
                None,
            )

        monkeypatch.setattr(ev, "_meta_query", fake_meta_query)

        ev.evolve(
            frame,
            20,
            tool_evidence_records=[old, boundary, current],
        )

        assert set(captured) == {"1", "skills", "subagents", "memory"}
        for user_prompt in captured.values():
            assert "run_subagent" in user_prompt
            assert "process_memory" in user_prompt
            assert "gate rule" in user_prompt
            assert "old_tool" not in user_prompt

    def test_skill_pass_adds_valid_skips_bad_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _RecordingSkillStore()
        ev = _make_evolver(tmp_path, monkeypatch, skills=store)
        payload = json.dumps(
            {
                "analysis": "x",
                "add": [
                    {"name": "good", "description": "d", "code": "result = 1", "tags": []},
                    {"name": "bad", "description": "d", "code": "import os\nresult = 1", "tags": []},
                ],
                "edit": [{"id": "skill_042", "code": "result = 2"}],
                "delete": ["skill_009"],
            }
        )
        monkeypatch.setattr(ev, "_meta_query", lambda *a, **k: (payload, None, None))

        out = ev._evolve_skills([], "traj", "trigger")

        # Skill whose code violates the sandbox AST policy is skipped.
        assert [s["name"] for s in store.added] == ["good"]
        assert store.edited == [("skill_042", "result = 2")]
        assert store.deleted == ["skill_009"]
        assert len(out["added"]) == 1

    def test_memory_pass_adds_valid_skips_bad_confidence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _RecordingMemoryStore()
        ev = _make_evolver(tmp_path, monkeypatch, memory=store)
        payload = json.dumps(
            {
                "analysis": "x",
                "add": [
                    {"title": "ok", "body": "b", "tags": [], "confidence": 2},
                    {"title": "bad", "body": "b", "tags": [], "confidence": 9},
                ],
                "edit": [{"id": "mem_001", "confidence": 4}],
                "delete": ["mem_005"],
            }
        )
        monkeypatch.setattr(ev, "_meta_query", lambda *a, **k: (payload, None, None))

        out = ev._evolve_memory([], "traj", "trigger")

        assert [m["title"] for m in store.added] == ["ok"]
        assert store.edited == [("mem_001", 4)]
        assert store.deleted == ["mem_005"]
        assert len(out["added"]) == 1

    def test_subagent_pass_add_edit_delete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _RecordingSubagentStore()
        ev = _make_evolver(tmp_path, monkeypatch, subagents=store)
        payload = json.dumps(
            {
                "analysis": "x",
                "add": [
                    {
                        "name": "explore", "description": "d", "instructions": "i",
                        "allowed_tools": ["take_actions"], "tags": [],
                    }
                ],
                "edit": [{"id": "subagent_001", "allowed_tools": ["process_memory"]}],
                "delete": ["subagent_002"],
            }
        )
        monkeypatch.setattr(ev, "_meta_query", lambda *a, **k: (payload, None, None))

        out = ev._evolve_subagents([], "traj", "trigger")

        assert [s["name"] for s in store.added] == ["explore"]
        assert store.edited == [("subagent_001", ["process_memory"])]
        assert store.deleted == ["subagent_002"]
        assert len(out["added"]) == 1

    def test_subagent_pass_normalizes_legacy_sa_aliases(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _RecordingSubagentStore()
        ev = _make_evolver(tmp_path, monkeypatch, subagents=store)
        payload = json.dumps(
            {
                "analysis": "x",
                "add": [],
                "edit": [{"id": "sa_001", "allowed_tools": ["process_memory"]}],
                "delete": ["sa_002"],
            }
        )
        monkeypatch.setattr(ev, "_meta_query", lambda *a, **k: (payload, None, None))

        ev._evolve_subagents([], "traj", "trigger")

        assert store.edited == [("subagent_001", ["process_memory"])]
        assert store.deleted == ["subagent_002"]

    def test_subagent_prompt_uses_real_subagent_ids(self) -> None:
        assert "subagent_002" in SUBAGENT_EVOLUTION_PROMPT
        assert "subagent_004" in SUBAGENT_EVOLUTION_PROMPT
        assert "sa_002" not in SUBAGENT_EVOLUTION_PROMPT
        assert "sa_004" not in SUBAGENT_EVOLUTION_PROMPT

    def test_pass_returns_error_on_malformed_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _RecordingSkillStore()
        ev = _make_evolver(tmp_path, monkeypatch, skills=store)
        monkeypatch.setattr(ev, "_meta_query", lambda *a, **k: ("not json", None, None))

        out = ev._evolve_skills([], "traj", "trigger")

        assert out == {"error": "failed_to_parse_response"}
        assert store.added == []

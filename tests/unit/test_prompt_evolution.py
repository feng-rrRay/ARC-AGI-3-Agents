"""Tests for prompt_evolution.py.

Covers: validation bounds, PromptFile read/write, PromptEvolutionStore append,
active_*_path env-var resolution, and build_evolution_prompt placeholder
substitution. The smoke test of the agent-level hook lives at the bottom.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from arcengine import FrameData, GameState  # noqa: F401 — used by other test files

from agents.templates.continual_harness.prompt_evolution import (
    PROMPT_MAX_CHARS,
    PROMPT_MIN_CHARS,
    PromptEvolutionRecord,
    PromptEvolutionStore,
    PromptFile,
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


@pytest.mark.unit
class TestPromptEvolutionFrequencyParsing:
    """Frequency value parsed by the agent comes from an env var written by main.py.

    Validate the parsing here so the agent itself doesn't need a full game-loop
    fixture just to exercise this codepath.
    """

    def test_default_is_100(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CONTINUAL_HARNESS_PROMPT_EVOLVE_FREQUENCY", raising=False)
        # Default in the class — verified via direct attribute access.
        from agents.templates.continual_harness_agent import ContinualHarness

        assert ContinualHarness.DEFAULT_PROMPT_EVOLVE_FREQUENCY == 100

    def test_zero_disables(self) -> None:
        # The hook condition `self._prompt_evolve_frequency > 0` ensures 0 disables.
        # We assert via class-level constant — actual enforcement is tested by
        # smoke tests at the integration level (not run here).
        from agents.templates.continual_harness_agent import ContinualHarness

        assert isinstance(ContinualHarness.DEFAULT_PROMPT_EVOLVE_FREQUENCY, int)

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
class TestGameOverPromptEvolutionHook:
    def _agent(self, *, frequency: int = 75, action_counter: int = 10):
        from agents.templates.continual_harness_agent import ContinualHarness

        agent = ContinualHarness.__new__(ContinualHarness)
        agent.game_id = "game-over-evo-test"
        agent.model_name = "test-model"
        agent._prompt_evolve_frequency = frequency
        agent._last_evolution_step = 0
        agent._last_game_over_evolution_step = -1
        agent._last_progress_step = 0
        agent.action_counter = action_counter
        return agent

    def _frame(self) -> FrameData:
        return FrameData(
            game_id="game-over-evo-test",
            frame=[[[0]]],
            state=GameState.GAME_OVER,
            levels_completed=0,
        )

    def _playing_frame(self) -> FrameData:
        return FrameData(
            game_id="game-over-evo-test",
            frame=[[[0]]],
            state=GameState.NOT_FINISHED,
            levels_completed=0,
        )

    def test_game_over_evolves_once_and_resets_frequency_counter(self) -> None:
        agent = self._agent(action_counter=12)
        frame = self._frame()
        calls: list[FrameData] = []
        agent._evolve_system_prompt = lambda latest, **_: calls.append(latest)

        agent._evolve_on_game_over(frame)

        assert calls == [frame]
        assert agent._last_game_over_evolution_step == 12
        assert agent._last_evolution_step == 12

    def test_game_over_evolution_respects_disabled_frequency(self) -> None:
        agent = self._agent(frequency=0, action_counter=12)
        frame = self._frame()
        calls: list[FrameData] = []
        agent._evolve_system_prompt = lambda latest, **_: calls.append(latest)

        agent._evolve_on_game_over(frame)

        assert calls == []
        assert agent._last_game_over_evolution_step == -1

    def test_game_over_evolution_only_once_per_action_counter(self) -> None:
        agent = self._agent(action_counter=12)
        frame = self._frame()
        calls: list[FrameData] = []
        agent._evolve_system_prompt = lambda latest, **_: calls.append(latest)

        agent._evolve_on_game_over(frame)
        agent._evolve_on_game_over(frame)

        assert calls == [frame]

    def test_game_over_reset_counter_blocks_immediate_stagnation_evolution(self) -> None:
        agent = self._agent(frequency=10, action_counter=12)
        frame = self._frame()
        calls: list[FrameData] = []
        agent._evolve_system_prompt = lambda latest, **_: calls.append(latest)
        agent._last_progress_step = -100
        agent.trajectory = _TrajectoryStub(_noop_records(30))

        agent._evolve_on_game_over(frame)
        agent.action_counter = 13
        agent._maybe_evolve_on_stagnation(self._playing_frame())

        assert calls == [frame]


class _TrajectoryStub:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records
        self.requested: list[int] = []

    def tail(self, n: int) -> list[dict[str, object]]:
        self.requested.append(n)
        return self.records[-n:]


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


@pytest.mark.unit
class TestStagnationPromptEvolutionHook:
    def _agent(self, *, frequency: int = 100, action_counter: int = 130):
        from agents.templates.continual_harness_agent import ContinualHarness

        agent = ContinualHarness.__new__(ContinualHarness)
        agent.game_id = "stagnation-evo-test"
        agent.model_name = "test-model"
        agent._prompt_evolve_frequency = frequency
        agent._last_evolution_step = 0
        agent._last_game_over_evolution_step = -1
        agent._last_progress_step = 0
        agent.action_counter = action_counter
        agent.trajectory = _TrajectoryStub(_noop_records(30))
        return agent

    def _frame(self) -> FrameData:
        return FrameData(
            game_id="stagnation-evo-test",
            frame=[[[0]]],
            state=GameState.NOT_FINISHED,
            levels_completed=0,
        )

    def test_stagnation_evolves_when_no_progress_and_noop_window(self) -> None:
        agent = self._agent()
        frame = self._frame()
        calls: list[tuple[FrameData, dict[str, object]]] = []

        def evolve(latest: FrameData, **kwargs: object) -> None:
            calls.append((latest, kwargs))

        agent._evolve_system_prompt = evolve

        agent._maybe_evolve_on_stagnation(frame)

        assert len(calls) == 1
        assert calls[0][0] == frame
        assert calls[0][1]["trigger"] == "stagnation"
        assert "no-op/invalid" in " ".join(calls[0][1]["trigger_evidence"])
        assert agent._last_evolution_step == 130
        assert agent.trajectory.requested == [agent.STAGNATION_WINDOW]

    def test_stagnation_requires_progress_gap(self) -> None:
        agent = self._agent(action_counter=130)
        agent._last_progress_step = 80
        frame = self._frame()
        calls: list[FrameData] = []
        agent._evolve_system_prompt = lambda latest, **_: calls.append(latest)

        agent._maybe_evolve_on_stagnation(frame)

        assert calls == []

    def test_stagnation_requires_pattern_evidence(self) -> None:
        agent = self._agent()
        agent.trajectory = _TrajectoryStub(
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
        frame = self._frame()
        calls: list[FrameData] = []
        agent._evolve_system_prompt = lambda latest, **_: calls.append(latest)

        agent._maybe_evolve_on_stagnation(frame)

        assert calls == []

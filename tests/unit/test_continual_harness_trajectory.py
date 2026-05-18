from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from arcengine import FrameData, GameState

from agents.templates.continual_harness.context import build_action_prompt
from agents.templates.continual_harness.models import StepRecord, ToolCallRecord
from agents.templates.continual_harness.trajectory import (
    TrajectoryStore,
    _frame_delta,
    default_trajectory_path,
    format_compact_history,
    format_full_history,
)


def _fake_frame(grid: list[list[int]]) -> SimpleNamespace:
    """Minimal stand-in for FrameData — only `.frame` is read by the formatter."""
    return SimpleNamespace(frame=[grid])


def _rec(
    action_counter: int,
    chosen_action: str = "ACTION1",
    *,
    state: str = "NOT_FINISHED",
    score: int = 0,
    reasoning: str | None = None,
    chosen_action_data: dict[str, Any] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a trajectory-row dict (matches what TrajectoryStore.tail() returns)."""
    return {
        "action_counter": action_counter,
        "chosen_action": chosen_action,
        "chosen_action_data": chosen_action_data or {},
        "state": state,
        "score": score,
        "reasoning": reasoning,
        "tool_calls": tool_calls or [],
    }


def _step(
    action_counter: int,
    chosen_action: str = "ACTION1",
    *,
    reasoning: str | None = None,
    tool_calls: list[ToolCallRecord] | None = None,
    chosen_action_data: dict[str, Any] | None = None,
) -> StepRecord:
    return StepRecord(
        game_id="traj-test",
        action_counter=action_counter,
        state="NOT_FINISHED",
        score=0,
        chosen_action=chosen_action,
        chosen_action_data=chosen_action_data or {},
        reasoning=reasoning,
        tool_calls=tool_calls or [],
    )


@pytest.mark.unit
class TestTrajectoryStore:
    def test_append_and_tail_round_trip(self, tmp_path: Path) -> None:
        store = TrajectoryStore(tmp_path / "run.trajectory.jsonl")
        store.append(_step(1, "ACTION1"))
        store.append(_step(2, "ACTION2"))
        store.append(_step(3, "ACTION3"))

        tail = store.tail(2)

        assert [row["action_counter"] for row in tail] == [2, 3]
        assert [row["chosen_action"] for row in tail] == ["ACTION2", "ACTION3"]

    def test_tail_on_missing_file_returns_empty(self, tmp_path: Path) -> None:
        store = TrajectoryStore(tmp_path / "absent.trajectory.jsonl")
        assert store.tail(5) == []

    def test_tail_with_zero_or_negative_returns_empty(self, tmp_path: Path) -> None:
        store = TrajectoryStore(tmp_path / "run.trajectory.jsonl")
        store.append(_step(1))
        assert store.tail(0) == []
        assert store.tail(-1) == []

    def test_append_writes_one_jsonl_line_per_record(self, tmp_path: Path) -> None:
        path = tmp_path / "run.trajectory.jsonl"
        store = TrajectoryStore(path)
        store.append(_step(1, "ACTION1"))
        store.append(_step(2, "ACTION2"))

        raw = path.read_text()
        lines = raw.splitlines()
        assert raw.endswith("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["chosen_action"] == "ACTION1"
        assert first["game_id"] == "traj-test"


@pytest.mark.unit
class TestFormatCompactHistory:
    def test_empty_records_returns_sentinel(self) -> None:
        assert format_compact_history([]) == "No previous actions recorded."

    def test_one_line_per_step(self) -> None:
        records = [
            _rec(i, f"ACTION{i}", reasoning=f"step {i}") for i in (1, 2, 3)
        ]
        out = format_compact_history(records)
        lines = out.splitlines()
        assert len(lines) == 3
        # No multi-line nesting (no "why:" or "tool:" line headers like format_full_history).
        assert all("why:" not in line for line in lines)
        assert all("tool:" not in line for line in lines)
        assert "[1] ACTION1" in lines[0]
        assert "[3] ACTION3" in lines[2]

    def test_reasoning_snippet_appears_inline_quoted(self) -> None:
        rec = _rec(7, "ACTION3", reasoning="trying to push north into the gate")
        out = format_compact_history([rec])
        assert '"trying to push north into the gate"' in out

    def test_reasoning_truncated_to_reasoning_chars(self) -> None:
        long_reason = "x" * 500
        out = format_compact_history(
            [_rec(1, "ACTION1", reasoning=long_reason)], reasoning_chars=50
        )
        # Truncated reasoning ends with the ellipsis marker.
        assert "…" in out
        # Anywhere we find the truncated chunk it's bounded by quotes.
        assert "x" * 60 not in out  # never the full string

    def test_missing_reasoning_omits_the_quoted_block(self) -> None:
        out = format_compact_history([_rec(1, "ACTION1", reasoning=None)])
        assert '"' not in out  # no quoted segment at all

    def test_truncates_oldest_when_over_max_chars(self) -> None:
        records = [
            _rec(i, "ACTION6", chosen_action_data={"x": 12, "y": 34}) for i in range(30)
        ]
        out = format_compact_history(records, max_chars=400)
        assert len(out) <= 400
        # Newest step (29) survives; oldest steps were dropped.
        assert "[29]" in out
        assert "[0]" not in out

    def test_includes_action_data_inline(self) -> None:
        out = format_compact_history(
            [_rec(5, "ACTION6", chosen_action_data={"x": 12, "y": 34})]
        )
        # New format is "ACTION6{x:12,y:34}", not "data={'x': 12, 'y': 34}".
        assert "ACTION6{x:12,y:34}" in out


@pytest.mark.unit
class TestEffectTags:
    def test_no_op_when_frames_identical(self) -> None:
        grid = [[1, 2], [3, 4]]
        frames = [_fake_frame(grid), _fake_frame([row[:] for row in grid])]
        out = format_compact_history([_rec(0, "ACTION1")], frames=frames)
        assert "NO_OP" in out

    def test_change_with_bbox(self) -> None:
        pre = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        post = [[1, 1, 1], [1, 2, 2], [1, 1, 1]]  # 2 cells changed at row 1
        out = format_compact_history(
            [_rec(0, "ACTION6", chosen_action_data={"x": 1, "y": 1})],
            frames=[_fake_frame(pre), _fake_frame(post)],
        )
        assert "CHANGE(" in out
        assert "rows 1-1" in out
        assert "cols 1-2" in out

    def test_level_up_outranks_change(self) -> None:
        # Frame also changed, but score went up — LEVEL_UP wins.
        pre = [[0, 0], [0, 0]]
        post = [[1, 0], [0, 0]]
        frames = [_fake_frame(pre), _fake_frame(pre), _fake_frame(post)]
        records = [
            _rec(0, "ACTION1", score=0),
            _rec(1, "ACTION1", score=1),  # score bump
        ]
        out = format_compact_history(records, frames=frames)
        # Row for action_counter=1 should carry LEVEL_UP, not CHANGE.
        line_for_1 = next(line for line in out.splitlines() if line.startswith("[1]"))
        assert "LEVEL_UP 0->1" in line_for_1
        assert "CHANGE" not in line_for_1

    def test_state_change_outranks_change(self) -> None:
        frames = [_fake_frame([[0]]), _fake_frame([[0]]), _fake_frame([[1]])]
        records = [
            _rec(0, "ACTION1", state="NOT_FINISHED"),
            _rec(1, "ACTION1", state="GAME_OVER"),
        ]
        out = format_compact_history(records, frames=frames)
        line_for_1 = next(line for line in out.splitlines() if line.startswith("[1]"))
        assert "STATE->GAME_OVER" in line_for_1

    def test_unknown_when_frames_missing(self) -> None:
        out = format_compact_history([_rec(0, "ACTION1")], frames=None)
        assert "UNKNOWN" in out

    def test_unknown_when_frame_index_out_of_range(self) -> None:
        # Only one frame supplied — no post-action frame for action_counter=0.
        out = format_compact_history(
            [_rec(0, "ACTION1")], frames=[_fake_frame([[0]])]
        )
        assert "UNKNOWN" in out


@pytest.mark.unit
class TestFrameDelta:
    def test_identical_grids_no_op(self) -> None:
        assert _frame_delta([[1, 2]], [[1, 2]])["kind"] == "NO_OP"

    def test_single_cell_change_bbox(self) -> None:
        d = _frame_delta([[0, 0], [0, 0]], [[0, 0], [0, 9]])
        assert d["kind"] == "CHANGE"
        assert d["n"] == 1
        assert d["bbox"] == (1, 1, 1, 1)

    def test_unknown_when_either_grid_none(self) -> None:
        assert _frame_delta(None, [[0]])["kind"] == "UNKNOWN"
        assert _frame_delta([[0]], None)["kind"] == "UNKNOWN"


@pytest.mark.unit
class TestFormatFullHistory:
    def test_includes_reasoning_and_tool_calls(self) -> None:
        records = [
            {
                "action_counter": 1,
                "chosen_action": "ACTION3",
                "chosen_action_data": {},
                "state": "NOT_FINISHED",
                "score": 0,
                "reasoning": "moving left to investigate",
                "tool_calls": [
                    {
                        "name": "get_recent_trajectory",
                        "args": {"limit": 5},
                        "result": {"count": 3, "history": "..."},
                    }
                ],
            }
        ]

        out = format_full_history(records)
        assert "why: moving left to investigate" in out
        assert "tool: get_recent_trajectory" in out
        assert "result:" in out

    def test_empty_records_returns_sentinel(self) -> None:
        assert format_full_history([]) == "No previous actions recorded."


@pytest.mark.unit
class TestDefaultTrajectoryPath:
    def test_uses_run_log_path_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_log = tmp_path / "logs" / "continualharness-x.log"
        monkeypatch.setenv("RUN_LOG_PATH", str(run_log))
        assert default_trajectory_path() == run_log.with_suffix(".trajectory.jsonl")

    def test_fallback_uses_logs_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RUN_LOG_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        path = default_trajectory_path()
        # Returned path is relative ("logs/trajectory-...jsonl"); it resolves under the cwd.
        assert path.parent == Path("logs")
        assert (tmp_path / "logs").is_dir()
        assert path.name.startswith("trajectory-")
        assert path.suffix == ".jsonl"


@pytest.mark.unit
class TestPromptInjection:
    def test_compact_history_lands_above_turn_line(self) -> None:
        records = [
            {
                "action_counter": 1,
                "chosen_action": "ACTION1",
                "chosen_action_data": {},
                "state": "NOT_FINISHED",
                "score": 0,
            }
        ]
        history = format_compact_history(records)
        frame = FrameData(
            game_id="inj-test", frame=[[[0]]], state=GameState.NOT_FINISHED
        )

        prompt = build_action_prompt(
            frame,
            extra_context=f"## RECENT STEPS\n{history}",
        )

        assert "## RECENT STEPS" in prompt
        assert "[1] ACTION1" in prompt
        assert prompt.index("## RECENT STEPS") < prompt.index("# TURN:")
        assert "# TURN:\nCall exactly one action." in prompt

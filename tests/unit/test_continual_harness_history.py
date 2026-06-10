"""Tests for the accumulative (append-only) RECENT HISTORY renderer."""
from __future__ import annotations

from typing import Any

import pytest

from agents.templates.continual_harness.trajectory import (
    render_recent_history,
    summarize_grid_transitions,
)


def _row(
    ac: int,
    action: str = "ACTION1",
    *,
    score: int = 0,
    score_delta: int | None = 0,
    state: str = "NOT_FINISHED",
    state_after: str = "NOT_FINISHED",
    source: str = "vlm",
    skill_id: str | None = None,
    data: dict[str, Any] | None = None,
    grid_change: list[list[int]] | None = None,
    grid_delta: list[list[int]] | None = None,
) -> dict[str, Any]:
    return {
        "action_counter": ac,
        "chosen_action": action,
        "chosen_action_data": data or {},
        "score": score,
        "score_delta": score_delta,
        "state": state,
        "state_after": state_after,
        "source": source,
        "skill_id": skill_id,
        "grid_change": grid_change,
        "grid_delta": grid_delta,
    }


def _reset(ac: int, score: int = 0) -> dict[str, Any]:
    return _row(
        ac, "RESET", score=score, source="auto_reset",
        state="GAME_OVER", state_after="NOT_FINISHED",
    )


@pytest.mark.unit
class TestSummarizeGridTransitions:
    def test_none_when_grid_missing(self) -> None:
        assert summarize_grid_transitions(None, [[0]]) is None
        assert summarize_grid_transitions([[0]], None) is None

    def test_none_when_no_change(self) -> None:
        assert summarize_grid_transitions([[1, 2], [3, 4]], [[1, 2], [3, 4]]) is None

    def test_groups_by_transition_with_bbox_and_count(self) -> None:
        pre = [[1, 1, 1], [1, 1, 1]]
        post = [[1, 2, 1], [1, 14, 1]]  # (0,1):1->2, (1,1):1->14
        out = summarize_grid_transitions(pre, post)
        assert [1, 2, 1, 0, 0, 1, 1] in out
        assert [1, 14, 1, 1, 1, 1, 1] in out

    def test_multi_cell_transition_bbox_and_sorted_by_count(self) -> None:
        pre = [[0, 0, 0, 0]]
        post = [[1, 1, 1, 9]]  # 0->1 x3, 0->9 x1
        out = summarize_grid_transitions(pre, post)
        # Most frequent transition first.
        assert out[0] == [0, 1, 3, 0, 0, 0, 2]
        assert out[1] == [0, 9, 1, 0, 0, 3, 3]


@pytest.mark.unit
class TestRenderRecentHistory:
    def test_empty(self) -> None:
        assert render_recent_history([]) == "No previous actions recorded."

    def test_single_level_detail_and_header(self) -> None:
        rows = [
            _row(1, "ACTION4", grid_change=[[0, 1, 1, 30, 30, 23, 23]]),
            _row(2, "ACTION1"),  # no change
        ]
        out = render_recent_history(rows)
        assert "══ Level 1 — started at step 1 ══" in out
        assert "[take_actions] step 1  ACTION4 → 1 cell changed: color 0→1: r30 c23" in out
        assert "[take_actions] step 2  ACTION1 → no change" in out
        assert "Levels cleared" not in out  # nothing cleared yet

    def test_action6_renders_coordinates_in_full(self) -> None:
        rows = [_row(5, "ACTION6", data={"x": 12, "y": 30}, grid_change=None)]
        out = render_recent_history(rows)
        assert "ACTION6(x=12, y=30) → no change" in out

    def test_level_archive_line(self) -> None:
        rows = [
            _row(1, "ACTION1", score=0, score_delta=1),  # clears level 1
            _row(2, "ACTION1", score=1),                  # now on level 2
        ]
        out = render_recent_history(rows)
        assert "Levels cleared: level 1 cleared in 1 steps." in out
        assert "══ Level 2 — started at step 2 ══" in out

    def test_attempts_split_on_reset_and_summarized(self) -> None:
        rows = [
            _row(1, "ACTION1", score=0, score_delta=1),     # clear L1
            _row(2, "ACTION2", score=1),
            _row(3, "ACTION2", score=1, state_after="GAME_OVER"),  # death 1
            _reset(4, score=1),
            _row(5, "ACTION2", score=1),
            _row(6, "ACTION2", score=1, state_after="GAME_OVER"),  # death 2
            _reset(7, score=1),
            _row(8, "ACTION1", score=1),                    # current attempt
        ]
        out = render_recent_history(rows)
        # Two identical deaths collapse into one summary line.
        assert "Attempts 1-2 (2 tries, steps 2-6): each GAME_OVER after ACTION2." in out
        assert "Attempt 3 — current (started step 8):" in out
        assert "[take_actions] step 8  ACTION1 → no change" in out

    def test_loop_collapse_with_progress_note(self) -> None:
        rows = [_row(i, "ACTION4", score=0) for i in range(1, 13)]  # 12 identical no-ops
        out = render_recent_history(rows)
        assert "[take_actions] steps 1-12  ACTION4 ×12 → no change (no progress — possible loop)" in out

    def test_run_skill_source_label(self) -> None:
        rows = [
            _row(1, "ACTION1", source="run_skill", skill_id="solve_l6",
                 grid_change=[[1, 14, 9, 30, 32, 27, 29]]),
        ]
        out = render_recent_history(rows)
        assert "[run_skill: solve_l6] step 1  ACTION1 → 9 cells changed: color 1→14 (×9): r30-32 c27-29" in out

    def test_game_over_effect_in_detail(self) -> None:
        rows = [_row(1, "ACTION2", state="NOT_FINISHED", state_after="GAME_OVER")]
        out = render_recent_history(rows)
        assert "ACTION2 → → GAME_OVER" in out

    def test_grid_delta_fallback_when_no_grid_change(self) -> None:
        # Old-format record: only grid_delta present.
        rows = [_row(1, "ACTION1", grid_change=None, grid_delta=[[5, 3, 0, 9]])]
        out = render_recent_history(rows)
        assert "1 cell changed: color 0→9: r3 c5" in out

    def test_cache_stability_prefix_is_byte_identical_on_append(self) -> None:
        """Appending one action must leave all prior rendered lines unchanged."""
        base = [
            _row(1, "ACTION4", grid_change=[[0, 1, 1, 5, 5, 5, 5]]),
            _row(2, "ACTION1"),
        ]
        out1 = render_recent_history(base)
        out2 = render_recent_history(base + [_row(3, "ACTION2")])
        # Everything in out1 is a prefix of out2 (only a new trailing line added).
        assert out2.startswith(out1)

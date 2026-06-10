"""Tests for the hex grid representation shared by the working prompt and the
sandbox `state` view (color cell -> hex char; 2D grid -> list[str]; frame stack
-> list[list[str]]). Engine storage stays int; hex is applied at the boundaries.
"""
from __future__ import annotations

import pytest

from agents.templates.continual_harness.helpers import (
    color_to_hex,
    frame_to_hex,
    grid_to_hex_lines,
    hex_line_to_ints,
)
from agents.templates.continual_harness.sandbox import (
    SandboxState,
    run_python_snippet,
)
from agents.templates.continual_harness.trajectory import hexify_record_colors


@pytest.mark.unit
class TestHexHelpers:
    def test_color_to_hex_maps_0_to_15(self) -> None:
        assert [color_to_hex(i) for i in range(16)] == list("0123456789abcdef")

    def test_color_to_hex_wraps_like_palette(self) -> None:
        # grid_to_image indexes palette with `% 16`; color_to_hex must match.
        assert color_to_hex(16) == "0"
        assert color_to_hex(30) == color_to_hex(14) == "e"

    def test_grid_to_hex_lines_one_string_per_row(self) -> None:
        lines = grid_to_hex_lines([[0, 1, 14, 14, 3], [15, 0, 0, 10, 2]])
        assert lines == ["01ee3", "f00a2"]
        assert all(isinstance(row, str) for row in lines)

    def test_frame_to_hex_stack(self) -> None:
        assert frame_to_hex([[[0, 14], [15, 0]]]) == [["0e", "f0"]]
        assert frame_to_hex(None) == []
        assert frame_to_hex([]) == []

    def test_hex_line_round_trips(self) -> None:
        for v in range(16):
            assert hex_line_to_ints(color_to_hex(v)) == [v]
        assert hex_line_to_ints("f00a2") == [15, 0, 0, 10, 2]


@pytest.mark.unit
class TestHexifyRecordColors:
    def test_grid_change_colors_become_hex_counts_stay_int(self) -> None:
        rec = {"grid_change": [[15, 0, 9, 30, 32, 27, 29]], "grid_delta": None}
        out = hexify_record_colors(rec)
        assert out["grid_change"] == [["f", "0", 9, 30, 32, 27, 29]]

    def test_grid_delta_colors_become_hex_coords_stay_int(self) -> None:
        rec = {"grid_change": None, "grid_delta": [[12, 30, 1, 14]]}
        out = hexify_record_colors(rec)
        assert out["grid_delta"] == [[12, 30, "1", "e"]]

    def test_does_not_mutate_input(self) -> None:
        rec = {"grid_change": [[15, 0, 1, 0, 0, 0, 0]], "grid_delta": None}
        _ = hexify_record_colors(rec)
        assert rec["grid_change"] == [[15, 0, 1, 0, 0, 0, 0]]  # original untouched

    def test_no_color_fields_returns_same_object(self) -> None:
        rec = {"action_counter": 1}
        assert hexify_record_colors(rec) is rec


@pytest.mark.unit
class TestHexSandboxState:
    """End-to-end: a skill reads the hex `state` exactly as the prompt shows it."""

    def _state(self) -> SandboxState:
        int_frame = [[[0, 1, 14], [15, 3, 0]]]  # one-layer animation stack
        return SandboxState(
            latest_frame={
                "state": "ONGOING",
                "score": 2,
                "available_actions": ["ACTION1"],
                "frame": frame_to_hex(int_frame),
            },
            observations=[
                {
                    "step": 1,
                    "action": "ACTION4",
                    "source": "vlm",
                    "state": "ONGOING",
                    "score": 2,
                    "frame": frame_to_hex(int_frame),
                }
            ],
        )

    def test_skill_reads_hex_cells_and_renders(self) -> None:
        code = (
            "g = state.latest_frame.frame[-1]\n"
            "assert isinstance(g[0], str), type(g[0])\n"
            "cell = g[0][2]\n"
            "img = render_grid(g)\n"  # hex input must render without error
            "obs_frame = state.observations[0].frame\n"
            "result = {'cell': cell, 'as_int': int(cell, 16),\n"
            "          'w': img.size[0], 'obs_rows': obs_frame[-1]}\n"
        )
        out = run_python_snippet(code, state=self._state(), timeout_s=15)
        assert out["success"] is True, out.get("error")
        assert out["result"]["cell"] == "e"
        assert out["result"]["as_int"] == 14
        assert out["result"]["w"] == 3
        assert out["result"]["obs_rows"] == ["01e", "f30"]

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from ...run_artifacts import RUN_ARTIFACTS_DIR_ENV, RUN_LOG_PATH_ENV, safe_slug
from .models import StepRecord


class TrajectoryStore:
    """Append-only thread-safe JSONL writer for per-step records.

    Mirrors TraceWriter's contract: parent dir auto-created, single lock, JSONL.
    `tail(n)` re-reads the file each call — fine at ARC's MAX_ACTIONS=80 and
    avoids keeping a parallel in-memory buffer.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: StepRecord) -> None:
        line = json.dumps(asdict(record), default=str)
        with self._lock, self.path.open("a") as f:
            f.write(line + "\n")

    def tail(self, n: int) -> list[dict[str, Any]]:
        if not self.path.exists() or n <= 0:
            return []
        with self._lock:
            lines = self.path.read_text().splitlines()
        return [json.loads(line) for line in lines[-n:] if line.strip()]


def default_trajectory_path(prefix: str | None = None, guid: str | None = None) -> Path:
    """Trajectory path for per-step action records."""
    artifacts_dir = os.getenv(RUN_ARTIFACTS_DIR_ENV)
    if artifacts_dir and prefix and guid:
        return (
            Path(artifacts_dir)
            / f"{safe_slug(prefix)}.{safe_slug(guid)}.trajectory.jsonl"
        )

    run_log = os.getenv(RUN_LOG_PATH_ENV)
    if run_log:
        return Path(run_log).with_suffix(".trajectory.jsonl")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir / f"trajectory-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"


def _grid_from_frame(frame: Any) -> list[list[int]] | None:
    """Extract the last grid layer of a FrameData as nested lists, or None if missing."""
    grids = getattr(frame, "frame", None)
    if not grids:
        return None
    last = grids[-1]
    try:
        return [list(row) for row in last]
    except TypeError:
        return None


def _frame_delta(
    pre_grid: list[list[int]] | None, post_grid: list[list[int]] | None
) -> dict[str, Any]:
    """Cheap cell-level diff between two grids.

    Returns one of:
      {"kind": "UNKNOWN"} when either grid is missing
      {"kind": "NO_OP",   "n": 0}
      {
        "kind": "CHANGE",
        "n": <count>,
        "bbox": (r0, r1, c0, c1),
        "components": [
            {
                "n": <count>,
                "bbox": (r0, r1, c0, c1),
                "transitions": {(before, after): count, ...},
                "cells": [{"r": r, "c": c, "from": before, "to": after}, ...],
            },
            ...
        ],
      }
    """
    if pre_grid is None or post_grid is None:
        return {"kind": "UNKNOWN", "n": 0}
    if pre_grid == post_grid:
        return {"kind": "NO_OP", "n": 0}
    diffs: list[tuple[int, int, int, int]] = []
    for r in range(min(len(pre_grid), len(post_grid))):
        pre_row, post_row = pre_grid[r], post_grid[r]
        for c in range(min(len(pre_row), len(post_row))):
            before, after = pre_row[c], post_row[c]
            if before != after:
                diffs.append((r, c, before, after))
    if not diffs:
        # Grids differ only in shape; still a change but no bbox.
        return {"kind": "CHANGE", "n": 0, "bbox": None, "components": []}
    rs = [r for r, _, _, _ in diffs]
    cs = [c for _, c, _, _ in diffs]
    return {
        "kind": "CHANGE",
        "n": len(diffs),
        "bbox": (min(rs), max(rs), min(cs), max(cs)),
        "components": _diff_components(diffs),
    }


def _diff_components(diffs: list[tuple[int, int, int, int]]) -> list[dict[str, Any]]:
    """Split changed cells into 4-connected regions, largest first."""
    values_by_position = {(r, c): (before, after) for r, c, before, after in diffs}
    remaining = set(values_by_position)
    components: list[dict[str, Any]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        positions = [start]
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    positions.append(neighbor)

        cells = [
            {
                "r": r,
                "c": c,
                "from": values_by_position[(r, c)][0],
                "to": values_by_position[(r, c)][1],
            }
            for r, c in sorted(positions)
        ]
        transitions: dict[tuple[int, int], int] = {}
        for cell in cells:
            key = (cell["from"], cell["to"])
            transitions[key] = transitions.get(key, 0) + 1

        rs = [cell["r"] for cell in cells]
        cs = [cell["c"] for cell in cells]
        components.append(
            {
                "n": len(cells),
                "bbox": (min(rs), max(rs), min(cs), max(cs)),
                "transitions": transitions,
                "cells": cells,
            }
        )
    components.sort(
        key=lambda item: (-item["n"], item["bbox"][0], item["bbox"][2])
    )
    return components


def _range_label(prefix: str, start: int, end: int) -> str:
    return f"{prefix}{start}" if start == end else f"{prefix}{start}-{end}"


def _format_transitions(component: dict[str, Any]) -> str:
    transitions = component.get("transitions") or {}
    if not transitions:
        return ""
    parts = [
        f"{before}->{after} x{count}"
        for (before, after), count in sorted(transitions.items())
    ]
    return "[" + ", ".join(parts) + "]"


def _component_label(component: dict[str, Any], *, index: int) -> str:
    bbox = component.get("bbox")
    if not bbox:
        return f"region {index + 1} {component.get('n', 0)} cells"
    r0, r1, c0, c1 = bbox
    label = f"region {index + 1}"
    suffix = _format_transitions(component)
    base = (
        f"{label} {component['n']} @ "
        f"{_range_label('r', r0, r1)} {_range_label('c', c0, c1)}"
    )
    return f"{base} {suffix}" if suffix else base


def _format_change_delta(delta: dict[str, Any], max_components: int = 3) -> str:
    n = delta.get("n", 0)
    components = delta.get("components") or []
    if not components:
        return f"CHANGE {n} cells"

    shown = components[:max_components]
    parts = [
        _component_label(component, index=i) for i, component in enumerate(shown)
    ]
    hidden = len(components) - len(shown)
    if hidden > 0:
        parts.append(f"+{hidden} regions")
    return f"CHANGE {n} cells: " + "; ".join(parts)


def _effect_tag(
    rec: dict[str, Any],
    prev_rec: dict[str, Any] | None,
    delta: dict[str, Any],
) -> str:
    """Pick the highest-priority human-readable tag for a step.

    Priority: LEVEL_UP > STATE->X > CHANGE ... > NO_OP > UNKNOWN.
    """
    if prev_rec is not None:
        prev_score = prev_rec.get("score")
        cur_score = rec.get("score")
        if (
            isinstance(prev_score, int)
            and isinstance(cur_score, int)
            and cur_score > prev_score
        ):
            return f"LEVEL_UP {prev_score}->{cur_score}"
        prev_state = prev_rec.get("state")
        cur_state = rec.get("state")
        if prev_state and cur_state and prev_state != cur_state:
            return f"STATE->{cur_state}"
    if delta["kind"] == "CHANGE":
        return _format_change_delta(delta)
    return str(delta["kind"])


def _action_with_data(rec: dict[str, Any]) -> str:
    action = rec.get("chosen_action") or "FAIL"
    data = rec.get("chosen_action_data") or {}
    if not data:
        return action
    items = ",".join(f"{k}:{v}" for k, v in data.items())
    return f"{action}{{{items}}}"


def format_compact_history(
    records: Iterable[dict[str, Any]],
    frames: Sequence[Any] | None = None,
    max_chars: int = 12000,
    reasoning_chars: int | None = None,
) -> str:
    """ONE line per step with action+data and effect tag.

    `frames` is the per-step frame history with frames[i] = pre-action-i and
    frames[i+1] = post-action-i. When provided, each row carries a CHANGE/NO_OP/
    LEVEL_UP/STATE tag derived from the frame delta. When omitted, the tag falls
    back to UNKNOWN.

    Truncation drops the OLDEST rows until total chars fit `max_chars`.
    """
    records = list(records)
    if not records:
        return "No previous actions recorded."

    rows: list[str] = []
    prev: dict[str, Any] | None = None
    for rec in records:
        ac = rec.get("action_counter")
        pre_grid: list[list[int]] | None = None
        post_grid: list[list[int]] | None = None
        if frames is not None and isinstance(ac, int):
            if 0 <= ac < len(frames):
                pre_grid = _grid_from_frame(frames[ac])
            if 0 <= ac + 1 < len(frames):
                post_grid = _grid_from_frame(frames[ac + 1])
        delta = _frame_delta(pre_grid, post_grid)
        tag = _effect_tag(rec, prev, delta)

        rows.append(f"[{ac}] {_action_with_data(rec):<22}  {tag}")
        prev = rec

    while rows and len("\n".join(rows)) > max_chars:
        rows.pop(0)
    return "\n".join(rows)


def format_full_history(
    records: Iterable[dict[str, Any]], max_chars: int = 4000
) -> str:
    """MULTI-line per step. Returned by get_recent_trajectory; includes reasoning + tool calls."""
    rows: list[str] = []
    for rec in records:
        rows.append(
            f"[{rec.get('action_counter')}] state={rec.get('state')} "
            f"score={rec.get('score')} action={rec.get('chosen_action')} "
            f"data={rec.get('chosen_action_data') or {}}"
        )
        if rec.get("reasoning"):
            rows.append(f"  why: {rec['reasoning']}")
        for call in rec.get("tool_calls", []) or []:
            rows.append(f"  tool: {call.get('name')} args={call.get('args')}")
            if call.get("result") is not None:
                rendered = json.dumps(call["result"], default=str)
                rows.append(f"    result: {rendered[:300]}")
    if not rows:
        return "No previous actions recorded."
    while rows and len("\n".join(rows)) > max_chars:
        rows.pop(0)
    return "\n".join(rows)

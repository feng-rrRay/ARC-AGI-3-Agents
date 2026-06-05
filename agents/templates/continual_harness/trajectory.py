from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from ...run_artifacts import RUN_DIR_ENV, RUN_LOG_PATH_ENV, game_artifacts
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


def default_trajectory_path(game_id: str | None = None) -> Path:
    """Trajectory path for per-step action records (per-game when inside a run)."""
    run_dir = os.getenv(RUN_DIR_ENV)
    if run_dir and game_id:
        return game_artifacts(run_dir, game_id).trajectory_path

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


def _short_action(rec: dict[str, Any]) -> str:
    """Short renderable action label: ACTION1 or ACTION6(12,30)."""
    name = rec.get("chosen_action") or "FAIL"
    data = rec.get("chosen_action_data") or {}
    if not data:
        return name
    if "x" in data and "y" in data and len(data) == 2:
        return f"{name}({data['x']},{data['y']})"
    items = ",".join(f"{k}={v}" for k, v in data.items())
    return f"{name}({items})"


def _per_action_line(
    rec: dict[str, Any],
    frames: Sequence[Any] | None,
) -> str:
    """Render one action row with score delta + frame-delta tag."""
    ac = rec.get("action_counter")
    pre_grid: list[list[int]] | None = None
    post_grid: list[list[int]] | None = None
    if frames is not None and isinstance(ac, int):
        if 0 <= ac < len(frames):
            pre_grid = _grid_from_frame(frames[ac])
        if 0 <= ac + 1 < len(frames):
            post_grid = _grid_from_frame(frames[ac + 1])
    delta = _frame_delta(pre_grid, post_grid)

    pre_score = rec.get("score")
    score_delta = rec.get("score_delta")
    if isinstance(pre_score, int) and isinstance(score_delta, int):
        score_label = f"score {pre_score}->{pre_score + score_delta}"
    elif isinstance(pre_score, int):
        score_label = f"score {pre_score}->?"
    else:
        score_label = "score ?"

    # Promote LEVEL_UP / STATE-change tags above the cell-change tag.
    if isinstance(score_delta, int) and score_delta > 0:
        tag = f"LEVEL_UP +{score_delta}"
    else:
        state_after = rec.get("state_after")
        if state_after and state_after != rec.get("state"):
            tag = f"STATE->{state_after}"
        elif delta["kind"] == "CHANGE":
            tag = _format_change_delta(delta)
        else:
            tag = str(delta["kind"])

    action = _short_action(rec)
    return f"    {action:<18}  {score_label:<14}  {tag}"


def _group_into_batches(
    records: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group consecutive rows by batch_id.

    A None batch_id (older / auto_reset / legacy rows) forms its own
    single-row batch. Consecutive rows sharing a non-None batch_id are
    grouped together.
    """
    batches: list[list[dict[str, Any]]] = []
    for rec in records:
        bid = rec.get("batch_id")
        if (
            batches
            and bid is not None
            and batches[-1]
            and batches[-1][-1].get("batch_id") == bid
        ):
            batches[-1].append(rec)
        else:
            batches.append([rec])
    return batches


def _render_batch_block(
    batch: list[dict[str, Any]],
    frames: Sequence[Any] | None,
) -> str:
    """Render one batch (1+ consecutive same-batch_id rows) as a block."""
    if not batch:
        return ""
    first = batch[0]
    source = first.get("source") or "vlm"
    bid = first.get("batch_id") or "-"
    step = first.get("action_counter")
    skill_id = first.get("skill_id")

    # Header line.
    header_bits = [f"batch {bid}", f"step {step}", f"source={source}"]
    if source == "run_skill" and skill_id:
        header_bits[-1] = f"source=run_skill {skill_id}"
    header = "[" + " | ".join(header_bits) + "]"

    # Abort marker if batch_total > number of executed rows in this batch.
    total = first.get("batch_total")
    executed = len(batch)
    abort_suffix = ""
    if isinstance(total, int) and total > executed:
        abort_suffix = f"  ⚠ ABORTED at {executed}/{total}"

    lines: list[str] = [header + abort_suffix]

    reasoning = first.get("batch_reasoning") or first.get("reasoning")
    if reasoning:
        # Single-line reasoning; collapse internal newlines.
        compact = " ".join(str(reasoning).split())
        if len(compact) > 220:
            compact = compact[:217] + "..."
        lines.append(f"  reasoning: {compact!r}")

    if isinstance(total, int):
        lines.append(f"  executed {executed}/{total}:")
    elif executed > 1:
        lines.append(f"  executed {executed}:")
    else:
        lines.append("  executed:")

    for rec in batch:
        lines.append(_per_action_line(rec, frames))

    # Show rejection notes only on the head (if present) — they all live there.
    rejected = first.get("batch_rejected") or []
    if rejected:
        for r in rejected[:4]:
            reason = r.get("reason") if isinstance(r, dict) else str(r)
            pos = r.get("position") if isinstance(r, dict) else "?"
            lines.append(f"    [rejected at position {pos}: {reason}]")
        if len(rejected) > 4:
            lines.append(f"    [+{len(rejected) - 4} more rejected]")

    if isinstance(total, int) and total > executed:
        skipped = total - executed
        lines.append(f"    [+{skipped} step(s) skipped after partial-execution abort]")

    return "\n".join(lines)


def format_compact_history(
    records: Iterable[dict[str, Any]],
    frames: Sequence[Any] | None = None,
    max_chars: int = 12000,
    max_batches: int = 5,
    reasoning_chars: int | None = None,  # kept for backward-compat; ignored
) -> str:
    """Batch-grouped action history, last `max_batches` batches.

    `frames` is the per-step frame history with frames[i] = pre-action-i and
    frames[i+1] = post-action-i. When provided, each row carries a CHANGE/
    NO_OP/LEVEL_UP/STATE tag derived from the frame delta plus score
    transition. Without frames, only score/state info is shown.

    Truncation drops the OLDEST batches until total chars fit `max_chars`.
    """
    del reasoning_chars  # signature kept for callers; new layout chooses its own width
    record_list = [r for r in records if isinstance(r, dict)]
    if not record_list:
        return "No previous actions recorded."

    batches = _group_into_batches(record_list)
    if len(batches) > max_batches:
        batches = batches[-max_batches:]

    blocks = [_render_batch_block(b, frames) for b in batches]
    while blocks and len("\n\n".join(blocks)) > max_chars:
        blocks.pop(0)
    return "\n\n".join(blocks) if blocks else "No previous actions recorded."


def format_full_history(
    records: Iterable[dict[str, Any]], max_chars: int = 4000
) -> str:
    """Full per-action detail with reasoning + tool calls, grouped by batch.

    Returned by `get_recent_trajectory`. Drops oldest batches to fit `max_chars`.
    """
    record_list = [r for r in records if isinstance(r, dict)]
    if not record_list:
        return "No previous actions recorded."

    batches = _group_into_batches(record_list)
    blocks: list[str] = []

    for batch in batches:
        first = batch[0]
        source = first.get("source") or "vlm"
        bid = first.get("batch_id") or "-"
        step = first.get("action_counter")
        header = f"[batch {bid} | step {step} | source={source}]"
        if source == "run_skill" and first.get("skill_id"):
            header = f"[batch {bid} | step {step} | source=run_skill {first['skill_id']}]"

        total = first.get("batch_total")
        executed = len(batch)
        if isinstance(total, int) and total > executed:
            header += f"  ⚠ ABORTED at {executed}/{total}"

        block_lines = [header]
        reasoning = first.get("batch_reasoning") or first.get("reasoning")
        if reasoning:
            block_lines.append(f"  reasoning: {reasoning}")

        for rec in batch:
            ac = rec.get("action_counter")
            block_lines.append(
                f"  [{ac}] state={rec.get('state')}->"
                f"{rec.get('state_after') or rec.get('state')} "
                f"score={rec.get('score')}->"
                f"{(rec.get('score') or 0) + (rec.get('score_delta') or 0)} "
                f"action={_short_action(rec)}"
            )
            gd = rec.get("grid_delta")
            if gd:
                shown = gd[:10]
                parts = [f"({c[0]},{c[1]}) {c[2]}→{c[3]}" for c in shown]
                more = f" +{len(gd) - 10} more" if len(gd) > 10 else ""
                block_lines.append(
                    f"    grid: {len(gd)} cells changed — {', '.join(parts)}{more}"
                )
            if rec.get("reasoning") and rec is not first:
                block_lines.append(f"    why: {rec['reasoning']}")
            for call in rec.get("tool_calls", []) or []:
                block_lines.append(
                    f"    tool: {call.get('name')} args={call.get('args')}"
                )
                if call.get("result") is not None:
                    rendered = json.dumps(call["result"], default=str)
                    block_lines.append(f"      result: {rendered[:300]}")

        rejected = first.get("batch_rejected") or []
        for r in rejected:
            reason = r.get("reason") if isinstance(r, dict) else str(r)
            pos = r.get("position") if isinstance(r, dict) else "?"
            block_lines.append(f"  [rejected pos {pos}: {reason}]")

        blocks.append("\n".join(block_lines))

    while blocks and len("\n\n".join(blocks)) > max_chars:
        blocks.pop(0)
    return "\n\n".join(blocks)

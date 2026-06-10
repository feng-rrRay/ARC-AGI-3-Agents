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


# --- COMMENTED OUT (dead; superseded by render_recent_history) ---
# def _grid_from_frame(frame: Any) -> list[list[int]] | None:
#     """Extract the last grid layer of a FrameData as nested lists, or None if missing."""
#     grids = getattr(frame, "frame", None)
#     if not grids:
#         return None
#     last = grids[-1]
#     try:
#         return [list(row) for row in last]
#     except TypeError:
#         return None


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


# --- COMMENTED OUT (dead; superseded by render_recent_history) ---
# def _range_label(prefix: str, start: int, end: int) -> str:
#     return f"{prefix}{start}" if start == end else f"{prefix}{start}-{end}"


# --- COMMENTED OUT (dead; superseded by render_recent_history) ---
# def _format_transitions(component: dict[str, Any]) -> str:
#     transitions = component.get("transitions") or {}
#     if not transitions:
#         return ""
#     parts = [
#         f"{before}->{after} x{count}"
#         for (before, after), count in sorted(transitions.items())
#     ]
#     return "[" + ", ".join(parts) + "]"


# --- COMMENTED OUT (dead; superseded by render_recent_history) ---
# def _component_label(component: dict[str, Any], *, index: int) -> str:
#     bbox = component.get("bbox")
#     if not bbox:
#         return f"region {index + 1} {component.get('n', 0)} cells"
#     r0, r1, c0, c1 = bbox
#     label = f"region {index + 1}"
#     suffix = _format_transitions(component)
#     base = (
#         f"{label} {component['n']} @ "
#         f"{_range_label('r', r0, r1)} {_range_label('c', c0, c1)}"
#     )
#     return f"{base} {suffix}" if suffix else base


# --- COMMENTED OUT (dead; superseded by render_recent_history) ---
# def _format_change_delta(delta: dict[str, Any], max_components: int = 3) -> str:
#     n = delta.get("n", 0)
#     components = delta.get("components") or []
#     if not components:
#         return f"CHANGE {n} cells"
#
#     shown = components[:max_components]
#     parts = [
#         _component_label(component, index=i) for i, component in enumerate(shown)
#     ]
#     hidden = len(components) - len(shown)
#     if hidden > 0:
#         parts.append(f"+{hidden} regions")
#     return f"CHANGE {n} cells: " + "; ".join(parts)


# --- COMMENTED OUT (dead; superseded by render_recent_history) ---
# def _effect_tag(
#     rec: dict[str, Any],
#     prev_rec: dict[str, Any] | None,
#     delta: dict[str, Any],
# ) -> str:
#     """Pick the highest-priority human-readable tag for a step.
#
#     Priority: LEVEL_UP > STATE->X > CHANGE ... > NO_OP > UNKNOWN.
#     """
#     if prev_rec is not None:
#         prev_score = prev_rec.get("score")
#         cur_score = rec.get("score")
#         if (
#             isinstance(prev_score, int)
#             and isinstance(cur_score, int)
#             and cur_score > prev_score
#         ):
#             return f"LEVEL_UP {prev_score}->{cur_score}"
#         prev_state = prev_rec.get("state")
#         cur_state = rec.get("state")
#         if prev_state and cur_state and prev_state != cur_state:
#             return f"STATE->{cur_state}"
#     if delta["kind"] == "CHANGE":
#         return _format_change_delta(delta)
#     return str(delta["kind"])


# --- COMMENTED OUT (dead; superseded by render_recent_history) ---
# def _action_with_data(rec: dict[str, Any]) -> str:
#     action = rec.get("chosen_action") or "FAIL"
#     data = rec.get("chosen_action_data") or {}
#     if not data:
#         return action
#     items = ",".join(f"{k}:{v}" for k, v in data.items())
#     return f"{action}{{{items}}}"


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


# --- COMMENTED OUT (dead; superseded by render_recent_history) ---
# def _per_action_line(
#     rec: dict[str, Any],
#     frames: Sequence[Any] | None,
# ) -> str:
#     """Render one action row with score delta + frame-delta tag."""
#     ac = rec.get("action_counter")
#     pre_grid: list[list[int]] | None = None
#     post_grid: list[list[int]] | None = None
#     if frames is not None and isinstance(ac, int):
#         if 0 <= ac < len(frames):
#             pre_grid = _grid_from_frame(frames[ac])
#         if 0 <= ac + 1 < len(frames):
#             post_grid = _grid_from_frame(frames[ac + 1])
#     delta = _frame_delta(pre_grid, post_grid)
#
#     pre_score = rec.get("score")
#     score_delta = rec.get("score_delta")
#     if isinstance(pre_score, int) and isinstance(score_delta, int):
#         score_label = f"score {pre_score}->{pre_score + score_delta}"
#     elif isinstance(pre_score, int):
#         score_label = f"score {pre_score}->?"
#     else:
#         score_label = "score ?"
#
#     # Promote LEVEL_UP / STATE-change tags above the cell-change tag.
#     if isinstance(score_delta, int) and score_delta > 0:
#         tag = f"LEVEL_UP +{score_delta}"
#     else:
#         state_after = rec.get("state_after")
#         if state_after and state_after != rec.get("state"):
#             tag = f"STATE->{state_after}"
#         elif delta["kind"] == "CHANGE":
#             tag = _format_change_delta(delta)
#         else:
#             tag = str(delta["kind"])
#
#     action = _short_action(rec)
#     return f"    {action:<18}  {score_label:<14}  {tag}"


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


# --- COMMENTED OUT (dead; superseded by render_recent_history) ---
# def _render_batch_block(
#     batch: list[dict[str, Any]],
#     frames: Sequence[Any] | None,
# ) -> str:
#     """Render one batch (1+ consecutive same-batch_id rows) as a block."""
#     if not batch:
#         return ""
#     first = batch[0]
#     source = first.get("source") or "vlm"
#     bid = first.get("batch_id") or "-"
#     step = first.get("action_counter")
#     skill_id = first.get("skill_id")
#
#     # Header line.
#     header_bits = [f"batch {bid}", f"step {step}", f"source={source}"]
#     if source == "run_skill" and skill_id:
#         header_bits[-1] = f"source=run_skill {skill_id}"
#     header = "[" + " | ".join(header_bits) + "]"
#
#     # Abort marker if batch_total > number of executed rows in this batch.
#     total = first.get("batch_total")
#     executed = len(batch)
#     abort_suffix = ""
#     if isinstance(total, int) and total > executed:
#         abort_suffix = f"  ⚠ ABORTED at {executed}/{total}"
#
#     lines: list[str] = [header + abort_suffix]
#
#     reasoning = first.get("batch_reasoning") or first.get("reasoning")
#     if reasoning:
#         # Single-line reasoning; collapse internal newlines.
#         compact = " ".join(str(reasoning).split())
#         if len(compact) > 220:
#             compact = compact[:217] + "..."
#         lines.append(f"  reasoning: {compact!r}")
#
#     if isinstance(total, int):
#         lines.append(f"  executed {executed}/{total}:")
#     elif executed > 1:
#         lines.append(f"  executed {executed}:")
#     else:
#         lines.append("  executed:")
#
#     for rec in batch:
#         lines.append(_per_action_line(rec, frames))
#
#     # Show rejection notes only on the head (if present) — they all live there.
#     rejected = first.get("batch_rejected") or []
#     if rejected:
#         for r in rejected[:4]:
#             reason = r.get("reason") if isinstance(r, dict) else str(r)
#             pos = r.get("position") if isinstance(r, dict) else "?"
#             lines.append(f"    [rejected at position {pos}: {reason}]")
#         if len(rejected) > 4:
#             lines.append(f"    [+{len(rejected) - 4} more rejected]")
#
#     if isinstance(total, int) and total > executed:
#         skipped = total - executed
#         lines.append(f"    [+{skipped} step(s) skipped after partial-execution abort]")
#
#     return "\n".join(lines)


# --- COMMENTED OUT (dead; superseded by render_recent_history) ---
# def format_compact_history(
#     records: Iterable[dict[str, Any]],
#     frames: Sequence[Any] | None = None,
#     max_chars: int = 12000,
#     max_batches: int = 5,
#     reasoning_chars: int | None = None,  # kept for backward-compat; ignored
# ) -> str:
#     """Batch-grouped action history, last `max_batches` batches.
#
#     `frames` is the per-step frame history with frames[i] = pre-action-i and
#     frames[i+1] = post-action-i. When provided, each row carries a CHANGE/
#     NO_OP/LEVEL_UP/STATE tag derived from the frame delta plus score
#     transition. Without frames, only score/state info is shown.
#
#     Truncation drops the OLDEST batches until total chars fit `max_chars`.
#     """
#     del reasoning_chars  # signature kept for callers; new layout chooses its own width
#     record_list = [r for r in records if isinstance(r, dict)]
#     if not record_list:
#         return "No previous actions recorded."
#
#     batches = _group_into_batches(record_list)
#     if len(batches) > max_batches:
#         batches = batches[-max_batches:]
#
#     blocks = [_render_batch_block(b, frames) for b in batches]
#     while blocks and len("\n\n".join(blocks)) > max_chars:
#         blocks.pop(0)
#     return "\n\n".join(blocks) if blocks else "No previous actions recorded."


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


# ---------------------------------------------------------------------------
# Accumulative (append-only) RECENT HISTORY rendering.
#
# A pure, deterministic function of the immutable trajectory records: the same
# records always render to the same bytes, so every line except the newest is
# byte-identical across VLM calls and is served from the model's KV cache. The
# log is segmented by level (grouped on the `score` field) and, within a level,
# by attempt (split on RESET rows). Cleared levels collapse to one archive line;
# finished attempts collapse to one summary line each (run-length-collapsed when
# identical); only the current attempt is rendered per-action, with consecutive
# identical (action, effect) rows collapsed.
# ---------------------------------------------------------------------------


def summarize_grid_transitions(
    pre_grid: list[list[int]] | None,
    post_grid: list[list[int]] | None,
) -> list[list[int]] | None:
    """Group changed cells by (from_color, to_color); return per-transition rows.

    Each row is ``[from, to, count, r0, r1, c0, c1]`` (count + bounding box of all
    cells sharing that transition), sorted by count descending. Uncapped — bounded
    only by the number of distinct colour transitions. Returns None when either
    grid is missing or nothing changed (the caller distinguishes "no change" from
    "invalid action" via the record's ``state_after``).
    """
    if pre_grid is None or post_grid is None:
        return None
    groups: dict[tuple[int, int], list[int]] = {}
    for y, (prow, qrow) in enumerate(zip(pre_grid, post_grid)):
        for x, (a, b) in enumerate(zip(prow, qrow)):
            if a == b:
                continue
            key = (int(a), int(b))
            g = groups.get(key)
            if g is None:
                groups[key] = [1, y, y, x, x]
            else:
                g[0] += 1
                g[1] = min(g[1], y)
                g[2] = max(g[2], y)
                g[3] = min(g[3], x)
                g[4] = max(g[4], x)
    if not groups:
        return None
    out = [[f, t, c, r0, r1, c0, c1] for (f, t), (c, r0, r1, c0, c1) in groups.items()]
    out.sort(key=lambda e: (-e[2], e[3], e[5]))
    return out


def _range_token(r0: int, r1: int, c0: int, c1: int) -> str:
    rs = f"r{r0}" if r0 == r1 else f"r{r0}-{r1}"
    cs = f"c{c0}" if c0 == c1 else f"c{c0}-{c1}"
    return f"{rs} {cs}"


def _render_transitions(transitions: list[list[int]], *, max_groups: int = 6) -> str:
    total = sum(int(e[2]) for e in transitions)
    parts: list[str] = []
    for e in transitions[:max_groups]:
        f, t, cnt, r0, r1, c0, c1 = e[0], e[1], e[2], e[3], e[4], e[5], e[6]
        times = f" (×{cnt})" if cnt > 1 else ""
        parts.append(f"color {f}→{t}{times}: {_range_token(r0, r1, c0, c1)}")
    if len(transitions) > max_groups:
        parts.append(f"+{len(transitions) - max_groups} more")
    noun = "cell" if total == 1 else "cells"
    return f"{total} {noun} changed: " + "; ".join(parts)


def _transitions_from_delta(grid_delta: list[list[int]] | None) -> list[list[int]] | None:
    """Fallback for older records that stored only per-cell ``grid_delta``."""
    if not grid_delta:
        return None
    groups: dict[tuple[int, int], list[int]] = {}
    for cell in grid_delta:
        if not (isinstance(cell, (list, tuple)) and len(cell) >= 4):
            continue
        x, y, a, b = int(cell[0]), int(cell[1]), int(cell[2]), int(cell[3])
        key = (a, b)
        g = groups.get(key)
        if g is None:
            groups[key] = [1, y, y, x, x]
        else:
            g[0] += 1
            g[1] = min(g[1], y)
            g[2] = max(g[2], y)
            g[3] = min(g[3], x)
            g[4] = max(g[4], x)
    if not groups:
        return None
    out = [[f, t, c, r0, r1, c0, c1] for (f, t), (c, r0, r1, c0, c1) in groups.items()]
    out.sort(key=lambda e: (-e[2], e[3], e[5]))
    return out


def _short_action_label(rec: dict[str, Any]) -> str:
    name = rec.get("chosen_action") or "?"
    data = rec.get("chosen_action_data") or {}
    if "x" in data and "y" in data:
        return f"{name}(x={data['x']}, y={data['y']})"
    if data:
        return f"{name}(" + ", ".join(f"{k}={v}" for k, v in data.items()) + ")"
    return str(name)


def _render_effect(rec: dict[str, Any]) -> str:
    score = rec.get("score")
    sd = rec.get("score_delta")
    if isinstance(sd, int) and sd > 0 and isinstance(score, int):
        return f"LEVEL UP {score}→{score + sd}"
    state, after = rec.get("state"), rec.get("state_after")
    if after == "INVALID":
        return "invalid (no frame returned)"
    if after and after != state and after in ("GAME_OVER", "WIN"):
        return f"→ {after}"
    transitions = rec.get("grid_change")
    if transitions is None:
        transitions = _transitions_from_delta(rec.get("grid_delta"))
    if not transitions:
        return "no change"
    return _render_transitions(transitions)


def _source_label(rec: dict[str, Any]) -> str:
    src = rec.get("source") or "vlm"
    if src == "vlm":
        return "take_actions"
    if src == "run_skill":
        sid = rec.get("skill_id")
        return f"run_skill: {sid}" if sid else "run_skill"
    if src == "auto_reset":
        return "reset"
    return str(src)


def _is_reset(rec: dict[str, Any]) -> bool:
    return rec.get("chosen_action") == "RESET" or rec.get("source") == "auto_reset"


def _group_levels(records: list[dict[str, Any]]) -> list[tuple[int, list[dict[str, Any]]]]:
    """Group records into contiguous runs sharing a ``score`` (= one level)."""
    levels: list[tuple[int, list[dict[str, Any]]]] = []
    for r in records:
        s = r.get("score")
        s = 0 if not isinstance(s, int) else s
        if levels and levels[-1][0] == s:
            levels[-1][1].append(r)
        else:
            levels.append((s, [r]))
    return levels


def _split_attempts(level_rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split a level's rows into attempts; RESET rows are dropped as separators."""
    attempts: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    for r in level_rows:
        if _is_reset(r):
            if cur:
                attempts.append(cur)
                cur = []
        else:
            cur.append(r)
    if cur:
        attempts.append(cur)
    return attempts


def _ac_range(rows: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    acs = [r.get("action_counter") for r in rows if isinstance(r.get("action_counter"), int)]
    return (min(acs), max(acs)) if acs else (None, None)


def _attempt_summary_key(attempt: list[dict[str, Any]]) -> tuple[str, str]:
    if not attempt:
        return ("empty", "")
    last = attempt[-1]
    after = last.get("state_after")
    if after == "GAME_OVER":
        return ("GAME_OVER", _short_action_label(last))
    return ("ended", str(after or "?"))


def _render_attempt_summaries(prior_attempts: list[list[dict[str, Any]]]) -> list[str]:
    lines: list[str] = []
    i, n, idx = 0, len(prior_attempts), 1
    while i < n:
        j = i
        key = _attempt_summary_key(prior_attempts[i])
        while j + 1 < n and _attempt_summary_key(prior_attempts[j + 1]) == key:
            j += 1
        kind = key[0]
        if i == j:
            a, b = _ac_range(prior_attempts[i])
            steps = len(prior_attempts[i])
            if kind == "GAME_OVER":
                lines.append(
                    f"Attempt {idx} (steps {a}-{b}, {steps} steps): "
                    f"GAME_OVER after {key[1]} at step {b}."
                )
            else:
                lines.append(
                    f"Attempt {idx} (steps {a}-{b}, {steps} steps): ended {key[1]}."
                )
        else:
            tries = j - i + 1
            a0, _ = _ac_range(prior_attempts[i])
            _, b1 = _ac_range(prior_attempts[j])
            if kind == "GAME_OVER":
                lines.append(
                    f"Attempts {idx}-{idx + tries - 1} ({tries} tries, steps "
                    f"{a0}-{b1}): each GAME_OVER after {key[1]}."
                )
            else:
                lines.append(
                    f"Attempts {idx}-{idx + tries - 1} ({tries} tries, steps "
                    f"{a0}-{b1}): each ended {key[1]}."
                )
        idx += j - i + 1
        i = j + 1
    return lines


def _render_attempt_detail(rows: list[dict[str, Any]], *, max_chars: int) -> list[str]:
    if not rows:
        return ["(no actions yet)"]
    # Run-length collapse consecutive rows sharing (source, action, effect).
    units: list[list[Any]] = []  # [src, action, effect, first_ac, last_ac, count]
    for r in rows:
        src, act, eff = _source_label(r), _short_action_label(r), _render_effect(r)
        ac = r.get("action_counter")
        if units and units[-1][0] == src and units[-1][1] == act and units[-1][2] == eff:
            units[-1][4] = ac
            units[-1][5] += 1
        else:
            units.append([src, act, eff, ac, ac, 1])
    lines: list[str] = []
    for src, act, eff, a, b, count in units:
        if count == 1:
            lines.append(f"[{src}] step {a}  {act} → {eff}")
        else:
            note = " (no progress — possible loop)" if eff == "no change" and count >= 10 else ""
            lines.append(f"[{src}] steps {a}-{b}  {act} ×{count} → {eff}{note}")
    if len("\n".join(lines)) > max_chars:
        while len(lines) > 1 and len("\n".join(lines)) > max_chars:
            lines.pop(0)
        lines.insert(0, "… (earlier steps of this attempt elided) …")
    return lines


def render_recent_history(
    records: Iterable[dict[str, Any]], max_chars: int = 8000
) -> str:
    """Append-only, level/attempt-segmented action log (see module note above)."""
    recs = [r for r in records if isinstance(r, dict)]
    if not recs:
        return "No previous actions recorded."

    levels = _group_levels(recs)
    out: list[str] = []

    prior_levels = levels[:-1]
    if prior_levels:
        archive: list[str] = []
        for score, rows in prior_levels:
            attempts = _split_attempts(rows)
            steps = sum(len(a) for a in attempts)
            tries = len(attempts)
            lvl = score + 1
            if tries > 1:
                archive.append(f"level {lvl} cleared in {tries} attempts / {steps} steps")
            else:
                archive.append(f"level {lvl} cleared in {steps} steps")
        out.append("Levels cleared: " + ", ".join(archive) + ".")
        out.append("")

    cur_score, cur_rows = levels[-1]
    attempts = _split_attempts(cur_rows)
    first_ac = next(
        (r.get("action_counter") for att in attempts for r in att
         if isinstance(r.get("action_counter"), int)),
        None,
    )
    out.append(f"══ Level {cur_score + 1} — started at step {first_ac} ══")

    prior_attempts = attempts[:-1] if attempts else []
    cur_attempt = attempts[-1] if attempts else []
    out.extend(_render_attempt_summaries(prior_attempts))
    if prior_attempts:
        ca0, _ = _ac_range(cur_attempt)
        out.append(f"Attempt {len(prior_attempts) + 1} — current (started step {ca0}):")
    out.extend(_render_attempt_detail(cur_attempt, max_chars=max_chars))

    return "\n".join(out)

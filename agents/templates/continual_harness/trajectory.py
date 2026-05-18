from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

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


def default_trajectory_path() -> Path:
    """Sibling of main.py's text log when RUN_LOG_PATH is set; else logs/trajectory-<ts>.jsonl."""
    run_log = os.getenv("RUN_LOG_PATH")
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
      {"kind": "CHANGE",  "n": <count>, "bbox": (r0, r1, c0, c1)}
    """
    if pre_grid is None or post_grid is None:
        return {"kind": "UNKNOWN", "n": 0}
    if pre_grid == post_grid:
        return {"kind": "NO_OP", "n": 0}
    diffs: list[tuple[int, int]] = []
    for r in range(min(len(pre_grid), len(post_grid))):
        pre_row, post_row = pre_grid[r], post_grid[r]
        for c in range(min(len(pre_row), len(post_row))):
            if pre_row[c] != post_row[c]:
                diffs.append((r, c))
    if not diffs:
        # Grids differ only in shape; still a change but no bbox.
        return {"kind": "CHANGE", "n": 0, "bbox": None}
    rs, cs = zip(*diffs)
    return {
        "kind": "CHANGE",
        "n": len(diffs),
        "bbox": (min(rs), max(rs), min(cs), max(cs)),
    }


def _effect_tag(
    rec: dict[str, Any],
    prev_rec: dict[str, Any] | None,
    delta: dict[str, Any],
) -> str:
    """Pick the highest-priority human-readable tag for a step.

    Priority: LEVEL_UP > STATE→X > CHANGE(...) > NO_OP > UNKNOWN.
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
        bbox = delta.get("bbox")
        if bbox:
            r0, r1, c0, c1 = bbox
            return f"CHANGE({delta['n']}px @ rows {r0}-{r1}, cols {c0}-{c1})"
        return f"CHANGE({delta['n']}px)"
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
    reasoning_chars: int = 150,
) -> str:
    """ONE line per step with action+data, effect tag, and a short reasoning snippet.

    `frames` is the per-step frame history with frames[i] = pre-action-i and
    frames[i+1] = post-action-i. When provided, each row carries a CHANGE/NO_OP/
    LEVEL_UP/STATE tag derived from the frame delta. When omitted, the tag falls
    back to UNKNOWN (still useful: reasoning and action+data are unaffected).

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

        why = (rec.get("reasoning") or "").strip().replace("\n", " ")
        if len(why) > reasoning_chars:
            why = why[: reasoning_chars - 1].rstrip() + "…"
        why_str = f'  "{why}"' if why else ""

        rows.append(f"[{ac}] {_action_with_data(rec):<22}  {tag}{why_str}")
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

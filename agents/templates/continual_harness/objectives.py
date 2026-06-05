"""Objective system for ContinualHarness — a maintained queue of near-term goals.

A deliberately simplified port of PokeAgent's DirectObjective system. ARC-AGI-3
games are short and fast, so this drops categories, legacy/categorized modes,
coords, priorities, and completion conditions: one flat ordered queue with
always-replace replanning. A dedicated built-in planner (see
``ContinualHarness._run_plan_objectives``) refills it with 3 fresh objectives
whenever the active queue runs low.

Backed by a single per-game JSON file (``logs/<run>/<game_id>/objectives.json``)
holding BOTH active and completed objectives, written atomically — the same
store pattern as ``memory.py`` / ``skills.py``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ...run_artifacts import game_artifacts
from ._locks import lock_for_path

logger = logging.getLogger(__name__)


DESCRIPTION_MAX_CHARS = 300
HINT_MAX_CHARS = 500
COMPLETED_SHOWN_DEFAULT = 5


@dataclass(slots=True)
class DirectObjective:
    """A single near-term goal in the objective queue.

    Minimal by design (see module docstring): a human-readable ``description``
    of what to accomplish plus an optional ``hint`` on how to approach it / what
    completion looks like. ``completed`` + ``completed_at`` track progress.
    """

    id: str
    description: str
    hint: str = ""
    completed: bool = False
    created_at: str = ""
    completed_at: str | None = None


RUN_DIR_ENV = "RUN_DIR"
RUN_LOG_PATH_ENV = "RUN_LOG_PATH"


def active_objectives_path(game_id: str | None = None) -> Path:
    """Return the backing objectives file for this agent.

    Per-game when a game_id and RUN_DIR are available (the swarm default), so
    games running in parallel threads never share an objectives file. Mirrors
    ``active_memory_path`` exactly.
    """
    run_dir = os.getenv(RUN_DIR_ENV)
    if run_dir and game_id:
        return game_artifacts(run_dir, game_id).objectives_path

    if run_dir:
        return Path(run_dir) / "objectives.json"

    run_log = os.getenv(RUN_LOG_PATH_ENV)
    if run_log:
        return Path(run_log).with_name("objectives.json")

    return Path("logs") / "continual_harness.objectives.json"


def coerce_objective_specs(raw: Any) -> list[dict[str, str]]:
    """Normalise a planner-supplied objectives array into ``{description, hint}`` dicts.

    Accepts a list of dicts (with ``description`` + optional ``hint``) or bare
    strings. Entries without a non-empty description are dropped.
    """
    specs: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return specs
    for item in raw:
        if isinstance(item, dict):
            description = str(item.get("description") or "").strip()
            hint = str(item.get("hint") or "").strip()
        elif isinstance(item, str):
            description, hint = item.strip(), ""
        else:
            continue
        if description:
            specs.append({"description": description, "hint": hint})
    return specs


class DirectObjectiveManager:
    """Single-file JSON-backed objective queue. Same store pattern as MemoryStore.

    State schema: ``{"next_id": 1, "objectives": [ <DirectObjective dict>, ... ]}``
    — one ordered list holding both active and completed objectives. The active
    queue is the sublist of objectives with ``completed=False``, in list order;
    the current objective is the first of those.
    """

    def __init__(self, path: Path, game_id: str) -> None:
        self.path = path
        self.game_id = game_id
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = lock_for_path(path)

    # ---- internal ----------------------------------------------------------

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {"next_id": 1, "objectives": []}

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty_state()
        try:
            with self.path.open() as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "objectives file %s unreadable (%s); starting fresh", self.path, exc
            )
            return self._empty_state()
        if not isinstance(state, dict):
            return self._empty_state()
        state.setdefault("next_id", 1)
        state.setdefault("objectives", [])
        # Advance next_id past the largest existing numeric suffix so hand-edited
        # files don't produce colliding IDs.
        max_n = 0
        for o in state["objectives"]:
            obj_id = o.get("id") if isinstance(o, dict) else None
            if isinstance(obj_id, str) and obj_id.startswith("obj_"):
                suffix = obj_id.split("_", 1)[1]
                if suffix.isdigit():
                    max_n = max(max_n, int(suffix))
        if state["next_id"] <= max_n:
            state["next_id"] = max_n + 1
        return state

    def _save(self, state: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(state, f, indent=2)
        tmp.replace(self.path)

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "…"

    # ---- public API --------------------------------------------------------

    def all_objectives(self) -> list[DirectObjective]:
        with self._lock:
            state = self._load()
        return [DirectObjective(**o) for o in state["objectives"]]

    def active(self) -> list[DirectObjective]:
        return [o for o in self.all_objectives() if not o.completed]

    def active_count(self) -> int:
        return len(self.active())

    def current(self) -> DirectObjective | None:
        for o in self.all_objectives():
            if not o.completed:
                return o
        return None

    def recent_completed(self, n: int = COMPLETED_SHOWN_DEFAULT) -> list[DirectObjective]:
        completed = [o for o in self.all_objectives() if o.completed]
        completed.sort(key=lambda o: o.completed_at or "", reverse=True)
        return completed[: max(0, n)]

    def replace_active(self, specs: list[dict[str, str]]) -> list[DirectObjective]:
        """Discard remaining active objectives, append fresh ones from ``specs``.

        Completed objectives are kept as history. ``specs`` is a list of
        ``{description, hint?}`` dicts (already coerced). Returns the newly
        created objectives. Persists immediately.
        """
        with self._lock:
            state = self._load()
            # Keep only completed objectives; drop the stale active queue.
            kept = [o for o in state["objectives"] if o.get("completed")]
            created: list[DirectObjective] = []
            now = self._now()
            for spec in specs:
                description = self._truncate(
                    str(spec.get("description") or ""), DESCRIPTION_MAX_CHARS
                )
                if not description:
                    continue
                hint = self._truncate(str(spec.get("hint") or ""), HINT_MAX_CHARS)
                obj = DirectObjective(
                    id=f"obj_{state['next_id']:03d}",
                    description=description,
                    hint=hint,
                    completed=False,
                    created_at=now,
                    completed_at=None,
                )
                kept.append(asdict(obj))
                created.append(obj)
                state["next_id"] += 1
            state["objectives"] = kept
            self._save(state)
            return created

    def complete_current(self) -> DirectObjective | None:
        """Mark the first active objective completed and persist. Returns it (or None)."""
        with self._lock:
            state = self._load()
            for o in state["objectives"]:
                if not o.get("completed"):
                    o["completed"] = True
                    o["completed_at"] = self._now()
                    self._save(state)
                    return DirectObjective(**o)
        return None


def format_objectives_section(
    manager: DirectObjectiveManager, completed_shown: int = COMPLETED_SHOWN_DEFAULT
) -> str:
    """Render the ``## CURRENT OBJECTIVE`` block auto-injected into the working prompt.

    The current objective is shown in full (description + hint); recently
    completed objectives are listed by description only (less detail).
    """
    objectives = manager.all_objectives()
    active = [o for o in objectives if not o.completed]
    current = active[0] if active else None

    lines = ["## CURRENT OBJECTIVE"]
    if current is None:
        lines.append("(No active objective — one will be planned automatically.)")
    else:
        lines.append(f"[{current.id}] {current.description}")
        if current.hint:
            lines.append(f"Hint: {current.hint}")
        if len(active) > 1:
            lines.append(f"({len(active) - 1} more objective(s) queued after this one.)")
        lines.append(
            "Call complete_direct_objective once this is verifiably done, or "
            "replan_objectives to discard and replan if it is no longer relevant."
        )

    completed = [o for o in objectives if o.completed]
    completed.sort(key=lambda o: o.completed_at or "", reverse=True)
    completed = completed[: max(0, completed_shown)]
    if completed:
        lines.append("")
        lines.append(f"Recently completed (last {len(completed)}):")
        for o in completed:
            lines.append(f"- {o.description}")
    return "\n".join(lines)

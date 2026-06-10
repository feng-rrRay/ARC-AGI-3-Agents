from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ...run_artifacts import game_artifacts
from ._locks import lock_for_path

logger = logging.getLogger(__name__)


TITLE_MAX_CHARS = 200
BODY_MAX_CHARS = 4000
MAX_ENTRIES = 50
SEARCH_MAX_MATCHES = 10


@dataclass(slots=True)
class MemoryEntry:
    id: str
    game_id: str
    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


RUN_DIR_ENV = "RUN_DIR"
RUN_LOG_PATH_ENV = "RUN_LOG_PATH"


def active_memory_path(game_id: str | None = None) -> Path:
    """Return the backing memory file for this agent.

    Per-game when a game_id and RUN_DIR are available (the swarm default), so
    games running in parallel threads never share a memory file. Falls back to
    a single run-local file only when there is no game_id (tests / non-run).
    """
    run_dir = os.getenv(RUN_DIR_ENV)
    if run_dir and game_id:
        return game_artifacts(run_dir, game_id).memory_path

    if run_dir:
        return Path(run_dir) / "memory.json"

    run_log = os.getenv(RUN_LOG_PATH_ENV)
    if run_log:
        return Path(run_log).with_name("memory.json")

    return Path("logs") / "continual_harness.memory.json"


class MemoryStore:
    """Single-file JSON-backed memory. The same path is both source and sink.

    Loaded on construction (or initialized empty if the file is missing).
    Every mutation rewrites the full state atomically via tempfile.replace.
    """

    def __init__(self, path: Path, game_id: str) -> None:
        self.path = path
        self.game_id = game_id
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = lock_for_path(path)

    # ---- internal ----------------------------------------------------------

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {"next_id": 1, "entries": []}

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty_state()
        try:
            with self.path.open() as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "memory file %s unreadable (%s); starting fresh", self.path, exc
            )
            return self._empty_state()
        if not isinstance(state, dict):
            return self._empty_state()
        state.setdefault("next_id", 1)
        state.setdefault("entries", [])
        # If the file was hand-edited and next_id is stale, advance past the largest
        # existing numeric suffix so new IDs don't collide.
        max_n = 0
        for e in state["entries"]:
            entry_id = e.get("id") if isinstance(e, dict) else None
            if isinstance(entry_id, str) and entry_id.startswith("mem_"):
                suffix = entry_id.split("_", 1)[1]
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

    def all_entries(self) -> list[MemoryEntry]:
        with self._lock:
            state = self._load()
        return [MemoryEntry(**e) for e in state["entries"]]

    def add(self, title: str, body: str, tags: list[str] | None = None) -> MemoryEntry:
        title = self._truncate(title, TITLE_MAX_CHARS)
        body = self._truncate(body, BODY_MAX_CHARS)
        if not title or not body:
            raise ValueError("add requires non-empty title and body")
        with self._lock:
            state = self._load()
            if len(state["entries"]) >= MAX_ENTRIES:
                raise ValueError(
                    f"memory full ({MAX_ENTRIES} entries); delete or edit an "
                    "existing entry first"
                )
            now = self._now()
            entry = MemoryEntry(
                id=f"mem_{state['next_id']:03d}",
                game_id=self.game_id,
                title=title,
                body=body,
                tags=list(tags or []),
                created_at=now,
                updated_at=now,
            )
            state["entries"].append(asdict(entry))
            state["next_id"] += 1
            self._save(state)
            return entry

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            state = self._load()
            before = len(state["entries"])
            state["entries"] = [e for e in state["entries"] if e.get("id") != entry_id]
            if len(state["entries"]) == before:
                return False
            self._save(state)
            return True

    def edit(
        self,
        entry_id: str,
        *,
        title: str | None = None,
        body: str | None = None,
        tags: list[str] | None = None,
    ) -> MemoryEntry | None:
        if title is None and body is None and tags is None:
            raise ValueError("edit requires at least one of title/body/tags")
        with self._lock:
            state = self._load()
            for entry in state["entries"]:
                if entry.get("id") == entry_id:
                    if title is not None:
                        entry["title"] = self._truncate(title, TITLE_MAX_CHARS)
                    if body is not None:
                        entry["body"] = self._truncate(body, BODY_MAX_CHARS)
                    if tags is not None:
                        entry["tags"] = list(tags)
                    entry["updated_at"] = self._now()
                    self._save(state)
                    return MemoryEntry(**entry)
        return None

    def search(self, query: str) -> tuple[list[MemoryEntry], int]:
        """Case-insensitive substring over title + body + tags.

        Returns (top_matches up to SEARCH_MAX_MATCHES, total_match_count).
        Empty query returns all entries.
        """
        with self._lock:
            state = self._load()
        entries = [MemoryEntry(**e) for e in state["entries"]]
        q = (query or "").lower().strip()
        if not q:
            matches = entries
        else:
            matches = [
                e
                for e in entries
                if q in e.title.lower()
                or q in e.body.lower()
                or any(q in t.lower() for t in e.tags)
            ]
        return matches[:SEARCH_MAX_MATCHES], len(matches)


def format_memory_overview(entries: list[MemoryEntry]) -> str:
    """Compact index for auto-injection. Body is fetched via process_memory(search)."""
    if not entries:
        return (
            "## LONG-TERM MEMORY (0 entries)\n"
            'No memories saved yet. Use process_memory(operation="add", '
            "title=..., body=..., tags=[...]) to record observations "
            "(player identity, action effects, level mechanics) that will help "
            "future steps. With --bootstrap-memory, they also persist across runs."
        )
    rows = [f"## LONG-TERM MEMORY ({len(entries)} entries)"]
    for e in entries:
        tag_str = f" ({', '.join(e.tags)})" if e.tags else ""
        first_line = f"{e.body.splitlines()[0]}" if e.body else ""
        rows.append(f"[{e.id}] {e.title}{tag_str}: {first_line}...")
    return "\n".join(rows)


def format_memory_full(entries: list[MemoryEntry]) -> str:
    """Full memory dump with bodies, for the evolution meta-call."""
    if not entries:
        return "## LONG-TERM MEMORY (0 entries)\nNo memories saved yet."
    rows = [f"## LONG-TERM MEMORY ({len(entries)} entries)"]
    for e in entries:
        tag_str = f" ({', '.join(e.tags)})" if e.tags else ""
        rows.append(f"### [{e.id}] {e.title}{tag_str}")
        rows.append(e.body)
    return "\n".join(rows)

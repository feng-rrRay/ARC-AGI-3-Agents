from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ._locks import lock_for_path

logger = logging.getLogger(__name__)


NAME_MAX_CHARS = 100
DESCRIPTION_MAX_CHARS = 500
CODE_MAX_CHARS = 8000
MAX_SKILLS = 50
SEARCH_MAX_MATCHES = 10


BOOTSTRAP_SKILLS_ENV = "CONTINUAL_HARNESS_BOOTSTRAP_SKILLS"
RUN_SKILLS_PATH_ENV = "RUN_SKILLS_PATH"
RUN_DIR_ENV = "RUN_DIR"
RUN_LOG_PATH_ENV = "RUN_LOG_PATH"

_VALID_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,99}$")


@dataclass(slots=True)
class SkillEntry:
    id: str
    game_id: str
    name: str
    description: str
    code: str
    tags: list[str] = field(default_factory=list)
    version: int = 1
    created_at: str = ""
    updated_at: str = ""


def bootstrap_skill_path() -> Path | None:
    """Explicit cross-run skill file passed via --bootstrap-skills, if any."""
    raw = os.getenv(BOOTSTRAP_SKILLS_ENV)
    return Path(raw) if raw else None


def active_skill_path() -> Path:
    """Resolve the backing skill file.

    --bootstrap-skills wins; without it, falls back to run-local storage so
    skills are always available regardless of CLI flags.
    """
    bootstrap = bootstrap_skill_path()
    if bootstrap is not None:
        return bootstrap

    raw = os.getenv(RUN_SKILLS_PATH_ENV)
    if raw:
        return Path(raw)

    run_dir = os.getenv(RUN_DIR_ENV)
    if run_dir:
        return Path(run_dir) / "skills.json"

    run_log = os.getenv(RUN_LOG_PATH_ENV)
    if run_log:
        return Path(run_log).with_name("skills.json")

    return Path("logs") / "continual_harness.skills.json"


class SkillStore:
    """Single-file JSON-backed skill registry. Atomic writes via tempfile.replace."""

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
                "skills file %s unreadable (%s); starting fresh", self.path, exc
            )
            return self._empty_state()
        if not isinstance(state, dict):
            return self._empty_state()
        state.setdefault("next_id", 1)
        state.setdefault("entries", [])
        max_n = 0
        for e in state["entries"]:
            entry_id = e.get("id") if isinstance(e, dict) else None
            if isinstance(entry_id, str) and entry_id.startswith("skill_"):
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

    def all_entries(self) -> list[SkillEntry]:
        with self._lock:
            state = self._load()
        return [SkillEntry(**e) for e in state["entries"]]

    def get(self, skill_id: str) -> SkillEntry | None:
        for entry in self.all_entries():
            if entry.id == skill_id:
                return entry
        return None

    def get_by_id_or_name(self, key: str) -> SkillEntry | None:
        """Resolve a skill by its canonical id (skill_NNN) or its unique name.

        Models frequently confuse the bracketed id shown in the overview with
        the human-readable name; accepting either avoids burning a turn on a
        retry. Name lookup is case-insensitive. Returns the FIRST match in
        store order — names are unique by add() invariants, but for legacy
        files we still favour id matches.
        """
        if not key:
            return None
        entries = self.all_entries()
        for entry in entries:
            if entry.id == key:
                return entry
        needle = key.lower()
        for entry in entries:
            if entry.name.lower() == needle:
                return entry
        return None

    def add(
        self,
        name: str,
        description: str,
        code: str,
        tags: list[str] | None = None,
    ) -> SkillEntry:
        name = (name or "").strip()
        if not _VALID_NAME.match(name):
            raise ValueError(
                f"name must match [A-Za-z][A-Za-z0-9_]{{0,99}} (got {name!r})"
            )
        description = self._truncate(description, DESCRIPTION_MAX_CHARS)
        code = (code or "").strip()
        if not code:
            raise ValueError("add requires non-empty code")
        if len(code) > CODE_MAX_CHARS:
            raise ValueError(f"code exceeds {CODE_MAX_CHARS} chars")
        with self._lock:
            state = self._load()
            if len(state["entries"]) >= MAX_SKILLS:
                raise ValueError(
                    f"skills full ({MAX_SKILLS} entries); delete or edit one first"
                )
            if any(e.get("name") == name for e in state["entries"]):
                raise ValueError(
                    f"skill name {name!r} already in use; use edit instead"
                )
            now = self._now()
            entry = SkillEntry(
                id=f"skill_{state['next_id']:03d}",
                game_id=self.game_id,
                name=name,
                description=description,
                code=code,
                tags=list(tags or []),
                version=1,
                created_at=now,
                updated_at=now,
            )
            state["entries"].append(asdict(entry))
            state["next_id"] += 1
            self._save(state)
            return entry

    def delete(self, skill_id: str) -> bool:
        with self._lock:
            state = self._load()
            before = len(state["entries"])
            state["entries"] = [e for e in state["entries"] if e.get("id") != skill_id]
            if len(state["entries"]) == before:
                return False
            self._save(state)
            return True

    def edit(
        self,
        skill_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        code: str | None = None,
        tags: list[str] | None = None,
    ) -> SkillEntry | None:
        if all(v is None for v in (name, description, code, tags)):
            raise ValueError("edit requires at least one of name/description/code/tags")
        if name is not None:
            stripped = name.strip()
            if not _VALID_NAME.match(stripped):
                raise ValueError(
                    f"name must match [A-Za-z][A-Za-z0-9_]{{0,99}} (got {name!r})"
                )
            name = stripped
        if code is not None:
            code = code.strip()
            if not code:
                raise ValueError("code must be non-empty when provided")
            if len(code) > CODE_MAX_CHARS:
                raise ValueError(f"code exceeds {CODE_MAX_CHARS} chars")
        with self._lock:
            state = self._load()
            for entry in state["entries"]:
                if entry.get("id") == skill_id:
                    if name is not None:
                        # Enforce uniqueness against OTHER entries.
                        if any(
                            e.get("name") == name and e.get("id") != skill_id
                            for e in state["entries"]
                        ):
                            raise ValueError(
                                f"skill name {name!r} already in use by another entry"
                            )
                        entry["name"] = name
                    if description is not None:
                        entry["description"] = self._truncate(
                            description, DESCRIPTION_MAX_CHARS
                        )
                    if code is not None:
                        entry["code"] = code
                    if tags is not None:
                        entry["tags"] = list(tags)
                    entry["version"] = int(entry.get("version", 1)) + 1
                    entry["updated_at"] = self._now()
                    self._save(state)
                    return SkillEntry(**entry)
        return None

    def search(self, query: str) -> tuple[list[SkillEntry], int]:
        """Case-insensitive substring over name + description + code + tags.

        Returns (top_matches up to SEARCH_MAX_MATCHES, total_match_count).
        Empty query returns all entries.
        """
        with self._lock:
            state = self._load()
        entries = [SkillEntry(**e) for e in state["entries"]]
        q = (query or "").lower().strip()
        if not q:
            matches = entries
        else:
            matches = [
                e
                for e in entries
                if q in e.name.lower()
                or q in e.description.lower()
                or q in e.code.lower()
                or any(q in t.lower() for t in e.tags)
            ]
        return matches[:SEARCH_MAX_MATCHES], len(matches)


def format_skill_overview(entries: list[SkillEntry]) -> str:
    """Compact index for auto-injection. Labels both `id` and `name` so the
    model can address a skill by either when calling run_skill / process_skill.
    """
    if not entries:
        return (
            "## SKILLS (0 saved)\n"
            'No skills saved yet. Use process_skill(operation="add", name=..., '
            "description=..., code=...) to save reusable analysis snippets, "
            "then run_skill(id_or_name) to execute one."
        )
    rows = [f"## SKILLS ({len(entries)} saved)"]
    for e in entries:
        tag_str = f" tags={','.join(e.tags)}" if e.tags else ""
        first_line = (
            (e.description or "").splitlines()[0][:120] if e.description else ""
        )
        sep = " — " if first_line else ""
        rows.append(f"- id={e.id} name={e.name}{tag_str}{sep}{first_line}")
    return "\n".join(rows)

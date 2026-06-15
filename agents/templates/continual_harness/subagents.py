from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from ...run_artifacts import game_artifacts
from ._locks import lock_for_path
from .tools import SUBAGENT_TOOL_ENUM

logger = logging.getLogger(__name__)


NAME_MAX_CHARS = 100
DESCRIPTION_MAX_CHARS = 500
SYSTEM_INSTRUCTIONS_MAX_CHARS = 4000
DIRECTIVE_MAX_CHARS = 4000
RETURN_CONDITION_MAX_CHARS = 1000
MAX_SUBAGENTS = 50
SEARCH_MAX_MATCHES = 10

# Execution modes, mirroring the PokeAgent design: "looping" runs a bounded
# action loop until subagent_return or max_turns; "one_step" runs a single VLM
# analysis turn and auto-returns.
HANDLER_TYPES: frozenset[str] = frozenset({"one_step", "looping"})
DEFAULT_HANDLER_TYPE = "looping"
# Per-subagent inner-loop bound. The agent additionally clamps to a hard ceiling
# for runaway protection; these are the store-level guards.
DEFAULT_MAX_TURNS = 25
MIN_MAX_TURNS = 1
MAX_MAX_TURNS = 50

# Empty by default: subagents already receive the compact RECENT STEPS block in
# their prompt, and get_recent_trajectory was removed. An omitted allowlist means
# the subagent only reasons over that in-prompt history and returns.
DEFAULT_SUBAGENT_ALLOWED_TOOLS: tuple[str, ...] = ()


RUN_DIR_ENV = "RUN_DIR"
RUN_LOG_PATH_ENV = "RUN_LOG_PATH"

_VALID_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,99}$")


@dataclass(slots=True)
class SubagentEntry:
    id: str
    game_id: str
    name: str
    description: str
    system_instructions: str  # static system prompt for the inner VLM
    directive: str = ""  # default per-invocation task framing (run_subagent.task overrides)
    return_condition: str = ""  # narrative "when to return" cue (not checked in code)
    handler_type: str = DEFAULT_HANDLER_TYPE  # "looping" | "one_step"
    max_turns: int = DEFAULT_MAX_TURNS  # inner-loop bound (clamped to [MIN, MAX])
    allowed_tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source: str = "orchestrator"  # "orchestrator" | "evolved"
    is_builtin: bool = False  # reserved (nothing is seeded today; delete-guarded)
    version: int = 1
    created_at: str = ""
    updated_at: str = ""


_ENTRY_FIELDS: frozenset[str] = frozenset(f.name for f in fields(SubagentEntry))


def _entry_from_dict(data: dict[str, Any]) -> SubagentEntry:
    """Reconstruct an entry, ignoring unknown keys and supplying defaults.

    Tolerant of stale files written by an earlier schema (e.g. a legacy
    ``instructions`` key) so a registry on disk never raises ``TypeError``.
    """
    known = {k: v for k, v in data.items() if k in _ENTRY_FIELDS}
    if "system_instructions" not in known and "instructions" in data:
        known["system_instructions"] = data["instructions"]
    return SubagentEntry(**known)


def active_subagent_path(game_id: str | None = None) -> Path:
    """Resolve the backing subagent file.

    Per-game when a game_id and RUN_DIR are available (the swarm default), so
    parallel games never share a subagent registry. Falls back to a single
    run-local file only when there is no game_id (tests / non-run).
    """
    run_dir = os.getenv(RUN_DIR_ENV)
    if run_dir and game_id:
        return game_artifacts(run_dir, game_id).subagents_path

    if run_dir:
        return Path(run_dir) / "subagents.json"

    run_log = os.getenv(RUN_LOG_PATH_ENV)
    if run_log:
        return Path(run_log).with_name("subagents.json")

    return Path("logs") / "continual_harness.subagents.json"


def _validate_allowed_tools(allowed: list[str]) -> list[str]:
    """Normalize, de-duplicate, and validate against SUBAGENT_TOOL_ENUM.

    Empty list is allowed: yields a prompt-only subagent that can reason but
    only ever calls subagent_return. Unknown names raise — keeps the registry
    consistent with what the inner loop will actually expose.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for name in allowed or []:
        if not isinstance(name, str):
            raise ValueError(f"allowed_tools entries must be strings (got {name!r})")
        n = name.strip()
        if not n:
            continue
        if n not in SUBAGENT_TOOL_ENUM:
            raise ValueError(
                f"allowed_tools entry {n!r} not in {sorted(SUBAGENT_TOOL_ENUM)}"
            )
        if n in seen:
            continue
        seen.add(n)
        cleaned.append(n)
    return cleaned


def _validate_handler_type(handler_type: str | None) -> str:
    ht = (handler_type or DEFAULT_HANDLER_TYPE).strip().lower()
    if ht not in HANDLER_TYPES:
        raise ValueError(
            f"handler_type must be one of {sorted(HANDLER_TYPES)} (got {handler_type!r})"
        )
    return ht


def _clamp_max_turns(max_turns: Any) -> int:
    try:
        n = int(max_turns)
    except (TypeError, ValueError):
        n = DEFAULT_MAX_TURNS
    return max(MIN_MAX_TURNS, min(n, MAX_MAX_TURNS))


class SubagentStore:
    """Single-file JSON-backed subagent registry. Atomic writes via tempfile.replace."""

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
                "subagents file %s unreadable (%s); starting fresh", self.path, exc
            )
            return self._empty_state()
        if not isinstance(state, dict):
            return self._empty_state()
        state.setdefault("next_id", 1)
        state.setdefault("entries", [])
        max_n = 0
        for e in state["entries"]:
            entry_id = e.get("id") if isinstance(e, dict) else None
            if isinstance(entry_id, str) and entry_id.startswith("subagent_"):
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

    def all_entries(self) -> list[SubagentEntry]:
        with self._lock:
            state = self._load()
        return [_entry_from_dict(e) for e in state["entries"]]

    def get(self, subagent_id: str) -> SubagentEntry | None:
        for entry in self.all_entries():
            if entry.id == subagent_id:
                return entry
        return None

    def add(
        self,
        name: str,
        description: str,
        system_instructions: str,
        *,
        directive: str = "",
        return_condition: str = "",
        handler_type: str = DEFAULT_HANDLER_TYPE,
        max_turns: int = DEFAULT_MAX_TURNS,
        allowed_tools: list[str] | None = None,
        tags: list[str] | None = None,
        source: str = "orchestrator",
    ) -> SubagentEntry:
        name = (name or "").strip()
        if not _VALID_NAME.match(name):
            raise ValueError(
                f"name must match [A-Za-z][A-Za-z0-9_]{{0,99}} (got {name!r})"
            )
        description = self._truncate(description, DESCRIPTION_MAX_CHARS)
        system_instructions = (system_instructions or "").strip()
        if not system_instructions:
            raise ValueError("add requires non-empty system_instructions")
        if len(system_instructions) > SYSTEM_INSTRUCTIONS_MAX_CHARS:
            raise ValueError(
                f"system_instructions exceeds {SYSTEM_INSTRUCTIONS_MAX_CHARS} chars"
            )
        directive = (directive or "").strip()
        if len(directive) > DIRECTIVE_MAX_CHARS:
            raise ValueError(f"directive exceeds {DIRECTIVE_MAX_CHARS} chars")
        return_condition = (return_condition or "").strip()
        if len(return_condition) > RETURN_CONDITION_MAX_CHARS:
            raise ValueError(
                f"return_condition exceeds {RETURN_CONDITION_MAX_CHARS} chars"
            )
        handler_type = _validate_handler_type(handler_type)
        max_turns = _clamp_max_turns(max_turns)
        allowed_tools_normalized = _validate_allowed_tools(
            list(DEFAULT_SUBAGENT_ALLOWED_TOOLS)
            if allowed_tools is None
            else allowed_tools
        )
        with self._lock:
            state = self._load()
            if len(state["entries"]) >= MAX_SUBAGENTS:
                raise ValueError(
                    f"subagents full ({MAX_SUBAGENTS} entries); delete or edit one first"
                )
            if any(e.get("name") == name for e in state["entries"]):
                raise ValueError(
                    f"subagent name {name!r} already in use; use edit instead"
                )
            now = self._now()
            entry = SubagentEntry(
                id=f"subagent_{state['next_id']:03d}",
                game_id=self.game_id,
                name=name,
                description=description,
                system_instructions=system_instructions,
                directive=directive,
                return_condition=return_condition,
                handler_type=handler_type,
                max_turns=max_turns,
                allowed_tools=allowed_tools_normalized,
                tags=list(tags or []),
                source=source,
                is_builtin=False,
                version=1,
                created_at=now,
                updated_at=now,
            )
            state["entries"].append(asdict(entry))
            state["next_id"] += 1
            self._save(state)
            return entry

    def delete(self, subagent_id: str) -> bool:
        with self._lock:
            state = self._load()
            target = next(
                (e for e in state["entries"] if e.get("id") == subagent_id), None
            )
            if target is None or target.get("is_builtin"):
                return False
            state["entries"] = [
                e for e in state["entries"] if e.get("id") != subagent_id
            ]
            self._save(state)
            return True

    def edit(
        self,
        subagent_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        system_instructions: str | None = None,
        directive: str | None = None,
        return_condition: str | None = None,
        handler_type: str | None = None,
        max_turns: int | None = None,
        allowed_tools: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> SubagentEntry | None:
        if all(
            v is None
            for v in (
                name,
                description,
                system_instructions,
                directive,
                return_condition,
                handler_type,
                max_turns,
                allowed_tools,
                tags,
            )
        ):
            raise ValueError(
                "edit requires at least one of name/description/system_instructions/"
                "directive/return_condition/handler_type/max_turns/allowed_tools/tags"
            )
        if name is not None:
            stripped = name.strip()
            if not _VALID_NAME.match(stripped):
                raise ValueError(
                    f"name must match [A-Za-z][A-Za-z0-9_]{{0,99}} (got {name!r})"
                )
            name = stripped
        if system_instructions is not None:
            system_instructions = system_instructions.strip()
            if not system_instructions:
                raise ValueError(
                    "system_instructions must be non-empty when provided"
                )
            if len(system_instructions) > SYSTEM_INSTRUCTIONS_MAX_CHARS:
                raise ValueError(
                    f"system_instructions exceeds {SYSTEM_INSTRUCTIONS_MAX_CHARS} chars"
                )
        if directive is not None:
            directive = directive.strip()
            if len(directive) > DIRECTIVE_MAX_CHARS:
                raise ValueError(f"directive exceeds {DIRECTIVE_MAX_CHARS} chars")
        if return_condition is not None:
            return_condition = return_condition.strip()
            if len(return_condition) > RETURN_CONDITION_MAX_CHARS:
                raise ValueError(
                    f"return_condition exceeds {RETURN_CONDITION_MAX_CHARS} chars"
                )
        if handler_type is not None:
            handler_type = _validate_handler_type(handler_type)
        if max_turns is not None:
            max_turns = _clamp_max_turns(max_turns)
        if allowed_tools is not None:
            allowed_tools = _validate_allowed_tools(allowed_tools)
        with self._lock:
            state = self._load()
            for entry in state["entries"]:
                if entry.get("id") == subagent_id:
                    if name is not None:
                        if any(
                            e.get("name") == name and e.get("id") != subagent_id
                            for e in state["entries"]
                        ):
                            raise ValueError(
                                f"subagent name {name!r} already in use by another entry"
                            )
                        entry["name"] = name
                    if description is not None:
                        entry["description"] = self._truncate(
                            description, DESCRIPTION_MAX_CHARS
                        )
                    if system_instructions is not None:
                        entry["system_instructions"] = system_instructions
                    if directive is not None:
                        entry["directive"] = directive
                    if return_condition is not None:
                        entry["return_condition"] = return_condition
                    if handler_type is not None:
                        entry["handler_type"] = handler_type
                    if max_turns is not None:
                        entry["max_turns"] = max_turns
                    if allowed_tools is not None:
                        entry["allowed_tools"] = allowed_tools
                    if tags is not None:
                        entry["tags"] = list(tags)
                    entry["version"] = int(entry.get("version", 1)) + 1
                    entry["updated_at"] = self._now()
                    self._save(state)
                    return _entry_from_dict(entry)
        return None

    def search(self, query: str) -> tuple[list[SubagentEntry], int]:
        """Case-insensitive substring over name + description + system_instructions
        + directive + tags.

        Returns (top_matches up to SEARCH_MAX_MATCHES, total_match_count).
        Empty query returns all entries.
        """
        with self._lock:
            state = self._load()
        entries = [_entry_from_dict(e) for e in state["entries"]]
        q = (query or "").lower().strip()
        if not q:
            matches = entries
        else:
            matches = [
                e
                for e in entries
                if q in e.name.lower()
                or q in e.description.lower()
                or q in e.system_instructions.lower()
                or q in e.directive.lower()
                or any(q in t.lower() for t in e.tags)
            ]
        return matches[:SEARCH_MAX_MATCHES], len(matches)


def format_subagent_overview(entries: list[SubagentEntry]) -> str:
    """Compact index for auto-injection. Shows id + name + handler/tools + first
    description line."""
    if not entries:
        return (
            "## SUBAGENTS (0 saved)\n"
            'No subagents saved yet. Use process_subagent(operation="add", name=..., '
            'description=..., system_instructions=..., directive=..., '
            'handler_type="looping"|"one_step", max_turns=..., allowed_tools=[...]) to '
            "register a focused inner agent for a complex task. If allowed_tools is "
            "omitted, it defaults to an empty allowlist (reason-and-return only). Then "
            "run_subagent(id, task) to invoke one. Include take_actions only for "
            "bounded action-capable subagents."
        )
    rows = [
        f"## SUBAGENTS ({len(entries)} saved)",
        "Only subagents whose allowed_tools include take_actions may commit actions; "
        "others analyze/use listed tools and return.",
    ]
    for e in entries:
        tool_str = ", ".join(e.allowed_tools) if e.allowed_tools else "no tools"
        first_line = (
            (e.description or "").splitlines()[0][:120] if e.description else ""
        )
        sep = " — " if first_line else ""
        rows.append(f"[{e.id}] {e.name} ({e.handler_type}, {tool_str}){sep}{first_line}")
    return "\n".join(rows)

"""Prompt-evolution scaffolding for ContinualHarness.

The harness runs a meta-VLM call every N steps that proposes a replacement for
its own system instruction. This module owns:

- `PromptFile` — atomic read/write of the current active prompt (single .md file).
- `PromptEvolutionStore` — append-only JSONL log of every evolution attempt
  (accepted or rejected) for offline inspection.
- `validate_evolved_prompt(text)` — length-bounds-only filter (200..6000 chars).
- `build_evolution_prompt(...)` — formats the user-facing half of the meta-call.

Path resolution mirrors memory/skills/subagents: `--bootstrap-prompt`
> `RUN_PROMPT_PATH` > `RUN_DIR/prompt.current.md` > fallback. The same single
file is both source and sink; mutations rewrite it atomically.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ...run_artifacts import game_artifacts
from ._locks import lock_for_path
from .prompts import EVOLUTION_USER_PROMPT
from .trajectory import format_full_history

logger = logging.getLogger(__name__)


# Bounds match the validator and the meta-prompt's stated contract.
PROMPT_MIN_CHARS = 200
PROMPT_MAX_CHARS = 12000

RUN_DIR_ENV = "RUN_DIR"
RUN_LOG_PATH_ENV = "RUN_LOG_PATH"


def active_prompt_path(game_id: str | None = None) -> Path:
    """Resolve the backing prompt file (the evolvable base prompt).

    Per-game when a game_id and RUN_DIR are available (the swarm default), so
    each game evolves its own prompt. Falls back to a single run-local file
    only when there is no game_id (tests / non-run).
    """
    run_dir = os.getenv(RUN_DIR_ENV)
    if run_dir and game_id:
        return game_artifacts(run_dir, game_id).prompt_path

    if run_dir:
        return Path(run_dir) / "prompt.current.md"

    run_log = os.getenv(RUN_LOG_PATH_ENV)
    if run_log:
        return Path(run_log).with_name("prompt.current.md")

    return Path("logs") / "continual_harness.prompt.current.md"


def active_prompt_evolution_path(game_id: str | None = None) -> Path:
    """Resolve the backing evolution-log file (a per-run, per-game audit trail)."""
    run_dir = os.getenv(RUN_DIR_ENV)
    if run_dir and game_id:
        return game_artifacts(run_dir, game_id).prompt_evolution_path

    if run_dir:
        return Path(run_dir) / "prompt_evolution.jsonl"

    run_log = os.getenv(RUN_LOG_PATH_ENV)
    if run_log:
        return Path(run_log).with_name("prompt_evolution.jsonl")

    return Path("logs") / "continual_harness.prompt_evolution.jsonl"


def validate_evolved_prompt(text: str) -> tuple[bool, str | None]:
    """Lenient validation: only length bounds.

    Returns (ok, error_msg). error_msg is None on success.
    """
    if not text or not text.strip():
        return False, "evolved prompt is empty"
    n = len(text)
    if n < PROMPT_MIN_CHARS:
        return False, f"too short ({n} < {PROMPT_MIN_CHARS})"
    if n > PROMPT_MAX_CHARS:
        return False, f"too long ({n} > {PROMPT_MAX_CHARS})"
    return True, None


class PromptFile:
    """Single-file string store for the current active system instruction.

    Atomic writes via tempfile.replace (parallel to MemoryStore). On
    construction, if the file is missing we seed it with `baseline` so a
    subsequent read returns a deterministic starting prompt.
    """

    def __init__(self, path: Path, *, baseline: str) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = lock_for_path(path)
        if not path.exists():
            self.write(baseline)

    def read(self) -> str:
        with self._lock:
            return self.path.read_text(encoding="utf-8")

    def write(self, text: str) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with self._lock:
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(self.path)


@dataclass(slots=True)
class PromptEvolutionRecord:
    generation: int
    action_counter: int
    accepted: bool
    reasoning: str
    proposed_prompt: str
    previous_prompt: str
    new_prompt: str  # == previous_prompt when accepted=False
    validation_error: str | None
    usage: dict[str, int | None] | None
    timestamp: str


class PromptEvolutionStore:
    """Append-only JSONL log of evolution attempts.

    One line per call to `_evolve_system_prompt`, whether the proposal was
    accepted or rejected. Survives concurrent appends inside a single process
    via a threading.Lock keyed by file path.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: PromptEvolutionRecord) -> None:
        line = json.dumps(asdict(record), default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def all_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]


def build_evolution_prompt(
    *,
    system_prompt: str,
    current_base_prompt: str,
    trajectory_rows: list[dict[str, Any]],
    memory_overview: str = "",
    skill_overview: str = "",
    subagent_overview: str = "",
) -> str:
    """Assemble the meta-call user prompt from the EVOLUTION_USER_PROMPT template.

    `trajectory_rows` are the raw dicts returned by `TrajectoryStore.tail(n)`;
    we render them with `format_full_history` so the meta-call sees reasoning
    + tool calls + grid deltas.
    """
    trajectory_text = format_full_history(trajectory_rows, max_chars=50000)

    return EVOLUTION_USER_PROMPT.format(
        system_prompt=system_prompt,
        current_base_prompt=current_base_prompt,
        n=len(trajectory_rows),
        trajectory=trajectory_text,
        memory_overview=memory_overview,
        skill_overview=skill_overview,
        subagent_overview=subagent_overview,
    )


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

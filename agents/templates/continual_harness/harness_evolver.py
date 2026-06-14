"""HarnessEvolver — full harness evolution for ContinualHarness.

The harness runs event- or stagnation-triggered meta-VLM calls that improve the
agent's harness from evidence in its recent trajectory. Each generation evolves
ALL FOUR components together, one independent meta-VLM call per pass:

1. the base orchestrator policy (the evolvable strategic-guidance prompt),
2. the skills library,
3. the subagents library,
4. memory.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from arcengine import FrameData, GameState

from ...run_artifacts import game_artifacts
from ._locks import lock_for_path
from .context import format_tool_evidence_markdown
from .helpers import frame_to_images
from .memory import format_memory_full
from .models import ToolEvidenceRecord
from .prompts import EVOLUTION_SYSTEM_INSTRUCTION, EVOLUTION_USER_PROMPT
from .sandbox import _validate_code
from .skills import format_skill_overview
from .subagents import format_subagent_overview
from .tools import SUBAGENT_TOOL_ENUM
from .trace import serialize_response
from .trajectory import format_full_history
from ..utils.vlm_backend import VLM, usage_token_count

logger = logging.getLogger(__name__)


# ======================================================================
# Prompt-persistence primitives (absorbed from the former prompt_evolution.py)
# ======================================================================

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
    """Single-file string store for the current active base prompt.

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
    """Append-only JSONL log of prompt-evolution attempts.

    One line per prompt pass, whether the proposal was accepted or rejected.
    Survives concurrent appends inside a single process via a threading.Lock.
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
    tool_evidence: str = "(none)",
    trigger_context: str = "Trigger: manual",
) -> str:
    """Assemble the base-policy meta-call user prompt from EVOLUTION_USER_PROMPT.

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
        tool_evidence=tool_evidence or "(none)",
        memory_overview=memory_overview,
        skill_overview=skill_overview,
        subagent_overview=subagent_overview,
        trigger_context=trigger_context,
    )


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ======================================================================
# Inline component meta-prompts (skills / subagents / memory)
# ======================================================================
#
# These use ``__TOKEN__`` placeholders filled via ``_fill`` (plain str.replace)
# rather than ``str.format`` so the literal JSON braces in the output schema do
# not need escaping. Each prompt is deliberately distinct: it states its
# component's job, the ARC-AGI-3 store schema, and demands a single JSON object.

COMPONENT_EVOLUTION_SYSTEM = (
    "You are a harness-evolution system for an AI agent playing __GAME__, a grid "
    "game whose rules are UNKNOWN and must be discovered from gameplay. Do NOT "
    "assume any familiar game mechanics (Sokoban, maze, snake, etc.). You improve "
    "ONE component of the agent's harness using evidence from its recent "
    "trajectory. Respond with ONLY a single JSON object — no prose outside it, no "
    "markdown fences."
)

# Shared skill-code contract, kept consistent with the orchestrator's view
# (prompts/harness_system.md → "Sandbox Environment").
SKILL_CODE_API = """\
- A skill's `code` is a top-level Python script (NOT a bare `def`): write logic
  at the top level, or define AND call your function. Set `result = <json-
  serializable>` to return data to the agent; use `print(...)` for debug (stdout).
- Read-only world view via `state` (both `obj.key` and `obj["key"]` work):
  - `state.latest_frame.frame` — hex animation stack `list[list[str]]`;
    `state.latest_frame.frame[-1]` is the current 2D grid (`list[str]`, one dense
    hex string per row; `int(grid[y][x], 16)` = color int 0-15).
  - `state.latest_frame.state` ("ONGOING"/"WIN"/"GAME_OVER"), `.score`,
    `.available_actions`; `state.observations`, `state.recent_trajectory`,
    `state.memory_entries`, `state.skill_entries`, `state.images`.
- Drive the engine with
  `tools.take_actions(actions=[{"name": "ACTION1"}, {"name": "ACTION6", "x": 3, "y": 4}])`
  → `{executed_count, last_frame, terminal, level_changed, state, score,
  available_actions}`. After each call check `terminal`/`level_changed` and read
  fresh grids from `last_frame.frame[-1]` (mid-skill `state` does not update).
- Pre-loaded, no import needed (imports are FORBIDDEN): np/numpy, collections,
  copy, dataclasses, functools, hashlib, heapq, itertools, json, math, random,
  re, statistics, Image/ImageDraw/ImageFilter/ImageOps/ImageChops; helpers
  render_grid(grid_2d), render_grids(grids_3d). No os/sys/subprocess/network."""

SKILL_EVOLUTION_PROMPT = """\
## Your job: identify reusable behavioral patterns (skills) from successful actions and evaluate existing skills

Skills are reusable Python snippets the agent runs via `run_skill`. From the
recent trajectory and the current library:
1. FIX skills whose `run_skill` calls errored — read the error and correct the
   code (provide FULL replacement source).
2. ADD a skill when the agent repeated the same multi-step `take_actions`
   sequence or manual grid analysis that should be reusable.
3. DELETE skills that are broken beyond repair or superseded.
Be conservative: prefer fixing or extending an existing skill over adding a
near-duplicate. Empty output is correct when nothing is worth changing.

## Skill code API (your `code` MUST follow this exactly)
__SKILL_API__

## Current skill library
__SKILL_OVERVIEW__

## Recent trajectory (oldest first, most recent last)
__TRAJECTORY__

## Recent tool evidence (same evolution window)
__TOOL_EVIDENCE__

## Evolution trigger
__TRIGGER__

## Output — a single JSON object, no markdown fences:
{
  "analysis": "one short paragraph: what you changed and the evidence for it",
  "add": [
    {"name": "scan_objects", "description": "what it does + I/O contract",
     "code": "<python source>", "tags": ["analysis"]}
  ],
  "edit": [
    {"id": "skill_003", "code": "<FULL replacement source>",
     "description": "optional updated description", "name": "optional new name"}
  ],
  "delete": ["skill_007"]
}
Rules: empty arrays are fine; `name` must match [A-Za-z][A-Za-z0-9_]*; `id` may be
a skill id (e.g. "skill_003") or its unique name."""

SUBAGENT_EVOLUTION_PROMPT = """\
## Your job: analyze recent trajectories and recommend changes to the agent's subagent library

A subagent is a focused inner agent with its own static `system_instructions`, a
per-invocation `directive`, and a small allowlist of tools. A `looping` subagent
runs a bounded action loop (up to `max_turns`) until it calls `subagent_return`;
a `one_step` subagent runs a single analysis turn and auto-returns. From the
recent trajectory and the current registry:
1. ADD a subagent when the orchestrator repeatedly did the same multi-step
   sub-task by hand that a focused inner agent should own (e.g. systematic
   exploration of the action space → a `looping` subagent; a recurring
   analyze-the-state routine → a `one_step` subagent).
2. EDIT a subagent whose `system_instructions`/`directive` led it astray, whose
   `handler_type`/`max_turns` was wrong, or whose toolset was wrong.
3. DELETE subagents that are unused or consistently fail.

## `allowed_tools` MUST be a subset of:
__ALLOWED_TOOLS__

## Current subagent registry
__SUBAGENT_OVERVIEW__

## Recent trajectory (oldest first, most recent last)
__TRAJECTORY__

## Recent tool evidence (same evolution window)
__TOOL_EVIDENCE__

## Evolution trigger
__TRIGGER__

## Output — a single JSON object, no markdown fences:
{
  "analysis": "one short paragraph: what you changed and the evidence for it",
  "add": [
    {"name": "explore_actions", "description": "what it is for",
     "system_instructions": "Static system prompt for the inner agent: who it is and how to approach its task, ending with when to call subagent_return(answer, status).",
     "directive": "Default per-invocation task framing (optional).",
     "return_condition": "When to return control (optional narrative).",
     "handler_type": "looping",
     "max_turns": 25,
     "allowed_tools": ["take_actions", "process_memory"], "tags": ["exploration"]}
  ],
  "edit": [
    {"id": "subagent_002", "system_instructions": "improved system prompt",
     "directive": "optional", "handler_type": "one_step", "max_turns": 1,
     "allowed_tools": ["take_actions"], "description": "optional"}
  ],
  "delete": ["subagent_004"]
}
Rules: empty arrays are fine; `name` must match [A-Za-z][A-Za-z0-9_]*;
`handler_type` is "looping" or "one_step"; `max_turns` is an integer 1-50; `id`
is an existing subagent id (e.g. "subagent_002")."""

MEMORY_EVOLUTION_PROMPT = """\
## Your job: review the agent's memory store and recent trajectories, then recommend targeted improvements. Be conservative — the agent itself writes memory during gameplay. You fill gaps and clean up.

Memory holds discovered facts about this game, each with a confidence 1-5
(1 = untested guess … 5 = repeatedly confirmed). The orchestrator already writes
memory inline during play, so DO NOT duplicate what is already there — only fill
gaps and fix contradictions. From the recent trajectory and the current memory:
1. ADD a fact the trajectory clearly demonstrates but memory is missing.
2. EDIT an entry the trajectory contradicts (correct it and lower confidence) or
   re-confirms (raise confidence). Set `confidence` to match the evidence.
3. DELETE entries the trajectory proves false.

## Current memory (full bodies)
__MEMORY_FULL__

## Recent trajectory (oldest first, most recent last)
__TRAJECTORY__

## Recent tool evidence (same evolution window)
__TOOL_EVIDENCE__

## Evolution trigger
__TRIGGER__

## Output — a single JSON object, no markdown fences:
{
  "analysis": "one short paragraph: what you changed and the evidence for it",
  "add": [
    {"title": "short fact title", "body": "the fact, stated precisely",
     "tags": ["rule"], "confidence": 2}
  ],
  "edit": [
    {"id": "mem_005", "body": "corrected fact", "confidence": 4,
     "title": "optional", "tags": ["optional"]}
  ],
  "delete": ["mem_009"]
}
Rules: empty arrays are fine; `confidence` is an integer 1-5 and is REQUIRED on
every add; `id` is an existing memory id (e.g. "mem_005")."""


# ======================================================================
# Small module helpers
# ======================================================================


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _cost_log_suffix(cost: dict[str, Any] | None) -> str:
    if not cost:
        return ""
    return (
        f" cost=${float(cost['current_usd']):.6f}"
        f" cum_cost=${float(cost['cumulative_usd']):.6f}"
    )


def _fill(template: str, **tokens: str) -> str:
    """Fill ``__TOKEN__`` placeholders via str.replace (no brace escaping)."""
    out = template
    for key, value in tokens.items():
        out = out.replace(f"__{key}__", value)
    return out


def _canonical_subagent_id(raw_id: Any) -> str:
    """Accept legacy prompt examples like ``sa_002`` for real ``subagent_002`` ids."""
    subagent_id = str(raw_id or "").strip()
    if subagent_id.startswith("sa_") and subagent_id[3:].isdigit():
        return f"subagent_{int(subagent_id[3:]):03d}"
    return subagent_id


def _tool_evidence_window(
    records: list[ToolEvidenceRecord] | None,
    *,
    start_action_counter: int,
    end_action_counter: int,
) -> list[ToolEvidenceRecord]:
    if not records:
        return []
    return [
        record
        for record in records
        if (
            record.action_counter_after > start_action_counter
            or (
                record.action_counter_after == start_action_counter
                and record.action_counter_before == start_action_counter
            )
        )
        and record.action_counter_before <= end_action_counter
    ]


# ======================================================================
# HarnessEvolver
# ======================================================================


class HarnessEvolver:
    """Evolves all four harness components: prompt, skills, subagents, memory.

    Constructed once by the agent. Holds shared references to the agent's
    memory/skills/subagents/trajectory stores (so inline orchestrator edits and
    meta-level edits operate on the same files) plus its own evolution
    bookkeeping (the base-prompt file, the prompt-evolution log, generation
    counters). The semantic triggers (level_up / game_over / stagnation) call
    ``evolve`` which runs every enabled pass in one generation.
    """

    # Stagnation-detection thresholds (migrated from the agent class body).
    STAGNATION_WINDOW = 30
    STAGNATION_MIN_WINDOW_RECORDS = 10
    STAGNATION_MIN_ACTIONS_SINCE_EVOLUTION = 50
    STAGNATION_NOOP_RATIO = 0.45
    STAGNATION_REPEAT_ACTION_RATIO = 0.60
    STAGNATION_MAX_CYCLE = 8
    STAGNATION_MIN_CYCLE_REPEATS = 3

    def __init__(
        self,
        *,
        model_name: str,
        system_instruction: str,
        game_id: str,
        agent_name: str,
        memory: Any,
        skills: Any,
        subagents: Any,
        trajectory: Any,
        trace: Any,
        record_usage: Callable[[dict[str, Any] | None], dict[str, Any] | None],
        baseline_prompt: str,
    ) -> None:
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.game_id = game_id
        self.agent_name = agent_name
        self.memory = memory
        self.skills = skills
        self.subagents = subagents
        self.trajectory = trajectory
        self.trace = trace
        self.record_usage = record_usage

        # How long without progress before stagnation evolution may fire.
        # 0 disables ALL evolution. (env name unchanged for back-compat.)
        self.stagnation_after = max(
            0, _int_env("CONTINUAL_HARNESS_PROMPT_EVOLVE_FREQUENCY", 100)
        )
        # Per-pass kill switches (default on) — a cost safety valve. The prompt
        # pass always runs when a trigger fires.
        self._evolve_skills_on = _bool_env("CONTINUAL_HARNESS_EVOLVE_SKILLS", True)
        self._evolve_subagents_on = _bool_env("CONTINUAL_HARNESS_EVOLVE_SUBAGENTS", True)
        self._evolve_memory_on = _bool_env("CONTINUAL_HARNESS_EVOLVE_MEMORY", True)

        # Evolvable base prompt — stored per-game so parallel games evolve
        # independent prompts. The evolver owns the file + the audit log.
        self._base_prompt_file = PromptFile(
            active_prompt_path(game_id), baseline=baseline_prompt
        )
        self._current_base_prompt = self._base_prompt_file.read()
        self.prompt_evolution = PromptEvolutionStore(
            active_prompt_evolution_path(game_id)
        )

        self._generation = 0
        self._last_evolution_step = -1
        self._last_game_over_evolution_step = -1

    # ------------------------------------------------------------------
    # Accessors used by the agent
    # ------------------------------------------------------------------

    def get_current_prompt(self) -> str:
        return self._current_base_prompt

    @property
    def base_prompt_path(self) -> Path:
        return self._base_prompt_file.path

    @property
    def evolution_log_path(self) -> Path:
        return self.prompt_evolution.path

    # ------------------------------------------------------------------
    # Triggers (semantic — preserved from the original agent)
    # ------------------------------------------------------------------

    def evolve_on_level_up(
        self,
        latest_frame: FrameData,
        action_counter: int,
        *,
        tool_evidence_records: list[ToolEvidenceRecord] | None = None,
    ) -> None:
        """Evolve unconditionally on level transition."""
        if self.stagnation_after <= 0:
            return
        evidence = [
            f"advanced to score/level {latest_frame.levels_completed}",
            f"action_counter={action_counter}",
        ]
        try:
            self.evolve(
                latest_frame, action_counter,
                trigger="level_up", trigger_evidence=evidence,
                tool_evidence_records=tool_evidence_records,
            )
        finally:
            self._last_evolution_step = action_counter

    def evolve_on_game_over(
        self,
        latest_frame: FrameData,
        action_counter: int,
        *,
        tool_evidence_records: list[ToolEvidenceRecord] | None = None,
    ) -> None:
        """Evolve once for a GAME_OVER state before auto-reset."""
        if self.stagnation_after <= 0:
            return
        if self._last_game_over_evolution_step == action_counter:
            return
        self._last_game_over_evolution_step = action_counter
        evidence = [
            "entered GAME_OVER before auto-reset",
            f"score/level={latest_frame.levels_completed}",
            f"action_counter={action_counter}",
        ]
        try:
            self.evolve(
                latest_frame, action_counter,
                trigger="game_over", trigger_evidence=evidence,
                tool_evidence_records=tool_evidence_records,
            )
        finally:
            self._last_evolution_step = action_counter

    def maybe_evolve_on_stagnation(
        self,
        latest_frame: FrameData,
        action_counter: int,
        last_progress_step: int,
        *,
        tool_evidence_records: list[ToolEvidenceRecord] | None = None,
    ) -> None:
        """Evolve only when the recent trajectory shows concrete stuck behavior."""
        if self.stagnation_after <= 0 or action_counter == 0:
            return
        if latest_frame.state in {
            GameState.GAME_OVER,
            GameState.NOT_PLAYED,
            GameState.WIN,
        }:
            return

        progress_gap = action_counter - max(last_progress_step, 0)
        if progress_gap < self.stagnation_after:
            return

        evolution_gap = action_counter - max(self._last_evolution_step, 0)
        if evolution_gap < self.STAGNATION_MIN_ACTIONS_SINCE_EVOLUTION:
            return

        pattern_evidence = self._stagnation_evidence(
            self.trajectory.tail(self.STAGNATION_WINDOW)
        )
        if not pattern_evidence:
            return

        evidence = [
            f"{progress_gap} actions since last score/level progress",
            f"{evolution_gap} actions since last prompt evolution",
            *pattern_evidence,
        ]
        logger.info("[%s] stagnation evolution: %s", self.agent_name, "; ".join(evidence))
        try:
            self.evolve(
                latest_frame, action_counter,
                trigger="stagnation", trigger_evidence=evidence,
                tool_evidence_records=tool_evidence_records,
            )
        finally:
            self._last_evolution_step = action_counter

    # ------------------------------------------------------------------
    # One generation = all enabled passes, each isolated
    # ------------------------------------------------------------------

    def evolve(
        self,
        latest_frame: FrameData,
        action_counter: int,
        *,
        trigger: str = "manual",
        trigger_evidence: list[str] | None = None,
        tool_evidence_records: list[ToolEvidenceRecord] | None = None,
    ) -> dict[str, Any]:
        self._generation += 1
        gen = self._generation
        logger.info(
            "=== HarnessEvolver generation %d at action %d (trigger=%s) ===",
            gen, action_counter, trigger,
        )

        steps_since = max(1, action_counter - max(self._last_evolution_step, 0))
        rows = self.trajectory.tail(steps_since)
        trajectory_text = format_full_history(rows, max_chars=50000)
        tool_window = _tool_evidence_window(
            tool_evidence_records,
            start_action_counter=max(self._last_evolution_step, 0),
            end_action_counter=action_counter,
        )
        tool_evidence_text = format_tool_evidence_markdown(
            tool_window, max_chars=50000
        )
        images = list(frame_to_images(latest_frame))
        trigger_ctx = self._format_trigger_context(trigger, trigger_evidence)

        passes: list[tuple[str, Callable[[], dict[str, Any]]]] = [
            (
                "prompt",
                lambda: self._evolve_prompt(
                    images,
                    rows,
                    action_counter,
                    gen,
                    trigger_ctx,
                    tool_evidence_text,
                ),
            ),
        ]
        if self._evolve_skills_on:
            passes.append(
                (
                    "skills",
                    lambda: self._evolve_skills(
                        images, trajectory_text, trigger_ctx, tool_evidence_text
                    ),
                )
            )
        if self._evolve_subagents_on:
            passes.append(
                (
                    "subagents",
                    lambda: self._evolve_subagents(
                        images, trajectory_text, trigger_ctx, tool_evidence_text
                    ),
                )
            )
        if self._evolve_memory_on:
            passes.append(
                (
                    "memory",
                    lambda: self._evolve_memory(
                        images, trajectory_text, trigger_ctx, tool_evidence_text
                    ),
                )
            )

        results: dict[str, Any] = {}
        for name, fn in passes:
            try:
                results[name] = fn()
            except Exception as exc:
                logger.error("evolution pass '%s' failed: %s", name, exc, exc_info=True)
                results[name] = {"error": repr(exc)}

        self._save_evolution_log(gen, action_counter, trigger, trigger_evidence, results)
        return results

    # ------------------------------------------------------------------
    # Meta-VLM call (one fresh text VLM per pass, mirrors the original code)
    # ------------------------------------------------------------------

    def _meta_query(
        self, system: str, user: str, images: list[Any], tag: Any
    ) -> tuple[Any, dict[str, Any] | None, dict[str, Any] | None]:
        vlm = VLM(self.model_name, backend="gemini", system_instruction=system)
        payload: Any = images if len(images) > 1 else (images[0] if images else None)
        response = vlm.get_query(
            payload, user, module_name=f"{self.agent_name}.evolve.{tag}"
        )
        usage = vlm.extract_usage(response)
        usage_cost = self.record_usage(usage)
        return response, usage, usage_cost

    # ------------------------------------------------------------------
    # Pass 1 — base orchestrator policy (relocated _evolve_system_prompt)
    # ------------------------------------------------------------------

    def _evolve_prompt(
        self,
        images: list[Any],
        rows: list[dict[str, Any]],
        action_counter: int,
        gen: int,
        trigger_ctx: str,
        tool_evidence_text: str = "(none)",
    ) -> dict[str, Any]:
        previous = self._current_base_prompt
        user_prompt = build_evolution_prompt(
            system_prompt=self.system_instruction,
            current_base_prompt=previous,
            trajectory_rows=rows,
            memory_overview=format_memory_full(self.memory.all_entries()),
            skill_overview=format_skill_overview(self.skills.all_entries()),
            subagent_overview=format_subagent_overview(self.subagents.all_entries()),
            tool_evidence=tool_evidence_text,
            trigger_context=trigger_ctx,
        )
        evolution_system = EVOLUTION_SYSTEM_INSTRUCTION.replace(
            "{game_name}", self.game_id
        )

        proposed = ""
        accepted = False
        validation_error: str | None = None
        usage: dict[str, Any] | None = None
        usage_cost: dict[str, Any] | None = None
        output: dict[str, Any] = {}
        error: str | None = None

        try:
            response, usage, usage_cost = self._meta_query(
                evolution_system, user_prompt, images, gen
            )
            output = serialize_response(response)
            proposed = self._extract_text(response)
            if not proposed:
                validation_error = "model returned empty text"
            else:
                ok, err = validate_evolved_prompt(proposed)
                accepted = ok
                validation_error = err
                if accepted:
                    self._current_base_prompt = proposed
                    self._base_prompt_file.write(proposed)
        except Exception as exc:
            error = repr(exc)
            logger.warning("Prompt evolution gen=%d failed: %s", gen, exc)
        finally:
            if usage is not None:
                logger.info(
                    "[%s] prompt_evolution gen=%d tokens=%d%s",
                    self.game_id,
                    gen,
                    usage_token_count(usage, "total"),
                    _cost_log_suffix(usage_cost),
                )

            record = PromptEvolutionRecord(
                generation=gen,
                action_counter=action_counter,
                accepted=accepted,
                reasoning=trigger_ctx,
                proposed_prompt=proposed,
                previous_prompt=previous,
                new_prompt=self._current_base_prompt,
                validation_error=validation_error or error,
                usage=usage,
                timestamp=now_iso(),
            )
            self.prompt_evolution.append(record)

            self.trace.write(
                {
                    "agent": self.agent_name,
                    "model": self.model_name,
                    "game_id": self.game_id,
                    "action_counter": action_counter,
                    "round": 0,
                    "tools_exposed": "evolution",
                    "evolution": {
                        "generation": gen,
                        "trigger_context": trigger_ctx,
                        "accepted": accepted,
                        "validation_error": validation_error,
                        "previous_len": len(previous),
                        "new_len": len(self._current_base_prompt),
                    },
                    "input": {
                        "system_instruction": evolution_system,
                        "user_prompt": user_prompt,
                        "images": [
                            {"width": img.width, "height": img.height, "mode": img.mode}
                            for img in images
                        ],
                    },
                    "output": output,
                    "usage": usage,
                    "usage_cost": usage_cost,
                    "error": error,
                }
            )

            logger.info(
                "[%s] Prompt evolution gen=%d accepted=%s len=%d (prev=%d) error=%s",
                self.agent_name,
                gen,
                accepted,
                len(self._current_base_prompt),
                len(previous),
                validation_error or error,
            )

        return {
            "accepted": accepted,
            "validation_error": validation_error or error,
            "previous_len": len(previous),
            "new_len": len(self._current_base_prompt),
        }

    # ------------------------------------------------------------------
    # Pass 2 — skills
    # ------------------------------------------------------------------

    def _evolve_skills(
        self,
        images: list[Any],
        trajectory_text: str,
        trigger_ctx: str,
        tool_evidence_text: str = "(none)",
    ) -> dict[str, Any]:
        overview = format_skill_overview(self.skills.all_entries())
        user = _fill(
            SKILL_EVOLUTION_PROMPT,
            SKILL_API=SKILL_CODE_API,
            SKILL_OVERVIEW=overview,
            TRAJECTORY=trajectory_text,
            TOOL_EVIDENCE=tool_evidence_text or "(none)",
            TRIGGER=trigger_ctx,
        )
        system = _fill(COMPONENT_EVOLUTION_SYSTEM, GAME=self.game_id)
        response, _, _ = self._meta_query(system, user, images, "skills")
        rec = self._parse_json_response(self._extract_text(response))
        if rec is None:
            return {"error": "failed_to_parse_response"}

        out: dict[str, Any] = {
            "added": [], "edited": [], "deleted": [],
            "analysis": rec.get("analysis", ""),
        }

        for spec in rec.get("add", []) or []:
            try:
                code = (spec.get("code") or "").strip()
                if code:
                    bad = _validate_code(code)
                    if bad:
                        logger.info("evolved skill add rejected (sandbox policy): %s", bad)
                        continue
                entry = self.skills.add(
                    name=spec.get("name", ""),
                    description=spec.get("description", ""),
                    code=code,
                    tags=list(spec.get("tags") or []),
                )
                out["added"].append(entry.id)
                logger.info("evolved skill added: %s (%s)", entry.id, entry.name)
            except Exception as exc:
                logger.error("evolved skill add failed (%s): %s", spec.get("name"), exc)

        for upd in rec.get("edit", []) or []:
            key = upd.get("id")
            if not key:
                continue
            target = self.skills.get_by_id_or_name(key)
            if target is None:
                continue
            try:
                code = upd.get("code")
                if code:
                    bad = _validate_code(code)
                    if bad:
                        logger.info("evolved skill edit rejected (sandbox policy): %s", bad)
                        continue
                edited = self.skills.edit(
                    target.id,
                    name=upd.get("name"),
                    description=upd.get("description"),
                    code=code,
                    tags=list(upd["tags"]) if "tags" in upd else None,
                )
                if edited is not None:
                    out["edited"].append(target.id)
            except Exception as exc:
                logger.error("evolved skill edit failed (%s): %s", key, exc)

        for key in rec.get("delete", []) or []:
            try:
                target = self.skills.get_by_id_or_name(key)
                if target is not None and self.skills.delete(target.id):
                    out["deleted"].append(target.id)
            except Exception as exc:
                logger.error("evolved skill delete failed (%s): %s", key, exc)

        return out

    # ------------------------------------------------------------------
    # Pass 3 — subagents
    # ------------------------------------------------------------------

    def _evolve_subagents(
        self,
        images: list[Any],
        trajectory_text: str,
        trigger_ctx: str,
        tool_evidence_text: str = "(none)",
    ) -> dict[str, Any]:
        overview = format_subagent_overview(self.subagents.all_entries())
        user = _fill(
            SUBAGENT_EVOLUTION_PROMPT,
            ALLOWED_TOOLS=", ".join(sorted(SUBAGENT_TOOL_ENUM)),
            SUBAGENT_OVERVIEW=overview,
            TRAJECTORY=trajectory_text,
            TOOL_EVIDENCE=tool_evidence_text or "(none)",
            TRIGGER=trigger_ctx,
        )
        system = _fill(COMPONENT_EVOLUTION_SYSTEM, GAME=self.game_id)
        response, _, _ = self._meta_query(system, user, images, "subagents")
        rec = self._parse_json_response(self._extract_text(response))
        if rec is None:
            return {"error": "failed_to_parse_response"}

        out: dict[str, Any] = {
            "added": [], "edited": [], "deleted": [],
            "analysis": rec.get("analysis", ""),
        }

        for spec in rec.get("add", []) or []:
            try:
                add_kwargs: dict[str, Any] = {
                    "directive": spec.get("directive", ""),
                    "return_condition": spec.get("return_condition", ""),
                    "source": "evolved",
                }
                if spec.get("handler_type"):
                    add_kwargs["handler_type"] = spec["handler_type"]
                if spec.get("max_turns") is not None:
                    add_kwargs["max_turns"] = spec["max_turns"]
                entry = self.subagents.add(
                    name=spec.get("name", ""),
                    description=spec.get("description", ""),
                    system_instructions=spec.get("system_instructions", ""),
                    allowed_tools=list(spec.get("allowed_tools") or []),
                    tags=list(spec.get("tags") or []),
                    **add_kwargs,
                )
                out["added"].append(entry.id)
                logger.info("evolved subagent added: %s (%s)", entry.id, entry.name)
            except ValueError as exc:  # invalid allowed_tools / name / handler_type
                logger.info("evolved subagent add rejected: %s", exc)
            except Exception as exc:
                logger.error("evolved subagent add failed (%s): %s", spec.get("name"), exc)

        for upd in rec.get("edit", []) or []:
            sid = _canonical_subagent_id(upd.get("id"))
            if not sid:
                continue
            try:
                edited = self.subagents.edit(
                    sid,
                    name=upd.get("name"),
                    description=upd.get("description"),
                    system_instructions=upd.get("system_instructions"),
                    directive=upd.get("directive"),
                    return_condition=upd.get("return_condition"),
                    handler_type=upd.get("handler_type"),
                    max_turns=upd.get("max_turns"),
                    allowed_tools=(
                        list(upd["allowed_tools"]) if "allowed_tools" in upd else None
                    ),
                    tags=list(upd["tags"]) if "tags" in upd else None,
                )
                if edited is not None:
                    out["edited"].append(edited.id)
            except ValueError as exc:
                logger.info("evolved subagent edit rejected: %s", exc)
            except Exception as exc:
                logger.error("evolved subagent edit failed (%s): %s", sid, exc)

        for raw_sid in rec.get("delete", []) or []:
            sid = _canonical_subagent_id(raw_sid)
            if not sid:
                continue
            try:
                if self.subagents.delete(sid):
                    out["deleted"].append(sid)
            except Exception as exc:
                logger.error("evolved subagent delete failed (%s): %s", sid, exc)

        return out

    # ------------------------------------------------------------------
    # Pass 4 — memory
    # ------------------------------------------------------------------

    def _evolve_memory(
        self,
        images: list[Any],
        trajectory_text: str,
        trigger_ctx: str,
        tool_evidence_text: str = "(none)",
    ) -> dict[str, Any]:
        overview = format_memory_full(self.memory.all_entries())
        user = _fill(
            MEMORY_EVOLUTION_PROMPT,
            MEMORY_FULL=overview,
            TRAJECTORY=trajectory_text,
            TOOL_EVIDENCE=tool_evidence_text or "(none)",
            TRIGGER=trigger_ctx,
        )
        system = _fill(COMPONENT_EVOLUTION_SYSTEM, GAME=self.game_id)
        response, _, _ = self._meta_query(system, user, images, "memory")
        rec = self._parse_json_response(self._extract_text(response))
        if rec is None:
            return {"error": "failed_to_parse_response"}

        out: dict[str, Any] = {
            "added": [], "edited": [], "deleted": [],
            "analysis": rec.get("analysis", ""),
        }

        for spec in rec.get("add", []) or []:
            try:
                entry = self.memory.add(
                    title=spec.get("title", ""),
                    body=spec.get("body", ""),
                    tags=list(spec.get("tags") or []),
                    confidence=spec.get("confidence"),
                )
                out["added"].append(entry.id)
                logger.info("evolved memory added: %s (%s)", entry.id, entry.title)
            except ValueError as exc:  # invalid / missing confidence
                logger.info("evolved memory add rejected: %s", exc)
            except Exception as exc:
                logger.error("evolved memory add failed: %s", exc)

        for upd in rec.get("edit", []) or []:
            mid = upd.get("id")
            if not mid:
                continue
            try:
                edited = self.memory.edit(
                    mid,
                    title=upd.get("title"),
                    body=upd.get("body"),
                    tags=list(upd["tags"]) if "tags" in upd else None,
                    confidence=upd.get("confidence"),
                )
                if edited is not None:
                    out["edited"].append(mid)
            except ValueError as exc:
                logger.info("evolved memory edit rejected: %s", exc)
            except Exception as exc:
                logger.error("evolved memory edit failed (%s): %s", mid, exc)

        for mid in rec.get("delete", []) or []:
            try:
                if self.memory.delete(mid):
                    out["deleted"].append(mid)
            except Exception as exc:
                logger.error("evolved memory delete failed (%s): %s", mid, exc)

        return out

    # ------------------------------------------------------------------
    # Stagnation detection (migrated verbatim from the agent)
    # ------------------------------------------------------------------

    @staticmethod
    def _trajectory_action_label(record: dict[str, Any]) -> str:
        name = str(record.get("chosen_action") or "UNKNOWN")
        data = record.get("chosen_action_data") or {}
        if not isinstance(data, dict) or not data:
            return name
        if "x" in data and "y" in data and len(data) == 2:
            return f"{name}({data['x']},{data['y']})"
        args = ",".join(f"{k}={v}" for k, v in sorted(data.items()))
        return f"{name}({args})"

    @staticmethod
    def _is_noop_or_invalid_record(record: dict[str, Any]) -> bool:
        state_after = record.get("state_after")
        if state_after == "INVALID":
            return True

        state_before = record.get("state")
        resolved_after = state_after or state_before
        score_delta = record.get("score_delta")
        score_changed = isinstance(score_delta, int) and score_delta != 0
        grid_changed = bool(record.get("grid_delta") or record.get("grid_change"))
        return resolved_after == state_before and not score_changed and not grid_changed

    def _repeated_cycle_evidence(self, labels: list[str]) -> str | None:
        max_cycle = min(self.STAGNATION_MAX_CYCLE, len(labels) // 2)
        for width in range(2, max_cycle + 1):
            pattern = labels[-width:]
            repeats = 1
            cursor = len(labels) - width
            while cursor - width >= 0 and labels[cursor - width : cursor] == pattern:
                repeats += 1
                cursor -= width
            if repeats >= self.STAGNATION_MIN_CYCLE_REPEATS:
                return (
                    f"last {width * repeats} actions repeat a {width}-action cycle: "
                    f"{', '.join(pattern)}"
                )
        return None

    def _stagnation_evidence(self, records: list[dict[str, Any]]) -> list[str]:
        actionable = [
            record
            for record in records
            if isinstance(record, dict) and record.get("source") != "auto_reset"
        ]
        if len(actionable) < self.STAGNATION_MIN_WINDOW_RECORDS:
            return []

        evidence: list[str] = []
        noops = sum(1 for record in actionable if self._is_noop_or_invalid_record(record))
        noop_ratio = noops / len(actionable)
        if noop_ratio >= self.STAGNATION_NOOP_RATIO:
            evidence.append(
                f"last {len(actionable)} actions include {noops} no-op/invalid "
                f"results ({noop_ratio:.0%})"
            )

        labels = [self._trajectory_action_label(record) for record in actionable]
        counts: dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        dominant_label, dominant_count = max(counts.items(), key=lambda item: item[1])
        repeat_ratio = dominant_count / len(labels)
        if repeat_ratio >= self.STAGNATION_REPEAT_ACTION_RATIO:
            evidence.append(
                f"action {dominant_label} appears {dominant_count}/"
                f"{len(labels)} times ({repeat_ratio:.0%})"
            )

        cycle = self._repeated_cycle_evidence(labels)
        if cycle:
            evidence.append(cycle)

        return evidence

    # ------------------------------------------------------------------
    # Helpers (migrated / new)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_trigger_context(
        trigger: str, trigger_evidence: list[str] | None
    ) -> str:
        evidence = [line for line in (trigger_evidence or []) if line]
        lines = [f"Trigger: {trigger}"]
        lines.extend(f"- {line}" for line in evidence)
        return "\n".join(lines)

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract plain text from a Gemini response or string.

        When no tools are set, the VLM backend returns a plain string. When
        tools are set, it returns a response object with candidates/parts.
        Handles preamble text before markdown fences.
        """
        import re

        if isinstance(response, str):
            text = response.strip()
        else:
            text = ""
            for cand in getattr(response, "candidates", None) or []:
                for part in getattr(getattr(cand, "content", None), "parts", []) or []:
                    t = getattr(part, "text", None)
                    if t:
                        text += t
            text = text.strip()
        m = re.search(r"```(?:markdown)?\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text

    @staticmethod
    def _parse_json_response(text: str) -> dict[str, Any] | None:
        """Extract a JSON object from a VLM response, handling markdown fences."""
        text = (text or "").strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if 0 <= start < end:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
        logger.error("Failed to parse JSON from evolution response: %s", text[:200])
        return None

    def _save_evolution_log(
        self,
        gen: int,
        action_counter: int,
        trigger: str,
        trigger_evidence: list[str] | None,
        results: dict[str, Any],
    ) -> None:
        """Write one combined generation entry summarizing every pass."""

        def _summ(name: str) -> dict[str, Any]:
            r = results.get(name, {}) or {}
            return {
                "added": r.get("added", []),
                "edited": r.get("edited", []),
                "deleted": r.get("deleted", []),
                "error": r.get("error"),
            }

        prompt_res = results.get("prompt", {}) or {}
        entry = {
            "agent": self.agent_name,
            "model": self.model_name,
            "game_id": self.game_id,
            "action_counter": action_counter,
            "round": 0,
            "tools_exposed": "harness_evolution",
            "harness_evolution": {
                "generation": gen,
                "trigger": trigger,
                "trigger_evidence": trigger_evidence or [],
                "prompt": {
                    "accepted": prompt_res.get("accepted"),
                    "validation_error": prompt_res.get("validation_error"),
                    "new_len": prompt_res.get("new_len"),
                },
                "skills": _summ("skills"),
                "subagents": _summ("subagents"),
                "memory": _summ("memory"),
                "store_counts": {
                    "memory": len(self.memory.all_entries()),
                    "skills": len(self.skills.all_entries()),
                    "subagents": len(self.subagents.all_entries()),
                },
            },
        }
        try:
            self.trace.write(entry)
        except Exception as exc:
            logger.error("failed to write harness-evolution log: %s", exc)

        logger.info(
            "[%s] harness evolution gen=%d trigger=%s skills=%s subagents=%s memory=%s",
            self.agent_name, gen, trigger,
            _summ("skills"), _summ("subagents"), _summ("memory"),
        )

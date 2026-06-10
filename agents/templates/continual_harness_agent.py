from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import asdict
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from ..recorder import Recorder
from ..run_artifacts import RUN_DIR_ENV, RUN_RECORDINGS_DIR_ENV, game_artifacts
from ..tracing import trace_agent_session
from .continual_harness.context import (
    build_observation_section,
    build_subagent_prompt,
    build_working_prompt,
    current_state_rendered_grid,
)
from .continual_harness.helpers import (
    available_game_actions,
    frame_to_hex,
    frame_to_images,
    grid_to_image,
    validate_action_sequence,
)
from .continual_harness.memory import (
    MemoryStore,
    active_memory_path,
    format_memory_full,
    format_memory_overview,
)
from .continual_harness.models import (
    PendingActionObservation,
    RenderedGrid,
    StepRecord,
    ToolCallRecord,
)
from .continual_harness.prompt_evolution import (
    PromptEvolutionRecord,
    PromptEvolutionStore,
    PromptFile,
    active_prompt_evolution_path,
    active_prompt_path,
    build_evolution_prompt,
    now_iso,
    validate_evolved_prompt,
)
from .continual_harness.prompts import (
    BASE_ORCHESTRATOR_POLICY,
    EVOLUTION_SYSTEM_INSTRUCTION,
    HARNESS_SYSTEM_INSTRUCTION,
)
from .continual_harness.sandbox import SandboxState, run_python_snippet
from .continual_harness.skills import (
    SkillStore,
    active_skill_path,
    format_skill_overview,
)
from .continual_harness.subagents import (
    SubagentStore,
    active_subagent_path,
    format_subagent_overview,
)
from .continual_harness.tools import (
    PROCESS_MEMORY_TOOL,
    PROCESS_SKILL_TOOL,
    PROCESS_SUBAGENT_TOOL,
    RUN_SKILL_TOOL,
    RUN_SUBAGENT_TOOL,
    TAKE_ACTIONS,
    TAKE_ACTIONS_TOOL,
    ContinualToolRouter,
    build_analysis_tools,
    build_subagent_tools,
    extract_function_calls,
    is_action_tool,
    is_subagent_return_call,
    is_take_actions_call,
    render_tool_results,
)
from .continual_harness.trace import (
    TraceWriter,
    default_trace_path,
    serialize_response,
)
from .continual_harness.trajectory import (
    TrajectoryStore,
    default_trajectory_path,
    # format_full_history,  # removed with get_recent_trajectory (still used by prompt_evolution)
    hexify_record_colors,
    render_recent_history,
    summarize_grid_transitions,
)
from .utils.vlm_backend import VLM

logger = logging.getLogger(__name__)


def _compute_grid_delta(
    pre_grid: list[list[int]] | None,
    post_grid: list[list[int]] | None,
) -> list[list[int]] | None:
    """Compare two 2D grids and return changed cells as [[x, y, old, new], ...].

    Returns None if either grid is missing. Caps at 50 entries to bound
    JSONL row size.
    """
    if pre_grid is None or post_grid is None:
        return None
    delta: list[list[int]] = []
    for y, (pre_row, post_row) in enumerate(zip(pre_grid, post_grid)):
        for x, (old, new) in enumerate(zip(pre_row, post_row)):
            if old != new:
                delta.append([x, y, old, new])
                if len(delta) >= 50:
                    return delta
    return delta if delta else None


def _action_data_dict(action: GameAction) -> dict[str, Any]:
    # GameAction.action_data is a pydantic model; fall back to {} if it's absent.
    # `game_id` is structural metadata (always present, always empty here) — strip it
    # so per-step history rows show only meaningful args (e.g. ACTION6's x/y).
    dump = getattr(getattr(action, "action_data", None), "model_dump", None)
    if not callable(dump):
        return {}
    return {k: v for k, v in dump().items() if k != "game_id"}


def _format_action_list(specs: Any) -> str:
    """Render a take_actions `actions` list as a compact log-friendly string.

    Output looks like `[A1,A1,A6(12,30),A5]` (with each item truncated to the
    short ARC action label). Returns "[?]" if the input isn't a list.
    """
    if not isinstance(specs, list):
        return "[?]"
    parts: list[str] = []
    for item in specs:
        if not isinstance(item, dict):
            parts.append("?")
            continue
        name = str(item.get("name") or "?")
        # Compact ACTION1 -> A1, ACTION6 -> A6 etc. for log readability.
        short = name
        if name.startswith("ACTION") and name[6:].isdigit():
            short = f"A{name[6:]}"
        if "x" in item and "y" in item:
            parts.append(f"{short}({item['x']},{item['y']})")
        else:
            parts.append(short)
    return "[" + ",".join(parts) + "]"


def _safe_frame_dump(frame: FrameData) -> dict[str, Any]:
    """JSON-safe dump of a FrameData for return to a sandboxed skill.

    The `frame` field is hex-rendered (list[list[str]]) so a skill reading
    `tools.take_actions(...).last_frame.frame` sees the same representation as
    `state.latest_frame.frame` and the working prompt. int→hex boundary.

    Falls back to a hand-built dict if model_dump fails (defensive — same
    pattern used when building the sandbox state at the top of an iter).
    """
    try:
        dump = frame.model_dump(mode="json")
    except (AttributeError, TypeError):
        dump = {
            "state": frame.state.name,
            "score": frame.levels_completed,
            "frame": frame.frame,
            "available_actions": list(frame.available_actions),
        }
    dump["frame"] = frame_to_hex(dump.get("frame"))
    return dump


def _build_sandbox_observations(
    pending: list[PendingActionObservation],
    frames: list[FrameData],
) -> list[dict[str, Any]]:
    """One hex entry per action since the last VLM query (mirrors OBSERVATIONS).

    Each entry is ``{step, action, source, state, score, frame}`` where ``frame``
    is the FULL hex animation stack (``list[list[str]]``) for that action —
    richer than the prompt's subsampled keyframes. int→hex boundary.
    """
    out: list[dict[str, Any]] = []
    n = len(frames)
    for obs in pending:
        idx = obs.post_frame_index
        if not (0 <= idx < n):
            continue
        post = frames[idx]
        data = obs.action_data or {}
        if "x" in data and "y" in data:
            label = f"{obs.action_name}(x={data['x']}, y={data['y']})"
        elif data:
            label = (
                obs.action_name
                + "("
                + ", ".join(f"{k}={v}" for k, v in data.items())
                + ")"
            )
        else:
            label = obs.action_name
        out.append(
            {
                "step": obs.action_counter,
                "action": label,
                "source": obs.source,
                "state": post.state.name,
                "score": post.levels_completed,
                "frame": frame_to_hex(post.frame),
            }
        )
    return out


def _optional_str_list_arg(args: dict[str, Any], key: str) -> list[str] | None:
    if key not in args or args[key] is None:
        return None
    value = args[key]
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, tuple):
        return [str(v) for v in value]
    raise ValueError(f"{key} must be an array of strings when provided")


class ContinualHarness(Agent):
    """Single-game VLM agent: one VLM call per outer iteration, unified `take_actions`.

    Overrides `Agent.main()` to drive the engine directly. Each outer iteration:
      1. If state is NOT_PLAYED, emit RESET and continue. If state is
         GAME_OVER, evolve the base prompt once for that terminal state, then
         emit RESET and continue.
      2. Maybe evolve the system prompt (boundary-gated by action_counter).
      3. Make ONE VLM call via `_vlm_loop_inner`. The response may contain
         analysis tool calls (process_memory, run_skill, etc.) and/or one
         `take_actions(actions=[...])` call. All are dispatched in emission
         order. take_actions executes its action list synchronously.
      4. If any actions were executed (orchestrator OR via a skill's inline
         tools.take_actions RPC), clear the carried tool-result block.
         Otherwise carry results forward to the next iteration's prompt
         (multi-round thinking across iterations).

    `choose_action` is unused (stubbed); the abstract method is satisfied but
    the engine is driven from `main()`.
    """

    MAX_ACTIONS = 5000
    MODEL = "gemini-3.1-pro-preview"  # default; override via GEMINI_MODEL
    HISTORY_MAX_CHARS = 12000  # budget for the RECENT HISTORY block
    HISTORY_BATCH_WINDOW = 5  # last N batches shown in compact history
    FULL_HISTORY_DEFAULT_LIMIT = 40
    FULL_HISTORY_MAX_LIMIT = 80
    MAX_SUBAGENT_CALLS_PER_STEP = 1  # distinct run_subagent invocations per outer step
    MAX_SUBAGENT_ROUNDS_PER_CALL = 20  # inner VLM rounds per invocation
    SUBAGENT_HISTORY_WINDOW = 20  # rows of compact history fed into a subagent's prompt
    RECENT_RESULTS_CAP = 16  # how many tool-result records to carry forward
    SKILL_TIMEOUT_S = 30.0  # wall-clock cap per run_skill (engine RPCs add latency)
    # Prompt-evolution defaults; the actual frequency is read from
    # CONTINUAL_HARNESS_PROMPT_EVOLVE_FREQUENCY (set by main.py from
    # --prompt-evolve-frequency). 0 disables; positive N means every N actions.
    DEFAULT_PROMPT_EVOLVE_FREQUENCY = 75
    PROMPT_EVOLVE_FREQUENCY_ENV = "CONTINUAL_HARNESS_PROMPT_EVOLVE_FREQUENCY"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Must resolve model_name BEFORE super().__init__(): Agent.__init__
        # calls start_recording() which reads self.name → self.model_name.
        self.model_name = os.getenv("GEMINI_MODEL", self.MODEL)
        super().__init__(*args, **kwargs)

        # Three-layer prompt architecture:
        # 1. _system_instruction: fixed, never evolved (tool schemas, game context)
        # 2. _current_base_prompt: evolved strategic guidance (rules, strategy)
        # 3. Per-step user prompt: game state, history, tool results
        freq_raw = os.getenv(
            self.PROMPT_EVOLVE_FREQUENCY_ENV,
            str(self.DEFAULT_PROMPT_EVOLVE_FREQUENCY),
        )
        try:
            self._prompt_evolve_frequency = max(0, int(freq_raw))
        except ValueError:
            self._prompt_evolve_frequency = self.DEFAULT_PROMPT_EVOLVE_FREQUENCY

        # Fixed system instruction — NEVER evolved. {game_name} substituted here.
        self._system_instruction: str = HARNESS_SYSTEM_INSTRUCTION.replace(
            "{game_name}", self.game_id
        )

        # Evolvable base prompt — strategic guidance, rule discoveries. Stored
        # per-game so parallel games evolve independent prompts.
        self._base_prompt_file = PromptFile(
            active_prompt_path(self.game_id), baseline=BASE_ORCHESTRATOR_POLICY
        )
        self._current_base_prompt: str = self._base_prompt_file.read()
        self._prompt_generation: int = 0
        self._last_evolution_step: int = -1
        self._last_game_over_evolution_step: int = -1
        self.prompt_evolution = PromptEvolutionStore(
            active_prompt_evolution_path(self.game_id)
        )

        self.vlm = VLM(
            self.model_name,
            backend="gemini",
            system_instruction=self._system_instruction,
        )
        self.vlm.set_tools(self._full_tool_list())
        # Per-game trace/trajectory JSONL under logs/<run>/<game_id>/.
        self.trace = TraceWriter(default_trace_path(self.game_id))
        self.trajectory = TrajectoryStore(default_trajectory_path(self.game_id))

        # --- COMMENTED OUT: get_recent_trajectory removed (superseded by the
        # always-in-prompt RECENT HISTORY block). format_full_history stays in
        # trajectory.py for prompt_evolution. ---
        # def _handle_get_recent_trajectory(args: dict[str, Any]) -> dict[str, Any]:
        #     try:
        #         limit = int(args.get("limit", self.FULL_HISTORY_DEFAULT_LIMIT))
        #     except (TypeError, ValueError):
        #         limit = self.FULL_HISTORY_DEFAULT_LIMIT
        #     limit = max(1, min(self.FULL_HISTORY_MAX_LIMIT, limit))
        #     records = self.trajectory.tail(limit)
        #     return {
        #         "success": True,
        #         "limit": limit,
        #         "count": len(records),
        #         "history": format_full_history(records),
        #     }

        # Memory is always available, backed per-game at
        # logs/<run_id>/<game_id>/memory.json (seeded from --bootstrap when a
        # single game was launched).
        self.memory = MemoryStore(
            active_memory_path(self.game_id), game_id=self.game_id
        )

        def _handle_process_memory(args: dict[str, Any]) -> dict[str, Any]:
            op = (args.get("operation") or "").strip().lower()
            try:
                if op == "add":
                    title = args.get("title") or ""
                    body = args.get("body") or ""
                    tags = list(args.get("tags") or [])
                    entry = self.memory.add(title=title, body=body, tags=tags)
                    return {
                        "success": True,
                        "operation": "add",
                        "id": entry.id,
                        "title": entry.title,
                        "tags": entry.tags,
                    }
                if op == "delete":
                    entry_id = args.get("id") or ""
                    if not entry_id:
                        return {
                            "success": False,
                            "operation": "delete",
                            "error": "delete requires an id",
                        }
                    deleted = self.memory.delete(entry_id)
                    result: dict[str, Any] = {
                        "success": deleted,
                        "operation": "delete",
                        "id": entry_id,
                        "deleted": deleted,
                    }
                    if not deleted:
                        result["error"] = f"no entry with id={entry_id}"
                    return result
                if op == "edit":
                    entry_id = args.get("id") or ""
                    if not entry_id:
                        return {
                            "success": False,
                            "operation": "edit",
                            "error": "edit requires an id",
                        }
                    edited = self.memory.edit(
                        entry_id,
                        title=args.get("title"),
                        body=args.get("body"),
                        tags=list(args["tags"]) if "tags" in args else None,
                    )
                    if edited is None:
                        return {
                            "success": False,
                            "operation": "edit",
                            "id": entry_id,
                            "error": f"no entry with id={entry_id}",
                        }
                    return {
                        "success": True,
                        "operation": "edit",
                        "id": entry_id,
                        "updated_at": edited.updated_at,
                    }
                if op == "search":
                    query = args.get("query") or ""
                    top, total = self.memory.search(query)
                    return {
                        "success": True,
                        "operation": "search",
                        "query": query,
                        "matches_returned": len(top),
                        "total_matches": total,
                        "entries": [asdict(e) for e in top],
                    }
                return {"success": False, "error": f"unknown operation: {op!r}"}
            except ValueError as exc:
                return {"success": False, "operation": op, "error": str(exc)}

        # Skills are always available too, backed per-game at
        # logs/<run_id>/<game_id>/skills.json (seeded from --bootstrap when a
        # single game was launched).
        self.skills = SkillStore(
            active_skill_path(self.game_id), game_id=self.game_id
        )

        # _current_sandbox_state is set at the top of each choose_action() so all
        # handlers (process_skill/run_skill) within that step share one
        # consistent JSON-only view of the world.
        self._current_sandbox_state: SandboxState | None = None

        def _handle_process_skill(args: dict[str, Any]) -> dict[str, Any]:
            op = (args.get("operation") or "").strip().lower()
            try:
                if op == "add":
                    name = args.get("name") or ""
                    description = args.get("description") or ""
                    code = args.get("code") or ""
                    tags = list(args.get("tags") or [])
                    entry = self.skills.add(
                        name=name, description=description, code=code, tags=tags
                    )
                    # Version > 1 means add() routed to upsert-by-name; tell
                    # the model so it knows the prior code was replaced.
                    upserted = entry.version > 1
                    return {
                        "success": True,
                        "operation": "add",
                        "id": entry.id,
                        "name": entry.name,
                        "tags": entry.tags,
                        "version": entry.version,
                        "upserted": upserted,
                        "message": (
                            f"updated existing skill {entry.name!r} "
                            f"(now version {entry.version})"
                            if upserted
                            else f"added new skill {entry.name!r}"
                        ),
                    }
                if op == "delete":
                    key = args.get("id") or ""
                    if not key:
                        return {
                            "success": False,
                            "operation": "delete",
                            "error": "delete requires an id (or name)",
                        }
                    target = self.skills.get_by_id_or_name(key)
                    if target is None:
                        return {
                            "success": False,
                            "operation": "delete",
                            "id": key,
                            "deleted": False,
                            "error": f"no skill with id-or-name={key}",
                        }
                    deleted = self.skills.delete(target.id)
                    return {
                        "success": deleted,
                        "operation": "delete",
                        "id": target.id,
                        "deleted": deleted,
                    }
                if op == "edit":
                    key = args.get("id") or ""
                    if not key:
                        return {
                            "success": False,
                            "operation": "edit",
                            "error": "edit requires an id (or name)",
                        }
                    target = self.skills.get_by_id_or_name(key)
                    if target is None:
                        return {
                            "success": False,
                            "operation": "edit",
                            "id": key,
                            "error": f"no skill with id-or-name={key}",
                        }
                    edited = self.skills.edit(
                        target.id,
                        name=args.get("name"),
                        description=args.get("description"),
                        code=args.get("code"),
                        tags=list(args["tags"]) if "tags" in args else None,
                    )
                    if edited is None:
                        return {
                            "success": False,
                            "operation": "edit",
                            "id": target.id,
                            "error": f"no skill with id={target.id}",
                        }
                    return {
                        "success": True,
                        "operation": "edit",
                        "id": target.id,
                        "version": edited.version,
                        "updated_at": edited.updated_at,
                    }
                if op == "search":
                    query = args.get("query") or ""
                    top, total = self.skills.search(query)
                    return {
                        "success": True,
                        "operation": "search",
                        "query": query,
                        "matches_returned": len(top),
                        "total_matches": total,
                        "entries": [asdict(e) for e in top],
                    }
                return {"success": False, "error": f"unknown operation: {op!r}"}
            except ValueError as exc:
                return {"success": False, "operation": op, "error": str(exc)}

        def _handle_run_skill(args: dict[str, Any]) -> dict[str, Any]:
            skill_id = args.get("id") or ""
            if not skill_id:
                return {"success": False, "error": "run_skill requires an id"}
            # Accept either canonical id (skill_NNN) or the human name shown
            # in the SKILLS overview — the model regularly confuses them and
            # eating a retry to disambiguate isn't worth it.
            skill = self.skills.get_by_id_or_name(skill_id)
            if skill is None:
                return {
                    "success": False,
                    "id": skill_id,
                    "error": f"no skill with id-or-name={skill_id}",
                }
            state = self._current_sandbox_state or SandboxState()

            # Per-skill sandbox bookkeeping. The on_rpc callback (self._sandbox_rpc)
            # uses these to tag trajectory rows and to enforce the terminal-state
            # latch: once an action returns WIN/GAME_OVER, subsequent RPCs from
            # the same skill are rejected so the skill must exit.
            self._sandbox_terminal_seen = False
            self._sandbox_skill_id = skill_id
            actions_before = self.action_counter
            try:
                out = run_python_snippet(
                    skill.code,
                    state=state,
                    args=dict(args.get("args") or {}),
                    images=self._current_images,
                    timeout_s=self.SKILL_TIMEOUT_S,
                    on_rpc=self._sandbox_rpc,
                )
            finally:
                self._sandbox_skill_id = None
                self._sandbox_terminal_seen = False

            out["id"] = skill_id
            out["name"] = skill.name
            out["version"] = skill.version
            out["actions_taken_inline"] = self.action_counter - actions_before
            return out

        # NOTE: run_code is intentionally NOT registered as a handler. It was
        # disabled after observation that the model wasted analysis rounds on
        # sandbox-policy violations (import statements) and wrong-schema
        # accesses on state.latest_frame. To re-enable, restore the closure,
        # add `"run_code": _handle_run_code` to `handlers`, add `RUN_CODE_TOOL`
        # back to `analysis_tools`, and put `"run_code"` back into
        # `SUBAGENT_TOOL_ENUM` in tools.py.

        # Subagents are always available too, backed per-game at
        # logs/<run_id>/<game_id>/subagents.json (seeded from --bootstrap when a
        # single game was launched).
        self.subagents = SubagentStore(
            active_subagent_path(self.game_id), game_id=self.game_id
        )

        # Per-iteration state used by _handle_run_subagent. Set at the top of
        # every _vlm_loop_inner call so any subagent invocation in that
        # iteration sees the live frame, images, and VLM-call counter, and
        # shares a single per-iteration call counter.
        self._current_latest_frame: FrameData | None = None
        self._current_images: list[Any] = []
        self._current_outer_round: int = 0
        self._subagent_call_count: int = 0

        # Cross-iteration state for the new VLM-call-driven loop.
        # `_recent_tool_results` carries forward analysis-tool outputs from one
        # outer iteration to the next (PokeAgent-style — cleared on action).
        # `_pending_observations` carries action transition/result frames from
        # executed actions into the next successful orchestrator VLM prompt.
        # `_consecutive_vlm_errors` is bookkeeping for hard failure detection.
        # `_sandbox_*` state is set per-skill by _handle_run_skill and read by
        # `_sandbox_rpc` to gate take_actions RPCs.
        # `_batch_counter` mints unique batch IDs for tagging trajectory rows.
        # `_vlm_call_count` is a monotonic counter set on `_current_outer_round`.
        self._recent_tool_results: list[ToolCallRecord] = []
        self._pending_observations: list[PendingActionObservation] = []
        self._consecutive_vlm_errors: int = 0
        self._sandbox_terminal_seen: bool = False
        self._sandbox_skill_id: str | None = None
        self._batch_counter: int = 0
        self._vlm_call_count: int = 0

        def _handle_process_subagent(args: dict[str, Any]) -> dict[str, Any]:
            op = (args.get("operation") or "").strip().lower()
            try:
                if op == "add":
                    name = args.get("name") or ""
                    description = args.get("description") or ""
                    instructions = args.get("instructions") or ""
                    allowed_tools = _optional_str_list_arg(args, "allowed_tools")
                    tags = _optional_str_list_arg(args, "tags") or []
                    entry = self.subagents.add(
                        name=name,
                        description=description,
                        instructions=instructions,
                        allowed_tools=allowed_tools,
                        tags=tags,
                    )
                    return {
                        "success": True,
                        "operation": "add",
                        "id": entry.id,
                        "name": entry.name,
                        "allowed_tools": entry.allowed_tools,
                        "tags": entry.tags,
                    }
                if op == "delete":
                    sub_id = args.get("id") or ""
                    if not sub_id:
                        return {
                            "success": False,
                            "operation": "delete",
                            "error": "delete requires an id",
                        }
                    deleted = self.subagents.delete(sub_id)
                    result: dict[str, Any] = {
                        "success": deleted,
                        "operation": "delete",
                        "id": sub_id,
                        "deleted": deleted,
                    }
                    if not deleted:
                        result["error"] = f"no subagent with id={sub_id}"
                    return result
                if op == "edit":
                    sub_id = args.get("id") or ""
                    if not sub_id:
                        return {
                            "success": False,
                            "operation": "edit",
                            "error": "edit requires an id",
                        }
                    edited = self.subagents.edit(
                        sub_id,
                        name=args.get("name"),
                        description=args.get("description"),
                        instructions=args.get("instructions"),
                        allowed_tools=_optional_str_list_arg(args, "allowed_tools"),
                        tags=_optional_str_list_arg(args, "tags"),
                    )
                    if edited is None:
                        return {
                            "success": False,
                            "operation": "edit",
                            "id": sub_id,
                            "error": f"no subagent with id={sub_id}",
                        }
                    return {
                        "success": True,
                        "operation": "edit",
                        "id": sub_id,
                        "version": edited.version,
                        "updated_at": edited.updated_at,
                    }
                if op == "search":
                    query = args.get("query") or ""
                    top, total = self.subagents.search(query)
                    return {
                        "success": True,
                        "operation": "search",
                        "query": query,
                        "matches_returned": len(top),
                        "total_matches": total,
                        "entries": [asdict(e) for e in top],
                    }
                return {"success": False, "error": f"unknown operation: {op!r}"}
            except ValueError as exc:
                return {"success": False, "operation": op, "error": str(exc)}

        def _handle_run_subagent(args: dict[str, Any]) -> dict[str, Any]:
            if self._subagent_call_count >= self.MAX_SUBAGENT_CALLS_PER_STEP:
                return {
                    "success": False,
                    "error": (
                        f"subagent budget exhausted this step "
                        f"(MAX_SUBAGENT_CALLS_PER_STEP={self.MAX_SUBAGENT_CALLS_PER_STEP})"
                    ),
                }
            self._subagent_call_count += 1

            sub_id = args.get("id") or ""
            if not sub_id:
                return {"success": False, "error": "run_subagent requires an id"}
            entry = self.subagents.get(sub_id)
            if entry is None:
                return {
                    "success": False,
                    "id": sub_id,
                    "error": f"no subagent with id={sub_id}",
                }
            if self._current_latest_frame is None:
                return {
                    "success": False,
                    "id": sub_id,
                    "error": "run_subagent invoked before choose_action stash; refusing",
                }

            task = str(args.get("task") or "")
            raw_context = args.get("context") or {}
            context = dict(raw_context) if isinstance(raw_context, dict) else {}

            tools = build_subagent_tools(entry.allowed_tools)
            sub_vlm = VLM(
                self.model_name,
                backend="gemini",
                system_instruction=entry.instructions,
            )
            sub_vlm.set_tools(tools)

            history = render_recent_history(
                self.trajectory.tail(self.MAX_ACTIONS + 1),
                max_chars=self.HISTORY_MAX_CHARS,
            )
            base_prompt = build_subagent_prompt(
                task=task,
                context=context,
                latest_frame=self._current_latest_frame,
                memory_overview=format_memory_overview(self.memory.all_entries()),
                skill_overview=format_skill_overview(self.skills.all_entries()),
                compact_history=history,
            )
            working_prompt = base_prompt
            images = self._current_images
            payload: Any = (
                images if len(images) > 1 else (images[0] if images else None)
            )

            tool_calls_log: list[ToolCallRecord] = []
            subagent_steps: list[dict[str, Any]] = []
            final_answer: dict[str, Any] | None = None
            last_error: str | None = None
            warning: str | None = None
            forced_return = False
            inner_round = 0
            total_subagent_actions = 0

            for inner_round in range(1, self.MAX_SUBAGENT_ROUNDS_PER_CALL + 1):
                round_records: list[ToolCallRecord] = []
                output: dict[str, Any] = {}
                usage: dict[str, int | None] | None = None
                round_error: str | None = None
                ret_call_args: dict[str, Any] | None = None
                round_actions = 0

                try:
                    response = sub_vlm.get_query(
                        payload,
                        working_prompt,
                        module_name=f"{self.name}.subagent.{entry.name}",
                    )
                    output = serialize_response(response)
                    usage = sub_vlm.extract_usage(response)
                    fcs = extract_function_calls(response)

                    for fc in fcs:
                        if is_subagent_return_call(fc.name):
                            ret_call_args = fc.args
                            final_answer = {
                                "answer": str(fc.args.get("answer") or ""),
                                "status": str(fc.args.get("status") or "success"),
                                "reasoning": str(fc.args.get("reasoning") or ""),
                            }
                            break

                        if is_take_actions_call(fc.name):
                            if "take_actions" not in (entry.allowed_tools or []):
                                round_records.append(
                                    ToolCallRecord(
                                        name=fc.name,
                                        args=fc.args,
                                        error="take_actions not in allowed_tools",
                                    )
                                )
                                continue
                            executed = self._dispatch_take_actions(
                                fc.args, source="subagent"
                            )
                            round_actions += executed
                            round_records.append(
                                ToolCallRecord(
                                    name=fc.name,
                                    args=fc.args,
                                    result={"executed": executed},
                                    actions_taken_inline=executed,
                                )
                            )
                            if self.frames[-1].state in (
                                GameState.WIN,
                                GameState.GAME_OVER,
                            ):
                                break
                            continue

                        if fc.name not in (entry.allowed_tools or []):
                            round_records.append(
                                ToolCallRecord(
                                    name=fc.name,
                                    args=fc.args,
                                    error=(
                                        f"tool {fc.name!r} not in allowed_tools "
                                        f"{entry.allowed_tools}"
                                    ),
                                )
                            )
                            continue
                        round_records.append(self.tool_router.execute(fc))
                except Exception as exc:
                    round_error = repr(exc)
                    last_error = round_error
                    logger.warning(
                        "Subagent %s inner round %d/%d failed: %s",
                        entry.name,
                        inner_round,
                        self.MAX_SUBAGENT_ROUNDS_PER_CALL,
                        exc,
                    )
                finally:
                    self.trace.write(
                        {
                            "agent": self.name,
                            "model": self.model_name,
                            "game_id": self.game_id,
                            "action_counter": self.action_counter,
                            "round": self._current_outer_round,
                            "tools_exposed": "subagent",
                            "subagent": {
                                "id": entry.id,
                                "name": entry.name,
                                "version": entry.version,
                                "inner_round": inner_round,
                                "max_inner_rounds": self.MAX_SUBAGENT_ROUNDS_PER_CALL,
                            },
                            "input": {
                                "system_instruction": entry.instructions,
                                "user_prompt": working_prompt,
                                "tools": tools,
                                "images": [
                                    {
                                        "width": img.width,
                                        "height": img.height,
                                        "mode": img.mode,
                                    }
                                    for img in images
                                ],
                            },
                            "output": output,
                            "usage": usage,
                            "chosen_action": None,
                            "reasoning": None,
                            "tool_calls": [asdict(r) for r in round_records],
                            "subagent_return": ret_call_args,
                            "error": round_error,
                        }
                    )
                    if usage is not None:
                        self.total_calls += 1
                        self.total_prompt_tokens += int(usage.get("prompt") or 0)
                        self.total_output_tokens += int(usage.get("output") or 0)
                        self.total_tokens += int(usage.get("total") or 0)

                total_subagent_actions += round_actions
                round_step = {
                    "inner_round": inner_round,
                    "tool_calls": [asdict(r) for r in round_records],
                    "subagent_return": ret_call_args,
                    "error": round_error,
                    "actions": round_actions,
                }
                if usage is not None:
                    round_step["usage"] = usage
                subagent_steps.append(round_step)
                tool_calls_log.extend(round_records)

                if final_answer is not None:
                    break
                if self.frames[-1].state in (GameState.WIN, GameState.GAME_OVER):
                    break
                if round_error is not None:
                    break
                if not round_records:
                    last_error = "subagent produced no tool calls and did not return"
                    break

                if round_actions > 0:
                    latest = self.frames[-1]
                    self._current_latest_frame = latest
                    self._current_images = list(frame_to_images(latest))
                    images = self._current_images
                    payload = (
                        images
                        if len(images) > 1
                        else (images[0] if images else None)
                    )
                    history = render_recent_history(
                        self.trajectory.tail(self.MAX_ACTIONS + 1),
                        max_chars=self.HISTORY_MAX_CHARS,
                    )
                    base_prompt = build_subagent_prompt(
                        task=task,
                        context=context,
                        latest_frame=latest,
                        memory_overview=format_memory_overview(
                            self.memory.all_entries()
                        ),
                        skill_overview=format_skill_overview(
                            self.skills.all_entries()
                        ),
                        compact_history=history,
                    )

                working_prompt = (
                    base_prompt + "\n\n" + render_tool_results(round_records)
                )

            if (
                final_answer is None
                and last_error is None
                and inner_round >= self.MAX_SUBAGENT_ROUNDS_PER_CALL
            ):
                warning = (
                    f"subagent exhausted max inner rounds "
                    f"({self.MAX_SUBAGENT_ROUNDS_PER_CALL}) without "
                    f"subagent_return; forcing return to orchestrator"
                )
                logger.warning(
                    "Subagent %s exhausted max inner rounds (%d) without "
                    "subagent_return; forcing return to orchestrator",
                    entry.name,
                    self.MAX_SUBAGENT_ROUNDS_PER_CALL,
                )
                last_error = warning
                forced_return = True
                final_answer = {
                    "answer": warning,
                    "status": "failure",
                    "reasoning": (
                        "Forced return because the subagent reached its inner "
                        "round limit without calling subagent_return."
                    ),
                }

            return {
                "success": final_answer is not None and not forced_return,
                "id": entry.id,
                "name": entry.name,
                "version": entry.version,
                "rounds_used": inner_round,
                "result": final_answer,
                "forced_return": forced_return,
                "warning": warning,
                "error": last_error,
                "steps": subagent_steps,
                "tool_calls": [asdict(r) for r in tool_calls_log],
                "actions_taken_inline": total_subagent_actions,
            }

        handlers: dict[str, Any] = {
            # "get_recent_trajectory": _handle_get_recent_trajectory,  # removed
            "process_memory": _handle_process_memory,
            "process_skill": _handle_process_skill,
            "run_skill": _handle_run_skill,
            "process_subagent": _handle_process_subagent,
            "run_subagent": _handle_run_subagent,
        }
        self.tool_router = ContinualToolRouter(handlers)

        logger.info(
            "[%s] Long-term memory enabled: %s (%d existing entries)",
            self.name,
            self.memory.path,
            len(self.memory.all_entries()),
        )
        logger.info(
            "[%s] Skills enabled: %s (%d existing entries)",
            self.name,
            self.skills.path,
            len(self.skills.all_entries()),
        )
        logger.info(
            "[%s] Subagents enabled: %s (%d existing entries)",
            self.name,
            self.subagents.path,
            len(self.subagents.all_entries()),
        )
        logger.info(
            "[%s] Prompt evolution: frequency=%d, baseline at %s, log at %s",
            self.name,
            self._prompt_evolve_frequency,
            self._base_prompt_file.path,
            self.prompt_evolution.path,
        )

        # Cumulative token usage; logged once on cleanup().
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0

    @property
    def name(self) -> str:
        sanitized = self.model_name.replace("/", "-").replace(":", "-")
        return f"{super().name}.{sanitized}"

    def start_recording(self) -> None:
        """Record into the per-game folder logs/<run>/<game_id>/recordings/.

        Overrides Agent.start_recording (called from Agent.__init__, after
        self.game_id is set) so parallel games don't share one recordings dir.
        Falls back to the run-wide RUN_RECORDINGS_DIR when no run dir is wired.
        """
        run_dir = os.environ.get(RUN_DIR_ENV)
        directory = (
            str(game_artifacts(run_dir, self.game_id).recordings_dir)
            if run_dir
            else os.environ.get(RUN_RECORDINGS_DIR_ENV)
        )
        self.recorder = Recorder(prefix=self.name, filename=None, directory=directory)
        logger.info(
            "created new recording for %s into %s", self.name, self.recorder.filename
        )

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

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

    def _evolve_system_prompt(self, latest_frame: FrameData) -> None:
        """One meta-VLM call that may rewrite the agent's base prompt.

        Spawns a fresh VLM with EVOLUTION_SYSTEM_INSTRUCTION. The model returns
        the improved prompt as plain text. Accepted only if
        `validate_evolved_prompt` returns ok=True; on accept, the new text is
        written through `self._base_prompt_file`. Every attempt is logged.
        """
        self._prompt_generation += 1
        gen = self._prompt_generation
        previous = self._current_base_prompt

        steps_since = max(1, self.action_counter - max(self._last_evolution_step, 0))
        trajectory_rows = self.trajectory.tail(steps_since)
        user_prompt = build_evolution_prompt(
            system_prompt=self._system_instruction,
            current_base_prompt=previous,
            trajectory_rows=trajectory_rows,
            memory_overview=format_memory_full(self.memory.all_entries()),
            skill_overview=format_skill_overview(self.skills.all_entries()),
            subagent_overview=format_subagent_overview(self.subagents.all_entries()),
        )

        evolution_system = EVOLUTION_SYSTEM_INSTRUCTION.replace(
            "{game_name}", self.game_id
        )
        meta_vlm = VLM(
            self.model_name,
            backend="gemini",
            system_instruction=evolution_system,
        )

        images = list(frame_to_images(latest_frame))
        payload: Any = images if len(images) > 1 else (images[0] if images else None)

        proposed = ""
        accepted = False
        validation_error: str | None = None
        usage: dict[str, int | None] | None = None
        output: dict[str, Any] = {}
        error: str | None = None

        try:
            response = meta_vlm.get_query(
                payload,
                user_prompt,
                module_name=f"{self.name}.evolve.{gen}",
            )
            output = serialize_response(response)
            usage = meta_vlm.extract_usage(response)
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
                self.total_calls += 1
                self.total_prompt_tokens += int(usage.get("prompt") or 0)
                self.total_output_tokens += int(usage.get("output") or 0)
                self.total_tokens += int(usage.get("total") or 0)

            record = PromptEvolutionRecord(
                generation=gen,
                action_counter=self.action_counter,
                accepted=accepted,
                reasoning="",
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
                    "agent": self.name,
                    "model": self.model_name,
                    "game_id": self.game_id,
                    "action_counter": self.action_counter,
                    "round": 0,
                    "tools_exposed": "evolution",
                    "evolution": {
                        "generation": gen,
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
                    "error": error,
                }
            )

            logger.info(
                "[%s] Prompt evolution gen=%d accepted=%s len=%d (prev=%d) error=%s",
                self.name,
                gen,
                accepted,
                len(self._current_base_prompt),
                len(previous),
                validation_error or error,
            )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # Abstract on `Agent`; satisfied here as a stub. ContinualHarness
        # drives the engine via the overridden `main()` and never calls
        # `choose_action`. If something does call it (e.g. an external
        # driver), surface that loudly.
        raise NotImplementedError(
            "ContinualHarness drives the engine via main(); "
            "choose_action is unused."
        )

    # ------------------------------------------------------------------
    # Main loop — overrides Agent.main() with @trace_agent_session re-applied.
    # ------------------------------------------------------------------

    @trace_agent_session
    def main(self) -> None:
        self.timer = time.time()
        while (
            not self.is_done(self.frames, self.frames[-1])
            and self.action_counter <= self.MAX_ACTIONS
        ):
            latest_frame = self.frames[-1]

            # Terminal failure: consolidate the failed trajectory before reset.
            if latest_frame.state is GameState.GAME_OVER:
                self._evolve_on_game_over(latest_frame)
                self._execute_one(GameAction.RESET, source="auto_reset")
                self._recent_tool_results = []
                continue

            # Idle initial state: emit RESET and restart the iteration.
            if latest_frame.state is GameState.NOT_PLAYED:
                self._execute_one(GameAction.RESET, source="auto_reset")
                self._recent_tool_results = []
                continue

            pre_step_level = latest_frame.levels_completed

            # Prompt-evolution hook — boundary-gated by action_counter rather
            # than modulo so multi-action batches that straddle a boundary
            # still fire exactly once.
            self._maybe_evolve_prompt(latest_frame)

            actions_executed = self._vlm_loop_inner(latest_frame)

            if actions_executed > 0:
                self._recent_tool_results = []

            # Level-up evolution: consolidate discovered rules immediately
            # after advancing to a new level.
            post_frame = self.frames[-1]
            if post_frame.levels_completed > pre_step_level:
                self._evolve_on_level_up(post_frame)

        self.cleanup()

    # ------------------------------------------------------------------
    # Inner VLM dispatch — one VLM call per outer iteration.
    # ------------------------------------------------------------------

    def _vlm_loop_inner(self, latest_frame: FrameData) -> int:
        """Make one VLM call, dispatch every function call, return # actions executed.

        - `take_actions` calls execute their action list synchronously via
          `_dispatch_take_actions`.
        - Analysis tool calls (process_*, run_skill, run_subagent, etc.) go
          through `self.tool_router`. A skill that called tools.take_actions()
          inside the sandbox contributes its inline action count via the
          `actions_taken_inline` field on the returned ToolCallRecord.
        - Stray legacy ACTION1..ACTION6 calls are rejected with an explicit
          error record so the model can self-correct next round.
        """
        # Stash per-iteration state used by sub-handlers (run_subagent, etc.).
        self._current_latest_frame = latest_frame
        self._current_images = list(frame_to_images(latest_frame))
        self._subagent_call_count = 0
        self._vlm_call_count += 1
        self._current_outer_round = self._vlm_call_count

        # Build a JSON-only sandbox snapshot. The sandbox sees this view at
        # the *start* of the skill call; subsequent take_actions RPCs update
        # the engine state but `state.latest_frame` inside the sandbox stays
        # fixed (the skill should read fresh frames from the RPC's return).
        try:
            frame_dump = latest_frame.model_dump(mode="json")
        except (AttributeError, TypeError):
            frame_dump = {
                "state": latest_frame.state.name,
                "score": latest_frame.levels_completed,
                "frame": latest_frame.frame,
            }
        # int→hex boundary: the sandbox `state` mirrors the working prompt's hex
        # view. `latest_frame.frame` becomes a hex stack; `observations` carries
        # the per-action hex frames; recent_trajectory colors are hex-mapped
        # (counts/coords stay int). Engine storage / trajectory.jsonl stay int.
        frame_dump["frame"] = frame_to_hex(frame_dump.get("frame"))
        self._current_sandbox_state = SandboxState(
            latest_frame=frame_dump,
            observations=_build_sandbox_observations(
                self._pending_observations, self.frames
            ),
            recent_trajectory=[
                hexify_record_colors(r)
                for r in self.trajectory.tail(self.FULL_HISTORY_MAX_LIMIT)
            ],
            memory_entries=[asdict(e) for e in self.memory.all_entries()],
            skill_entries=[asdict(e) for e in self.skills.all_entries()],
        )

        prompt, rendered_grids = self._build_working_prompt(latest_frame)
        tools = self._full_tool_list()

        # The orchestrator image payload mirrors the grids rendered in the
        # prompt text, in order. Skills still see every grid from the current
        # latest_frame via `self._current_images` -> `state.images`.
        vlm_images = [grid_to_image(item.grid) for item in rendered_grids]
        payload: Any = (
            vlm_images
            if len(vlm_images) > 1
            else (vlm_images[0] if vlm_images else [])
        )

        output: dict[str, Any] = {}
        usage: dict[str, int | None] | None = None
        error: str | None = None
        try:
            response = self.vlm.get_query(payload, prompt, module_name=self.name)
            output = serialize_response(response)
            usage = self.vlm.extract_usage(response)
        except Exception as exc:
            self._consecutive_vlm_errors += 1
            self._write_orchestrator_trace(
                prompt=prompt, output={}, usage=None, tools=tools,
                tool_calls=[], actions_executed=0, rendered_grids=rendered_grids,
                error=repr(exc),
            )
            logger.warning("VLM call failed: %s", exc)
            return 0
        self._consecutive_vlm_errors = 0
        # Observations shown in this prompt have now been delivered. Clear
        # before dispatching the response so any actions emitted by this query
        # become the next prompt's observations.
        self._pending_observations = []

        fcs = extract_function_calls(response)
        actions_executed = 0
        new_results: list[ToolCallRecord] = []
        # Track the human-readable name of every tool call this round, in
        # emission order, for the run.log summary at the end. take_actions
        # entries include the emitted action list inline so a `grep
        # take_actions` against run.log shows exactly what was committed.
        round_tool_log: list[str] = []

        for fc in fcs:
            if is_take_actions_call(fc.name):
                action_specs = fc.args.get("actions") if isinstance(fc.args, dict) else None
                spec_label = _format_action_list(action_specs)
                executed = self._dispatch_take_actions(fc.args, source="vlm")
                actions_executed += executed
                round_tool_log.append(f"take_actions{spec_label}={executed}")
                # We don't add a TOOL RESULTS entry for take_actions — the
                # next prompt's RECENT HISTORY block already shows what ran.
            elif is_action_tool(fc.name):
                # Legacy per-action tool emitted directly — reject and
                # surface so the model corrects.
                new_results.append(
                    ToolCallRecord(
                        name=fc.name,
                        args=fc.args,
                        error=(
                            f"per-action tool deprecated; emit {TAKE_ACTIONS} "
                            "with an actions=[...] list"
                        ),
                    )
                )
                round_tool_log.append(f"REJECTED:{fc.name}")
            else:
                record = self.tool_router.execute(fc)
                inline = record.actions_taken_inline or 0
                actions_executed += inline
                new_results.append(record)
                if fc.name == "run_skill" and inline > 0:
                    skill_id = (fc.args or {}).get("id") if isinstance(fc.args, dict) else None
                    round_tool_log.append(f"run_skill({skill_id})={inline}")
                else:
                    round_tool_log.append(fc.name)

            if self.frames[-1].state in (GameState.WIN, GameState.GAME_OVER):
                # Terminal mid-response — stop processing further fcs.
                break

        if usage is not None:
            self.total_calls += 1
            self.total_prompt_tokens += int(usage.get("prompt") or 0)
            self.total_output_tokens += int(usage.get("output") or 0)
            self.total_tokens += int(usage.get("total") or 0)

        self._write_orchestrator_trace(
            prompt=prompt, output=output, usage=usage, tools=tools,
            tool_calls=new_results, actions_executed=actions_executed,
            rendered_grids=rendered_grids, error=error,
        )

        # Carry analysis results to the next iteration's prompt. Cap to avoid
        # unbounded growth across consecutive no-action iters. If actions
        # executed, main() will clear this list anyway.
        if new_results:
            self._recent_tool_results = (
                self._recent_tool_results + new_results
            )[-self.RECENT_RESULTS_CAP:]

        # Per-VLM-call run.log summary. One line per outer iteration.
        latest = self.frames[-1]
        tokens_total = int((usage or {}).get("total") or 0) if usage else 0
        tools_label = ", ".join(round_tool_log) if round_tool_log else "(no fcs)"
        logger.info(
            "[%s] vlm#%d step=%d lvl=%d state=%s tokens=%d actions=%d tools=[%s]",
            self.game_id,
            self._vlm_call_count,
            self.action_counter,
            latest.levels_completed,
            latest.state.name,
            tokens_total,
            actions_executed,
            tools_label,
        )

        return actions_executed

    # ------------------------------------------------------------------
    # take_actions dispatch — used by both the VLM path and the sandbox RPC.
    # ------------------------------------------------------------------

    def _dispatch_take_actions(
        self,
        args: dict[str, Any],
        *,
        source: str,
        skill_id: str | None = None,
    ) -> int:
        """Validate + execute an actions list synchronously. Returns # executed.

        No length-cap truncation — the soft 1-8 guideline lives in the prompt
        only. Real bounds come from per-step revalidation (level transitions
        / available_actions changes), terminal-state detection, and the global
        `MAX_ACTIONS` counter checked after each step.
        """
        available = available_game_actions(self.frames[-1].available_actions)
        steps, _rejected = validate_action_sequence(
            args.get("actions"), available
        )
        if not steps:
            return 0
        batch_id = self._mint_batch_id()
        batch_reasoning = str(args.get("reasoning") or "")
        rejected_payload = (
            [r.to_dict() for r in _rejected] if _rejected else None
        )
        total = len(steps)
        skill_id = skill_id if skill_id is not None else self._sandbox_skill_id
        executed = 0
        # Snapshot the level before the batch so we can abort the remainder if a
        # mid-batch level transition happens. The available_actions check below
        # already covers most cases, but a level can advance without changing
        # the action set, and the rest of the queue was planned for the OLD
        # level — running it blindly burns the budget on the wrong puzzle.
        pre_level = self.frames[-1].levels_completed

        for step in steps:
            # Per-step revalidation: state may have moved (e.g., level
            # transition) so the next action may no longer be available.
            live_avail = available_game_actions(
                self.frames[-1].available_actions
            )
            if step.action not in live_avail:
                break
            self._execute_one(
                step.action,
                source=source,
                batch_id=batch_id,
                batch_position=step.position,
                batch_total=total,
                batch_reasoning=batch_reasoning if step.position == 1 else None,
                batch_rejected=rejected_payload if step.position == 1 else None,
                skill_id=skill_id,
            )
            executed += 1
            if self.frames[-1].state in (GameState.WIN, GameState.GAME_OVER):
                break
            if self.frames[-1].levels_completed != pre_level:
                break
            if self.action_counter > self.MAX_ACTIONS:
                break
        return executed

    # ------------------------------------------------------------------
    # _execute_one — the single execution path (drain, vlm, skill, reset, backstop).
    # ------------------------------------------------------------------

    def _execute_one(
        self,
        action: GameAction,
        *,
        source: str,
        batch_id: str | None = None,
        batch_position: int | None = None,
        batch_total: int | None = None,
        batch_reasoning: str | None = None,
        batch_rejected: list[dict[str, Any]] | None = None,
        skill_id: str | None = None,
    ) -> FrameData | None:
        pre_frame_index = len(self.frames) - 1
        pre = self.frames[-1]
        pre_state, pre_score = pre.state.name, pre.levels_completed
        pre_grid = pre.frame[-1] if pre.frame else None

        frame = self.take_action(action)        # base class: arc_env.step + validate
        if frame is not None:
            self.append_frame(frame)            # base class: also calls recorder.record
        post_frame_index = len(self.frames) - 1
        self.action_counter += 1                # mirror base Agent.main() semantics

        post = frame if frame is not None else pre
        score_after = post.levels_completed
        state_after = post.state.name if frame is not None else "INVALID"

        post_grid = post.frame[-1] if post.frame else None
        grid_delta = _compute_grid_delta(pre_grid, post_grid)
        grid_change = summarize_grid_transitions(pre_grid, post_grid)

        self.trajectory.append(
            StepRecord(
                game_id=self.game_id,
                action_counter=self.action_counter - 1,
                state=pre_state,
                score=pre_score,
                state_after=state_after,
                score_delta=(score_after - pre_score) if frame is not None else None,
                chosen_action=action.name,
                chosen_action_data=_action_data_dict(action),
                reasoning=getattr(action, "reasoning", None),
                source=source,
                batch_id=batch_id,
                batch_position=batch_position,
                batch_total=batch_total,
                batch_reasoning=batch_reasoning,
                batch_rejected=batch_rejected,
                skill_id=skill_id,
                tool_calls=[],
                grid_delta=grid_delta,
                grid_change=grid_change,
            )
        )
        self._pending_observations.append(
            PendingActionObservation(
                action_counter=self.action_counter - 1,
                action_name=action.name,
                action_data=_action_data_dict(action),
                source=source,
                batch_id=batch_id,
                batch_position=batch_position,
                batch_total=batch_total,
                pre_frame_index=pre_frame_index,
                post_frame_index=post_frame_index,
                valid_frame=frame is not None,
            )
        )

        # Per-action progress log (visible in run.log). Replaces the
        # base-class line we lose by overriding main(). Includes the batch
        # context, levels_completed, score delta, and avg fps so a single
        # tail of run.log shows engine progression.
        action_label = action.name
        data = _action_data_dict(action)
        if data:
            action_label = f"{action.name}({','.join(f'{k}={v}' for k, v in data.items())})"
        batch_ctx = ""
        if batch_id and batch_position is not None and batch_total is not None:
            batch_ctx = f" {batch_id}[{batch_position}/{batch_total}]"
        score_label = f"lvl {pre_score}"
        if frame is not None and score_after != pre_score:
            score_label = f"lvl {pre_score}->{score_after}"
        state_label = state_after if state_after != pre_state else pre_state
        logger.info(
            "[%s] step=%d %s%s src=%s %s state=%s fps=%.2f",
            self.game_id,
            self.action_counter - 1,
            action_label,
            batch_ctx,
            source,
            score_label,
            state_label,
            self.fps,
        )
        return frame

    # ------------------------------------------------------------------
    # Sandbox RPC dispatcher — invoked from a skill's tools.take_actions().
    # ------------------------------------------------------------------

    def _sandbox_rpc(self, method: str, args: dict[str, Any]) -> dict[str, Any]:
        """Handle one RPC from a sandboxed skill.

        Returns the response dict that the sandbox parent will JSON-encode and
        send back over stdin to the worker.
        """
        if method != "take_actions":
            return {"ok": False, "error": f"unknown rpc method: {method!r}"}
        if self._sandbox_terminal_seen:
            return {
                "ok": False,
                "error": "terminal state already reached; skill must return",
                "terminal": True,
            }
        pre_level = self.frames[-1].levels_completed
        executed = self._dispatch_take_actions(args, source="run_skill")
        last = self.frames[-1]
        terminal = last.state in (GameState.WIN, GameState.GAME_OVER)
        level_changed = last.levels_completed != pre_level
        if terminal:
            self._sandbox_terminal_seen = True
        frame_dump = _safe_frame_dump(last)
        return {
            "ok": executed > 0,
            "value": {
                "executed_count": executed,
                "last_frame": frame_dump,
                "frame": frame_dump,
                "terminal": terminal,
                "level_changed": level_changed,
                "state": last.state.name,
                "score": last.levels_completed,
                "available_actions": [
                    a.name for a in available_game_actions(last.available_actions)
                ],
            },
            "terminal": terminal,
        }

    # ------------------------------------------------------------------
    # Working-prompt assembly.
    # ------------------------------------------------------------------

    def _build_working_prompt(
        self, latest_frame: FrameData
    ) -> tuple[str, list[RenderedGrid]]:
        history_block = render_recent_history(
            self.trajectory.tail(self.MAX_ACTIONS + 1),
            max_chars=self.HISTORY_MAX_CHARS,
        )
        observation_block, observation_grids = build_observation_section(
            self._pending_observations,
            self.frames,
        )
        memory_overview = format_memory_overview(self.memory.all_entries())
        skill_overview = format_skill_overview(self.skills.all_entries())
        subagent_overview = format_subagent_overview(self.subagents.all_entries())
        current_grid = current_state_rendered_grid(latest_frame)
        rendered_grids = list(observation_grids)
        if current_grid is not None:
            rendered_grids.append(current_grid)
        prompt = build_working_prompt(
            latest_frame,
            action_counter=self.action_counter,
            recent_tool_results=self._recent_tool_results,
            history_block=history_block,
            memory_overview=memory_overview,
            skill_overview=skill_overview,
            subagent_overview=subagent_overview,
            observation_block=observation_block,
            base_prompt=self._current_base_prompt,
        )
        return prompt, rendered_grids

    # ------------------------------------------------------------------
    # Tool list, prompt-evolution gate, batch IDs, trace writer.
    # ------------------------------------------------------------------

    def _full_tool_list(self) -> list[dict[str, Any]]:
        """Unified action tool + analysis tools — exposed every iteration."""
        tools: list[dict[str, Any]] = [TAKE_ACTIONS_TOOL]
        tools.extend(build_analysis_tools())  # currently empty (get_recent_trajectory removed)
        tools.append(PROCESS_MEMORY_TOOL)
        tools.extend([PROCESS_SKILL_TOOL, RUN_SKILL_TOOL])
        tools.extend([PROCESS_SUBAGENT_TOOL, RUN_SUBAGENT_TOOL])
        return tools

    def _maybe_evolve_prompt(self, latest_frame: FrameData) -> None:
        if self._prompt_evolve_frequency <= 0:
            return
        if self.action_counter == 0:
            return
        gap = self.action_counter - self._last_evolution_step
        if gap < self._prompt_evolve_frequency:
            return
        self._last_evolution_step = self.action_counter
        self._evolve_system_prompt(latest_frame)

    def _evolve_on_level_up(self, latest_frame: FrameData) -> None:
        """Trigger evolution unconditionally on level transition."""
        if self._prompt_evolve_frequency <= 0:
            return
        self._last_evolution_step = self.action_counter
        self._evolve_system_prompt(latest_frame)

    def _evolve_on_game_over(self, latest_frame: FrameData) -> None:
        """Trigger evolution once for a GAME_OVER state before auto-reset."""
        if self._prompt_evolve_frequency <= 0:
            return
        if self._last_game_over_evolution_step == self.action_counter:
            return
        self._last_game_over_evolution_step = self.action_counter
        self._last_evolution_step = self.action_counter
        self._evolve_system_prompt(latest_frame)

    def _mint_batch_id(self) -> str:
        self._batch_counter += 1
        # Suffix with a short uuid fragment so batch IDs are unique even
        # across restarts within the same trajectory file.
        return f"b_{self._batch_counter:04d}_{uuid.uuid4().hex[:4]}"

    def _write_orchestrator_trace(
        self,
        *,
        prompt: str,
        output: dict[str, Any],
        usage: dict[str, int | None] | None,
        tools: list[dict[str, Any]],
        tool_calls: list[ToolCallRecord],
        actions_executed: int,
        rendered_grids: list[RenderedGrid],
        error: str | None = None,
    ) -> None:
        self.trace.write(
            {
                "agent": self.name,
                "model": self.model_name,
                "game_id": self.game_id,
                "action_counter": self.action_counter,
                "vlm_call": self._vlm_call_count,
                "round": self._current_outer_round,
                "tools_exposed": "full",
                "input": {
                    "system_instruction": self._system_instruction,
                    "base_prompt": self._current_base_prompt,
                    "user_prompt": prompt,
                    "tools": tools,
                    # `images` mirrors the grids rendered in this prompt, in
                    # prompt order. The PNG payload is built from the same
                    # RenderedGrid list.
                    "images": [
                        {
                            "label": item.label,
                            "width": len(item.grid[0]) if item.grid else 0,
                            "height": len(item.grid),
                        }
                        for item in rendered_grids
                    ],
                    "images_attached_count": len(rendered_grids),
                },
                "output": output,
                "usage": usage,
                "actions_executed": actions_executed,
                "tool_calls": [asdict(r) for r in tool_calls],
                "error": error,
            }
        )

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        logger.info(
            "[%s] Final token usage: calls=%d prompt=%d output=%d total=%d",
            self.name,
            self.total_calls,
            self.total_prompt_tokens,
            self.total_output_tokens,
            self.total_tokens,
        )
        logger.info(
            "[%s] Trajectory: %d steps recorded at %s",
            self.name,
            len(self.trajectory.tail(self.MAX_ACTIONS + 1)),
            self.trajectory.path,
        )
        super().cleanup(*args, **kwargs)

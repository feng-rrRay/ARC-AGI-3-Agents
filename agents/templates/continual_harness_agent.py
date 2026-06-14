from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import asdict, replace
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
    format_tool_record_md,
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
    format_memory_overview,
)
from .continual_harness.models import (
    PendingActionObservation,
    RenderedGrid,
    StepRecord,
    ToolCallRecord,
    ToolEvidenceRecord,
)
from .continual_harness.harness_evolver import HarnessEvolver
from .continual_harness.prompts import (
    BASE_ORCHESTRATOR_POLICY,
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
from .utils.vlm_backend import VLM, estimate_vlm_usage_cost, usage_token_count

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


def _cost_log_suffix(cost: dict[str, Any] | None) -> str:
    if not cost:
        return ""
    return (
        f" cost=${float(cost['current_usd']):.6f}"
        f" cum_cost=${float(cost['cumulative_usd']):.6f}"
    )


def _tool_result_pairs(
    records: list[ToolCallRecord],
) -> list[tuple[str, dict[str, Any]]]:
    """Map tool-call records to (name, response) pairs for a function-response turn.

    Each non-action tool record becomes one Gemini function_response part whose
    ``output`` is the markdown rendering (JSON fields + fenced code/text), so the
    model reads code print-style rather than as a ``\\n``-escaped JSON string.
    """
    return [(r.name, {"output": format_tool_record_md(r)}) for r in records]


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
      2. Make ONE VLM call via `_vlm_loop_inner`. The response may contain
         analysis tool calls (process_memory, run_skill, etc.) and/or one
         `take_actions(actions=[...])` call. All are dispatched in emission
         order. take_actions executes its action list synchronously.
      3. If the action result advances a level, evolve immediately. Otherwise,
         evolve only when recent trajectory evidence shows stagnation.
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
    MAX_SUBAGENT_ROUNDS_CEILING = 50  # hard cap on inner VLM rounds (runaway protection)
    SUBAGENT_HISTORY_WINDOW = 20  # rows of compact history fed into a subagent's prompt
    RECENT_RESULTS_CAP = 16  # how many tool-result records to carry forward
    TOOL_EVIDENCE_CAP = 512  # non-action tool records retained for evolution windows
    MAX_CONVERSATION_TURNS = 24  # max tool-only VLM turns per decision before forcing a break
    CONVERSATION_CONTEXT_GUARD_RATIO = 0.80
    SKILL_TIMEOUT_S = 30.0  # wall-clock cap per run_skill (engine RPCs add latency)
    # Prompt/skill/subagent/memory evolution is owned by HarnessEvolver, which
    # reads CONTINUAL_HARNESS_PROMPT_EVOLVE_FREQUENCY (set by main.py from
    # --prompt-evolve-frequency; 0 disables all evolution) and holds the
    # stagnation thresholds.

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Must resolve model_name BEFORE super().__init__(): Agent.__init__
        # calls start_recording() which reads self.name → self.model_name.
        self.model_name = os.getenv("GEMINI_MODEL", self.MODEL)
        super().__init__(*args, **kwargs)

        # Three-layer prompt architecture:
        # 1. _system_instruction: fixed, never evolved (tool schemas, game context)
        # 2. base orchestrator policy: evolved strategic guidance — owned by the
        #    HarnessEvolver (read via get_current_prompt()), constructed below
        #    once the memory/skill/subagent/trajectory stores exist.
        # 3. Per-step user prompt: game state, history, tool results

        # Fixed system instruction — NEVER evolved. {game_name} substituted here.
        self._system_instruction: str = HARNESS_SYSTEM_INSTRUCTION.replace(
            "{game_name}", self.game_id
        )

        # Action at which score/level last advanced; updated inline on progress
        # and read by the evolver's stagnation trigger.
        self._last_progress_step: int = 0

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
                    entry = self.memory.add(
                        title=title,
                        body=body,
                        tags=tags,
                        confidence=args.get("confidence"),
                    )
                    return {
                        "success": True,
                        "operation": "add",
                        "id": entry.id,
                        "title": entry.title,
                        "tags": entry.tags,
                        "confidence": entry.confidence,
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
                        confidence=args.get("confidence"),
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
                        "confidence": edited.confidence,
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
            self._refresh_current_sandbox_stores()
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
            # Echo the executed source so the result renders a ```python section
            # (helps the model edit-then-rerun a failing skill without a search).
            out["code"] = skill.code
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
        # `_recent_tool_results` carries the previous conversation's analysis-tool
        # outputs into the next decision's prompt (older ones as a recap line,
        # the newest in full — see format_tool_results_markdown).
        # `_tool_evidence` keeps a longer window for harness evolution prompts.
        # `_pending_observations` carries action transition/result frames from
        # executed actions into the next successful orchestrator VLM prompt.
        # `_consecutive_vlm_errors` is bookkeeping for hard failure detection.
        # `_sandbox_*` state is set per-skill by _handle_run_skill and read by
        # `_sandbox_rpc` to gate take_actions RPCs.
        # `_batch_counter` mints unique batch IDs for tagging trajectory rows.
        # `_vlm_call_count` is a monotonic counter set on `_current_outer_round`.
        self._recent_tool_results: list[ToolCallRecord] = []
        self._tool_evidence: list[ToolEvidenceRecord] = []
        self._pending_observations: list[PendingActionObservation] = []
        self._consecutive_vlm_errors: int = 0
        self._sandbox_terminal_seen: bool = False
        self._sandbox_skill_id: str | None = None
        self._batch_counter: int = 0
        self._vlm_call_count: int = 0
        # `_conversation_count` increments once per decision (one `_vlm_loop_inner`
        # call), grouping the multi-turn conversation's per-VLM-call trace records.
        self._conversation_count: int = 0
        self._previous_no_action_reason: str | None = None

        def _handle_process_subagent(args: dict[str, Any]) -> dict[str, Any]:
            op = (args.get("operation") or "").strip().lower()
            try:
                if op == "add":
                    name = args.get("name") or ""
                    description = args.get("description") or ""
                    system_instructions = args.get("system_instructions") or ""
                    allowed_tools = _optional_str_list_arg(args, "allowed_tools")
                    tags = _optional_str_list_arg(args, "tags") or []
                    add_kwargs: dict[str, Any] = {
                        "directive": args.get("directive") or "",
                        "return_condition": args.get("return_condition") or "",
                    }
                    if args.get("handler_type"):
                        add_kwargs["handler_type"] = args["handler_type"]
                    if args.get("max_turns") is not None:
                        add_kwargs["max_turns"] = args["max_turns"]
                    entry = self.subagents.add(
                        name=name,
                        description=description,
                        system_instructions=system_instructions,
                        allowed_tools=allowed_tools,
                        tags=tags,
                        **add_kwargs,
                    )
                    return {
                        "success": True,
                        "operation": "add",
                        "id": entry.id,
                        "name": entry.name,
                        "handler_type": entry.handler_type,
                        "max_turns": entry.max_turns,
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
                        system_instructions=args.get("system_instructions"),
                        directive=args.get("directive"),
                        return_condition=args.get("return_condition"),
                        handler_type=args.get("handler_type"),
                        max_turns=args.get("max_turns"),
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

            directive = (str(args.get("task") or "").strip()) or entry.directive
            if not directive.strip():
                return {
                    "success": False,
                    "id": sub_id,
                    "error": (
                        "run_subagent requires a task, or the subagent must have a "
                        "stored directive"
                    ),
                }
            raw_context = args.get("context") or {}
            context = dict(raw_context) if isinstance(raw_context, dict) else {}

            # one_step runs a single analysis turn; looping runs a bounded action
            # loop, clamped to the runaway ceiling.
            inner_rounds = (
                1
                if entry.handler_type == "one_step"
                else max(1, min(entry.max_turns, self.MAX_SUBAGENT_ROUNDS_CEILING))
            )

            tools = build_subagent_tools(entry.allowed_tools)
            sub_vlm = VLM(
                self.model_name,
                backend="gemini",
                system_instruction=entry.system_instructions,
            )
            sub_vlm.set_tools(tools)

            history = render_recent_history(
                self.trajectory.tail(self.MAX_ACTIONS + 1),
                max_chars=self.HISTORY_MAX_CHARS,
            )
            base_prompt = build_subagent_prompt(
                directive=directive,
                return_condition=entry.return_condition,
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
            last_output: dict[str, Any] = {}
            completed_response = False

            for inner_round in range(1, inner_rounds + 1):
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
                    last_output = output
                    completed_response = True
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
                        inner_rounds,
                        exc,
                    )
                finally:
                    usage_cost = self._record_vlm_usage(usage)
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
                                "handler_type": entry.handler_type,
                                "inner_round": inner_round,
                                "max_inner_rounds": inner_rounds,
                            },
                            "input": {
                                "system_instruction": entry.system_instructions,
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
                            "usage_cost": usage_cost,
                            "chosen_action": None,
                            "reasoning": None,
                            "tool_calls": [asdict(r) for r in round_records],
                            "subagent_return": ret_call_args,
                            "error": round_error,
                        }
                    )
                    if usage is not None:
                        logger.info(
                            "[%s] subagent=%s inner=%d/%d tokens=%d actions=%d "
                            "tools=%d return=%s error=%s%s",
                            self.game_id,
                            entry.name,
                            inner_round,
                            inner_rounds,
                            usage_token_count(usage, "total"),
                            round_actions,
                            len(round_records),
                            ret_call_args is not None,
                            round_error,
                            _cost_log_suffix(usage_cost),
                        )

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
                    if entry.handler_type == "one_step":
                        break
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
                        directive=directive,
                        return_condition=entry.return_condition,
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
                and entry.handler_type == "one_step"
                and last_error is None
                and completed_response
            ):
                # one_step subagents are read-only analysers: they either call
                # subagent_return or simply produce a text analysis. Either way,
                # surface their output as a successful return rather than treating
                # the absence of subagent_return as a failure.
                answer_text = (
                    str(last_output.get("text") or "").strip()
                    or "(one_step subagent produced no analysis)"
                )
                final_answer = {
                    "answer": answer_text,
                    "status": "success",
                    "reasoning": "one_step auto-return",
                }
            elif (
                final_answer is None
                and last_error is None
                and inner_round >= inner_rounds
            ):
                warning = (
                    f"subagent exhausted max inner rounds "
                    f"({inner_rounds}) without "
                    f"subagent_return; forcing return to orchestrator"
                )
                logger.warning(
                    "Subagent %s exhausted max inner rounds (%d) without "
                    "subagent_return; forcing return to orchestrator",
                    entry.name,
                    inner_rounds,
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
        # All evolution state (base prompt file, evolution log, generation
        # counters) lives in the evolver; it shares the stores built above so
        # inline orchestrator edits and meta-level edits hit the same files.
        self.harness_evolver = HarnessEvolver(
            model_name=self.model_name,
            system_instruction=self._system_instruction,
            game_id=self.game_id,
            agent_name=self.name,
            memory=self.memory,
            skills=self.skills,
            subagents=self.subagents,
            trajectory=self.trajectory,
            trace=self.trace,
            record_usage=self._record_vlm_usage,
            baseline_prompt=BASE_ORCHESTRATOR_POLICY,
        )
        logger.info(
            "[%s] Harness evolution: stagnation_after=%d, base prompt at %s, log at %s",
            self.name,
            self.harness_evolver.stagnation_after,
            self.harness_evolver.base_prompt_path,
            self.harness_evolver.evolution_log_path,
        )

        # Cumulative token usage; logged once on cleanup().
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.total_vlm_cost_usd = 0.0
        self.total_priced_calls = 0

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

    def _refresh_current_sandbox_stores(self) -> None:
        """Refresh mutable stores while preserving the per-decision frame snapshot."""
        if self._current_sandbox_state is None:
            return
        self._current_sandbox_state = replace(
            self._current_sandbox_state,
            memory_entries=[asdict(e) for e in self.memory.all_entries()],
            skill_entries=[asdict(e) for e in self.skills.all_entries()],
        )

    def _conversation_context_guard_reason(
        self, contents: list[Any]
    ) -> tuple[str | None, dict[str, int] | None]:
        context_window = self.vlm.context_window_tokens()
        if context_window is None or context_window <= 0:
            return None, None
        guard_tokens = int(context_window * self.CONVERSATION_CONTEXT_GUARD_RATIO)
        input_tokens = self.vlm.count_input_tokens(contents, module_name=self.name)
        usage = {
            "prompt": input_tokens,
            "total": input_tokens,
            "context_window": context_window,
            "context_guard": guard_tokens,
        }
        if input_tokens < guard_tokens:
            return None, usage
        reason = (
            "context_window_guard: conversation input estimated at "
            f"{input_tokens} tokens, exceeding {self.CONVERSATION_CONTEXT_GUARD_RATIO:.0%} "
            f"of the model context window ({guard_tokens}/{context_window})"
        )
        return reason, usage

    def _record_vlm_usage(
        self, usage: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if usage is None:
            return None

        prompt_tokens = usage_token_count(usage, "prompt")
        output_tokens = usage_token_count(usage, "output")
        total_tokens = usage_token_count(usage, "total")

        self.total_calls += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += total_tokens

        cost = estimate_vlm_usage_cost(
            self.model_name,
            usage,
            cumulative_usd_before=self.total_vlm_cost_usd,
        )
        if cost is None:
            return None

        self.total_vlm_cost_usd = float(cost["cumulative_usd"])
        self.total_priced_calls += 1
        return cost

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
                self.harness_evolver.evolve_on_game_over(
                    latest_frame,
                    self.action_counter,
                    tool_evidence_records=list(self._tool_evidence),
                )
                self._execute_one(GameAction.RESET, source="auto_reset")
                self._recent_tool_results = []
                continue

            # Idle initial state: emit RESET and restart the iteration.
            if latest_frame.state is GameState.NOT_PLAYED:
                self._execute_one(GameAction.RESET, source="auto_reset")
                self._recent_tool_results = []
                continue

            pre_step_level = latest_frame.levels_completed

            # `_vlm_loop_inner` runs a full conversation until an action fires and
            # owns `_recent_tool_results` (carrying all of the conversation's tool
            # results into the next decision's prompt as recap + full tail), so
            # main() no longer clears it here.
            self._vlm_loop_inner(latest_frame)

            # Level-up evolution: consolidate discovered rules immediately
            # after advancing to a new level. Otherwise, evolve only when the
            # recent trajectory shows concrete stuck behavior. Progress tracking
            # stays on the agent (also set inline in dispatch on score gains).
            post_frame = self.frames[-1]
            if post_frame.levels_completed > pre_step_level:
                self._last_progress_step = self.action_counter
                self.harness_evolver.evolve_on_level_up(
                    post_frame,
                    self.action_counter,
                    tool_evidence_records=list(self._tool_evidence),
                )
            else:
                self.harness_evolver.maybe_evolve_on_stagnation(
                    post_frame,
                    self.action_counter,
                    self._last_progress_step,
                    tool_evidence_records=list(self._tool_evidence),
                )

        self.cleanup()

    # ------------------------------------------------------------------
    # Inner VLM dispatch — a conversation per decision, one trace record per turn.
    # ------------------------------------------------------------------

    def _vlm_loop_inner(self, latest_frame: FrameData) -> int:
        """Run one decision as a multi-turn conversation until an action fires.

        Turn 0 sends the freshly rendered working prompt (observations + current
        state + grid images). If the model emits only non-action tool calls, the
        model turn and the tool-result function-responses are appended to the SAME
        conversation and re-queried — the game state cannot change without an
        action, so the observation/images are never re-rendered mid-decision. The
        loop ends on the first executed action (state changed → must re-observe),
        a terminal state, an empty/dead response, or `MAX_CONVERSATION_TURNS`.

        Each VLM turn writes one orchestrator trace record sharing a
        `conversation_id`. Returns the total actions executed this decision.

        - `take_actions` calls execute synchronously via `_dispatch_take_actions`.
        - Analysis tool calls (process_*, run_skill, run_subagent, etc.) go
          through `self.tool_router`; a skill's inline take_actions count is
          carried via the `actions_taken_inline` field on the ToolCallRecord.
        - Stray legacy ACTION1..ACTION6 calls are rejected with an explicit
          error record so the model can self-correct next turn.
        """
        self._conversation_count += 1
        conversation_id = self._conversation_count

        # Stash per-decision state used by sub-handlers (run_subagent, skills).
        # Built once from `latest_frame`: no action runs mid-conversation, so the
        # frame — and this snapshot — stay valid for every turn of the decision.
        self._current_latest_frame = latest_frame
        self._current_images = list(frame_to_images(latest_frame))
        self._subagent_call_count = 0

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

        # Turn 0: render the working prompt + grid images once for this decision.
        # Skills still see every grid from the current latest_frame via
        # `self._current_images` -> `state.images`.
        prompt, rendered_grids = self._build_working_prompt(latest_frame)
        tools = self._full_tool_list()
        vlm_images = [grid_to_image(item.grid) for item in rendered_grids]
        contents: list[Any] = [self.vlm.build_user_turn(prompt, vlm_images)]

        actions_executed = 0
        all_new_results: list[ToolCallRecord] = []
        prev_results: list[ToolCallRecord] = []
        no_action_reason: str | None = None
        turn = 0
        while True:
            self._vlm_call_count += 1
            self._current_outer_round = self._vlm_call_count

            # Trace input fidelity: turn 0 carries the working prompt + its grids;
            # follow-up turns carry only the tool-result function-responses the
            # model is now answering (no new images attached).
            turn_prompt = prompt if turn == 0 else render_tool_results(prev_results)
            turn_grids = rendered_grids if turn == 0 else []

            guard_reason, guard_usage = self._conversation_context_guard_reason(contents)
            if guard_reason is not None:
                no_action_reason = guard_reason
                self._write_orchestrator_trace(
                    prompt=turn_prompt, output={}, usage=guard_usage, tools=tools,
                    tool_calls=[], actions_executed=0, rendered_grids=turn_grids,
                    error=guard_reason,
                    conversation_id=conversation_id, conversation_turn=turn,
                )
                logger.warning("[%s] %s", self.game_id, guard_reason)
                break

            try:
                response = self.vlm.get_query_contents(contents, module_name=self.name)
                output = serialize_response(response)
                usage = self.vlm.extract_usage(response)
            except Exception as exc:
                self._consecutive_vlm_errors += 1
                no_action_reason = f"vlm_error: {exc!r}"
                self._write_orchestrator_trace(
                    prompt=turn_prompt, output={}, usage=None, tools=tools,
                    tool_calls=[], actions_executed=0, rendered_grids=turn_grids,
                    error=repr(exc),
                    conversation_id=conversation_id, conversation_turn=turn,
                )
                logger.warning("VLM call failed: %s", exc)
                break
            self._consecutive_vlm_errors = 0

            # Observations shown in turn 0 have now been delivered. Clear before
            # dispatch so any actions emitted become the NEXT decision's prompt.
            if turn == 0:
                self._pending_observations = []

            action_counter_before = self.action_counter
            (
                turn_actions,
                new_results,
                round_tool_log,
                terminal,
                had_fcs,
            ) = self._dispatch_response(response)
            self._append_tool_evidence(
                new_results,
                conversation_id=conversation_id,
                conversation_turn=turn,
                action_counter_before=action_counter_before,
                action_counter_after=self.action_counter,
            )
            actions_executed += turn_actions
            all_new_results.extend(new_results)

            usage_cost = self._record_vlm_usage(usage)

            self._write_orchestrator_trace(
                prompt=turn_prompt, output=output, usage=usage, tools=tools,
                tool_calls=new_results, actions_executed=turn_actions,
                rendered_grids=turn_grids, error=None,
                conversation_id=conversation_id, conversation_turn=turn,
                usage_cost=usage_cost,
            )

            # Per-VLM-call run.log summary. One line per conversation turn.
            latest = self.frames[-1]
            tokens_total = usage_token_count(usage, "total")
            tools_label = ", ".join(round_tool_log) if round_tool_log else "(no fcs)"
            logger.info(
                "[%s] vlm#%d conv=%d.t%d step=%d lvl=%d state=%s tokens=%d "
                "actions=%d tools=[%s]%s",
                self.game_id,
                self._vlm_call_count,
                conversation_id,
                turn,
                self.action_counter,
                latest.levels_completed,
                latest.state.name,
                tokens_total,
                turn_actions,
                tools_label,
                _cost_log_suffix(usage_cost),
            )

            # End the conversation when the game state changed (action fired) or a
            # terminal state was reached — both require a fresh observation.
            if turn_actions > 0 or terminal:
                break
            # No function calls at all (e.g. safety-filtered/empty response): we
            # can't extend the conversation with a model turn, so end here.
            if not had_fcs:
                no_action_reason = "model response contained no function calls"
                break
            # Tool-only turn: extend the SAME conversation (cached prefix) and
            # re-query without re-rendering the unchanged observation/state.
            contents.append(self.vlm.model_turn(response))
            contents.append(
                self.vlm.build_tool_results_turn(_tool_result_pairs(new_results))
            )
            prev_results = new_results
            turn += 1
            if turn >= self.MAX_CONVERSATION_TURNS:
                no_action_reason = (
                    f"max_conversation_turns: hit MAX_CONVERSATION_TURNS="
                    f"{self.MAX_CONVERSATION_TURNS} without an action"
                )
                logger.warning("[%s] conv=%d %s", self.game_id, conversation_id, no_action_reason)
                break

        # Carry every non-action tool result from this conversation into the next
        # working prompt's TOOL RESULTS block: the newest render in full, older
        # ones as a one-line recap (see format_tool_results_markdown), so the
        # next decision does not re-run tools to re-derive outputs it already saw.
        self._recent_tool_results = all_new_results[-self.RECENT_RESULTS_CAP:]
        if actions_executed > 0 or self.frames[-1].state in (
            GameState.WIN,
            GameState.GAME_OVER,
        ):
            self._previous_no_action_reason = None
        else:
            self._previous_no_action_reason = (
                no_action_reason or "conversation ended without an action"
            )
        return actions_executed

    def _append_tool_evidence(
        self,
        records: list[ToolCallRecord],
        *,
        conversation_id: int,
        conversation_turn: int,
        action_counter_before: int,
        action_counter_after: int,
    ) -> None:
        for record in records:
            self._tool_evidence.append(
                ToolEvidenceRecord(
                    conversation_id=conversation_id,
                    conversation_turn=conversation_turn,
                    round=self._current_outer_round,
                    action_counter_before=action_counter_before,
                    action_counter_after=action_counter_after,
                    tool_call=record,
                )
            )
        overflow = len(self._tool_evidence) - self.TOOL_EVIDENCE_CAP
        if overflow > 0:
            del self._tool_evidence[:overflow]

    def _dispatch_response(
        self, response: Any
    ) -> tuple[int, list[ToolCallRecord], list[str], bool, bool]:
        """Dispatch every function call in one VLM response.

        Returns ``(actions_executed, new_results, round_tool_log, terminal,
        had_fcs)``. ``new_results`` holds the non-action / rejected tool records
        for this turn. Successful take_actions calls are not recorded — the next
        prompt's observation shows what ran — but zero-action take_actions calls
        are returned so Gemini receives a response for the pending function call.
        ``terminal`` is True if a WIN/GAME_OVER was reached mid-response.
        ``had_fcs`` is False only when the response carried no function calls at
        all.
        """
        fcs = extract_function_calls(response)
        actions_executed = 0
        new_results: list[ToolCallRecord] = []
        # Track the human-readable name of every tool call this turn, in emission
        # order, for the run.log summary. take_actions entries include the emitted
        # action list inline so a `grep take_actions` against run.log shows what
        # was committed.
        round_tool_log: list[str] = []
        terminal = False

        for fc in fcs:
            if is_take_actions_call(fc.name):
                action_specs = (
                    fc.args.get("actions") if isinstance(fc.args, dict) else None
                )
                spec_label = _format_action_list(action_specs)
                executed = self._dispatch_take_actions(fc.args, source="vlm")
                actions_executed += executed
                round_tool_log.append(f"take_actions{spec_label}={executed}")
                if executed == 0:
                    new_results.append(
                        ToolCallRecord(
                            name=fc.name,
                            args=fc.args,
                            result={
                                "success": False,
                                "executed": 0,
                                "available_actions": [
                                    a.name
                                    for a in available_game_actions(
                                        self.frames[-1].available_actions
                                    )
                                ],
                                "message": (
                                    "take_actions executed no actions; send a "
                                    "non-empty valid actions list using the "
                                    "currently available actions"
                                ),
                            },
                        )
                    )
                # Successful take_actions calls do not need a TOOL RESULTS entry:
                # the conversation stops and the next prompt shows what ran.
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
                terminal = True
                break

        return actions_executed, new_results, round_tool_log, terminal, bool(fcs)

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
        if frame is not None and (
            score_after > pre_score or post.state is GameState.WIN
        ):
            self._last_progress_step = self.action_counter

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
            base_prompt=self.harness_evolver.get_current_prompt(),
            previous_no_action_reason=self._previous_no_action_reason,
            max_deliberation_turns=self.MAX_CONVERSATION_TURNS,
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
        conversation_id: int | None = None,
        conversation_turn: int | None = None,
        usage_cost: dict[str, Any] | None = None,
    ) -> None:
        self.trace.write(
            {
                "agent": self.name,
                "model": self.model_name,
                "game_id": self.game_id,
                "action_counter": self.action_counter,
                "vlm_call": self._vlm_call_count,
                "round": self._current_outer_round,
                # Group a decision's multi-turn conversation: all turns share
                # `conversation_id`; `conversation_turn` is the 0-based turn index.
                "conversation_id": conversation_id,
                "conversation_turn": conversation_turn,
                "tools_exposed": "full",
                "input": {
                    "system_instruction": self._system_instruction,
                    "base_prompt": self.harness_evolver.get_current_prompt(),
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
                "usage_cost": usage_cost,
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
        if self.total_priced_calls:
            logger.info(
                "[%s] Final priced VLM usage: priced_calls=%d cost=$%.6f",
                self.name,
                self.total_priced_calls,
                self.total_vlm_cost_usd,
            )
        logger.info(
            "[%s] Trajectory: %d steps recorded at %s",
            self.name,
            len(self.trajectory.tail(self.MAX_ACTIONS + 1)),
            self.trajectory.path,
        )
        super().cleanup(*args, **kwargs)

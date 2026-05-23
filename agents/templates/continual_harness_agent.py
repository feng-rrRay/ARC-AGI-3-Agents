from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import asdict
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from ..tracing import trace_agent_session
from .continual_harness.context import build_subagent_prompt, build_working_prompt
from .continual_harness.helpers import (
    available_game_actions,
    frame_to_images,
    validate_action_sequence,
)
from .continual_harness.memory import (
    MemoryStore,
    active_memory_path,
    format_memory_overview,
)
from .continual_harness.models import StepRecord, ToolCallRecord
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
    EVOLVE_SYSTEM_PROMPT_TOOL,
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
    is_evolve_prompt_call,
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
    format_compact_history,
    format_full_history,
)
from .utils.vlm_backend import VLM

logger = logging.getLogger(__name__)


def _action_data_dict(action: GameAction) -> dict[str, Any]:
    # GameAction.action_data is a pydantic model; fall back to {} if it's absent.
    # `game_id` is structural metadata (always present, always empty here) — strip it
    # so per-step history rows show only meaningful args (e.g. ACTION6's x/y).
    dump = getattr(getattr(action, "action_data", None), "model_dump", None)
    if not callable(dump):
        return {}
    return {k: v for k, v in dump().items() if k != "game_id"}


def _sample_keyframes(items: list[Any], k: int) -> list[Any]:
    """Pick up to k items from `items`, biased toward keyframes.

    For animation sequences shorter than or equal to k, returns everything
    unchanged. For longer sequences, returns the first item, the last item,
    and k-2 evenly-spaced intermediates. Duplicate indices are collapsed,
    so the result may be shorter than k if k > len(items).
    """
    n = len(items)
    if n <= k:
        return list(items)
    if k <= 1:
        return [items[-1]]
    seen: set[int] = set()
    indices: list[int] = []
    for i in range(k):
        idx = round(i * (n - 1) / (k - 1))
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    indices.sort()
    return [items[i] for i in indices]


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

    Falls back to a hand-built dict if model_dump fails (defensive — same
    pattern used when building the sandbox state at the top of an iter).
    """
    try:
        return frame.model_dump(mode="json")
    except (AttributeError, TypeError):
        return {
            "state": frame.state.name,
            "score": frame.levels_completed,
            "frame": frame.frame,
            "available_actions": list(frame.available_actions),
        }


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
      1. If state is NOT_PLAYED or GAME_OVER, emit RESET and continue.
      2. Maybe evolve the system prompt (boundary-gated by action_counter).
      3. Make ONE VLM call via `_vlm_loop_inner`. The response may contain
         analysis tool calls (process_memory, run_skill, etc.) and/or one
         `take_actions(actions=[...])` call. All are dispatched in emission
         order. take_actions executes its action list synchronously.
      4. If any actions were executed (orchestrator OR via a skill's inline
         tools["take_actions"] RPC), clear the carried tool-result block.
         Otherwise carry results forward to the next iteration's prompt
         (multi-round thinking across iterations).
      5. If too many consecutive iterations passed without action, force-mode
         the next iter: only take_actions exposed, with a BACKSTOP block.

    `choose_action` is unused (stubbed); the abstract method is satisfied but
    the engine is driven from `main()`.
    """

    MAX_ACTIONS = 1000
    MODEL = "gemini-3.1-pro-preview"  # default; override via GEMINI_MODEL
    HISTORY_MAX_CHARS = 12000  # budget for the RECENT HISTORY block
    HISTORY_BATCH_WINDOW = 5  # last N batches shown in compact history
    FULL_HISTORY_DEFAULT_LIMIT = 40
    FULL_HISTORY_MAX_LIMIT = 80
    MAX_SUBAGENT_CALLS_PER_STEP = 1  # distinct run_subagent invocations per outer step
    MAX_SUBAGENT_ROUNDS_PER_CALL = 20  # inner VLM rounds per invocation
    SUBAGENT_HISTORY_WINDOW = 20  # rows of compact history fed into a subagent's prompt
    # No-action-iteration backstop: after this many consecutive outer iters with
    # zero actions executed, the next iter is force-mode (only take_actions
    # exposed + BACKSTOP block in prompt). Force-mode is a soft nudge — the
    # loop keeps going even if force still produces no action.
    MAX_CONSECUTIVE_NO_ACTION_ITERS = 5
    RECENT_RESULTS_CAP = 16  # how many tool-result records to carry forward
    SKILL_TIMEOUT_S = 30.0  # wall-clock cap per run_skill (engine RPCs add latency)
    # Max number of grid-image attachments per orchestrator VLM call. Frames
    # with more grids than this (e.g., long animation sequences or the 118-grid
    # level-transition case) get keyframe-sampled: first + last + evenly-spaced
    # middles. Skills are unaffected — they still see every grid via
    # `state.images` (capped by sandbox.MAX_IMAGES=16 independently).
    MAX_VLM_PAYLOAD_IMAGES = 8
    # Prompt-evolution defaults; the actual frequency is read from
    # CONTINUAL_HARNESS_PROMPT_EVOLVE_FREQUENCY (set by main.py from
    # --prompt-evolve-frequency). 0 disables; positive N means every N actions.
    DEFAULT_PROMPT_EVOLVE_FREQUENCY = 25
    PROMPT_EVOLVE_FREQUENCY_ENV = "CONTINUAL_HARNESS_PROMPT_EVOLVE_FREQUENCY"
    EVOLUTION_TRAJECTORY_WINDOW = 25  # how many recent steps the meta-call sees

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Must resolve model_name BEFORE super().__init__(): Agent.__init__
        # calls start_recording() which reads self.name → self.model_name.
        self.model_name = os.getenv("GEMINI_MODEL", self.MODEL)
        super().__init__(*args, **kwargs)

        # Prompt-evolution wiring: the active system instruction lives in a
        # single .md file (run-local by default; --bootstrap-prompt opts into
        # cross-run persistence). On first use the baseline is seeded into the
        # file; every successful evolution rewrites it.
        freq_raw = os.getenv(
            self.PROMPT_EVOLVE_FREQUENCY_ENV,
            str(self.DEFAULT_PROMPT_EVOLVE_FREQUENCY),
        )
        try:
            self._prompt_evolve_frequency = max(0, int(freq_raw))
        except ValueError:
            self._prompt_evolve_frequency = self.DEFAULT_PROMPT_EVOLVE_FREQUENCY
        self._prompt_file = PromptFile(
            active_prompt_path(), baseline=HARNESS_SYSTEM_INSTRUCTION
        )
        self._current_system_instruction: str = self._prompt_file.read()
        self._prompt_generation: int = 0
        self._last_evolution_step: int = -1
        self.prompt_evolution = PromptEvolutionStore(active_prompt_evolution_path())

        self.vlm = VLM(
            self.model_name,
            backend="gemini",
            system_instruction=self._current_system_instruction,
        )
        recorder = getattr(self, "recorder", None)
        guid = getattr(recorder, "guid", None)
        # JSONL artifacts share the recording stem when run-dir wiring is active.
        self.trace = TraceWriter(default_trace_path(prefix=self.name, guid=guid))
        self.trajectory = TrajectoryStore(
            default_trajectory_path(prefix=self.name, guid=guid)
        )

        def _handle_get_recent_trajectory(args: dict[str, Any]) -> dict[str, Any]:
            try:
                limit = int(args.get("limit", self.FULL_HISTORY_DEFAULT_LIMIT))
            except (TypeError, ValueError):
                limit = self.FULL_HISTORY_DEFAULT_LIMIT
            limit = max(1, min(self.FULL_HISTORY_MAX_LIMIT, limit))
            records = self.trajectory.tail(limit)
            return {
                "success": True,
                "limit": limit,
                "count": len(records),
                "history": format_full_history(records),
            }

        # Memory is always available. --bootstrap-memory selects a cross-run
        # backing file; otherwise active_memory_path() falls back to run-local
        # storage under logs/<run_id>/memory.json.
        self.memory = MemoryStore(active_memory_path(), game_id=self.game_id)

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

        # Skills are always available too. --bootstrap-skills selects a cross-run
        # backing file; otherwise active_skill_path() falls back to run-local
        # storage under logs/<run_id>/skills.json.
        self.skills = SkillStore(active_skill_path(), game_id=self.game_id)

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
                    return {
                        "success": True,
                        "operation": "add",
                        "id": entry.id,
                        "name": entry.name,
                        "tags": entry.tags,
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

        # Subagents are always available too. --bootstrap-subagents selects a
        # cross-run backing file; otherwise active_subagent_path() falls back
        # to run-local storage under logs/<run_id>/subagents.json.
        self.subagents = SubagentStore(active_subagent_path(), game_id=self.game_id)

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
        # `_consecutive_no_action_iters` drives the backstop force-mode.
        # `_consecutive_vlm_errors` is bookkeeping for hard failure detection.
        # `_sandbox_*` state is set per-skill by _handle_run_skill and read by
        # `_sandbox_rpc` to gate take_actions RPCs.
        # `_batch_counter` mints unique batch IDs for tagging trajectory rows.
        # `_vlm_call_count` is a monotonic counter set on `_current_outer_round`.
        self._recent_tool_results: list[ToolCallRecord] = []
        self._consecutive_no_action_iters: int = 0
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

            history = format_compact_history(
                self.trajectory.tail(self.SUBAGENT_HISTORY_WINDOW),
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

            for inner_round in range(1, self.MAX_SUBAGENT_ROUNDS_PER_CALL + 1):
                round_records: list[ToolCallRecord] = []
                output: dict[str, Any] = {}
                usage: dict[str, int | None] | None = None
                round_error: str | None = None
                ret_call_args: dict[str, Any] | None = None

                try:
                    response = sub_vlm.get_query(
                        payload,
                        working_prompt,
                        module_name=f"{self.name}.subagent.{entry.name}",
                    )
                    output = serialize_response(response)
                    usage = sub_vlm.extract_usage(response)
                    fcs = extract_function_calls(response)

                    ret_call = next(
                        (fc for fc in fcs if is_subagent_return_call(fc.name)),
                        None,
                    )
                    if ret_call is not None:
                        ret_call_args = ret_call.args
                        final_answer = {
                            "answer": str(ret_call.args.get("answer") or ""),
                            "status": str(ret_call.args.get("status") or "success"),
                            "reasoning": str(ret_call.args.get("reasoning") or ""),
                        }
                    else:
                        for fc in fcs:
                            if fc.name not in entry.allowed_tools:
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

                round_step = {
                    "inner_round": inner_round,
                    "tool_calls": [asdict(r) for r in round_records],
                    "subagent_return": ret_call_args,
                    "error": round_error,
                }
                if usage is not None:
                    round_step["usage"] = usage
                subagent_steps.append(round_step)
                tool_calls_log.extend(round_records)

                if final_answer is not None:
                    break
                if round_error is not None:
                    # Network/VLM error — bail (no retry inside the subagent loop;
                    # the orchestrator can decide whether to invoke again next round).
                    break
                if not round_records:
                    last_error = "subagent produced no tool calls and did not return"
                    break
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
            }

        handlers: dict[str, Any] = {
            "get_recent_trajectory": _handle_get_recent_trajectory,
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
            self._prompt_file.path,
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

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def _evolve_system_prompt(self, latest_frame: FrameData) -> None:
        """One meta-VLM call that may rewrite the agent's system instruction.

        Spawns a fresh VLM with EVOLUTION_SYSTEM_INSTRUCTION + a single tool
        (evolve_system_prompt). Single round, no retry. The proposal is
        accepted only if `validate_evolved_prompt` returns ok=True; on accept,
        the new text is written through `self._prompt_file` and `self.vlm`
        picks it up on the next get_query call. Every attempt (accepted or
        rejected) is appended to `self.prompt_evolution`.
        """
        self._prompt_generation += 1
        gen = self._prompt_generation
        previous = self._current_system_instruction

        trajectory_rows = self.trajectory.tail(self.EVOLUTION_TRAJECTORY_WINDOW)
        user_prompt = build_evolution_prompt(
            current_prompt=previous,
            latest_frame=latest_frame,
            generation=gen,
            action_counter=self.action_counter,
            trajectory_rows=trajectory_rows,
            memory_overview=format_memory_overview(self.memory.all_entries()),
            skill_overview=format_skill_overview(self.skills.all_entries()),
            subagent_overview=format_subagent_overview(self.subagents.all_entries()),
        )

        meta_vlm = VLM(
            self.model_name,
            backend="gemini",
            system_instruction=EVOLUTION_SYSTEM_INSTRUCTION,
        )
        meta_vlm.set_tools([EVOLVE_SYSTEM_PROMPT_TOOL])

        # This hook runs before choose_action refreshes `_current_images`, so
        # render from latest_frame directly to avoid using a stale previous-step
        # image payload.
        images = list(frame_to_images(latest_frame))
        payload: Any = images if len(images) > 1 else (images[0] if images else None)

        proposed = ""
        reasoning = ""
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
            fcs = extract_function_calls(response)
            call = next((fc for fc in fcs if is_evolve_prompt_call(fc.name)), None)
            if call is None:
                validation_error = "model did not call evolve_system_prompt"
            else:
                reasoning = str(call.args.get("reasoning") or "")
                proposed = str(call.args.get("new_prompt") or "")
                ok, err = validate_evolved_prompt(proposed)
                accepted = ok
                validation_error = err
                if accepted:
                    self._current_system_instruction = proposed
                    self.vlm.set_system_instruction(proposed)
                    self._prompt_file.write(proposed)
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
                reasoning=reasoning,
                proposed_prompt=proposed,
                previous_prompt=previous,
                new_prompt=self._current_system_instruction,
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
                        "new_len": len(self._current_system_instruction),
                    },
                    "input": {
                        "system_instruction": EVOLUTION_SYSTEM_INSTRUCTION,
                        "user_prompt": user_prompt,
                        "tools": [EVOLVE_SYSTEM_PROMPT_TOOL],
                        "images": [
                            {"width": img.width, "height": img.height, "mode": img.mode}
                            for img in images
                        ],
                    },
                    "output": output,
                    "usage": usage,
                    "chosen_action": None,
                    "reasoning": reasoning,
                    "tool_calls": [],
                    "error": error,
                }
            )

            logger.info(
                "[%s] Prompt evolution gen=%d accepted=%s len=%d (prev=%d) error=%s",
                self.name,
                gen,
                accepted,
                len(self._current_system_instruction),
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

            # Terminal / idle states: emit RESET and restart the iteration.
            if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                self._execute_one(GameAction.RESET, source="auto_reset")
                self._recent_tool_results = []
                self._consecutive_no_action_iters = 0
                continue

            # Prompt-evolution hook — boundary-gated by action_counter rather
            # than modulo so multi-action batches that straddle a boundary
            # still fire exactly once.
            self._maybe_evolve_prompt(latest_frame)

            force = (
                self._consecutive_no_action_iters
                >= self.MAX_CONSECUTIVE_NO_ACTION_ITERS
            )

            actions_executed = self._vlm_loop_inner(latest_frame, force=force)

            if actions_executed > 0:
                self._consecutive_no_action_iters = 0
                self._recent_tool_results = []
            else:
                self._consecutive_no_action_iters += 1

        self.cleanup()

    # ------------------------------------------------------------------
    # Inner VLM dispatch — one VLM call per outer iteration.
    # ------------------------------------------------------------------

    def _vlm_loop_inner(self, latest_frame: FrameData, *, force: bool) -> int:
        """Make one VLM call, dispatch every function call, return # actions executed.

        - `take_actions` calls execute their action list synchronously via
          `_dispatch_take_actions`.
        - Analysis tool calls (process_*, run_skill, run_subagent, etc.) go
          through `self.tool_router`. A skill that called tools["take_actions"]
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
        self._current_sandbox_state = SandboxState(
            latest_frame=frame_dump,
            recent_trajectory=self.trajectory.tail(self.FULL_HISTORY_MAX_LIMIT),
            memory_entries=[asdict(e) for e in self.memory.all_entries()],
            skill_entries=[asdict(e) for e in self.skills.all_entries()],
        )

        prompt = self._build_working_prompt(latest_frame, force_take_actions=force)
        tools = [TAKE_ACTIONS_TOOL] if force else self._full_tool_list()
        self.vlm.set_tools(tools)

        # Keyframe-sample the image list before sending to the VLM. Most frames
        # carry 1 grid (sample is a no-op). Long animation sequences (and the
        # pathological 118-grid level-transition case) get reduced to at most
        # MAX_VLM_PAYLOAD_IMAGES = first + last + evenly-spaced middles —
        # enough to read the animation's success/failure signal without
        # multipart-payload bloat. Skills still see every grid via
        # `self._current_images` → `state.images` in the sandbox.
        vlm_images = _sample_keyframes(
            self._current_images, self.MAX_VLM_PAYLOAD_IMAGES
        )
        payload: Any = (
            vlm_images
            if len(vlm_images) > 1
            else (vlm_images[0] if vlm_images else None)
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
                prompt=prompt, output={}, usage=None, tools=tools, force=force,
                tool_calls=[], actions_executed=0, error=repr(exc),
            )
            logger.warning("VLM call failed: %s", exc)
            return 0
        self._consecutive_vlm_errors = 0

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
            prompt=prompt, output=output, usage=usage, tools=tools, force=force,
            tool_calls=new_results, actions_executed=actions_executed, error=error,
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
            "[%s] vlm#%d step=%d lvl=%d state=%s tokens=%d actions=%d tools=[%s]%s",
            self.game_id,
            self._vlm_call_count,
            self.action_counter,
            latest.levels_completed,
            latest.state.name,
            tokens_total,
            actions_executed,
            tools_label,
            " (FORCE)" if force else "",
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
        pre = self.frames[-1]
        pre_state, pre_score = pre.state.name, pre.levels_completed

        frame = self.take_action(action)        # base class: arc_env.step + validate
        if frame is not None:
            self.append_frame(frame)            # base class: also calls recorder.record
        self.action_counter += 1                # mirror base Agent.main() semantics

        post = frame if frame is not None else pre
        score_after = post.levels_completed
        state_after = post.state.name if frame is not None else "INVALID"

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
    # Sandbox RPC dispatcher — invoked from a skill's tools["take_actions"].
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
        return {
            "ok": executed > 0,
            "value": {
                "executed_count": executed,
                "last_frame": _safe_frame_dump(last),
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
        self, latest_frame: FrameData, *, force_take_actions: bool
    ) -> str:
        history_block = format_compact_history(
            self.trajectory.tail(self.FULL_HISTORY_MAX_LIMIT),
            frames=self.frames,
            max_chars=self.HISTORY_MAX_CHARS,
            max_batches=self.HISTORY_BATCH_WINDOW,
        )
        memory_overview = format_memory_overview(self.memory.all_entries())
        skill_overview = format_skill_overview(self.skills.all_entries())
        subagent_overview = format_subagent_overview(self.subagents.all_entries())
        return build_working_prompt(
            latest_frame,
            action_counter=self.action_counter,
            recent_tool_results=self._recent_tool_results,
            history_block=history_block,
            memory_overview=memory_overview,
            skill_overview=skill_overview,
            subagent_overview=subagent_overview,
            force_take_actions=force_take_actions,
            no_action_iters=self._consecutive_no_action_iters,
        )

    # ------------------------------------------------------------------
    # Tool list, prompt-evolution gate, batch IDs, trace writer.
    # ------------------------------------------------------------------

    def _full_tool_list(self) -> list[dict[str, Any]]:
        """Unified action tool + analysis tools. Used in normal (non-force) mode."""
        tools: list[dict[str, Any]] = [TAKE_ACTIONS_TOOL]
        tools.extend(build_analysis_tools())  # get_recent_trajectory
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
        force: bool,
        tool_calls: list[ToolCallRecord],
        actions_executed: int,
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
                "tools_exposed": "take_actions_only" if force else "full",
                "force_take_actions": force,
                "consecutive_no_action_iters": self._consecutive_no_action_iters,
                "input": {
                    "system_instruction": self._current_system_instruction,
                    "user_prompt": prompt,
                    "tools": tools,
                    # `images` reflects every grid rendered for the frame
                    # (post-skill-cap), so traces show the full set the
                    # orchestrator had access to. `images_attached_count` is
                    # how many were actually sent in this VLM call after
                    # keyframe-sampling at MAX_VLM_PAYLOAD_IMAGES.
                    "images": [
                        {"width": img.width, "height": img.height, "mode": img.mode}
                        for img in self._current_images
                    ],
                    "images_attached_count": len(
                        _sample_keyframes(
                            self._current_images, self.MAX_VLM_PAYLOAD_IMAGES
                        )
                    ),
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

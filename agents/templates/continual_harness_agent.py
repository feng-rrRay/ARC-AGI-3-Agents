from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from .continual_harness.context import build_action_prompt, build_subagent_prompt
from .continual_harness.helpers import (
    _action_from_name,
    available_game_actions,
    build_action_tools,
    frame_to_images,
)
from .continual_harness.memory import (
    MemoryStore,
    active_memory_path,
    format_memory_overview,
)
from .continual_harness.models import StepRecord, ToolCallRecord
from .continual_harness.prompts import SYSTEM_INSTRUCTION
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
    RUN_CODE_TOOL,
    RUN_SKILL_TOOL,
    RUN_SUBAGENT_TOOL,
    ContinualToolRouter,
    build_analysis_tools,
    build_subagent_tools,
    extract_function_calls,
    is_action_tool,
    is_subagent_return_call,
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
    """Single-game VLM agent with a bounded multi-round tool router.

    Each step the VLM may run up to MAX_TOOL_ROUNDS rounds; in each round it
    can call any subset of analysis tools (currently get_recent_trajectory) or
    commit to an ARC action. A per-step analysis-call budget and a forced-action
    final round guarantee that every step ends with a GameAction or RuntimeError.
    """

    MAX_ACTIONS = 80
    MODEL = "gemini-3.1-pro-preview"  # default; override via GEMINI_MODEL
    MAX_TOOL_ROUNDS = 3  # VLM round-trips per step
    MAX_ANALYSIS_CALLS_PER_STEP = 5  # total analysis tool calls per step
    HISTORY_WINDOW = 5  # max trajectory rows fed into the auto-injected tail
    HISTORY_MAX_CHARS = 12000  # ~4k tokens budget for the RECENT STEPS block
    HISTORY_REASONING_CHARS = 300  # per-row reasoning truncation cap
    FULL_HISTORY_DEFAULT_LIMIT = 40
    FULL_HISTORY_MAX_LIMIT = 80  # = MAX_ACTIONS, so the model can request the whole run
    MAX_SUBAGENT_CALLS_PER_STEP = 1  # distinct run_subagent invocations per outer step
    MAX_SUBAGENT_ROUNDS_PER_CALL = 20  # inner VLM rounds per invocation
    SUBAGENT_HISTORY_WINDOW = 10  # rows of compact history fed into a subagent's prompt

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Must resolve model_name BEFORE super().__init__(): Agent.__init__
        # calls start_recording() which reads self.name → self.model_name.
        self.model_name = os.getenv("GEMINI_MODEL", self.MODEL)
        super().__init__(*args, **kwargs)
        self.vlm = VLM(
            self.model_name,
            backend="gemini",
            system_instruction=SYSTEM_INSTRUCTION,
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
        # handlers (process_skill/run_skill/run_code) within that step share one
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
                    skill_id = args.get("id") or ""
                    if not skill_id:
                        return {
                            "success": False,
                            "operation": "delete",
                            "error": "delete requires an id",
                        }
                    deleted = self.skills.delete(skill_id)
                    result: dict[str, Any] = {
                        "success": deleted,
                        "operation": "delete",
                        "id": skill_id,
                        "deleted": deleted,
                    }
                    if not deleted:
                        result["error"] = f"no skill with id={skill_id}"
                    return result
                if op == "edit":
                    skill_id = args.get("id") or ""
                    if not skill_id:
                        return {
                            "success": False,
                            "operation": "edit",
                            "error": "edit requires an id",
                        }
                    edited = self.skills.edit(
                        skill_id,
                        name=args.get("name"),
                        description=args.get("description"),
                        code=args.get("code"),
                        tags=list(args["tags"]) if "tags" in args else None,
                    )
                    if edited is None:
                        return {
                            "success": False,
                            "operation": "edit",
                            "id": skill_id,
                            "error": f"no skill with id={skill_id}",
                        }
                    return {
                        "success": True,
                        "operation": "edit",
                        "id": skill_id,
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
            skill = self.skills.get(skill_id)
            if skill is None:
                return {
                    "success": False,
                    "id": skill_id,
                    "error": f"no skill with id={skill_id}",
                }
            state = self._current_sandbox_state or SandboxState()
            out = run_python_snippet(
                skill.code,
                state=state,
                args=dict(args.get("args") or {}),
            )
            out["id"] = skill_id
            out["name"] = skill.name
            out["version"] = skill.version
            return out

        def _handle_run_code(args: dict[str, Any]) -> dict[str, Any]:
            code = args.get("code") or ""
            state = self._current_sandbox_state or SandboxState()
            return run_python_snippet(
                code,
                state=state,
                args=dict(args.get("args") or {}),
            )

        # Subagents are always available too. --bootstrap-subagents selects a
        # cross-run backing file; otherwise active_subagent_path() falls back
        # to run-local storage under logs/<run_id>/subagents.json.
        self.subagents = SubagentStore(active_subagent_path(), game_id=self.game_id)

        # Per-step state used by _handle_run_subagent. Set at the top of every
        # choose_action() so each subagent invocation in that step sees the same
        # frame, images, outer-round index, and shares a single per-step call
        # counter.
        self._current_latest_frame: FrameData | None = None
        self._current_images: list[Any] = []
        self._current_outer_round: int = 0
        self._subagent_call_count: int = 0

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
                reasoning_chars=self.HISTORY_REASONING_CHARS,
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
            "run_code": _handle_run_code,
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

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # Base loop calls choose_action even on terminal states; reset first.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        available = available_game_actions(latest_frame.available_actions)
        action_tools = build_action_tools(available)
        analysis_tools = list(build_analysis_tools())
        analysis_tools.append(PROCESS_MEMORY_TOOL)
        analysis_tools.extend([PROCESS_SKILL_TOOL, RUN_SKILL_TOOL, RUN_CODE_TOOL])
        analysis_tools.extend([PROCESS_SUBAGENT_TOOL, RUN_SUBAGENT_TOOL])
        full_tools = action_tools + analysis_tools
        self.vlm.set_tools(full_tools)
        current_tools = full_tools

        images = frame_to_images(latest_frame)
        payload: Any = images if len(images) > 1 else images[0]

        # Stash per-step state for _handle_run_subagent (latest_frame + images
        # + outer-round index) and reset the per-step invocation counter.
        self._current_latest_frame = latest_frame
        self._current_images = list(images)
        self._subagent_call_count = 0

        # Build a JSON-only sandbox snapshot once per step. All sandbox tool
        # invocations within this step share the same view of the world.
        # `mode="json"` ensures enum fields (GameAction, GameState) serialize
        # to plain strings/ints — without it, the sandbox payload would carry
        # enum objects that json.dumps refuses.
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

        history = format_compact_history(
            self.trajectory.tail(self.HISTORY_WINDOW),
            frames=frames,
            max_chars=self.HISTORY_MAX_CHARS,
            reasoning_chars=self.HISTORY_REASONING_CHARS,
        )
        history_block = (
            "## RECENT STEPS (compact; call get_recent_trajectory for full detail)\n"
            + history
        )
        # Tool results accumulate across rounds; each round rebuilds the prompt so the
        # `# TURN:` line stays the LAST thing the model reads. Memory overview is
        # re-read each round so add/edit/delete made by a previous round show up.
        tool_results_blocks: list[str] = []

        def _build_working_prompt() -> str:
            extras: list[str] = []
            extras.append(format_memory_overview(self.memory.all_entries()))
            extras.append(format_skill_overview(self.skills.all_entries()))
            extras.append(format_subagent_overview(self.subagents.all_entries()))
            extras.append(history_block)
            extras.extend(tool_results_blocks)
            return build_action_prompt(latest_frame, extra_context="\n\n".join(extras))

        working_prompt = _build_working_prompt()

        tool_calls_this_step: list[ToolCallRecord] = []
        analysis_budget = self.MAX_ANALYSIS_CALLS_PER_STEP
        last_exc: Exception | None = None
        chosen: GameAction | None = None
        round_idx = 0

        for round_idx in range(1, self.MAX_TOOL_ROUNDS + 1):
            # Invariant: every step must end with an action call. Strip analysis
            # tools when (a) the per-step budget is exhausted or (b) this is the
            # final round (force-action — guarantees the last VLM call commits).
            self._current_outer_round = round_idx
            is_final_round = round_idx == self.MAX_TOOL_ROUNDS
            expose_analysis = analysis_budget > 0 and not is_final_round
            desired_tools = full_tools if expose_analysis else action_tools
            if desired_tools is not current_tools:
                self.vlm.set_tools(desired_tools)
                current_tools = desired_tools

            output: dict[str, Any] = {}
            usage: dict[str, int | None] | None = None
            round_chosen: GameAction | None = None
            round_error: str | None = None
            round_tool_records: list[ToolCallRecord] = []

            try:
                response = self.vlm.get_query(
                    payload, working_prompt, module_name=self.name
                )
                output = serialize_response(response)
                usage = self.vlm.extract_usage(response)
                fcs = extract_function_calls(response)

                # Priority: any usable action tool call wins immediately.
                for fc in fcs:
                    if is_action_tool(fc.name):
                        round_chosen = _action_from_name(fc.name, fc.args, available)
                        if round_chosen is not None:
                            break

                # No usable action → run analysis tool calls (in order), up to budget.
                if round_chosen is None:
                    for fc in fcs:
                        if is_action_tool(fc.name):
                            continue
                        if analysis_budget <= 0:
                            round_tool_records.append(
                                ToolCallRecord(
                                    name=fc.name,
                                    args=fc.args,
                                    error="analysis budget exhausted; commit to an action next round",
                                )
                            )
                            continue
                        round_tool_records.append(self.tool_router.execute(fc))
                        analysis_budget -= 1
            except Exception as exc:
                last_exc = exc
                round_error = repr(exc)
                logger.warning(
                    "Round %d/%d: VLM call failed: %s",
                    round_idx,
                    self.MAX_TOOL_ROUNDS,
                    exc,
                )
            finally:
                self.trace.write(
                    {
                        "agent": self.name,
                        "model": self.model_name,
                        "game_id": self.game_id,
                        "action_counter": self.action_counter,
                        "round": round_idx,
                        "analysis_budget_remaining": analysis_budget,
                        "tools_exposed": (
                            "action_only" if current_tools is action_tools else "full"
                        ),
                        "input": {
                            "system_instruction": SYSTEM_INSTRUCTION,
                            "user_prompt": working_prompt,
                            "tools": current_tools,
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
                        "chosen_action": round_chosen.name if round_chosen else None,
                        "reasoning": (
                            getattr(round_chosen, "reasoning", None)
                            if round_chosen
                            else None
                        ),
                        "tool_calls": [asdict(r) for r in round_tool_records],
                        "error": round_error,
                    }
                )
                if usage is not None:
                    self.total_calls += 1
                    self.total_prompt_tokens += int(usage.get("prompt") or 0)
                    self.total_output_tokens += int(usage.get("output") or 0)
                    self.total_tokens += int(usage.get("total") or 0)

            tool_calls_this_step.extend(round_tool_records)

            if round_chosen is not None:
                chosen = round_chosen
                break

            if round_error is not None:
                # Plain VLM/network error — retry the same prompt next round.
                continue

            if not round_tool_records:
                # Model produced no function calls at all → next round won't have
                # new info; bail out and fail hard.
                break

            # Feed analysis results back into the prompt for the next round.
            # Splice them ABOVE `# TURN:` (via _build_working_prompt) so the
            # action cue stays the last thing the model reads.
            tool_results_blocks.append(render_tool_results(round_tool_records))
            working_prompt = _build_working_prompt()

        if chosen is None:
            self.trajectory.append(
                StepRecord(
                    game_id=self.game_id,
                    action_counter=self.action_counter,
                    state=latest_frame.state.name,
                    score=latest_frame.levels_completed,
                    chosen_action=None,
                    tool_calls=tool_calls_this_step,
                    vlm_attempts=round_idx,
                    error=repr(last_exc) if last_exc else "no action chosen",
                )
            )
            raise RuntimeError(
                f"ContinualHarness could not select an action after "
                f"{self.MAX_TOOL_ROUNDS} rounds"
            ) from last_exc

        self.trajectory.append(
            StepRecord(
                game_id=self.game_id,
                action_counter=self.action_counter,
                state=latest_frame.state.name,
                score=latest_frame.levels_completed,
                chosen_action=chosen.name,
                chosen_action_data=_action_data_dict(chosen),
                reasoning=getattr(chosen, "reasoning", None),
                tool_calls=tool_calls_this_step,
                vlm_attempts=round_idx,
            )
        )
        return chosen

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

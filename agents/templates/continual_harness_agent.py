from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from .continual_harness.context import build_action_prompt
from .continual_harness.helpers import (
    _action_from_name,
    available_game_actions,
    build_action_tools,
    frame_to_images,
)
from .continual_harness.memory import (
    MemoryStore,
    bootstrap_memory_path,
    format_memory_overview,
)
from .continual_harness.models import StepRecord, ToolCallRecord
from .continual_harness.prompts import SYSTEM_INSTRUCTION
from .continual_harness.tools import (
    PROCESS_MEMORY_TOOL,
    ContinualToolRouter,
    build_analysis_tools,
    extract_function_calls,
    is_action_tool,
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

        # Long-term memory is opt-in via --bootstrap-memory (see main.py).
        bootstrap = bootstrap_memory_path()
        self.memory: MemoryStore | None = (
            MemoryStore(bootstrap, game_id=self.game_id)
            if bootstrap is not None
            else None
        )

        def _handle_process_memory(args: dict[str, Any]) -> dict[str, Any]:
            # Defensive: tool only registered when self.memory is not None, but be
            # explicit so a stale registration can't crash a step.
            if self.memory is None:
                return {
                    "success": False,
                    "error": "memory is disabled; pass --bootstrap-memory to enable",
                }
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

        handlers: dict[str, Any] = {
            "get_recent_trajectory": _handle_get_recent_trajectory,
        }
        if self.memory is not None:
            handlers["process_memory"] = _handle_process_memory
        self.tool_router = ContinualToolRouter(handlers)

        if self.memory is not None:
            logger.info(
                "[%s] Long-term memory enabled: %s (%d existing entries)",
                self.name,
                self.memory.path,
                len(self.memory.all_entries()),
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
        if self.memory is not None:
            analysis_tools.append(PROCESS_MEMORY_TOOL)
        full_tools = action_tools + analysis_tools
        self.vlm.set_tools(full_tools)
        current_tools = full_tools

        images = frame_to_images(latest_frame)
        payload: Any = images if len(images) > 1 else images[0]

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
            if self.memory is not None:
                extras.append(format_memory_overview(self.memory.all_entries()))
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

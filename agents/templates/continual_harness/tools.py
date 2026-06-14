from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from arcengine import GameAction

from .models import ToolCallRecord

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FunctionCall:
    name: str
    args: dict[str, Any]


PROCESS_MEMORY_TOOL: dict[str, Any] = {
    "name": "process_memory",
    "description": "Manage the fact scratchpad (add/edit/delete/search). One small fact per entry — a confirmed action effect, an object identity, or a hypothesis to test — each with a confidence score. Do NOT write monolithic entries that mix confirmed facts with guesses. Memory index is auto-injected into every prompt; use search to read full bodies.",
    "parameters": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
            "description": "Required. Brief justification for this memory operation (what you are trying to learn or change and why)",
            },
            "operation": {
                "type": "string",
                "enum": ["add", "delete", "edit", "search"],
                "description": "Required. Which memory operation to perform.",
            },
            "title": {
                "type": "string",
                "description": "Required for add; optional for edit. Max 200 chars. Short label shown in the auto-injected overview.",
            },
            "body": {
                "type": "string",
                "description": "Required for add; optional for edit. The fact itself (returned by search). Keep it short — 1-3 sentences stating one claim.",
            },
            "confidence": {
                "type": "integer",
                "description": "Required for add; optional for edit. How sure you are this fact is true: 1=untested guess/should explore, 2=weak evidence, 3=unverified inference, 4=confirmed once, 5=repeatedly confirmed. Update it as evidence accrues.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional. Labels for grouping/search (e.g. ['player_identity', 'action6']).",
            },
            "id": {
                "type": "string",
                "description": "Required for delete and edit. Existing entry id (e.g. 'mem_003').",
            },
            "query": {
                "type": "string",
                "description": "Required for search. Substring matched case-insensitively against title + body + tags. Empty string returns all.",
            },
        },
        "required": ["reasoning", "operation"],
    },
}


PROCESS_SKILL_TOOL: dict[str, Any] = {
    "name": "process_skill",
    "description": "Manage saved skills (add/edit/delete/search). Re-saving an existing name updates in place. Skills run in a subprocess sandbox with pre-loaded numpy/PIL/etc.",
    "parameters": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Required. Brief justification for this skill operation (what strategy you are recording or updating and why)",
            },
            "operation": {
                "type": "string",
                "enum": ["add", "delete", "edit", "search"],
                "description": "Which skill operation to perform. Required.",
            },
            "name": {
                "type": "string",
                "description": "Short identifier shown in the overview. Required for add; optional for edit. Max 100 chars, must match [A-Za-z][A-Za-z0-9_]*.",
            },
            "description": {
                "type": "string",
                "description": "What the skill does + its I/O contract. Required for add; optional for edit. Max 500 chars.",
            },
            "code": {
                "type": "string",
                "description": "Python source. The body of run_skill. Required for add; optional for edit.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional labels for grouping/search (e.g. ['analysis', 'geometry']).",
            },
            "id": {
                "type": "string",
                "description": (
                    "Existing skill id (e.g. 'skill_007') or unique skill "
                    "name (e.g. 'eval_python'). Required for delete and edit."
                ),
            },
            "query": {
                "type": "string",
                "description": "Substring matched case-insensitively against name + description + code + tags. Required for search; empty string returns all.",
            },
        },
        "required": ["reasoning", "operation"],
    },
}


RUN_SKILL_TOOL: dict[str, Any] = {
    "name": "run_skill",
    "description": "Execute a saved skill by id-or-name in the subprocess sandbox. Returns {success, result, stdout, stderr, error?, actions_taken_inline}. 30s wall-clock cap. If it errors, edit the skill before re-running.",
    "parameters": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Why this skill is the right move right now. Required.",
            },
            "id": {
                "type": "string",
                "description": (
                    "Existing skill id (e.g. 'skill_007') OR unique skill "
                    "name (e.g. 'eval_python'). Required. The `## SKILLS` "
                    "overview lists both fields for every saved skill."
                ),
            },
            "args": {
                "type": "object",
                "description": "Free-form dict bound as `args` inside the skill.",
            },
        },
        "required": ["reasoning", "id"],
    },
}


RUN_CODE_TOOL: dict[str, Any] = {
    "name": "run_code",
    "description": (
        "Execute an ad-hoc Python snippet in the same sandbox as run_skill "
        "(use for one-off analysis before saving as a skill). Allowed "
        "modules: math, json, re, collections, itertools, functools, "
        "statistics, copy, dataclasses, hashlib, random. NO "
        "os/sys/subprocess/io/network. Read state.latest_frame / "
        "state.recent_trajectory / state.memory_entries / "
        "state.skill_entries. Set `result = ...` to return data. 30s "
        "wall-clock cap."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Why this snippet is worth a turn. Required.",
            },
            "code": {
                "type": "string",
                "description": "Python source to execute in the sandbox. Required.",
            },
            "args": {
                "type": "object",
                "description": "Free-form dict bound as `args` inside the snippet.",
            },
        },
        "required": ["reasoning", "code"],
    },
}


# --- Subagent tool surface ---
SUBAGENT_TOOL_ENUM: frozenset[str] = frozenset(
    {
        # "get_recent_trajectory",  # removed — superseded by RECENT HISTORY block
        "process_memory",
        "process_skill",
        "run_skill",
        "take_actions",
    }
)


PROCESS_SUBAGENT_TOOL: dict[str, Any] = {
    "name": "process_subagent",
    "description": "Manage the subagent registry (add/edit/delete/search). A subagent is a focused inner agent with a system prompt and an allowlist of tools. Subagent index is auto-injected into every prompt.",
    "parameters": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Why this subagent operation is worth a turn. Required.",
            },
            "operation": {
                "type": "string",
                "enum": ["add", "delete", "edit", "search"],
                "description": "Which subagent operation to perform. Required.",
            },
            "name": {
                "type": "string",
                "description": "Short identifier shown in the overview. Required for add; optional for edit. Max 100 chars, must match [A-Za-z][A-Za-z0-9_]*.",
            },
            "description": {
                "type": "string",
                "description": "What the subagent does + when to invoke it. Required for add; optional for edit. Max 500 chars.",
            },
            "instructions": {
                "type": "string",
                "description": "The subagent's system prompt: how it should approach its task. Required for add; optional for edit. Max 4000 chars.",
            },
            "allowed_tools": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": sorted(SUBAGENT_TOOL_ENUM),
                },
                "description": "Tools the subagent may call (subset of "
                "process_memory/process_skill/run_skill/take_actions). "
                "Optional for add/edit; omitted on add defaults to an empty "
                "allowlist — the subagent only reasons over the history already "
                "in its prompt and returns.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional labels for grouping/search.",
            },
            "id": {
                "type": "string",
                "description": "Existing subagent id (e.g. 'subagent_003'). Required for delete and edit.",
            },
            "query": {
                "type": "string",
                "description": "Substring matched case-insensitively against name + description + instructions + tags. Required for search; empty returns all.",
            },
        },
        "required": ["reasoning", "operation"],
    },
}


RUN_SUBAGENT_TOOL: dict[str, Any] = {
    "name": "run_subagent",
    "description": "Invoke a registered subagent on a task. Runs a bounded inner loop (up to 20 rounds) using its allowed tools. Mutations to memory/skills persist immediately. Max 1 per step.",
    "parameters": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Why this subagent fits this step. Required.",
            },
            "id": {
                "type": "string",
                "description": "Existing subagent id (e.g. 'subagent_003'). Required.",
            },
            "task": {
                "type": "string",
                "description": "Natural-language description of the one task to perform. Required.",
            },
            "context": {
                "type": "object",
                "description": "Optional free-form dict bound into the subagent's prompt as JSON.",
            },
        },
        "required": ["reasoning", "id", "task"],
    },
}


# subagent_return is the terminator the inner loop watches for. It is exposed
# ONLY to subagents (never to the orchestrator) and is NOT in SUBAGENT_TOOL_ENUM
# because that enum gates `allowed_tools` validation — subagent_return is
# unconditionally available and not configurable.
SUBAGENT_RETURN_NAME = "subagent_return"
SUBAGENT_RETURN_TOOL: dict[str, Any] = {
    "name": SUBAGENT_RETURN_NAME,
    "description": (
        "Terminate this subagent and return a final answer to the orchestrator. "
        "Call this when your task is complete (status='success'), cannot be "
        "completed (status='failure'), or has partial progress worth surfacing "
        "(status='partial'). After this call the inner loop ends; further tool "
        "calls in the same response are ignored."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Short summary of what you did and why you're returning now. Required.",
            },
            "answer": {
                "type": "string",
                "description": "The structured result to return to the orchestrator. Required.",
            },
            "status": {
                "type": "string",
                "enum": ["success", "failure", "partial"],
                "description": "Outcome of the task. Defaults to 'success'.",
            },
        },
        "required": ["reasoning", "answer"],
    },
}


def is_subagent_return_call(name: str) -> bool:
    """True iff `name` is the terminator the inner loop watches for."""
    return name == SUBAGENT_RETURN_NAME



def build_analysis_tools() -> list[dict[str, Any]]:
    """Read-only analysis tools. process_memory + skill tools are appended by the agent.

    Currently empty: the only analysis tool was get_recent_trajectory, removed
    once render_recent_history made it redundant. Kept as a seam for future
    analysis tools.
    """
    # return [GET_RECENT_TRAJECTORY_TOOL]  # removed — superseded by RECENT HISTORY block
    return []


# --- Function-call extraction -------------------------------------------------


def _plain_args(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    items = getattr(raw, "items", None)
    if callable(items):
        return {str(k): v for k, v in items()}
    return {}


def extract_function_calls(response: Any) -> list[FunctionCall]:
    """Walk all candidates × parts and surface every function_call in order.

    Shape-only — no GameAction validation, no JSON fallback. The orchestrator
    decides what to do with the list (commit on first action, run all analysis
    calls, etc.).
    """
    calls: list[FunctionCall] = []
    for cand in getattr(response, "candidates", None) or []:
        for part in getattr(getattr(cand, "content", None), "parts", []) or []:
            fc = getattr(part, "function_call", None)
            name = getattr(fc, "name", None)
            if name:
                calls.append(
                    FunctionCall(str(name), _plain_args(getattr(fc, "args", None)))
                )
    return calls


# --- Action / analysis discrimination -----------------------------------------


def is_action_tool(name: str) -> bool:
    """True iff `name` is a GameAction enum name.

    Retained as a stray-call detector: in the new design the orchestrator
    exposes a single `take_actions` tool instead of one per GameAction. If the
    model still emits a bare ACTION1..ACTION6 call, the harness logs an error
    record so the model can self-correct on the next round.
    """
    try:
        GameAction.from_name(name)
    except ValueError:
        return False
    return True


# --- Single unified action tool ----------------------------------------------
# Used by the orchestrator VLM (function call) AND by skills via the sandbox
# `tools.take_actions` callback. Same name everywhere. The orchestrator's
# tool surface exposes TAKE_ACTIONS_TOOL; the sandbox builds an equivalent
# callable. See feedback-tool-unification memory.

TAKE_ACTIONS = "take_actions"

TAKE_ACTIONS_TOOL: dict[str, Any] = {
    "name": TAKE_ACTIONS,
    "description": "Advance the game by executing a list of actions in order. Each runs synchronously; if a step becomes invalid mid-sequence (level transition, game ended), the remainder is skipped. Keep lists short (1-4, prefer 1).",
    "parameters": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": (
                    "Why this sequence of actions is the right next step. Required."
                ),
            },
            "actions": {
                "type": "array",
                "minItems": 1,
                # No maxItems on purpose: the soft length guideline lives in
                # the description / system prompt only (see feedback-soft-limits).
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "ARC GameAction name (ACTION1..ACTION6).",
                        },
                        "x": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 63,
                            "description": "X coordinate for ACTION6 (column).",
                        },
                        "y": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 63,
                            "description": "Y coordinate for ACTION6 (row).",
                        },
                    },
                    "required": ["name"],
                },
                "description": (
                    "Ordered list of actions. Each entry must include `name`; "
                    "ACTION6 additionally requires `x` and `y` in 0-63."
                ),
            },
        },
        "required": ["reasoning", "actions"],
    },
}


def is_take_actions_call(name: str) -> bool:
    """True iff `name` is the unified action-commit tool."""
    return name == TAKE_ACTIONS


# Maps a SUBAGENT_TOOL_ENUM name to the concrete tool spec the orchestrator
# already defines. Keeping this lookup in one place avoids drift between the
# allowlist enum and the actual spec shapes shown to the subagent VLM.
_SUBAGENT_TOOL_SPECS: dict[str, dict[str, Any]] = {
    # "get_recent_trajectory": GET_RECENT_TRAJECTORY_TOOL,  # removed
    "process_memory": PROCESS_MEMORY_TOOL,
    "process_skill": PROCESS_SKILL_TOOL,
    "run_skill": RUN_SKILL_TOOL,
    "take_actions": TAKE_ACTIONS_TOOL,
}


def build_subagent_tools(allowed: list[str]) -> list[dict[str, Any]]:
    """Tool list for a subagent's inner loop = allowlist + subagent_return.

    Unknown names are skipped silently; the store has already validated them at
    registration time. subagent_return is always appended last so the model
    can always terminate even if `allowed` is empty.
    """
    out: list[dict[str, Any]] = []
    for name in allowed or []:
        spec = _SUBAGENT_TOOL_SPECS.get(name)
        if spec is not None:
            out.append(spec)
    out.append(SUBAGENT_RETURN_TOOL)
    return out


# --- Router -------------------------------------------------------------------


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


class ContinualToolRouter:
    """Dispatches non-action tool calls to read-only handlers.

    Unknown tools return an error record; the next round can show that error to
    the model so it can correct itself. Handler exceptions are caught so a
    misbehaving analysis tool never kills a step.
    """

    def __init__(self, handlers: Mapping[str, ToolHandler]) -> None:
        self._handlers: dict[str, ToolHandler] = dict(handlers)

    def execute(self, call: FunctionCall) -> ToolCallRecord:
        handler = self._handlers.get(call.name)
        if handler is None:
            return ToolCallRecord(
                name=call.name, args=call.args, error=f"unknown tool: {call.name}"
            )
        try:
            result = handler(call.args)
            record = ToolCallRecord(name=call.name, args=call.args, result=result)
            # Hoist the inline-action count the handler reports inside its result
            # (run_skill / run_subagent) onto the record field the orchestrator
            # reads, so skill/subagent-fired actions count toward the loop's
            # actions_executed (and the conversation breaks to re-observe).
            if isinstance(result, dict) and "actions_taken_inline" in result:
                try:
                    record.actions_taken_inline = int(result.get("actions_taken_inline") or 0)
                except (TypeError, ValueError):
                    record.actions_taken_inline = 0
            return record
        except Exception as exc:
            logger.warning("tool %s raised: %s", call.name, exc)
            return ToolCallRecord(name=call.name, args=call.args, error=repr(exc))


def render_tool_results(records: list[ToolCallRecord]) -> str:
    """Format ToolCallRecords for injection into the working prompt."""
    payload = [
        {"name": r.name, "args": r.args, "result": r.result, "error": r.error}
        for r in records
    ]
    return "## TOOL RESULTS\n" + json.dumps(payload, default=str, indent=2)

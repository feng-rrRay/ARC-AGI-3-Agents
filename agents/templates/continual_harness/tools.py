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


# --- Analysis tool specs ------------------------------------------------------
# Schema shape matches build_action_tools so the backend treats them uniformly.

GET_RECENT_TRAJECTORY_TOOL: dict[str, Any] = {
    "name": "get_recent_trajectory",
    "description": (
        "Return the FULL step history (reasoning + tool calls + results) for "
        "this game. The prompt already shows a compact view of the last few "
        "batches; call this to reach further back, see older reasoning, or "
        "inspect partial-execution detail."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Why the compact view isn't enough. Required.",
            },
            "limit": {
                "type": "integer",
                "description": "How many recent actions to retrieve (1-80). Defaults to 40.",
            },
        },
        "required": ["reasoning"],
    },
}


PROCESS_MEMORY_TOOL: dict[str, Any] = {
    "name": "process_memory",
    "description": (
        "Manage long-term memory for this game. Memory persists for the "
        "current run by default, and across runs when --bootstrap-memory is "
        "provided. The current index is auto-injected into every prompt "
        "under ## LONG-TERM MEMORY (id + title + tags); you only need this "
        "tool to mutate or to read full bodies. Operations: add (title, "
        "body, tags?), edit (id, plus any of title/body/tags), delete (id), "
        "search (query — substring over title/body/tags; returns full bodies)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Why this memory operation is worth a turn. Required.",
            },
            "operation": {
                "type": "string",
                "enum": ["add", "delete", "edit", "search"],
                "description": "Which memory operation to perform. Required.",
            },
            "title": {
                "type": "string",
                "description": "Short label shown in the auto-injected overview. Required for add; optional for edit. Max 200 chars.",
            },
            "body": {
                "type": "string",
                "description": "Full memory content (returned by search). Required for add; optional for edit. Max 4000 chars.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional labels for grouping/search (e.g. ['player_identity', 'action6']).",
            },
            "id": {
                "type": "string",
                "description": "Existing entry id (e.g. 'mem_003'). Required for delete and edit.",
            },
            "query": {
                "type": "string",
                "description": "Substring matched case-insensitively against title + body + tags. Required for search; empty string returns all.",
            },
        },
        "required": ["reasoning", "operation"],
    },
}


PROCESS_SKILL_TOOL: dict[str, Any] = {
    "name": "process_skill",
    "description": (
        "Manage saved Python skills. Skills persist for the current run by "
        "default (logs/<run-id>/skills.json) and across runs when "
        "--bootstrap-skills is provided. The current registry is "
        "auto-injected under ## SKILLS (one row per skill listing both `id` "
        "— e.g. skill_007 — and `name`); call this tool to mutate or to read "
        "full code via search. Operations: add (name, description, code, "
        "tags?), edit (id-or-name + any of name/description/code/tags), "
        "delete (id-or-name), search (substring over "
        "name+description+code+tags).\n\n"
        "## Skill code rules\n"
        "The `code` you provide becomes the body of the skill and runs in a "
        "subprocess sandbox.\n"
        "- Pre-loaded names (no `import` needed): `np`, `numpy`, "
        "`collections`, `copy`, `dataclasses`, `functools`, `hashlib`, "
        "`itertools`, `json`, `math`, `random`, `re`, `statistics`, `Image`, "
        "`ImageDraw`, `ImageFilter`, `ImageOps`, `ImageChops`, plus helpers "
        "`render_grid(grid_2d)` and `render_grids(grids_3d)`. `import X` / "
        "`from X import Y` is accepted ONLY when X is one of those roots "
        "(or `PIL`); imports of anything else are rejected at parse time.\n"
        "- `state` is attribute-access only: use `state.latest_frame`, "
        "`state.images`, `state.recent_trajectory`, `state.memory_entries`, "
        "`state.skill_entries`. Bracket access (`state['latest_frame']`) "
        "raises TypeError. `args`, by contrast, is a plain dict.\n"
        "- Banned at parse time: `exec`, `eval`, `open`, `compile`, "
        "`getattr`, `setattr`, `globals`, `locals`, `__import__`, any "
        "dunder name, any `_`-prefixed attribute, filesystem/network I/O.\n"
        "- Engine-driving skills: call `tools[\"take_actions\"](actions=[...])` "
        "to advance the engine inline. The call returns "
        "`{executed_count, last_frame, terminal, level_changed, state, "
        "score, available_actions}`. After a call you MUST check `terminal` "
        "and `level_changed` before queueing more actions — once `terminal` "
        "is True the parent latches and further calls return an error. On "
        "`level_changed=True` the new level needs a fresh plan; do not "
        "re-send a sequence you computed for the previous level.\n"
        "- Return shape: assign `result = ...` for analysis output. Output "
        "is JSON-serialized with size caps; keep results compact."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Why this skill operation is worth a turn. Required.",
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
                "description": "Python source. The body of run_skill. Required for add; optional for edit. Max 8000 chars.",
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
    "description": (
        "Execute a saved skill in a subprocess sandbox. Address it by `id` "
        "(e.g. 'skill_007') or by `name` (e.g. 'eval_python') — the "
        "registry resolves both. The skill receives `state` (latest_frame, "
        "recent_trajectory, memory_entries, skill_entries, images) and your "
        "`args` dict. It may set `result = ...` to return data.\n\n"
        "`state.images` is a list of PIL.Image objects pre-rendered from "
        "`latest_frame.frame` (one image per grid layer; capped at 16). "
        "Use the `render_grid(grid_2d)` and `render_grids(grids_3d)` "
        "helpers (in skill globals) to render any other grid data — for "
        "example, the post-action `last_frame.frame` returned by a "
        "tools['take_actions'] RPC.\n\n"
        "Skills CAN drive the engine inline by calling "
        "tools['take_actions'](actions=[...]) — this RPCs back to the "
        "harness, executes the actions, and returns {executed_count, "
        "last_frame, terminal, level_changed, state, score, "
        "available_actions} so the skill can read the new frame and branch "
        "on the result. Use this for deterministic sub-routines "
        "(pathfinding, scanning loops, etc.) where one VLM call per action "
        "would be wasteful. Check both `terminal` and `level_changed` "
        "before sending another batch — a level transition means the rest "
        "of any precomputed plan is stale.\n\n"
        "Sandbox: 30s wall-clock cap; no filesystem or network; no `exec`/"
        "`eval`/`open`/`__`-anything. Pre-loaded modules are usable WITHOUT "
        "`import`: math, json, re, collections, itertools, functools, "
        "statistics, copy, dataclasses, hashlib, random; plus numpy as "
        "`np` (with file-I/O removed) and Pillow as `Image`, `ImageDraw`, "
        "`ImageFilter`, `ImageOps`, `ImageChops`. `import X` / `from X "
        "import Y` is accepted for the listed names (and `PIL`); imports "
        "of anything else are rejected at parse time."
    ),
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
                "description": "Python source to execute in the sandbox. Required. Max 8000 chars.",
            },
            "args": {
                "type": "object",
                "description": "Free-form dict bound as `args` inside the snippet.",
            },
        },
        "required": ["reasoning", "code"],
    },
}


# --- Subagent tool surface ---------------------------------------------------
# The single source of truth for what a subagent's inner loop may call.
# process_subagent and run_subagent are intentionally excluded: no recursion.
# run_code is intentionally excluded for now (see continual_harness_agent.py
# for the rationale: too many wasted rounds on schema/import failures).
SUBAGENT_TOOL_ENUM: frozenset[str] = frozenset(
    {
        "get_recent_trajectory",
        "process_memory",
        "process_skill",
        "run_skill",
    }
)


PROCESS_SUBAGENT_TOOL: dict[str, Any] = {
    "name": "process_subagent",
    "description": (
        "Manage the subagent registry. A subagent is a focused inner agent "
        "registered with a system prompt (instructions) + an allowlist of "
        "tools it can call during its bounded inner loop. The orchestrator "
        "invokes one via run_subagent(id, task). Subagents cannot directly "
        "commit ARC actions and cannot invoke other subagents — they return "
        "text the orchestrator may use to inform its own take_actions call. "
        "The current registry is auto-injected under ## SUBAGENT REGISTRY "
        "(id + name + allowed_tools + first description line). Operations: "
        "add (name, description, instructions, allowed_tools, tags?), edit "
        "(id + any of name/description/instructions/allowed_tools/tags), "
        "delete (id), search (substring over "
        "name+description+instructions+tags). If allowed_tools is omitted "
        "on add, it defaults to get_recent_trajectory."
    ),
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
                "get_recent_trajectory/process_memory/process_skill/"
                "run_skill). Optional for add/edit; omitted on add defaults "
                "to get_recent_trajectory. Empty list = subagent that only "
                "reasons and returns.",
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
    "description": (
        "Invoke a registered subagent on a single task. The subagent runs "
        "its own bounded inner loop (up to MAX_SUBAGENT_ROUNDS_PER_CALL=20) "
        "using only the tools in its allowlist, then calls "
        "subagent_return(answer, status) to terminate. Returns {success, "
        "result, rounds_used, id, name, version, warning?, error?, steps}. "
        "Mutations the subagent makes to memory / skills via its allowed "
        "tools persist immediately. The subagent CANNOT directly commit "
        "ARC actions — the orchestrator still owns take_actions. Counts "
        "against MAX_SUBAGENT_CALLS_PER_STEP=1 per outer iteration."
    ),
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


# --- Prompt-evolution tool ---------------------------------------------------
# Exposed ONLY to the meta-VLM call inside ContinualHarness._evolve_system_prompt.
# Never appears in the orchestrator's tool list, never in any subagent allowlist.
EVOLVE_PROMPT_NAME = "evolve_system_prompt"
EVOLVE_SYSTEM_PROMPT_TOOL: dict[str, Any] = {
    "name": EVOLVE_PROMPT_NAME,
    "description": (
        "Replace the agent's system instruction with an improved version. The "
        "new prompt must be 200-6000 characters; proposals outside that range "
        "are rejected and the agent keeps using the previous prompt."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": (
                    "What you saw in the trajectory and what you're changing "
                    "in the prompt. Required."
                ),
            },
            "new_prompt": {
                "type": "string",
                "description": (
                    "The full replacement system instruction (200-6000 chars). "
                    "Required."
                ),
            },
        },
        "required": ["reasoning", "new_prompt"],
    },
}


def is_evolve_prompt_call(name: str) -> bool:
    """True iff `name` is the prompt-evolution terminator the meta-call watches for."""
    return name == EVOLVE_PROMPT_NAME


# Maps a SUBAGENT_TOOL_ENUM name to the concrete tool spec the orchestrator
# already defines. Keeping this lookup in one place avoids drift between the
# allowlist enum and the actual spec shapes shown to the subagent VLM.
_SUBAGENT_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "get_recent_trajectory": GET_RECENT_TRAJECTORY_TOOL,
    "process_memory": PROCESS_MEMORY_TOOL,
    "process_skill": PROCESS_SKILL_TOOL,
    "run_skill": RUN_SKILL_TOOL,
    # "run_code": RUN_CODE_TOOL — disabled (see SUBAGENT_TOOL_ENUM note).
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


def build_analysis_tools() -> list[dict[str, Any]]:
    """Always-on read-only analysis tool. process_memory + skill tools are appended by the agent."""
    return [GET_RECENT_TRAJECTORY_TOOL]


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
# `tools["take_actions"]` callback. Same name everywhere. The orchestrator's
# tool surface exposes TAKE_ACTIONS_TOOL; the sandbox builds an equivalent
# callable. See feedback-tool-unification memory.

TAKE_ACTIONS = "take_actions"

TAKE_ACTIONS_TOOL: dict[str, Any] = {
    "name": TAKE_ACTIONS,
    "description": (
        "Advance the game by executing a list of one or more actions in order. "
        "Each action runs synchronously; after each one the harness records "
        "the resulting frame. If a step becomes invalid mid-sequence because "
        "the world changed (level transition, available_actions changed, or "
        "the game ended), the remaining steps are skipped and you'll see a "
        "partial-execution block in the next prompt's history.\n\n"
        "Action key: ACTION1 = Up/W · ACTION2 = Down/S · ACTION3 = Left/A · "
        "ACTION4 = Right/D · ACTION5 = Enter/Space/Delete · ACTION6 = Click "
        "(requires x, y in 0..63) · ACTION7 = Undo/Back. The current frame's "
        "prompt lists which of these are actually available this turn — "
        "calling an unavailable action will be rejected.\n\n"
        "Guideline: keep the list short (typically 1-8 actions). Include "
        "enough actions to make real progress, but not so many that the late "
        "steps depend on conditions you can't reasonably predict.\n\n"
        "Skills may also drive the engine inline by calling "
        "tools['take_actions'](actions=[...]) inside their code; that call "
        "returns the resulting frame so the skill can react before the next "
        "step."
    ),
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
            return ToolCallRecord(name=call.name, args=call.args, result=result)
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

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
        "Return the FULL step history (reasoning + analysis tool calls + results) for this "
        "game. The prompt already shows a compact one-line tail of the last few steps; call "
        "this to reach further back and read the original reasoning for older steps. Each "
        "call consumes one of your MAX_ANALYSIS_CALLS_PER_STEP=5 analysis-call budget."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Why the compact tail is insufficient and you need full detail. Required.",
            },
            "limit": {
                "type": "integer",
                "description": "How many recent steps to retrieve (1-80). Defaults to 40.",
            },
        },
        "required": ["reasoning"],
    },
}


def build_analysis_tools() -> list[dict[str, Any]]:
    """Stage 3 ships exactly one analysis tool; later stages append memory/skill tools here."""
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
    """True iff `name` is a GameAction enum name."""
    try:
        GameAction.from_name(name)
    except ValueError:
        return False
    return True


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

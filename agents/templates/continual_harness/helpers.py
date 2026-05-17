from __future__ import annotations

import json
import logging
import re
from typing import Any, Mapping, Sequence, TypeAlias

from arcengine import FrameData, GameAction
from PIL import Image

from .prompts import ACTION_DESCRIPTIONS

logger = logging.getLogger(__name__)

ActionIdentifier: TypeAlias = int | GameAction

# 16-colour ARC-AGI palette indexed by cell value modulo 16.
_PALETTE: list[tuple[int, int, int, int]] = [
    (0xFF, 0xFF, 0xFF, 0xFF),
    (0xCC, 0xCC, 0xCC, 0xFF),
    (0x99, 0x99, 0x99, 0xFF),
    (0x66, 0x66, 0x66, 0xFF),
    (0x33, 0x33, 0x33, 0xFF),
    (0x00, 0x00, 0x00, 0xFF),
    (0xE5, 0x3A, 0xA3, 0xFF),
    (0xFF, 0x7B, 0xCC, 0xFF),
    (0xF9, 0x3C, 0x31, 0xFF),
    (0x1E, 0x93, 0xFF, 0xFF),
    (0x88, 0xD8, 0xF1, 0xFF),
    (0xFF, 0xDC, 0x00, 0xFF),
    (0xFF, 0x85, 0x1B, 0xFF),
    (0x92, 0x12, 0x31, 0xFF),
    (0x4F, 0xCC, 0x30, 0xFF),
    (0xA3, 0x56, 0xD6, 0xFF),
]

# Reasoning is a required tool-level field on every action: forces Gemini to
# emit a short justification before choosing, captured separately from action_data.
_REASONING_PROP: dict[str, Any] = {
    "type": "string",
    "description": "1-3 sentences explaining why you chose this action. Required.",
}
_ACTION6_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": _REASONING_PROP,
        "x": {"type": "integer", "description": "Screen x in 0..63."},
        "y": {"type": "integer", "description": "Screen y in 0..63."},
    },
    "required": ["reasoning", "x", "y"],
}
_EMPTY_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {"reasoning": _REASONING_PROP},
    "required": ["reasoning"],
}


def describe_action(action: GameAction) -> str:
    return ACTION_DESCRIPTIONS.get(
        action.name, f"{action.name}: available game action."
    )


def available_game_actions(action_ids: Sequence[ActionIdentifier]) -> list[GameAction]:
    """Normalise frame.available_actions (ids or enums) into a unique action list, excluding RESET."""
    actions: list[GameAction] = []
    seen: set[GameAction] = set()

    for raw_action in action_ids:
        try:
            action = (
                raw_action
                if isinstance(raw_action, GameAction)
                else GameAction.from_id(int(raw_action))
            )
        except (TypeError, ValueError):
            logger.warning("Ignoring unknown available action id: %s", raw_action)
            continue
        if action is GameAction.RESET or action in seen:
            continue
        seen.add(action)
        actions.append(action)

    if actions:
        return actions
    # Fall back to every non-RESET action if the frame reported none.
    return [action for action in GameAction if action is not GameAction.RESET]


def build_action_tools(
    available_actions: Sequence[GameAction],
) -> list[dict[str, Any]]:
    """Backend-agnostic tool specs: one {name, description, parameters} per action.

    The VLM backend is responsible for wrapping these into its provider format.
    """
    return [
        {
            "name": action.name,
            "description": describe_action(action),
            "parameters": _ACTION6_PARAMS
            if action is GameAction.ACTION6
            else _EMPTY_PARAMS,
        }
        for action in available_actions
    ]


def grid_to_image(grid: Sequence[Sequence[int]]) -> Image.Image:
    """Render a 2-D grid of palette indices into an RGBA PIL image."""
    height = len(grid)
    width = max((len(row) for row in grid), default=0)
    if height == 0 or width == 0:
        return Image.new("RGBA", (64, 64), _PALETTE[0])

    raw = bytearray()
    for row in grid:
        for x in range(width):
            value = row[x] if x < len(row) else 0
            raw.extend(_PALETTE[int(value) % len(_PALETTE)])
    return Image.frombytes("RGBA", (width, height), bytes(raw))


def frame_to_images(latest_frame: FrameData) -> list[Image.Image]:
    """Render every layer of a FrameData into PIL images (one per grid)."""
    if not latest_frame.frame:
        return [Image.new("RGBA", (64, 64), _PALETTE[0])]
    return [grid_to_image(grid) for grid in latest_frame.frame]


def _extract_text_from_response(response: Any) -> str:
    # Concatenate every text part across every candidate.
    text_parts: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                text_parts.append(str(text))
    return "\n".join(text_parts)


def _as_plain_args(raw_args: Any) -> dict[str, Any]:
    # Function-call args may arrive as a Mapping, a Pydantic-ish object with .items(), or None.
    if raw_args is None:
        return {}
    if isinstance(raw_args, Mapping):
        return dict(raw_args)
    items = getattr(raw_args, "items", None)
    if callable(items):
        return {str(key): value for key, value in items()}
    return {}


def _extract_json_object(text: str) -> dict[str, Any] | None:
    # Try a fenced ```json ... ``` block first; otherwise grab the outermost {...}.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.IGNORECASE | re.S)
    json_text = fence.group(1) if fence else ""
    if not json_text:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        json_text = text[start : end + 1]

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _coerce_action_data(action: GameAction, args: Mapping[str, Any]) -> dict[str, Any]:
    # ACTION6 takes screen coordinates; clamp x, y into the 0..63 grid.
    if action is not GameAction.ACTION6:
        return {}

    def coord(name: str) -> int:
        try:
            value = int(args.get(name, 0))
        except (TypeError, ValueError):
            value = 0
        return max(0, min(63, value))

    return {"x": coord("x"), "y": coord("y")}


def _action_from_name(
    name: Any,
    args: Mapping[str, Any],
    available_actions: Sequence[GameAction],
) -> GameAction | None:
    if not isinstance(name, str):
        return None

    try:
        action = GameAction.from_name(name)
    except ValueError:
        return None

    if action not in available_actions:
        logger.warning("Model chose unavailable action: %s", action.name)
        return None

    # Pull reasoning out before passing args to action-data coercion: it's a
    # tool-level field and would otherwise pollute action_data.
    reasoning = args.get("reasoning")
    data_args = {k: v for k, v in args.items() if k != "reasoning"}
    action.set_data(_coerce_action_data(action, data_args))
    if reasoning is not None:
        action.reasoning = str(reasoning)
    return action


def parse_action_response(
    response: Any,
    available_actions: Sequence[GameAction],
) -> GameAction | None:
    """Parse a VLM response into a GameAction, or None if no valid choice was made.

    Tries the structured function-call path first, then falls back to JSON-in-text.
    Returning None lets the caller decide whether to retry or raise.
    """
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            fc = getattr(part, "function_call", None)
            name = getattr(fc, "name", None)
            if name:
                action = _action_from_name(
                    name,
                    _as_plain_args(getattr(fc, "args", None)),
                    available_actions,
                )
                if action is not None:
                    return action

    data = _extract_json_object(_extract_text_from_response(response))
    if not data:
        return None
    raw_args = data.get("arguments", data.get("data", {}))
    args = dict(raw_args) if isinstance(raw_args, Mapping) else {}
    return _action_from_name(
        data.get("action", data.get("name")), args, available_actions
    )

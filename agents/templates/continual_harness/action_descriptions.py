"""Static descriptions for ARC GameAction enum values.

Lifted verbatim from llm_agents.py:261-320 (plus ACTION7 for ARC-AGI-3 parity).
Consumed by helpers.build_action_tools.
"""

from __future__ import annotations

ACTION_DESCRIPTIONS: dict[str, str] = {
    "RESET": "Start or restart a game. Must be called first when NOT_PLAYED or after GAME_OVER to play again.",
    "ACTION1": "(1, W, Up).",
    "ACTION2": "(2, S, Down).",
    "ACTION3": "(3, A, Left).",
    "ACTION4": "(4, D, Right).",
    "ACTION5": "(5, Enter, Spacebar, Delete).",
    "ACTION6": "(6, Click, Point).",
    "ACTION7": "(7, Undo, Back).",
}

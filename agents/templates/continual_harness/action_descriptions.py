"""Static descriptions for ARC GameAction enum values.

Lifted verbatim from llm_agents.py:261-320 (plus ACTION7 for ARC-AGI-3 parity).
Consumed by helpers.build_action_tools.
"""

from __future__ import annotations

ACTION_DESCRIPTIONS: dict[str, str] = {
    "RESET": "Start or restart a game. Must be called first when NOT_PLAYED or after GAME_OVER to play again.",
    "ACTION1": "Send this simple input action (1, W, Up).",
    "ACTION2": "Send this simple input action (2, S, Down).",
    "ACTION3": "Send this simple input action (3, A, Left).",
    "ACTION4": "Send this simple input action (4, D, Right).",
    "ACTION5": "Send this simple input action (5, Enter, Spacebar, Delete).",
    "ACTION6": "Send this complex input action (6, Click, Point).",
    "ACTION7": "Send this simple input action (7, Undo, Back).",
}

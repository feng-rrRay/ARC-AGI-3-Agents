# Lifted verbatim from llm_agents.py:365-371 (the CONTEXT block of LLM.build_user_prompt).
SYSTEM_INSTRUCTION = """\
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.
"""

# Byte-for-byte copies of llm_agents.py:261-320 descriptions, plus ACTION7 for ARC-AGI-3 parity.
ACTION_DESCRIPTIONS = {
    "RESET": "Start or restart a game. Must be called first when NOT_PLAYED or after GAME_OVER to play again.",
    "ACTION1": "Send this simple input action (1, W, Up).",
    "ACTION2": "Send this simple input action (2, S, Down).",
    "ACTION3": "Send this simple input action (3, A, Left).",
    "ACTION4": "Send this simple input action (4, D, Right).",
    "ACTION5": "Send this simple input action (5, Enter, Spacebar, Delete).",
    "ACTION6": "Send this complex input action (6, Click, Point).",
    "ACTION7": "Send this simple input action (7, Undo, Back).",
}

# USER_PROMPT = LLM build_func_resp_prompt (State/Score/Frame) + our Previous-Action
# block (no message history) + LLM build_user_prompt TURN line.
USER_PROMPT = """\
# State:
{state}

# Score:
{score}

# Frame:
{latest_frame}

# Previous Action:
{previous_action}

# Previous Action Data:
{previous_action_data}

# TURN:
Call exactly one action.
"""

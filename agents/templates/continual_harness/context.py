from __future__ import annotations

from typing import Any

from arcengine import FrameData

from .prompts import USER_PROMPT


def pretty_print_3d(array_3d: list[list[list[Any]]]) -> str:
    # Mirrors LLM.pretty_print_3d in llm_agents.py to keep frame rendering identical.
    lines: list[str] = []
    for i, block in enumerate(array_3d):
        lines.append(f"Grid {i}:")
        for row in block:
            lines.append(f"  {row}")
        lines.append("")
    return "\n".join(lines)


def build_action_prompt(latest_frame: FrameData, extra_context: str = "") -> str:
    prompt = USER_PROMPT.format(
        state=latest_frame.state.name,
        score=latest_frame.levels_completed,
        latest_frame=pretty_print_3d(latest_frame.frame),
        previous_action=latest_frame.action_input.id.name,
        previous_action_data=latest_frame.action_input.data,
    )
    if extra_context.strip():
        prompt = prompt.replace("# TURN:", f"{extra_context.rstrip()}\n\n# TURN:")
    return prompt

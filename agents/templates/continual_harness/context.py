from __future__ import annotations

import json
from typing import Any

from arcengine import FrameData

from .prompts import HARNESS_USER_PROMPT


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
    prompt = HARNESS_USER_PROMPT.format(
        state=latest_frame.state.name,
        score=latest_frame.levels_completed,
        latest_frame=pretty_print_3d(latest_frame.frame),
        previous_action=latest_frame.action_input.id.name,
        previous_action_data=latest_frame.action_input.data,
    )
    if extra_context.strip():
        prompt = prompt.replace("# TURN:", f"{extra_context.rstrip()}\n\n# TURN:")
    return prompt


def build_subagent_prompt(
    *,
    task: str,
    context: dict[str, Any] | None,
    latest_frame: FrameData,
    memory_overview: str,
    skill_overview: str,
    compact_history: str,
) -> str:
    """Assemble the user-prompt half of a subagent invocation.

    The subagent's `instructions` are passed as system_instruction by the
    orchestrator; this function only builds the per-call user prompt. The
    termination cue at the bottom mirrors how the orchestrator's USER_PROMPT
    ends with `# TURN:` — keeping the action cue last (here, subagent_return).
    """
    parts: list[str] = []
    parts.append("## TASK")
    parts.append((task or "").strip() or "(no task supplied)")
    parts.append("")

    parts.append("## CONTEXT")
    if context:
        parts.append(json.dumps(context, indent=2, default=str))
    else:
        parts.append("(none)")
    parts.append("")

    parts.append(memory_overview)
    parts.append("")
    parts.append(skill_overview)
    parts.append("")

    parts.append("## RECENT STEPS")
    parts.append(compact_history)
    parts.append("")

    parts.append("## CURRENT FRAME")
    parts.append(
        f"state={latest_frame.state.name} score={latest_frame.levels_completed}"
    )
    parts.append(pretty_print_3d(latest_frame.frame))
    parts.append("")

    parts.append(
        "When you have completed your task, call "
        "subagent_return(reasoning=..., answer=..., status=...). You cannot "
        "commit ARC actions; only the orchestrator can."
    )
    return "\n".join(parts)

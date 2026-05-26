from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from arcengine import FrameData, GameAction

from .action_descriptions import ACTION_DESCRIPTIONS
from .helpers import available_game_actions
from .models import ToolCallRecord


def pretty_print_3d(array_3d: list[list[list[Any]]]) -> str:
    """Render a 3D grid stack as integer lists, one row per line.

    Output format matches ``state.latest_frame.frame`` exactly — each row is
    a Python-style list of ints — so the model sees the same representation
    in the prompt and in skill code.
    """
    lines: list[str] = []
    for i, block in enumerate(array_3d):
        if not block:
            lines.append(f"Grid {i}: (empty)")
            lines.append("")
            continue
        height = len(block)
        width = max((len(row) for row in block), default=0)
        lines.append(f"Grid {i} ({height}x{width}):")
        for row in block:
            lines.append("  " + str(list(row)))
        lines.append("")
    return "\n".join(lines)




def _render_tool_results(records: Iterable[ToolCallRecord]) -> str:
    payload = [
        {
            "name": r.name,
            "args": r.args,
            "result": r.result,
            "error": r.error,
            "actions_taken_inline": r.actions_taken_inline,
        }
        for r in records
    ]
    if not payload:
        return "(none)"
    return json.dumps(payload, default=str, indent=2)


def _format_available_actions(actions: Sequence[GameAction]) -> str:
    """Per-game list of usable actions WITH their semantic meanings.

    Pulls descriptions from `action_descriptions.ACTION_DESCRIPTIONS`. ACTION6
    is special-cased to remind the model that x/y are required (the static
    description there only says "Click, Point").
    """
    if not actions:
        return "(none — wait for the next frame)"
    lines: list[str] = []
    for a in actions:
        if a is GameAction.ACTION6:
            desc = "Complex click — provide x (column 0-63) and y (row 0-63)."
        else:
            desc = ACTION_DESCRIPTIONS.get(a.name, "")
        lines.append(f"  {a.name}: {desc}" if desc else f"  {a.name}")
    return "\n" + "\n".join(lines)


def build_working_prompt(
    latest_frame: FrameData,
    *,
    action_counter: int,
    recent_tool_results: Sequence[ToolCallRecord],
    history_block: str,
    memory_overview: str,
    skill_overview: str,
    subagent_overview: str,
    base_prompt: str = "",
) -> str:
    """Assemble the per-VLM-call working prompt."""
    available = available_game_actions(latest_frame.available_actions)
    frame_text = pretty_print_3d(latest_frame.frame) or "(empty frame)"

    sections: list[str] = []
    if base_prompt.strip():
        sections.append(base_prompt.strip())
    sections.append(f"# Step: {action_counter}")
    sections.append(
        "## RECENT HISTORY (batch-grouped; call get_recent_trajectory for older detail)\n"
        + (history_block or "No previous actions recorded.")
    )
    sections.append(
        "## TOOL RESULTS FROM PREVIOUS STEP\n"
        + _render_tool_results(recent_tool_results)
    )
    if memory_overview.strip():
        sections.append(memory_overview.rstrip())
    if skill_overview.strip():
        sections.append(skill_overview.rstrip())
    if subagent_overview.strip():
        sections.append(subagent_overview.rstrip())

    state_block = (
        "## CURRENT STATE\n"
        f"state: {latest_frame.state.name}\n"
        f"score (levels completed): {latest_frame.levels_completed}\n"
        f"available actions: {_format_available_actions(available)}\n"
        f"frame:\n{frame_text}"
    )
    sections.append(state_block)

    sections.append(
        "## TURN\n"
        "Decide your next move. Keep responses to at most 2 tool calls, "
        "and predict each action's effect before committing."
    )
    return "\n\n".join(sections)


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
        "subagent_return(reasoning=..., answer=..., status=...)."
    )
    return "\n".join(parts)

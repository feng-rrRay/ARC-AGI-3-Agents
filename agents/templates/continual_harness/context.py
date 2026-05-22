from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from arcengine import FrameData, GameAction

from .action_descriptions import ACTION_DESCRIPTIONS
from .helpers import available_game_actions
from .models import ToolCallRecord
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
    """Legacy builder retained for any callers still on the old prompt shape.

    The orchestrator no longer uses this — see `build_working_prompt` below.
    """
    prompt = HARNESS_USER_PROMPT
    if extra_context.strip():
        prompt = f"{extra_context.rstrip()}\n\n{prompt}"
    return prompt


# --- New orchestrator working-prompt builder ---------------------------------
# Replaces the old `build_action_prompt` for orchestrator use. Assembles the
# full per-step prompt directly from data, without placeholder substitution.
# Layout intentionally mirrors PokeAgent._build_structured_prompt:
#   [RECENT HISTORY] [TOOL RESULTS FROM PREVIOUS STEP] [LONG-TERM MEMORY]
#   [SKILL LIBRARY] [SUBAGENT REGISTRY] [CURRENT STATE] [BACKSTOP?]
#   [TURN instructions from harness_user.md]


_BACKSTOP_TEMPLATE = (
    "## BACKSTOP\n"
    "You have spent {n} consecutive iteration(s) on analysis without "
    "committing actions. Tools other than take_actions are not available "
    "this round; you must call take_actions to advance the game."
)


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
    turn_block: str = HARNESS_USER_PROMPT,
    force_take_actions: bool = False,
    no_action_iters: int = 0,
) -> str:
    """Assemble the per-VLM-call working prompt.

    `force_take_actions=True` appends a BACKSTOP block telling the model that
    only `take_actions` is exposed this round; the caller is responsible for
    actually restricting the tool list before the VLM query.
    """
    available = available_game_actions(latest_frame.available_actions)
    frame_text = pretty_print_3d(latest_frame.frame) or "(empty frame)"

    sections: list[str] = []
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

    if force_take_actions:
        sections.append(
            _BACKSTOP_TEMPLATE.format(n=max(1, no_action_iters))
        )

    sections.append(turn_block.strip())
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
        "subagent_return(reasoning=..., answer=..., status=...). You cannot "
        "commit ARC actions; only the orchestrator can."
    )
    return "\n".join(parts)

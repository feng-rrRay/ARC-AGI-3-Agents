from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolCallRecord:
    """One non-action tool call made within a single VLM round."""

    name: str
    args: dict[str, Any]
    result: dict[str, Any] | str | None = None
    error: str | None = None
    # Number of engine actions this tool call drove inline (only nonzero for
    # run_skill calls where the skill code invoked tools.take_actions().
    actions_taken_inline: int = 0


@dataclass(slots=True)
class StepRecord:
    """One executed game action. Append-only; one row per take_action() call."""

    game_id: str
    action_counter: int
    state: str
    score: int
    chosen_action: str | None
    chosen_action_data: dict[str, Any] = field(default_factory=dict)
    reasoning: str | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    vlm_attempts: int = 1
    error: str | None = None

    # Source of the action and batch metadata. batch_id groups actions emitted
    # by a single take_actions tool call (or a single run_skill RPC batch).
    # batch_reasoning / batch_rejected are populated only on the position == 1
    # row to avoid duplication across drain rows of the same batch.
    source: str = "vlm"  # "vlm" | "run_skill" | "auto_reset" | "backstop"
    batch_id: str | None = None
    batch_position: int | None = None
    batch_total: int | None = None
    batch_reasoning: str | None = None
    batch_rejected: list[dict[str, Any]] | None = None
    skill_id: str | None = None
    score_delta: int | None = None
    state_after: str | None = None

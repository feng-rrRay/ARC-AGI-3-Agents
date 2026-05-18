from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolCallRecord:
    """One non-action tool call made within a single choose_action() invocation."""

    name: str
    args: dict[str, Any]
    result: dict[str, Any] | str | None = None
    error: str | None = None


@dataclass(slots=True)
class StepRecord:
    """One choose_action() outcome. Append-only; one row per game step."""

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

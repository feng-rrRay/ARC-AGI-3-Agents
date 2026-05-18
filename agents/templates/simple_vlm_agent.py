from __future__ import annotations

import logging
import os
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from .continual_harness.helpers import (
    available_game_actions,
    build_action_tools,
    frame_to_images,
    parse_action_response,
)
from .continual_harness.prompts import SYSTEM_INSTRUCTION, USER_PROMPT
from .continual_harness.trace import (
    TraceWriter,
    default_trace_path,
    serialize_response,
)
from .utils.vlm_backend import VLM

logger = logging.getLogger(__name__)


def pretty_print_3d(array_3d: list[list[list[Any]]]) -> str:
    # Mirrors LLM.pretty_print_3d in llm_agents.py to keep frame rendering identical.
    lines = []
    for i, block in enumerate(array_3d):
        lines.append(f"Grid {i}:")
        for row in block:
            lines.append(f"  {row}")
        lines.append("")
    return "\n".join(lines)


def build_action_prompt(latest_frame: FrameData) -> str:
    return str(
        USER_PROMPT.format(
            state=latest_frame.state.name,
            score=latest_frame.levels_completed,
            latest_frame=pretty_print_3d(latest_frame.frame),
            previous_action=latest_frame.action_input.id.name,
            previous_action_data=latest_frame.action_input.data,
        )
    )


class VLMSimple(Agent):
    """Single-game VLM agent: each step queries the VLM with the rendered frame.

    On every step we (1) advertise the currently available actions as tools,
    (2) send the frame + prompt, and (3) parse the response into a GameAction.
    A bounded retry covers both VLM call errors and unparseable outputs;
    persistent failure raises rather than silently picking a default action.
    """

    MAX_ACTIONS = 80
    MODEL = "gemini-3.1-pro-preview"  # default; override via GEMINI_MODEL
    MAX_PARSE_RETRIES = 3  # attempts before raising

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Must resolve model_name BEFORE super().__init__(): Agent.__init__
        # calls start_recording() which reads self.name → self.model_name.
        self.model_name = os.getenv("GEMINI_MODEL", self.MODEL)
        super().__init__(*args, **kwargs)
        self.vlm = VLM(
            self.model_name,
            backend="gemini",
            system_instruction=SYSTEM_INSTRUCTION,
        )
        recorder = getattr(self, "recorder", None)
        guid = getattr(recorder, "guid", None)
        # One JSONL record per VLM call lands here (paired with the recording stem).
        self.trace = TraceWriter(default_trace_path(prefix=self.name, guid=guid))
        # Cumulative token usage; logged once on cleanup().
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0

    @property
    def name(self) -> str:
        sanitized = self.model_name.replace("/", "-").replace(":", "-")
        return f"{super().name}.{sanitized}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # Base loop calls choose_action even on terminal states; reset first.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        available = available_game_actions(latest_frame.available_actions)
        tools = build_action_tools(available)
        self.vlm.set_tools(tools)
        prompt = build_action_prompt(latest_frame)
        images = frame_to_images(latest_frame)
        payload: Any = images if len(images) > 1 else images[0]

        trace_input = {
            "system_instruction": SYSTEM_INSTRUCTION,
            "user_prompt": prompt,
            "tools": tools,
            "images": [
                {"width": img.width, "height": img.height, "mode": img.mode}
                for img in images
            ],
        }

        last_exc: Exception | None = None
        for attempt in range(1, self.MAX_PARSE_RETRIES + 1):
            output: dict[str, Any] = {}
            usage: dict[str, int | None] | None = None
            chosen: GameAction | None = None
            error: str | None = None
            try:
                response = self.vlm.get_query(payload, prompt, module_name=self.name)
                output = serialize_response(response)
                usage = self.vlm.extract_usage(response)
                chosen = parse_action_response(response, available)
                if chosen is None:
                    logger.warning(
                        "Attempt %d/%d: VLM response could not be parsed",
                        attempt,
                        self.MAX_PARSE_RETRIES,
                    )
            except Exception as exc:
                last_exc = exc
                error = repr(exc)
                logger.warning(
                    "Attempt %d/%d: VLM call failed: %s",
                    attempt,
                    self.MAX_PARSE_RETRIES,
                    exc,
                )
            finally:
                self.trace.write(
                    {
                        "agent": self.name,
                        "model": self.model_name,
                        "game_id": self.game_id,
                        "action_counter": self.action_counter,
                        "attempt": attempt,
                        "input": trace_input,
                        "output": output,
                        "usage": usage,
                        "chosen_action": chosen.name if chosen else None,
                        "reasoning": getattr(chosen, "reasoning", None)
                        if chosen
                        else None,
                        "error": error,
                    }
                )
                if usage is not None:
                    self.total_calls += 1
                    self.total_prompt_tokens += int(usage.get("prompt") or 0)
                    self.total_output_tokens += int(usage.get("output") or 0)
                    self.total_tokens += int(usage.get("total") or 0)
            if chosen is not None:
                return chosen

        raise RuntimeError(
            f"VLMSimple could not select an action after "
            f"{self.MAX_PARSE_RETRIES} attempts"
        ) from last_exc

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        logger.info(
            "[%s] Final token usage: calls=%d prompt=%d output=%d total=%d",
            self.name,
            self.total_calls,
            self.total_prompt_tokens,
            self.total_output_tokens,
            self.total_tokens,
        )
        super().cleanup(*args, **kwargs)

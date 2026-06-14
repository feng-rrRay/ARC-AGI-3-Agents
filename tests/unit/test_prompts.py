"""Tests for the markdown-backed prompt loader.

Covers: loader smoke test, naive-prompt golden snapshot (so simple_vlm_agent
stays byte-identical to the pre-restructure behaviour), harness keyword
checks, and the action_descriptions module port.
"""

from __future__ import annotations

import pytest

from agents.templates.continual_harness.action_descriptions import (
    ACTION_DESCRIPTIONS,
)
from agents.templates.continual_harness.prompts import (
    BASE_ORCHESTRATOR_POLICY,
    EVOLUTION_SYSTEM_INSTRUCTION,
    EVOLUTION_USER_PROMPT,
    HARNESS_SYSTEM_INSTRUCTION,
    NAIVE_SYSTEM_INSTRUCTION,
    NAIVE_USER_PROMPT,
    load_prompt,
)

# Pre-restructure literals — these are the byte-identical strings that used
# to live in agents/templates/continual_harness/prompts.py. Locking them here
# means a future edit to naive_*.md that drifts from this snapshot will fail
# the test (preventing accidental regressions in simple_vlm_agent).
_NAIVE_SYSTEM_GOLDEN = (
    "# CONTEXT:\n"
    "You are an agent playing a dynamic game. Your objective is to\n"
    "WIN and avoid GAME_OVER while minimizing actions.\n"
    "\n"
    "One action produces one Frame. One Frame is made of one or more sequential\n"
    "Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with\n"
    "INT<0,15> values."
)

_NAIVE_USER_GOLDEN = (
    "# State:\n"
    "{state}\n"
    "\n"
    "# Score:\n"
    "{score}\n"
    "\n"
    "# Frame:\n"
    "{latest_frame}\n"
    "\n"
    "# Previous Action:\n"
    "{previous_action}\n"
    "\n"
    "# Previous Action Data:\n"
    "{previous_action_data}\n"
    "\n"
    "# TURN:\n"
    "Call exactly one action."
)


@pytest.mark.unit
class TestPromptLoader:
    def test_all_constants_load_and_are_nonempty(self) -> None:
        for name, value in [
            ("NAIVE_SYSTEM_INSTRUCTION", NAIVE_SYSTEM_INSTRUCTION),
            ("NAIVE_USER_PROMPT", NAIVE_USER_PROMPT),
            ("HARNESS_SYSTEM_INSTRUCTION", HARNESS_SYSTEM_INSTRUCTION),
            ("BASE_ORCHESTRATOR_POLICY", BASE_ORCHESTRATOR_POLICY),
            ("EVOLUTION_SYSTEM_INSTRUCTION", EVOLUTION_SYSTEM_INSTRUCTION),
            ("EVOLUTION_USER_PROMPT", EVOLUTION_USER_PROMPT),
        ]:
            assert isinstance(value, str), f"{name} is not a string"
            assert value.strip(), f"{name} is empty"

    def test_load_prompt_returns_freshly_read_content(self) -> None:
        # load_prompt should hit disk and return the same as the cached const.
        assert load_prompt("naive_system.md") == NAIVE_SYSTEM_INSTRUCTION
        assert load_prompt("harness_system.md") == HARNESS_SYSTEM_INSTRUCTION


@pytest.mark.unit
class TestNaivePromptsGolden:
    """Lock the simple_vlm_agent prompts byte-for-byte to the pre-restructure text."""

    def test_naive_system_matches_golden(self) -> None:
        assert NAIVE_SYSTEM_INSTRUCTION == _NAIVE_SYSTEM_GOLDEN

    def test_naive_user_matches_golden(self) -> None:
        assert NAIVE_USER_PROMPT == _NAIVE_USER_GOLDEN

    def test_naive_user_format_substitutes_placeholders(self) -> None:
        # The simple agent calls .format(...) on this; ensure the keys match.
        rendered = NAIVE_USER_PROMPT.format(
            state="IN_PROGRESS",
            score=0,
            latest_frame="(grid)",
            previous_action="RESET",
            previous_action_data={},
        )
        assert "IN_PROGRESS" in rendered
        assert "(grid)" in rendered
        assert "Call exactly one action." in rendered


@pytest.mark.unit
class TestHarnessSystemContent:
    """Smoke check that the harness prompt documents what it must."""

    def test_mentions_required_concepts(self) -> None:
        text = HARNESS_SYSTEM_INSTRUCTION.lower()
        # run_code is intentionally NOT required here — it's currently disabled
        # in the orchestrator and dropped from the prompt.
        for token in ["action", "memory", "skill", "subagent"]:
            assert token in text, f"harness_system.md must mention {token!r}"

    def test_run_code_not_mentioned(self) -> None:
        # Guardrail: run_code is disabled in the registry; the prompt must not
        # advertise it (otherwise the model will try to call it and fail).
        assert "run_code" not in HARNESS_SYSTEM_INSTRUCTION

    def test_mentions_play_pace(self) -> None:
        text = HARNESS_SYSTEM_INSTRUCTION.lower()
        assert "action" in text and "reasoning" in text


@pytest.mark.unit
class TestEvolutionPromptContent:
    def test_evolution_system_mentions_game_name_placeholder(self) -> None:
        assert "{game_name}" in EVOLUTION_SYSTEM_INSTRUCTION

    def test_evolution_user_has_all_placeholders(self) -> None:
        for placeholder in [
            "{system_prompt}",
            "{current_base_prompt}",
            "{n}",
            "{trajectory}",
            "{tool_evidence}",
            "{memory_overview}",
            "{skill_overview}",
            "{subagent_overview}",
        ]:
            assert placeholder in EVOLUTION_USER_PROMPT, (
                f"evolution_user.md missing placeholder {placeholder}"
            )


@pytest.mark.unit
class TestActionDescriptions:
    def test_all_actions_present(self) -> None:
        for name in [
            "RESET",
            "ACTION1",
            "ACTION2",
            "ACTION3",
            "ACTION4",
            "ACTION5",
            "ACTION6",
            "ACTION7",
        ]:
            assert name in ACTION_DESCRIPTIONS

    def test_descriptions_unchanged_post_move(self) -> None:
        # Lock the pre-move text so consumers (helpers.build_action_tools)
        # don't silently drift.
        assert ACTION_DESCRIPTIONS["ACTION6"] == (
            "Send this complex input action (6, Click, Point)."
        )
        assert ACTION_DESCRIPTIONS["RESET"].startswith("Start or restart a game.")

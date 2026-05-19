"""Markdown-backed prompt loader for ContinualHarness + VLMSimple.

Each markdown file is read once at import time and exposed as a module-level
string. `load_prompt(name)` re-reads from disk for paths that may have been
overwritten at runtime (e.g. evolved prompts in tests).
"""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def _read(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def load_prompt(name: str) -> str:
    """Re-read a prompt file from disk. Useful for tests and rollback paths."""
    return _read(name)


# Naive prompts — byte-identical to the pre-restructure SYSTEM_INSTRUCTION /
# USER_PROMPT constants. Consumed by simple_vlm_agent.py.
NAIVE_SYSTEM_INSTRUCTION: str = _read("naive_system.md")
NAIVE_USER_PROMPT: str = _read("naive_user.md")

# Harness prompts — extended system instruction that documents the full tool
# surface (memory / skills / subagents / run_code) and the orchestrator's
# behaviour rules. Consumed by continual_harness_agent.py.
HARNESS_SYSTEM_INSTRUCTION: str = _read("harness_system.md")
HARNESS_USER_PROMPT: str = _read("harness_user.md")

# Evolution prompts — meta-prompt used by ContinualHarness's prompt-evolution
# step. Only loaded when --prompt-evolve-frequency > 0.
EVOLUTION_SYSTEM_INSTRUCTION: str = _read("evolution_system.md")
EVOLUTION_USER_PROMPT: str = _read("evolution_user.md")


__all__ = [
    "NAIVE_SYSTEM_INSTRUCTION",
    "NAIVE_USER_PROMPT",
    "HARNESS_SYSTEM_INSTRUCTION",
    "HARNESS_USER_PROMPT",
    "EVOLUTION_SYSTEM_INSTRUCTION",
    "EVOLUTION_USER_PROMPT",
    "load_prompt",
]

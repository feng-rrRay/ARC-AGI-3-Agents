"""Minimal Hermes AIAgent wrapper for ARC-AGI-3 evaluation.

Runs inside the Docker container. Imports Hermes internals, constructs an
AIAgent with the arc-agi-3 MCP toolset, and runs the autonomous conversation
loop. Emits basic JSONL events to stdout (system + result) for the host
stream reader.

Phase-2 parity items (not yet implemented):
  - thinking / tool_use JSONL callbacks
  - usage_events.jsonl capture (token/cost per API call)
  - Multimodal image injection patch (if native Gemini path is insufficient)
  - Session resume via --resume-session-id

Usage (inside container):
    python3 hermes_wrapper.py \
        --directive-path /workspace/.agent_directive.txt \
        --working-dir /workspace \
        --server-url http://host.docker.internal:8000 \
        --hermes-home /home/hermes-agent/.hermes \
        --model gemini-3.1-pro-preview \
        --provider gemini \
        --api-key-env GEMINI_API_KEY \
        --max-turns 90
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _emit(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def _read_text(path: str | None) -> str:
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def _build_initial_prompt(directive_text: str, server_url: str, is_resume: bool) -> str:
    runtime_ctx = (
        "Runtime context:\n"
        f"- ARC game server URL (context only — do NOT call it directly): {server_url}\n"
        "- You are in a long-running autonomous session.\n"
        "- Observe and act using only the two MCP tools: get_game_state and take_actions.\n"
        "- Do not wait for follow-up prompts; continue until WIN or external termination.\n"
    )
    if is_resume:
        return f"Continue the ARC-AGI-3 session.\n\n{runtime_ctx}"
    if directive_text.strip():
        return directive_text.rstrip()
    return f"Start the ARC-AGI-3 session.\n\n{runtime_ctx}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal Hermes wrapper for ARC-AGI-3")
    parser.add_argument("--directive-path",    default="")
    parser.add_argument("--working-dir",       required=True)
    parser.add_argument("--server-url",        default="http://localhost:8000")
    parser.add_argument("--hermes-home",       required=True)
    parser.add_argument("--model",             default="")
    parser.add_argument("--provider",          default="")
    parser.add_argument("--base-url",          default="")
    parser.add_argument("--api-key-env",       default="")
    parser.add_argument("--max-turns",         type=int, default=90)
    parser.add_argument("--resume-session-id", default="")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stderr,
    )

    hermes_home = Path(args.hermes_home)
    hermes_home.mkdir(parents=True, exist_ok=True)
    os.environ["HERMES_HOME"] = str(hermes_home)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    # Resolve model / provider / api_key from args then env vars
    model    = args.model.strip()    or os.environ.get("HERMES_MODEL",    "").strip() or "gemini-3.1-pro-preview"
    provider = args.provider.strip() or os.environ.get("HERMES_PROVIDER", "").strip() or "gemini"
    base_url = args.base_url.strip() or os.environ.get("HERMES_BASE_URL", "").strip() or None

    api_key: str | None = None
    api_key_env = args.api_key_env.strip() or os.environ.get("HERMES_API_KEY_ENV", "").strip()
    if api_key_env:
        api_key = os.environ.get(api_key_env) or None
    if api_key is None:
        # Fallback: try well-known Gemini env vars directly
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or None

    resume_session_id = args.resume_session_id.strip() or None

    # Import Hermes internals only after HERMES_HOME is set (so Hermes picks up
    # the run-local config.yaml that was injected by the host backend).
    try:
        from hermes_state import SessionDB      # type: ignore[import]
        from run_agent import AIAgent           # type: ignore[import]
    except ImportError as exc:
        _emit({
            "type": "error",
            "message": (
                "Failed to import Hermes internals. "
                "Ensure hermes-agent is installed at /opt/hermes-agent with pip install .[mcp]."
            ),
            "error": str(exc),
        })
        return 1

    session_db = SessionDB(hermes_home / "state.db")
    conversation_history = None
    if resume_session_id:
        try:
            conversation_history = session_db.get_messages_as_conversation(resume_session_id)
        except Exception:
            conversation_history = None

    directive_text = _read_text(args.directive_path)
    user_message = _build_initial_prompt(
        directive_text, args.server_url, is_resume=bool(resume_session_id and conversation_history)
    )

    try:
        os.chdir(args.working_dir)
    except OSError as exc:
        logger.warning("Could not chdir to %s: %s", args.working_dir, exc)

    agent = AIAgent(
        base_url=base_url,
        api_key=api_key,
        provider=provider,
        model=model,
        max_iterations=args.max_turns,
        quiet_mode=True,
        session_id=resume_session_id,
        session_db=session_db,
        enabled_toolsets=["mcp-arc-agi-3"],
        pass_session_id=True,
    )

    _emit({
        "type": "system",
        "session_id": agent.session_id,
        "model": agent.model,
        "mcp_servers": ["arc-agi-3"],
        "tools": [
            (t.get("function", {}) if isinstance(t, dict) else {}).get("name", "")
            for t in (agent.tools or [])
        ],
    })

    start_time = time.time()
    result: dict = {}
    try:
        result = agent.run_conversation(
            user_message=user_message,
            conversation_history=conversation_history,
            persist_user_message=user_message,
        ) or {}
    except Exception as exc:
        import traceback
        _emit({
            "type": "error",
            "message": "Hermes wrapper: run_conversation raised an exception",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        return 1

    duration_ms = int((time.time() - start_time) * 1000)
    is_error = bool(result.get("failed"))
    _emit({
        "type": "result",
        "session_id": agent.session_id,
        "model": agent.model,
        "is_error": is_error,
        "error": str(result.get("error") or ""),
        "content": result.get("final_response") if isinstance(result, dict) else None,
        "duration_ms": duration_ms,
    })
    return 1 if is_error else 0


if __name__ == "__main__":
    sys.exit(main())

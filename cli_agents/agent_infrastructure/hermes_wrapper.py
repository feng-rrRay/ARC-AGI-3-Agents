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
        --max-turns 5000
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

MIN_TOOLSETS = ["mcp-arc-agi-3"]
FULL_TOOLSETS = [
    "mcp-arc-agi-3",
    "skills",
    "memory",
    "session_search",
    "file",
    "terminal",
    "code_execution",
]

# Full-toolset code-execution guidance, kept as a directive markdown (appended to
# SOUL.md in full mode) rather than inlined here.
_FULL_TOOLSET_DIRECTIVE = (
    Path(__file__).resolve().parents[1] / "directives" / "arc_full_toolset.md"
)

# ARC MCP tools to expose inside the Hermes code-execution sandbox (registered
# tool names — what handle_function_call dispatches on).
ARC_SANDBOX_TOOLS = (
    "mcp_arc_agi_3_get_game_state",
    "mcp_arc_agi_3_take_actions",
)

# Friendly stub functions appended to the sandbox's generated hermes_tools module
# so code can call get_game_state()/take_actions(...) which RPC to the registered
# MCP tools above. (_call is defined by the module's transport header.)
_ARC_SANDBOX_STUBS = '''

# --- ARC game tools (injected by hermes_wrapper) ---
def get_game_state():
    """Observe the ARC game; returns the observation payload dict."""
    return _call("mcp_arc_agi_3_get_game_state", {})

def take_actions(actions, reasoning=""):
    """Apply ARC actions, e.g. actions=[{"name": "ACTION1"}]; returns the result dict."""
    return _call("mcp_arc_agi_3_take_actions", {"actions": actions, "reasoning": reasoning})
'''


def _emit(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def _read_text(path: str | None) -> str:
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def _normalize_tool_name(name: str) -> str:
    """Strip Hermes's MCP prefix (mcp_arc_agi_3_) for readable trajectory events."""
    for prefix in ("mcp_arc_agi_3_", "mcp__arc-agi-3__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name.split("__")[-1] if "__" in name else name


def _payload_from_result(result: Any) -> dict | None:
    """Extract an ARC MCP tool's payload dict from a tool-call result.

    Used for both get_game_state and take_actions. The result arrives as the MCP
    wrapper ``{"result": "<json string>"}`` (or an already-decoded dict). Returns
    the inner payload dict, or None if it can't be parsed — the trace callback
    must never raise.
    """
    try:
        obj = result if isinstance(result, (dict, list)) else json.loads(result)
    except (ValueError, TypeError):
        return None
    inner = obj.get("result", obj) if isinstance(obj, dict) else obj
    if isinstance(inner, str):
        try:
            inner = json.loads(inner)
        except ValueError:
            return None
    return inner if isinstance(inner, dict) else None


def _normalise_toolset_mode(value: str) -> str:
    mode = (value or "").strip().lower()
    return mode if mode in {"min", "full"} else "min"


def _enabled_toolsets(mode: str) -> list[str]:
    return FULL_TOOLSETS if _normalise_toolset_mode(mode) == "full" else MIN_TOOLSETS


def _full_toolset_runtime_note() -> str:
    return (
        "\n\n## Full Hermes Toolset Mode\n"
        "In this run, additional Hermes built-in tools are available for analysis, "
        "skill management, memory, session search, file inspection/editing, terminal "
        "commands, and Python code execution. Use the ARC MCP tools only for game "
        "observation and scored game actions. Do not call the ARC game server, "
        "three.arcprize.org, or external scoring APIs directly."
    )


def _enable_arc_tools_in_sandbox() -> bool:
    """Expose the ARC MCP tools inside Hermes's execute_code sandbox.

    Hermes only stubs an allow-list of built-ins (SANDBOX_ALLOWED_TOOLS) into the
    sandbox; the dynamically-registered ARC MCP tools aren't included, so code
    can't drive the game. We add the MCP tool names to the allow-list (so the RPC
    server permits them — handle_function_call already routes by registered name)
    and append friendly get_game_state()/take_actions() stubs to the generated
    hermes_tools module. Runtime monkeypatch (no Hermes fork / image rebuild);
    returns False if the sandbox module isn't importable.
    """
    try:
        import tools.code_execution_tool as cet  # type: ignore[import]
    except Exception as exc:
        logger.warning("Could not patch code-exec sandbox for ARC tools: %s", exc)
        return False

    cet.SANDBOX_ALLOWED_TOOLS = frozenset(
        set(cet.SANDBOX_ALLOWED_TOOLS) | set(ARC_SANDBOX_TOOLS)
    )
    _orig_generate = cet.generate_hermes_tools_module

    def _generate_with_arc(enabled_tools, transport="uds"):
        return _orig_generate(enabled_tools, transport) + _ARC_SANDBOX_STUBS

    _generate_with_arc.__name__ = "generate_hermes_tools_module"
    cet.generate_hermes_tools_module = _generate_with_arc
    logger.info("Enabled ARC tools in execute_code sandbox: %s", list(ARC_SANDBOX_TOOLS))
    return True


def _extend_rate_limit_resilience(agent: Any) -> None:
    """Make a Gemini 429 / rate limit (e.g. the 8M-tokens-per-minute cap) survivable.

    A 429 is transient — it clears once the per-minute quota window resets — so we
    want the in-session API retry loop to be patient instead of giving up after a
    few seconds. Two knobs, both env-overridable:

      * ``agent._api_max_retries`` — how many times each model API call is retried
        before ``run_conversation`` returns ``failed=True``. Hermes defaults to 3
        (~20s total backoff via base=2s, cap=60s), far too short for a per-minute
        token quota to reset. We raise it (default 8) so the retry loop rides
        through a minutes-long throttle.
      * ``agent.conversation_loop.jittered_backoff`` — raise the per-retry wait
        floor/cap so early retries don't burn out in 2-12s. Guarded monkeypatch
        (no-op if Hermes internals differ), same runtime-patch pattern as
        ``_enable_arc_tools_in_sandbox``. Set HERMES_RATE_LIMIT_BACKOFF_BASE=0 to
        leave Hermes's native backoff untouched.

    When the retries are still exhausted, ``run_conversation`` returns
    ``failure_reason="rate_limit"``; the host orchestrator treats that as
    non-fatal and resumes the session rather than ending it.
    """
    # 1) More attempts.
    try:
        retries = int(os.environ.get("HERMES_API_MAX_RETRIES", "8"))
    except ValueError:
        retries = 8
    retries = max(retries, 1)
    if hasattr(agent, "_api_max_retries"):
        agent._api_max_retries = retries
        logger.info("Rate-limit resilience: api_max_retries=%d", retries)
    else:
        logger.warning("agent._api_max_retries not present; cannot raise retry count")

    # 2) Longer waits between attempts.
    try:
        floor = float(os.environ.get("HERMES_RATE_LIMIT_BACKOFF_BASE", "10"))
        cap = float(os.environ.get("HERMES_RATE_LIMIT_BACKOFF_MAX", "90"))
    except ValueError:
        floor, cap = 10.0, 90.0
    if floor <= 0:
        return
    try:
        import importlib

        cl = importlib.import_module("agent.conversation_loop")
        _orig_backoff = cl.jittered_backoff

        def _patient_backoff(attempt, *, base_delay=5.0, max_delay=120.0, jitter_ratio=0.5):
            # Only ever raise the wait, never shorten Hermes's own choice.
            return _orig_backoff(
                attempt,
                base_delay=max(base_delay, floor),
                max_delay=max(max_delay, cap),
                jitter_ratio=jitter_ratio,
            )

        _patient_backoff.__name__ = "jittered_backoff"
        cl.jittered_backoff = _patient_backoff
        logger.info("Rate-limit resilience: retry backoff floor=%.0fs cap=%.0fs", floor, cap)
    except Exception as exc:
        logger.warning("Could not extend retry backoff for rate limits: %s", exc)


def _build_initial_prompt(
    server_url: str,
    is_resume: bool,
    toolset_mode: str,
) -> str:
    # The full ARC directive is installed as SOUL.md (system prompt), so this
    # opening user turn only carries runtime context + a kickoff to start acting.
    full_note = _full_toolset_runtime_note() if _normalise_toolset_mode(toolset_mode) == "full" else ""
    runtime_ctx = (
        "Runtime context:\n"
        f"- ARC game server URL (context only — do NOT call it directly): {server_url}\n"
        "- You are in a long-running autonomous session.\n"
        "- Observe and act using only the two MCP tools: get_game_state and take_actions.\n"
        "- Do not wait for follow-up prompts; continue until WIN or external termination.\n"
    )
    lead = "Continue the ARC-AGI-3 session." if is_resume else "Start the ARC-AGI-3 session."
    return f"{lead}\n\n{runtime_ctx}{full_note}"


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
    parser.add_argument("--max-turns",         type=int, default=5000)
    parser.add_argument("--resume-session-id", default="")
    parser.add_argument(
        "--toolset",
        choices=["min", "full"],
        default=os.environ.get("HERMES_TOOLSET", "min"),
    )
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

    # Install the ARC directive as Hermes's SOUL.md (identity / system prompt).
    # Hermes loads HERMES_HOME/SOUL.md into the cache-stable `stable` layer of the
    # system prompt (agent/system_prompt.py build_system_prompt_parts -> load_soul_md),
    # so unlike a first user message it SURVIVES context compaction and is present
    # on every relaunch/resume. Written every launch (idempotent overwrite) so the
    # directive stays in sync; this also pre-empts Hermes seeding its default
    # persona (config.py _ensure_default_soul_md only seeds when SOUL.md is absent).
    directive_text = _read_text(args.directive_path)
    if directive_text.strip():
        soul = directive_text.rstrip()
        # In full toolset mode, append the code-execution directive markdown so the
        # guidance is durable (survives context compaction) and lives next to the
        # main directive rather than inline in code.
        if _normalise_toolset_mode(args.toolset) == "full":
            code_note = _read_text(_FULL_TOOLSET_DIRECTIVE)
            if code_note.strip():
                soul += "\n\n" + code_note.rstrip()
        (hermes_home / "SOUL.md").write_text(soul, encoding="utf-8")

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
    toolset_mode = _normalise_toolset_mode(args.toolset)
    enabled_toolsets = _enabled_toolsets(toolset_mode)

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

    # In full toolset mode, let sandboxed code call the ARC game tools directly.
    if toolset_mode == "full":
        _enable_arc_tools_in_sandbox()

    session_db = SessionDB(hermes_home / "state.db")
    conversation_history = None
    if resume_session_id:
        try:
            conversation_history = session_db.get_messages_as_conversation(resume_session_id)
        except Exception:
            conversation_history = None

    # The directive is the SOUL.md system prompt now (see above), so the opening
    # user turn is just a short kickoff — no need to duplicate ~4.6KB of directive
    # into the conversation on every (re)launch.
    user_message = _build_initial_prompt(
        args.server_url,
        is_resume=bool(resume_session_id and conversation_history),
        toolset_mode=toolset_mode,
    )

    try:
        os.chdir(args.working_dir)
    except OSError as exc:
        logger.warning("Could not chdir to %s: %s", args.working_dir, exc)

    # On hermes-agent main, AIAgent.__init__ does NOT connect MCP servers — the
    # `hermes` CLI bootstrap calls register_mcp_servers() before building the
    # agent. Since we construct AIAgent directly, we must register the configured
    # MCP servers into the tool registry FIRST; otherwise agent.tools is empty and
    # enabled_toolsets=["mcp-arc-agi-3"] filters down to nothing.
    mcp_tool_names: list[str] = []
    try:
        from hermes_cli.config import load_config       # type: ignore[import]
        from tools.mcp_tool import register_mcp_servers  # type: ignore[import]

        mcp_servers = (load_config() or {}).get("mcp_servers") or {}
        if mcp_servers:
            mcp_tool_names = register_mcp_servers(mcp_servers) or []
            logger.info("Registered MCP tools from %s: %s",
                        list(mcp_servers.keys()), mcp_tool_names)
            if not mcp_tool_names:
                _emit({
                    "type": "error",
                    "message": "register_mcp_servers returned no tools — MCP "
                               "connection likely failed (check the proxy is up "
                               "and reachable at the configured url).",
                    "mcp_servers": list(mcp_servers.keys()),
                })
        else:
            _emit({
                "type": "error",
                "message": "No mcp_servers found in config.yaml; the agent will "
                           "have no game tools.",
                "hermes_home": str(hermes_home),
            })
    except Exception as exc:
        import traceback
        _emit({
            "type": "error",
            "message": "MCP server registration failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })

    # Trajectory callbacks: stream the agent's reasoning and each tool call as
    # JSONL events to stdout. The host stream reader writes them to
    # trajectory.jsonl, giving a documented per-step action trajectory.
    #
    # IMPORTANT (hermes-agent main signatures, verified against agent/
    # tool_executor.py): use the fixed-arity callbacks, NOT tool_progress_callback
    # (which main calls with a variable arg count and an event-kind first arg —
    # a 3-arg handler raises TypeError that Hermes silently swallows, so nothing
    # is emitted).
    #   tool_start_callback(tool_id, name, args)
    #   tool_complete_callback(tool_id, name, args, result)
    #   reasoning_callback(reasoning_text)
    _tool_counter = {"n": 0}

    def tool_start_callback(tool_id: Any, name: Any, args: Any) -> None:
        _tool_counter["n"] += 1
        _emit({
            "type": "tool_use",
            "index": _tool_counter["n"],
            "tool_id": str(tool_id),
            "name": _normalize_tool_name(str(name)),
            "raw_tool_name": str(name),
            "arguments": args if isinstance(args, dict) else {},
        })

    def tool_complete_callback(tool_id: Any, name: Any, args: Any, result: Any) -> None:
        tool = _normalize_tool_name(str(name))

        def _emit_preview(limit: int = 5000) -> None:
            preview = result if isinstance(result, str) else str(result)
            if len(preview) > limit:
                preview = preview[:limit] + f"...(+{len(preview) - limit} chars)"
            _emit({
                "type": "tool_result",
                "tool_id": str(tool_id),
                "name": tool,
                "result_preview": preview,
            })

        if tool == "get_game_state":
            # The agent's observation: emit the full, untruncated rendered
            # observation (text + saved image paths, no inline base64) as a
            # dedicated event so the per-step observation is preserved in the trace.
            payload = _payload_from_result(result)
            if payload is None:
                _emit_preview()
                return
            _emit({
                "type": "observation",
                "tool_id": str(tool_id),
                "observe_index": payload.get("observe_index"),
                "state": payload.get("state"),
                "levels_completed": payload.get("levels_completed"),
                "available_actions": payload.get("available_actions"),
                "observations_since_last_query": payload.get(
                    "observations_since_last_query"
                ),
                "current_grid": payload.get("current_grid"),
                "screenshot_files": payload.get("screenshot_files"),
            })
        elif tool == "take_actions":
            # The agent's action batch: emit the structured outcome (what applied
            # vs rejected, resulting state/score, budget) rather than a string
            # preview, so the action log is machine-readable in the trajectory.
            payload = _payload_from_result(result)
            if payload is None:
                _emit_preview()
                return
            _emit({
                "type": "action_result",
                "tool_id": str(tool_id),
                "success": payload.get("success"),
                "applied_count": payload.get("applied_count"),
                "applied_actions": payload.get("applied_actions"),
                "levels_gained": payload.get("levels_gained"),
                "rejected": payload.get("rejected"),
                "state": payload.get("state"),
                "levels_completed": payload.get("levels_completed"),
                "available_actions": payload.get("available_actions"),
                "budget_remaining": payload.get("budget_remaining"),
                "done": payload.get("done"),
                "error": payload.get("error"),
            })
        else:
            # Any other tool (full-toolset code/file/terminal/etc.): capped preview.
            _emit_preview()

    def reasoning_callback(reasoning_text: Any) -> None:
        if isinstance(reasoning_text, str) and reasoning_text.strip():
            _emit({"type": "thinking", "content": reasoning_text.strip()})

    agent = AIAgent(
        base_url=base_url,
        api_key=api_key,
        provider=provider,
        model=model,
        max_iterations=args.max_turns,
        quiet_mode=True,
        session_id=resume_session_id,
        session_db=session_db,
        enabled_toolsets=enabled_toolsets,
        tool_start_callback=tool_start_callback,
        tool_complete_callback=tool_complete_callback,
        reasoning_callback=reasoning_callback,
        pass_session_id=True,
    )

    # Be patient with Gemini 429 / per-minute token throttling: more in-session
    # retries + longer backoff before run_conversation gives up (see helper).
    _extend_rate_limit_resilience(agent)

    agent_tool_names = [
        (t.get("function", {}) if isinstance(t, dict) else {}).get("name", "")
        for t in (agent.tools or [])
    ]
    _emit({
        "type": "system",
        "session_id": agent.session_id,
        "model": agent.model,
        "mcp_servers": ["arc-agi-3"],
        "toolset": toolset_mode,
        "enabled_toolsets": enabled_toolsets,
        "registered_mcp_tools": mcp_tool_names,
        "tools": agent_tool_names,
    })
    if not agent_tool_names:
        _emit({
            "type": "error",
            "message": "agent.tools is empty after MCP registration — the agent "
                       "cannot act on the game. Aborting.",
        })
        return 1
    if not any(name.startswith("mcp_arc_agi_3_") for name in agent_tool_names):
        _emit({
            "type": "error",
            "message": "ARC MCP tools are missing after MCP registration — the agent "
                       "cannot act on the game. Aborting.",
            "tools": agent_tool_names,
        })
        return 1

    start_time = time.time()
    result: dict = {}
    try:
        result = agent.run_conversation(
            user_message=user_message,
            # Directive is installed as SOUL.md (system prompt) in main(), so it is
            # not passed as system_message here.
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
        # Classified reason from conversation_loop (e.g. "rate_limit", "billing").
        # The host uses "rate_limit" to resume rather than count a hard failure.
        "failure_reason": (result.get("failure_reason") if isinstance(result, dict) else None),
        "content": result.get("final_response") if isinstance(result, dict) else None,
        "duration_ms": duration_ms,
    })
    return 1 if is_error else 0


if __name__ == "__main__":
    sys.exit(main())

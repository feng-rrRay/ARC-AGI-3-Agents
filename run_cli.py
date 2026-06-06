"""ARC-AGI-3 Hermes evaluation orchestrator.

Mirrors run_cli.py from sethkarten/continual-harness, adapted for ARC-AGI-3.
Starts one MCP proxy (stateless, shared across games), then loops over the
requested game IDs: per game it starts the ARC game server, launches the
Hermes container, monitors the termination condition, and collects the
per-game scorecard. Results are aggregated into logs/<run_id>/summary.json.

Usage:
    python run_cli.py --game <game_id1>[,<game_id2>,...] [options]

Example:
    python run_cli.py --game my_game --build --model gemini-3.1-pro-preview
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load .env like main.py so ARC_API_KEY / GEMINI_API_KEY are available to this
# orchestrator and inherited by the game-server and MCP-proxy subprocesses.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(dotenv_path=str(_REPO_ROOT / ".env.example"))
load_dotenv(dotenv_path=str(_REPO_ROOT / ".env"), override=True)

from cli_agents.agent_infrastructure.cli_agent_backends import (  # noqa: E402
    CliSessionMetrics,
    HermesCliBackend,
)

logger = logging.getLogger(__name__)

DEFAULT_DIRECTIVE = str(
    _REPO_ROOT / "cli_agents/directives/arc_directive.md"
)
DEFAULT_MCP_PORT_OFFSET = 2  # MCP port = game port + 2
MAX_CONSECUTIVE_FAILURES = 3  # stop relaunching a game after this many failed sessions


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARC-AGI-3 Hermes evaluation")
    parser.add_argument(
        "--game", required=True,
        help="Comma-separated game_id(s) to play (e.g. game1,game2)",
    )
    parser.add_argument("--backend", default="hermes", choices=["hermes"],
                        help="Agent backend (currently only 'hermes')")
    parser.add_argument("--build", action="store_true",
                        help="Build the Docker image before running")
    parser.add_argument("--hermes-commit", default="",
                        help="Optional Hermes git commit to pin (passed to docker build)")
    parser.add_argument("--directive", default=DEFAULT_DIRECTIVE,
                        help="Path to the agent directive file")
    parser.add_argument("--port", type=int, default=8000,
                        help="Base port for the ARC game server (default: 8000)")
    parser.add_argument("--mcp-port", type=int, default=None,
                        help="Port for the MCP proxy (default: --port + 2)")
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument(
        "--toolset",
        choices=["min", "full"],
        default="min",
        help=(
            "Hermes tool exposure: 'min' exposes only mcp-arc-agi-3; "
            "'full' also exposes Hermes built-in skills, memory, session search, "
            "file, terminal, and code execution tools (default: min)"
        ),
    )
    parser.add_argument("--max-actions", type=int, default=5000,
                        help="Per-game ARC action budget (default: 5000)")
    parser.add_argument("--max-turns", type=int, default=5000,
                        help="Max Hermes tool-calling iterations per session "
                             "(maps to AIAgent max_iterations; default: 5000)")
    parser.add_argument("--operation-mode", default="online",
                        choices=["normal", "online", "offline"],
                        help="ARC operation mode (default: online)")
    parser.add_argument("--poll-interval", type=int, default=10,
                        help="Seconds between termination condition polls (default: 10)")
    parser.add_argument("--max-sessions", type=int, default=100,
                        help=("Max Hermes container relaunches per game. Each "
                              "run_conversation returns when the agent stops "
                              "calling tools or hits --max-turns; the agent is "
                              "relaunched (resuming its session) until WIN / "
                              "budget exhaustion / this cap (default: 100)."))
    parser.add_argument("--logs-dir", default="logs",
                        help="Directory for run logs (default: logs/)")
    parser.add_argument("--tags", default="",
                        help="Comma-separated extra scorecard tags")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Service helpers
# ---------------------------------------------------------------------------

@dataclass
class GameServerHandle:
    process: subprocess.Popen
    port: int
    game_id: str
    run_dir: Path


@dataclass
class ProcessStream:
    thread: threading.Thread
    log_file: io.TextIOWrapper
    label: str


def _start_mcp_proxy(port: int, game_port: int, project_root: Path) -> subprocess.Popen:
    """Start the FastMCP proxy server."""
    env = os.environ.copy()
    env["MCP_PORT"]             = str(port)
    env["ARC_GAME_SERVER_URL"]  = f"http://localhost:{game_port}"
    env["PYTHONPATH"]           = str(project_root)
    env["MCP_TRANSPORT"]        = "combined"
    env["PYTHONUNBUFFERED"]     = "1"

    proc = subprocess.Popen(
        [sys.executable, "-m", "cli_agents.server.arc_mcp_server"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    logger.info("MCP proxy started on port %d (pid %d)", port, proc.pid)
    return proc


def _start_game_server(
    game_id: str,
    port: int,
    run_dir: Path,
    tags: str,
    operation_mode: str,
    max_actions: int,
    project_root: Path,
) -> GameServerHandle:
    """Start the per-game FastAPI ARC server."""
    env = os.environ.copy()
    env["OPERATION_MODE"]  = operation_mode
    env["ARC_MAX_ACTIONS"] = str(max_actions)
    env["PYTHONPATH"]      = str(project_root)
    env["PYTHONUNBUFFERED"] = "1"

    # Ensure ARC_API_KEY is present for online mode
    if operation_mode != "offline" and not env.get("ARC_API_KEY"):
        logger.warning(
            "ARC_API_KEY is not set; the game server may fail for online mode."
        )

    recordings_dir = run_dir / "recordings"
    cmd = [
        sys.executable, "-m", "cli_agents.server.app",
        "--game",           game_id,
        "--port",           str(port),
        "--run-dir",        str(run_dir),
        "--tags",           tags,
        "--recordings-dir", str(recordings_dir),
    ]
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    logger.info("Game server started for %s on port %d (pid %d)", game_id, port, proc.pid)
    return GameServerHandle(process=proc, port=port, game_id=game_id, run_dir=run_dir)


def _drain_process_stream(
    stdout_pipe: io.RawIOBase,
    log_file: io.TextIOWrapper,
    label: str,
) -> None:
    """Drain a child stdout pipe into a log file so subprocesses cannot block."""
    try:
        buffered = io.BufferedReader(stdout_pipe)  # type: ignore[arg-type]
        for raw_line in buffered:
            line = raw_line.decode("utf-8", errors="replace")
            log_file.write(line)
            log_file.flush()
    except Exception as exc:
        logger.debug("%s stream reader exited: %s", label, exc)


def _start_process_stream(
    proc: subprocess.Popen,
    label: str,
    log_path: Path,
) -> ProcessStream | None:
    if proc.stdout is None:
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    thread = threading.Thread(
        target=_drain_process_stream,
        args=(proc.stdout, log_file, label),
        daemon=True,
    )
    thread.start()
    return ProcessStream(thread=thread, log_file=log_file, label=label)


def _close_process_stream(stream: ProcessStream | None, timeout: int = 10) -> None:
    if stream is None:
        return
    stream.thread.join(timeout=timeout)
    stream.log_file.close()


def _wait_for_server(url: str, timeout: int = 30) -> bool:
    """Poll GET <url>/health until 200 or timeout."""
    import requests as _req
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = _req.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _check_termination(server_url: str) -> dict[str, Any]:
    import requests as _req
    try:
        r = _req.get(f"{server_url}/termination_condition", timeout=5)
        return r.json()
    except Exception as exc:
        return {"condition_met": False, "error": str(exc)}


def _close_game_scorecard(server_url: str, game_id: str) -> dict[str, Any] | None:
    import requests as _req
    try:
        r = _req.post(f"{server_url}/close_scorecard", timeout=60)
        if r.status_code != 200:
            logger.warning(
                "Game %s: close_scorecard returned HTTP %s: %s",
                game_id, r.status_code, r.text[:300],
            )
            return None
        payload = r.json().get("scorecard")
        if isinstance(payload, dict):
            logger.info(
                "Game %s: scorecard closed score=%s card_id=%s",
                game_id, payload.get("score"), payload.get("card_id"),
            )
            return payload
        logger.warning("Game %s: close_scorecard response missing scorecard", game_id)
        return None
    except Exception as exc:
        logger.warning("Game %s: close_scorecard request failed: %s", game_id, exc)
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _log_progress_if_changed(
    game_id: str,
    tc: dict[str, Any],
    max_actions: int,
    last_progress: tuple[Any, ...] | None,
) -> tuple[Any, ...] | None:
    if tc.get("error"):
        key = ("error", tc.get("error"))
        if key != last_progress:
            logger.warning("Game %s progress unavailable: %s", game_id, tc.get("error"))
        return key

    state = tc.get("state", "?")
    levels = _to_int(tc.get("levels_completed"))
    win_levels = _to_int(tc.get("win_levels"))
    budget = _to_int(tc.get("budget_remaining"))
    actions_used = max_actions - budget if budget is not None else None
    key = (state, levels, win_levels, budget)

    if key != last_progress:
        logger.info(
            "Game %s progress: levels=%s/%s state=%s actions=%s/%s budget=%s",
            game_id,
            levels if levels is not None else "?",
            win_levels if win_levels is not None else "?",
            state,
            actions_used if actions_used is not None else "?",
            max_actions,
            budget if budget is not None else "?",
        )
    return key


def _terminate_process(proc: subprocess.Popen, label: str, timeout: int = 15) -> None:
    if proc.poll() is not None:
        return
    logger.info("Stopping %s (pid %d)…", label, proc.pid)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except Exception:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("%s did not stop gracefully; killing", label)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        proc.wait()


# ---------------------------------------------------------------------------
# Per-game agent loop
# ---------------------------------------------------------------------------

def _run_game(
    game_id: str,
    backend: HermesCliBackend,
    run_id: str,
    run_dir: Path,
    directive_path: str,
    game_port: int,
    mcp_port: int,
    model: str,
    provider: str,
    api_key_env: str,
    max_turns: int,
    poll_interval: int,
    project_root: Path,
    tags: str,
    operation_mode: str,
    max_actions: int,
    max_sessions: int,
    toolset: str,
) -> dict[str, Any]:
    """Run Hermes for one game until WIN / budget exhaustion / caps.

    Hermes's run_conversation returns whenever the model stops calling tools or
    hits --max-turns; that does NOT mean the game is over. So we relaunch the
    container (resuming the same Hermes session, which persists in the mounted
    hermes_memory) in an outer loop, and let the game server's
    /termination_condition (WIN or budget<=0) decide when to actually stop.
    """
    game_dir     = run_dir / game_id
    memory_dir   = game_dir / "hermes_memory"
    scratch_dir  = game_dir / "scratch"
    game_log_dir = game_dir / "logs"
    for d in (memory_dir, scratch_dir, game_log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Write directive and MCP config before starting the container
    directive_text = Path(directive_path).read_text(encoding="utf-8")
    (scratch_dir / ".agent_directive.txt").write_text(directive_text, encoding="utf-8")
    backend.inject_mcp_config(memory_dir, mcp_port)

    server_url = f"http://localhost:{game_port}"
    result: dict[str, Any] = {
        "game_id": game_id, "status": "error",
        "score": None, "scorecard_url": None,
    }

    game_server: GameServerHandle | None = None
    game_server_stream: ProcessStream | None = None
    log_file: io.TextIOWrapper | None = None
    trajectory_file: io.TextIOWrapper | None = None
    active_proc: subprocess.Popen | None = None
    active_stop_event: threading.Event | None = None
    active_stream_thread: threading.Thread | None = None
    active_session_label = ""
    last_progress: tuple[Any, ...] | None = None
    termination_reason = "unknown"
    sessions_run = 0

    try:
        # --- Start game server (once; owns the scorecard for the whole game) ---
        game_server = _start_game_server(
            game_id, game_port, game_dir, tags, operation_mode, max_actions, project_root
        )
        game_server_stream = _start_process_stream(
            game_server.process,
            f"game-server-{game_id}",
            game_log_dir / "game-server.log",
        )
        if not _wait_for_server(f"{server_url}/health", timeout=30):
            logger.error("Game server for %s did not become healthy", game_id)
            result["error"] = "game server failed to start"
            return result

        log_file = open(game_log_dir / "hermes.log", "w", encoding="utf-8")
        trajectory_file = open(game_log_dir / "trajectory.jsonl", "w", encoding="utf-8")

        resume_session_id = ""
        consecutive_failures = 0

        # --- Outer relaunch loop ---
        while True:
            # Stop conditions checked BEFORE (re)launching
            tc = _check_termination(server_url)
            last_progress = _log_progress_if_changed(
                game_id, tc, max_actions, last_progress
            )
            if tc.get("condition_met"):
                termination_reason = "condition_met"
                break
            if game_server.process.poll() is not None:
                termination_reason = "server_died"
                break
            if sessions_run >= max_sessions:
                termination_reason = "max_sessions_reached"
                break

            sessions_run += 1
            banner = (f"\n===== Hermes session {sessions_run} "
                      f"(resume={resume_session_id or 'none'}) =====\n")
            log_file.write(banner)
            log_file.flush()
            logger.info("Game %s: launching Hermes session %d (resume=%s)",
                        game_id, sessions_run, resume_session_id or "none")

            docker_cmd = backend.build_launch_cmd(
                directive_path=directive_path, game_id=game_id,
                hermes_memory_dir=memory_dir, scratch_dir=scratch_dir,
                project_root=project_root, run_id=run_id,
                mcp_port=mcp_port, game_port=game_port,
                model=model, provider=provider, api_key_env=api_key_env,
                max_turns=max_turns, resume_session_id=resume_session_id,
                toolset=toolset,
            )
            logger.debug("docker cmd: %s", " ".join(docker_cmd))

            metrics = CliSessionMetrics()
            stop_event = threading.Event()
            proc = subprocess.Popen(
                docker_cmd, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            active_proc = proc
            active_stop_event = stop_event
            active_session_label = f"hermes-{game_id}-s{sessions_run}"
            stream_thread = threading.Thread(
                target=backend.run_stream_reader,
                args=(proc.stdout, stop_event, log_file, metrics, server_url,
                      trajectory_file),
                daemon=True,
            )
            active_stream_thread = stream_thread
            stream_thread.start()

            # --- Monitor THIS session ---
            killed_for_termination = False
            while proc.poll() is None:
                if game_server.process.poll() is not None:
                    termination_reason = "server_died"
                    stop_event.set()
                    _terminate_process(proc, active_session_label)
                    break
                tc = _check_termination(server_url)
                last_progress = _log_progress_if_changed(
                    game_id, tc, max_actions, last_progress
                )
                if tc.get("condition_met"):
                    logger.info("Game %s: termination met mid-session "
                                "(state=%s levels=%s budget=%s)",
                                game_id, tc.get("state"), tc.get("levels_completed"),
                                tc.get("budget_remaining"))
                    killed_for_termination = True
                    stop_event.set()
                    _terminate_process(proc, active_session_label)
                    break
                time.sleep(poll_interval)

            stop_event.set()
            stream_thread.join(timeout=10)
            rc = proc.poll()
            active_proc = None
            active_stop_event = None
            active_stream_thread = None
            active_session_label = ""
            if metrics.session_id:
                resume_session_id = metrics.session_id

            # Decide whether to relaunch
            if game_server.process.poll() is not None:
                termination_reason = "server_died"
                break
            tc = _check_termination(server_url)
            if killed_for_termination or tc.get("condition_met"):
                termination_reason = "condition_met"
                break
            if metrics.auth_fatal_error:
                termination_reason = "auth_error"
                break
            if metrics.tool_count == 0:
                # MCP tools never loaded; relaunching won't help.
                termination_reason = "no_tools"
                logger.error("Game %s: agent loaded 0 tools — aborting "
                             "(check MCP proxy reachability / config).", game_id)
                break
            if rc not in (0, None):
                consecutive_failures += 1
                logger.warning("Game %s: session %d exited rc=%s (failure %d/%d)",
                               game_id, sessions_run, rc,
                               consecutive_failures, MAX_CONSECUTIVE_FAILURES)
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    termination_reason = "consecutive_failures"
                    break
            else:
                consecutive_failures = 0
                logger.info("Game %s: session %d ended cleanly but game not "
                            "terminal; relaunching (resume).", game_id, sessions_run)
            # loop → relaunch

        logger.info("Game %s finished: reason=%s sessions=%d",
                    game_id, termination_reason, sessions_run)
        result["termination_reason"] = termination_reason
        result["sessions_run"] = sessions_run

    except KeyboardInterrupt:
        termination_reason = "interrupted"
        result["termination_reason"] = termination_reason
        result["sessions_run"] = sessions_run
        result["status"] = "interrupted"
        result["interrupted"] = True
        logger.warning("Game %s interrupted; shutting down cleanly", game_id)

    finally:
        if active_proc is not None and active_proc.poll() is None:
            if active_stop_event is not None:
                active_stop_event.set()
            _terminate_process(active_proc, active_session_label or f"hermes-{game_id}")
        if active_stream_thread is not None:
            active_stream_thread.join(timeout=10)

        # Stop Hermes before closing the game so no in-flight tool call can mutate
        # the environment while the scorecard is being finalized.
        if game_server is not None:
            if game_server.process.poll() is None:
                _close_game_scorecard(server_url, game_id)
                _terminate_process(game_server.process, f"game-server-{game_id}")
            _close_process_stream(game_server_stream)
        if log_file:
            log_file.close()
        if trajectory_file:
            trajectory_file.close()

    # Read scorecard written by the game server on shutdown
    scorecard_path = game_dir / "scorecard.json"
    if scorecard_path.exists():
        try:
            sc = json.loads(scorecard_path.read_text())
            if termination_reason == "condition_met":
                result["status"] = "completed"
            elif termination_reason == "interrupted":
                result["status"] = "interrupted"
            elif termination_reason in {
                "auth_error", "no_tools", "server_died", "consecutive_failures",
            }:
                result["status"] = "error"
            else:
                result["status"] = "stopped"
            result["score"]         = sc.get("score")
            result["scorecard_url"] = sc.get("scorecard_url")
            result["card_id"]       = sc.get("card_id")
        except Exception as exc:
            logger.warning("Could not parse scorecard.json for %s: %s", game_id, exc)
    else:
        logger.warning("No scorecard.json found for %s", game_id)

    return result


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    log_level = logging.INFO
    if os.environ.get("DEBUG") == "True":
        log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    run_id   = f"hermes-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir  = (Path(args.logs_dir) / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run ID: %s  →  %s", run_id, run_dir)

    game_ids = [g.strip() for g in args.game.split(",") if g.strip()]
    if not game_ids:
        logger.error("No game IDs specified")
        sys.exit(1)

    mcp_port  = args.mcp_port or (args.port + DEFAULT_MCP_PORT_OFFSET)
    game_port = args.port

    # Propagate operation-mode to the environment so the game server reads it
    os.environ["OPERATION_MODE"] = args.operation_mode

    backend = HermesCliBackend()
    logger.info("Hermes toolset mode: %s", args.toolset)

    # --- Build Docker image if requested ---
    if args.build:
        backend.build_image(
            _REPO_ROOT,
            hermes_commit=args.hermes_commit,
        )

    # --- Start MCP proxy once (shared across all games) ---
    mcp_proxy: subprocess.Popen | None = None
    mcp_proxy_stream: ProcessStream | None = None
    game_results: list[dict[str, Any]] = []
    interrupted = False
    try:
        mcp_proxy = _start_mcp_proxy(mcp_port, game_port, _REPO_ROOT)  # game_port updated per-game if needed
        mcp_proxy_stream = _start_process_stream(
            mcp_proxy,
            "mcp-proxy",
            run_dir / "mcp-proxy.log",
        )
        # Give the proxy a moment to bind
        time.sleep(2)

        extra_tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        tags_str   = ",".join(["hermes-eval", args.model] + extra_tags)

        # --- Per-game loop ---
        for game_id in game_ids:
            logger.info("=" * 60)
            logger.info("Starting game: %s", game_id)
            logger.info("=" * 60)
            game_result = _run_game(
                game_id       = game_id,
                backend       = backend,
                run_id        = run_id,
                run_dir       = run_dir,
                directive_path= args.directive,
                game_port     = game_port,
                mcp_port      = mcp_port,
                model         = args.model,
                provider      = args.provider,
                api_key_env   = args.api_key_env,
                max_turns     = args.max_turns,
                poll_interval = args.poll_interval,
                project_root  = _REPO_ROOT,
                tags          = tags_str,
                operation_mode= args.operation_mode,
                max_actions   = args.max_actions,
                max_sessions  = args.max_sessions,
                toolset       = args.toolset,
            )
            game_results.append(game_result)
            logger.info(
                "Game %s → status=%s score=%s url=%s",
                game_id,
                game_result.get("status"),
                game_result.get("score"),
                game_result.get("scorecard_url"),
            )
            if game_result.get("interrupted"):
                interrupted = True
                break

    except KeyboardInterrupt:
        interrupted = True
        logger.warning("Run interrupted; writing partial summary")
    finally:
        if mcp_proxy and mcp_proxy.poll() is None:
            _terminate_process(mcp_proxy, "mcp-proxy")
        _close_process_stream(mcp_proxy_stream)

    # --- Write summary ---
    summary = {
        "run_id":       run_id,
        "status":       "interrupted" if interrupted else "completed",
        "games":        game_ids,
        "model":        args.model,
        "provider":     args.provider,
        "toolset":      args.toolset,
        "operation_mode": args.operation_mode,
        "results":      game_results,
        "total_games":  len(game_results),
        "completed":    sum(1 for r in game_results if r.get("status") == "completed"),
        "mean_score": (
            sum(r["score"] for r in game_results if r.get("score") is not None)
            / max(1, sum(1 for r in game_results if r.get("score") is not None))
        ) if any(r.get("score") is not None for r in game_results) else None,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, default=str, indent=2))
    logger.info("Summary written to %s", summary_path)
    logger.info(
        "Run complete: %d/%d games completed, mean_score=%s",
        summary["completed"], summary["total_games"], summary["mean_score"],
    )

    # Print the summary for easy inspection
    print(json.dumps(summary, default=str, indent=2))


if __name__ == "__main__":
    main()

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
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
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
    CliSession,
    CliSessionMetrics,
    HermesCliBackend,
)

logger = logging.getLogger(__name__)

DEFAULT_DIRECTIVE = str(
    _REPO_ROOT / "cli_agents/directives/arc_directive.md"
)
DEFAULT_MCP_PORT_OFFSET = 2  # MCP port = game port + 2


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
    parser.add_argument("--max-actions", type=int, default=80,
                        help="Per-game ARC action budget (default: 80)")
    parser.add_argument("--max-turns", type=int, default=90,
                        help="Max Hermes loop iterations per game (default: 90)")
    parser.add_argument("--operation-mode", default="online",
                        choices=["normal", "online", "offline"],
                        help="ARC operation mode (default: online)")
    parser.add_argument("--poll-interval", type=int, default=10,
                        help="Seconds between termination condition polls (default: 10)")
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


def _start_mcp_proxy(port: int, game_port: int, project_root: Path) -> subprocess.Popen:
    """Start the FastMCP proxy server."""
    env = os.environ.copy()
    env["MCP_PORT"]             = str(port)
    env["ARC_GAME_SERVER_URL"]  = f"http://localhost:{game_port}"
    env["PYTHONPATH"]           = str(project_root)
    env["MCP_TRANSPORT"]        = "combined"

    proc = subprocess.Popen(
        [sys.executable, "-m", "cli_agents.server.arc_mcp_server"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
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

    # Ensure ARC_API_KEY is present for online mode
    if operation_mode != "offline" and not env.get("ARC_API_KEY"):
        logger.warning(
            "ARC_API_KEY is not set; the game server may fail for online mode."
        )

    cmd = [
        sys.executable, "-m", "cli_agents.server.app",
        "--game",    game_id,
        "--port",    str(port),
        "--run-dir", str(run_dir),
        "--tags",    tags,
    ]
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    logger.info("Game server started for %s on port %d (pid %d)", game_id, port, proc.pid)
    return GameServerHandle(process=proc, port=port, game_id=game_id, run_dir=run_dir)


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


def _terminate_process(proc: subprocess.Popen, label: str, timeout: int = 15) -> None:
    if proc.poll() is not None:
        return
    logger.info("Stopping %s (pid %d)…", label, proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("%s did not stop gracefully; killing", label)
        proc.kill()
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
) -> dict[str, Any]:
    """Run Hermes for one game. Returns a result dict."""
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
    cli_session: CliSession | None = None
    log_file: io.TextIOWrapper | None = None

    try:
        # --- Start game server ---
        game_server = _start_game_server(
            game_id, game_port, game_dir, tags, operation_mode, max_actions, project_root
        )
        if not _wait_for_server(f"{server_url}/health", timeout=30):
            logger.error("Game server for %s did not become healthy", game_id)
            result["error"] = "game server failed to start"
            return result

        # --- Launch Hermes container ---
        log_path = game_log_dir / "hermes.log"
        log_file = open(log_path, "w", encoding="utf-8")

        docker_cmd = backend.build_launch_cmd(
            directive_path=directive_path,
            game_id=game_id,
            hermes_memory_dir=memory_dir,
            scratch_dir=scratch_dir,
            project_root=project_root,
            run_id=run_id,
            mcp_port=mcp_port,
            game_port=game_port,
            model=model,
            provider=provider,
            api_key_env=api_key_env,
            max_turns=max_turns,
        )
        logger.info("Launching Hermes for %s", game_id)
        logger.debug("docker cmd: %s", " ".join(docker_cmd))

        metrics = CliSessionMetrics()
        stop_event = threading.Event()

        proc = subprocess.Popen(
            docker_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stream_thread = threading.Thread(
            target=backend.run_stream_reader,
            args=(proc.stdout, stop_event, log_file, metrics, server_url),
            daemon=True,
        )
        stream_thread.start()
        cli_session = CliSession(
            process=proc, stop_event=stop_event,
            stream_thread=stream_thread, metrics=metrics,
        )

        # --- Monitor loop ---
        termination_reason = "container_exited"
        while cli_session.process.poll() is None:
            if game_server.process.poll() is not None:
                logger.error("Game server for %s died unexpectedly", game_id)
                termination_reason = "server_died"
                break
            tc = _check_termination(server_url)
            if tc.get("condition_met"):
                logger.info(
                    "Termination condition met for %s: state=%s levels=%s",
                    game_id, tc.get("state"), tc.get("levels_completed"),
                )
                termination_reason = "condition_met"
                break
            time.sleep(poll_interval)

        logger.info(
            "Game %s finished: reason=%s container_rc=%s",
            game_id, termination_reason,
            cli_session.process.poll(),
        )

        result["termination_reason"] = termination_reason
        result["auth_error"] = metrics.auth_fatal_error
        result["is_error"] = metrics.is_error

    finally:
        # Stop the container
        if cli_session is not None:
            if cli_session.process.poll() is None:
                cli_session.stop_event.set()
                _terminate_process(cli_session.process, f"hermes-container-{game_id}")
            cli_session.stream_thread.join(timeout=5)

        # Stop the game server (triggers scorecard close + scorecard.json write)
        if game_server is not None:
            _terminate_process(game_server.process, f"game-server-{game_id}")

        if log_file:
            log_file.close()

    # Read scorecard written by the game server on shutdown
    scorecard_path = game_dir / "scorecard.json"
    if scorecard_path.exists():
        try:
            sc = json.loads(scorecard_path.read_text())
            result["status"]        = "completed"
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

    run_id   = f"hermes-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    run_dir  = Path(args.logs_dir) / run_id
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

    # --- Build Docker image if requested ---
    if args.build:
        backend.build_image(
            _REPO_ROOT,
            hermes_commit=args.hermes_commit,
        )

    # --- Start MCP proxy once (shared across all games) ---
    mcp_proxy: subprocess.Popen | None = None
    try:
        mcp_proxy = _start_mcp_proxy(mcp_port, game_port, _REPO_ROOT)  # game_port updated per-game if needed
        # Give the proxy a moment to bind
        time.sleep(2)

        extra_tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        tags_str   = ",".join(["hermes-eval", args.model] + extra_tags)

        # --- Per-game loop ---
        game_results: list[dict[str, Any]] = []
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
            )
            game_results.append(game_result)
            logger.info(
                "Game %s → status=%s score=%s url=%s",
                game_id,
                game_result.get("status"),
                game_result.get("score"),
                game_result.get("scorecard_url"),
            )

    finally:
        if mcp_proxy and mcp_proxy.poll() is None:
            _terminate_process(mcp_proxy, "mcp-proxy")

    # --- Write summary ---
    summary = {
        "run_id":       run_id,
        "games":        game_ids,
        "model":        args.model,
        "provider":     args.provider,
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

# ruff: noqa: E402
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.example")
load_dotenv(dotenv_path=".env", override=True)

import argparse
import json
import logging
import os
import signal
import sys
import threading
from functools import partial
from pathlib import Path
from types import FrameType
from typing import Optional

import requests

from agents import AVAILABLE_AGENTS, Swarm
from agents.run_artifacts import (
    RunArtifacts,
    create_run_artifacts,
    export_run_env,
    snapshot_memory,
    write_manifest,
    write_scorecard,
)
from agents.tracing import initialize as init_agentops

logger = logging.getLogger()

SCHEME = os.environ.get("SCHEME", "http")
HOST = os.environ.get("HOST", "localhost")
PORT = os.environ.get("PORT", 8001)

# Hide standard ports in URL
if (SCHEME == "http" and str(PORT) == "80") or (
    SCHEME == "https" and str(PORT) == "443"
):
    ROOT_URL = f"{SCHEME}://{HOST}"
else:
    ROOT_URL = f"{SCHEME}://{HOST}:{PORT}"
HEADERS = {
    "X-API-Key": os.getenv("ARC_API_KEY", ""),
    "Accept": "application/json",
}


def run_agent(
    swarm: Swarm,
    run_artifacts: RunArtifacts,
    bootstrap_memory: Path | None,
) -> None:
    scorecard = swarm.main()
    finalize_run_artifacts(
        run_artifacts,
        swarm=swarm,
        scorecard=scorecard,
        bootstrap_memory=bootstrap_memory,
        status="completed",
    )
    os.kill(os.getpid(), signal.SIGINT)


def finalize_run_artifacts(
    run_artifacts: RunArtifacts,
    *,
    swarm: Swarm,
    scorecard: object | None,
    bootstrap_memory: Path | None,
    status: str,
) -> None:
    card_id = _scorecard_id(scorecard)
    if scorecard is not None:
        write_scorecard(run_artifacts, scorecard)
    if bootstrap_memory is not None:
        snapshot_memory(bootstrap_memory, run_artifacts.memory_final_path)
    write_manifest(
        run_artifacts,
        agent=swarm.agent_name,
        games=swarm.GAMES,
        tags=swarm.tags,
        card_id=card_id,
        status=status,
        bootstrap_memory=bootstrap_memory,
    )


def _scorecard_id(scorecard: object | None) -> str | None:
    if scorecard is None:
        return None
    dump = getattr(scorecard, "model_dump", None)
    payload = dump() if callable(dump) else scorecard
    if isinstance(payload, dict):
        for key in ("card_id", "scorecard_id", "id"):
            value = payload.get(key)
            if value is not None:
                return str(value)
    return None


def cleanup(
    swarm: Swarm,
    run_artifacts: RunArtifacts,
    bootstrap_memory: Path | None,
    signum: Optional[int],
    frame: Optional[FrameType],
) -> None:
    logger.info("Received SIGINT, exiting...")
    card_id = swarm.card_id
    if card_id:
        scorecard = swarm.close_scorecard(card_id)
        if scorecard:
            logger.info("--- EXISTING SCORECARD REPORT ---")
            logger.info(json.dumps(scorecard.model_dump(), indent=2))
            swarm.cleanup(scorecard)
        finalize_run_artifacts(
            run_artifacts,
            swarm=swarm,
            scorecard=scorecard,
            bootstrap_memory=bootstrap_memory,
            status="interrupted",
        )

        # Provide web link to scorecard
        if card_id:
            scorecard_url = f"{ROOT_URL}/scorecards/{card_id}"
            logger.info(f"View your scorecard online: {scorecard_url}")

    sys.exit(0)


def main() -> None:
    log_level = logging.INFO
    if os.environ.get("DEBUG", "False") == "True":
        log_level = logging.DEBUG

    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # logging.getLogger("requests").setLevel(logging.CRITICAL)
    # logging.getLogger("werkzeug").setLevel(logging.CRITICAL)

    parser = argparse.ArgumentParser(description="ARC-AGI-3-Agents")
    parser.add_argument(
        "-a",
        "--agent",
        choices=AVAILABLE_AGENTS.keys(),
        help="Choose which agent to run.",
    )
    parser.add_argument(
        "-g",
        "--game",
        help="Choose a specific game_id for the agent to play. If none specified, an agent swarm will play all available games.",
    )
    parser.add_argument(
        "-t",
        "--tags",
        type=str,
        help="Comma-separated list of tags for the scorecard (e.g., 'experiment,v1.0')",
        default=None,
    )
    parser.add_argument(
        "--bootstrap-memory",
        type=str,
        default=None,
        help=(
            "Optional JSON file backing ContinualHarness long-term memory. "
            "When provided, the agent loads it on start and writes mutations "
            "atomically on every change. When omitted, memory is disabled "
            "(no process_memory tool, no LONG-TERM MEMORY block)."
        ),
    )

    args = parser.parse_args()

    run_artifacts = create_run_artifacts(args.agent or "no-agent")
    export_run_env(run_artifacts)

    bootstrap_memory = (
        Path(args.bootstrap_memory).resolve() if args.bootstrap_memory else None
    )
    # Memory persistence is opt-in: agents read this env var when constructing their store.
    if bootstrap_memory is not None:
        os.environ["CONTINUAL_HARNESS_BOOTSTRAP_MEMORY"] = str(bootstrap_memory)
        snapshot_memory(bootstrap_memory, run_artifacts.memory_initial_path)

    file_handler = logging.FileHandler(run_artifacts.log_path, mode="w")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging this run to {run_artifacts.log_path}")

    if not args.agent:
        logger.error("An Agent must be specified")
        write_manifest(
            run_artifacts,
            agent=None,
            games=[],
            tags=[],
            status="error",
            bootstrap_memory=bootstrap_memory,
        )
        return

    # Start with Empty tags, "agent" and agent name will be added by the Swarm later
    tags: list[str] = []

    # Append user-provided tags if any
    if args.tags:
        user_tags = [tag.strip() for tag in args.tags.split(",")]
        tags.extend(user_tags)

    print(f"{ROOT_URL}/api/games")

    # Get the list of games from the API
    full_games = []
    try:
        with requests.Session() as session:
            session.headers.update(HEADERS)
            r = session.get(f"{ROOT_URL}/api/games", timeout=10)

        if r.status_code == 200:
            try:
                full_games = [g["game_id"] for g in r.json()]
            except (ValueError, KeyError) as e:
                logger.error(f"Failed to parse games response: {e}")
                logger.error(f"Response content: {r.text[:200]}")
        else:
            logger.error(
                f"API request failed with status {r.status_code}: {r.text[:200]}"
            )

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to API server: {e}")

    # For playback agents, we can derive the game from the recording filename
    if not full_games and args.agent and args.agent.endswith(".recording.jsonl"):
        from agents.recorder import Recorder

        game_prefix = Recorder.get_prefix_one(args.agent)
        full_games = [game_prefix]
        logger.info(
            f"Using game '{game_prefix}' derived from playback recording filename"
        )
    games = full_games[:]
    if args.game:
        filters = args.game.split(",")
        games = [
            gid
            for gid in full_games
            if any(gid.startswith(prefix) for prefix in filters)
        ]

    logger.info(f"Game list: {games}")

    if not games:
        if full_games:
            logger.error(
                f"The specified game '{args.game}' does not exist or is not available with your API key. Please try a different game."
            )
        else:
            logger.error(
                "No games available to play. Check API connection or recording file."
            )
        write_manifest(
            run_artifacts,
            agent=args.agent,
            games=games,
            tags=tags,
            status="error",
            bootstrap_memory=bootstrap_memory,
        )
        return

    # Initialize AgentOps client
    init_agentops(api_key=os.getenv("AGENTOPS_API_KEY"), log_level=log_level)

    swarm = Swarm(
        args.agent,
        ROOT_URL,
        games,
        tags=tags,  # Pass tags as keyword argument
    )
    write_manifest(
        run_artifacts,
        agent=args.agent,
        games=games,
        tags=swarm.tags,
        status="running",
        bootstrap_memory=bootstrap_memory,
    )
    agent_thread = threading.Thread(
        target=partial(run_agent, swarm, run_artifacts, bootstrap_memory)
    )
    agent_thread.daemon = True  # die when the main thread dies
    signal.signal(
        signal.SIGINT,
        partial(cleanup, swarm, run_artifacts, bootstrap_memory),
    )  # handler for Ctrl+C
    agent_thread.start()

    try:
        # Wait for the agent thread to complete
        while agent_thread.is_alive():
            agent_thread.join(timeout=5)  # Check every 5 second
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received in main thread")
        cleanup(swarm, run_artifacts, bootstrap_memory, signal.SIGINT, None)
    except Exception as e:
        logger.error(f"Unexpected error in main thread: {e}")
        cleanup(swarm, run_artifacts, bootstrap_memory, None, None)


if __name__ == "__main__":
    os.environ["TESTING"] = "False"
    main()

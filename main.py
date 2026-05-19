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
    snapshot_skills,
    snapshot_subagents,
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
    bootstrap_skills: Path | None,
    bootstrap_subagents: Path | None,
) -> None:
    scorecard = swarm.main()
    finalize_run_artifacts(
        run_artifacts,
        swarm=swarm,
        scorecard=scorecard,
        bootstrap_memory=bootstrap_memory,
        bootstrap_skills=bootstrap_skills,
        bootstrap_subagents=bootstrap_subagents,
        status="completed",
    )
    os.kill(os.getpid(), signal.SIGINT)


def finalize_run_artifacts(
    run_artifacts: RunArtifacts,
    *,
    swarm: Swarm,
    scorecard: object | None,
    bootstrap_memory: Path | None,
    bootstrap_skills: Path | None,
    bootstrap_subagents: Path | None,
    status: str,
) -> None:
    card_id = _scorecard_id(scorecard)
    if scorecard is not None:
        write_scorecard(run_artifacts, scorecard)
    memory_source = bootstrap_memory or run_artifacts.memory_path
    snapshot_memory(memory_source, run_artifacts.memory_final_path)
    skills_source = bootstrap_skills or run_artifacts.skills_path
    snapshot_skills(skills_source, run_artifacts.skills_final_path)
    subagents_source = bootstrap_subagents or run_artifacts.subagents_path
    snapshot_subagents(subagents_source, run_artifacts.subagents_final_path)
    write_manifest(
        run_artifacts,
        agent=swarm.agent_name,
        games=swarm.GAMES,
        tags=swarm.tags,
        card_id=card_id,
        status=status,
        bootstrap_memory=bootstrap_memory,
        bootstrap_skills=bootstrap_skills,
        bootstrap_subagents=bootstrap_subagents,
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
    bootstrap_skills: Path | None,
    bootstrap_subagents: Path | None,
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
            bootstrap_skills=bootstrap_skills,
            bootstrap_subagents=bootstrap_subagents,
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
            "atomically on every change. When omitted, memory still runs with "
            "run-local backing at logs/<run_id>/memory.json."
        ),
    )
    parser.add_argument(
        "--bootstrap-skills",
        type=str,
        default=None,
        help=(
            "Optional JSON file backing the ContinualHarness skill registry. "
            "When provided, skills load from and write back to this file "
            "(cross-run persistence). When omitted, skills live in "
            "logs/<run_id>/skills.json. process_skill, run_skill, and run_code "
            "are ALWAYS available; this flag only changes the backing file."
        ),
    )
    parser.add_argument(
        "--bootstrap-subagents",
        type=str,
        default=None,
        help=(
            "Optional JSON file backing the ContinualHarness subagent "
            "registry. When provided, subagents load from and write back to "
            "this file (cross-run persistence). When omitted, subagents live "
            "in logs/<run_id>/subagents.json. process_subagent and "
            "run_subagent are ALWAYS available; this flag only changes the "
            "backing file."
        ),
    )

    args = parser.parse_args()

    run_artifacts = create_run_artifacts(args.agent or "no-agent")
    export_run_env(run_artifacts)

    bootstrap_memory_raw = args.bootstrap_memory or os.getenv(
        "CONTINUAL_HARNESS_BOOTSTRAP_MEMORY"
    )
    bootstrap_memory = (
        Path(bootstrap_memory_raw).resolve() if bootstrap_memory_raw else None
    )
    if bootstrap_memory is not None:
        os.environ["CONTINUAL_HARNESS_BOOTSTRAP_MEMORY"] = str(bootstrap_memory)
    memory_source = bootstrap_memory or run_artifacts.memory_path
    snapshot_memory(memory_source, run_artifacts.memory_initial_path)

    bootstrap_skills_raw = args.bootstrap_skills or os.getenv(
        "CONTINUAL_HARNESS_BOOTSTRAP_SKILLS"
    )
    bootstrap_skills = (
        Path(bootstrap_skills_raw).resolve() if bootstrap_skills_raw else None
    )
    if bootstrap_skills is not None:
        os.environ["CONTINUAL_HARNESS_BOOTSTRAP_SKILLS"] = str(bootstrap_skills)
    skills_source = bootstrap_skills or run_artifacts.skills_path
    snapshot_skills(skills_source, run_artifacts.skills_initial_path)

    bootstrap_subagents_raw = args.bootstrap_subagents or os.getenv(
        "CONTINUAL_HARNESS_BOOTSTRAP_SUBAGENTS"
    )
    bootstrap_subagents = (
        Path(bootstrap_subagents_raw).resolve() if bootstrap_subagents_raw else None
    )
    if bootstrap_subagents is not None:
        os.environ["CONTINUAL_HARNESS_BOOTSTRAP_SUBAGENTS"] = str(bootstrap_subagents)
    subagents_source = bootstrap_subagents or run_artifacts.subagents_path
    snapshot_subagents(subagents_source, run_artifacts.subagents_initial_path)

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
            bootstrap_skills=bootstrap_skills,
            bootstrap_subagents=bootstrap_subagents,
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
            bootstrap_skills=bootstrap_skills,
            bootstrap_subagents=bootstrap_subagents,
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
        bootstrap_skills=bootstrap_skills,
        bootstrap_subagents=bootstrap_subagents,
    )
    agent_thread = threading.Thread(
        target=partial(
            run_agent,
            swarm,
            run_artifacts,
            bootstrap_memory,
            bootstrap_skills,
            bootstrap_subagents,
        )
    )
    agent_thread.daemon = True  # die when the main thread dies
    signal.signal(
        signal.SIGINT,
        partial(
            cleanup,
            swarm,
            run_artifacts,
            bootstrap_memory,
            bootstrap_skills,
            bootstrap_subagents,
        ),
    )  # handler for Ctrl+C
    agent_thread.start()

    try:
        # Wait for the agent thread to complete
        while agent_thread.is_alive():
            agent_thread.join(timeout=5)  # Check every 5 second
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received in main thread")
        cleanup(
            swarm,
            run_artifacts,
            bootstrap_memory,
            bootstrap_skills,
            bootstrap_subagents,
            signal.SIGINT,
            None,
        )
    except Exception as e:
        logger.error(f"Unexpected error in main thread: {e}")
        cleanup(
            swarm,
            run_artifacts,
            bootstrap_memory,
            bootstrap_skills,
            bootstrap_subagents,
            None,
            None,
        )


if __name__ == "__main__":
    os.environ["TESTING"] = "False"
    main()

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

RUN_ID_ENV = "RUN_ID"
RUN_DIR_ENV = "RUN_DIR"
RUN_LOG_PATH_ENV = "RUN_LOG_PATH"
RUN_RECORDINGS_DIR_ENV = "RUN_RECORDINGS_DIR"
RUN_ARTIFACTS_DIR_ENV = "RUN_ARTIFACTS_DIR"
RUN_MEMORY_PATH_ENV = "RUN_MEMORY_PATH"
RUN_SKILLS_PATH_ENV = "RUN_SKILLS_PATH"
RUN_SUBAGENTS_PATH_ENV = "RUN_SUBAGENTS_PATH"
RUN_PROMPT_PATH_ENV = "RUN_PROMPT_PATH"
RUN_PROMPT_EVOLUTION_PATH_ENV = "RUN_PROMPT_EVOLUTION_PATH"

EMPTY_MEMORY_STATE: dict[str, Any] = {"next_id": 1, "entries": []}
EMPTY_SKILLS_STATE: dict[str, Any] = {"next_id": 1, "entries": []}
EMPTY_SUBAGENTS_STATE: dict[str, Any] = {"next_id": 1, "entries": []}
_SAFE_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    log_path: Path
    recordings_dir: Path
    artifacts_dir: Path
    manifest_path: Path

    @property
    def scorecard_path(self) -> Path:
        return self.run_dir / "scorecard.final.json"

    @property
    def memory_initial_path(self) -> Path:
        return self.run_dir / "memory.initial.json"

    @property
    def memory_final_path(self) -> Path:
        return self.run_dir / "memory.final.json"

    @property
    def memory_path(self) -> Path:
        return self.run_dir / "memory.json"

    @property
    def skills_initial_path(self) -> Path:
        return self.run_dir / "skills.initial.json"

    @property
    def skills_final_path(self) -> Path:
        return self.run_dir / "skills.final.json"

    @property
    def skills_path(self) -> Path:
        return self.run_dir / "skills.json"

    @property
    def subagents_initial_path(self) -> Path:
        return self.run_dir / "subagents.initial.json"

    @property
    def subagents_final_path(self) -> Path:
        return self.run_dir / "subagents.final.json"

    @property
    def subagents_path(self) -> Path:
        return self.run_dir / "subagents.json"

    @property
    def prompt_initial_path(self) -> Path:
        return self.run_dir / "prompt.initial.md"

    @property
    def prompt_final_path(self) -> Path:
        return self.run_dir / "prompt.final.md"

    @property
    def prompt_path(self) -> Path:
        return self.run_dir / "prompt.current.md"

    @property
    def prompt_evolution_path(self) -> Path:
        return self.run_dir / "prompt_evolution.jsonl"


@dataclass(frozen=True)
class GameArtifacts:
    """Per-game artifact layout inside a run directory.

    Each game in a Swarm gets its own subfolder so the ContinualHarness stores
    (memory/skills/subagents/prompt) and per-game logs (trace/trajectory/
    recordings) never collide across games running in parallel threads.
    """

    game_id: str
    game_dir: Path

    @property
    def memory_path(self) -> Path:
        return self.game_dir / "memory.json"

    @property
    def memory_initial_path(self) -> Path:
        return self.game_dir / "memory.initial.json"

    @property
    def memory_final_path(self) -> Path:
        return self.game_dir / "memory.final.json"

    @property
    def skills_path(self) -> Path:
        return self.game_dir / "skills.json"

    @property
    def skills_initial_path(self) -> Path:
        return self.game_dir / "skills.initial.json"

    @property
    def skills_final_path(self) -> Path:
        return self.game_dir / "skills.final.json"

    @property
    def subagents_path(self) -> Path:
        return self.game_dir / "subagents.json"

    @property
    def subagents_initial_path(self) -> Path:
        return self.game_dir / "subagents.initial.json"

    @property
    def subagents_final_path(self) -> Path:
        return self.game_dir / "subagents.final.json"

    @property
    def prompt_path(self) -> Path:
        return self.game_dir / "prompt.current.md"

    @property
    def prompt_initial_path(self) -> Path:
        return self.game_dir / "prompt.initial.md"

    @property
    def prompt_final_path(self) -> Path:
        return self.game_dir / "prompt.final.md"

    @property
    def prompt_evolution_path(self) -> Path:
        return self.game_dir / "prompt_evolution.jsonl"

    @property
    def trace_path(self) -> Path:
        return self.game_dir / "trace.jsonl"

    @property
    def trajectory_path(self) -> Path:
        return self.game_dir / "trajectory.jsonl"

    @property
    def recordings_dir(self) -> Path:
        return self.game_dir / "recordings"


def game_artifacts(run_dir: str | Path, game_id: str) -> GameArtifacts:
    """Resolve the per-game artifact layout for `game_id` under `run_dir`."""
    return GameArtifacts(
        game_id=game_id, game_dir=Path(run_dir) / safe_slug(game_id)
    )


# Filenames in a --bootstrap source mapped to the GameArtifacts attribute they
# seed. Only these four files are restored; trace/trajectory/recordings and the
# prompt_evolution audit log always start fresh per run.
_BOOTSTRAP_DEST = {
    "memory.json": "memory_path",
    "skills.json": "skills_path",
    "subagents.json": "subagents_path",
    "prompt.current.md": "prompt_path",
}


def seed_game_from_bootstrap(bootstrap: str | Path, ga: GameArtifacts) -> list[str]:
    """Copy the four state files from a .zip or directory into a game folder.

    Missing files are skipped (the agent starts that store empty / at baseline).
    A zip that wraps everything in a single top-level folder is unwrapped.
    Returns the list of filenames actually seeded.
    """
    bootstrap = Path(bootstrap)
    ga.game_dir.mkdir(parents=True, exist_ok=True)
    seeded: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        if zipfile.is_zipfile(bootstrap):
            with zipfile.ZipFile(bootstrap) as zf:
                zf.extractall(tmp)
            src = Path(tmp)
            inner = list(src.iterdir())
            if len(inner) == 1 and inner[0].is_dir():
                src = inner[0]
        elif bootstrap.is_dir():
            src = bootstrap
        else:
            raise ValueError(f"--bootstrap must be a .zip or directory: {bootstrap}")
        for name, attr in _BOOTSTRAP_DEST.items():
            f = src / name
            if f.exists():
                shutil.copyfile(f, getattr(ga, attr))
                seeded.append(name)
    return seeded


def safe_slug(value: str) -> str:
    """Return a filesystem-friendly slug while preserving useful dots."""
    slug = _SAFE_SLUG_RE.sub("-", value.strip()).strip(".-")
    return slug or "run"


def create_run_artifacts(
    agent_name: str | None,
    *,
    game: str | None = None,
    logs_dir: str | Path = "logs",
    timestamp: str | None = None,
) -> RunArtifacts:
    """Create and return the per-invocation artifact directory."""
    root = Path(logs_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [safe_slug(agent_name or "no-agent")]
    if game:
        parts.append(safe_slug(game))
    parts.append(stamp)
    base_run_id = "-".join(parts)
    run_id, run_dir = _unique_run_dir(root, base_run_id)
    recordings_dir = run_dir / "recordings"
    artifacts_dir = run_dir / "artifacts"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        log_path=run_dir / "run.log",
        recordings_dir=recordings_dir,
        artifacts_dir=artifacts_dir,
        manifest_path=run_dir / "manifest.json",
    )


def export_run_env(artifacts: RunArtifacts) -> None:
    """Expose run paths to recorders and agent-local artifact writers."""
    os.environ[RUN_ID_ENV] = artifacts.run_id
    os.environ[RUN_DIR_ENV] = str(artifacts.run_dir)
    os.environ[RUN_LOG_PATH_ENV] = str(artifacts.log_path)
    os.environ[RUN_RECORDINGS_DIR_ENV] = str(artifacts.recordings_dir)
    os.environ[RUN_ARTIFACTS_DIR_ENV] = str(artifacts.artifacts_dir)
    os.environ[RUN_MEMORY_PATH_ENV] = str(artifacts.memory_path)
    os.environ[RUN_SKILLS_PATH_ENV] = str(artifacts.skills_path)
    os.environ[RUN_SUBAGENTS_PATH_ENV] = str(artifacts.subagents_path)
    os.environ[RUN_PROMPT_PATH_ENV] = str(artifacts.prompt_path)
    os.environ[RUN_PROMPT_EVOLUTION_PATH_ENV] = str(artifacts.prompt_evolution_path)


def write_manifest(
    artifacts: RunArtifacts,
    *,
    agent: str | None,
    games: list[str],
    tags: list[str],
    status: str,
    card_id: str | None = None,
    bootstrap: str | Path | None = None,
    prompt_evolve_frequency: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the current run manifest atomically.

    `paths.games` maps each game_id to its per-game artifact folder and the
    files inside it — the ContinualHarness stores are isolated per game.
    """
    per_game: dict[str, dict[str, str]] = {}
    for g in games:
        ga = game_artifacts(artifacts.run_dir, g)
        per_game[g] = {
            "dir": str(ga.game_dir),
            "memory": str(ga.memory_path),
            "skills": str(ga.skills_path),
            "subagents": str(ga.subagents_path),
            "prompt": str(ga.prompt_path),
            "prompt_evolution": str(ga.prompt_evolution_path),
            "trace": str(ga.trace_path),
            "trajectory": str(ga.trajectory_path),
            "recordings": str(ga.recordings_dir),
        }
    payload: dict[str, Any] = {
        "run_id": artifacts.run_id,
        "status": status,
        "agent": agent,
        "games": games,
        "tags": tags,
        "card_id": card_id,
        "bootstrap": str(bootstrap) if bootstrap else None,
        "prompt_evolve_frequency": prompt_evolve_frequency,
        "paths": {
            "run_dir": str(artifacts.run_dir),
            "log": str(artifacts.log_path),
            "scorecard": str(artifacts.scorecard_path),
            "games": per_game,
        },
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    return write_json_atomic(artifacts.manifest_path, payload)


def write_scorecard(artifacts: RunArtifacts, scorecard: Any) -> Path:
    """Write the final scorecard payload."""
    dump = getattr(scorecard, "model_dump", None)
    payload = dump() if callable(dump) else scorecard
    return write_json_atomic(artifacts.scorecard_path, payload)


def snapshot_json_file(
    source: str | Path,
    destination: Path,
    *,
    empty_state: dict[str, Any],
) -> Path:
    """Copy a JSON file to a run snapshot, or write `empty_state` if absent."""
    source_path = Path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_path.exists():
        shutil.copyfile(source_path, destination)
        return destination
    return write_json_atomic(destination, empty_state)


def snapshot_memory(source: str | Path, destination: Path) -> Path:
    """Thin wrapper around snapshot_json_file with the memory empty state."""
    return snapshot_json_file(source, destination, empty_state=EMPTY_MEMORY_STATE)


def snapshot_skills(source: str | Path, destination: Path) -> Path:
    """Thin wrapper around snapshot_json_file with the skills empty state."""
    return snapshot_json_file(source, destination, empty_state=EMPTY_SKILLS_STATE)


def snapshot_subagents(source: str | Path, destination: Path) -> Path:
    """Thin wrapper around snapshot_json_file with the subagents empty state."""
    return snapshot_json_file(source, destination, empty_state=EMPTY_SUBAGENTS_STATE)


def snapshot_prompt(source: str | Path, destination: Path, *, baseline: str) -> Path:
    """Copy the source prompt markdown to a run snapshot, or write `baseline`.

    Mirrors snapshot_memory/skills/subagents but for plain markdown — the
    prompt file is not JSON.
    """
    source_path = Path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_path.exists():
        shutil.copyfile(source_path, destination)
        return destination
    destination.write_text(baseline, encoding="utf-8")
    return destination


def write_json_atomic(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)
        file.write("\n")
    tmp.replace(path)
    return path


def _unique_run_dir(root: Path, base_run_id: str) -> tuple[str, Path]:
    run_id = base_run_id
    run_dir = root / run_id
    counter = 2
    while run_dir.exists():
        run_id = f"{base_run_id}-{counter}"
        run_dir = root / run_id
        counter += 1
    run_dir.mkdir(parents=True)
    return run_id, run_dir

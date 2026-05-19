from __future__ import annotations

import json
import os
import re
import shutil
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


def safe_slug(value: str) -> str:
    """Return a filesystem-friendly slug while preserving useful dots."""
    slug = _SAFE_SLUG_RE.sub("-", value.strip()).strip(".-")
    return slug or "run"


def create_run_artifacts(
    agent_name: str | None,
    *,
    logs_dir: str | Path = "logs",
    timestamp: str | None = None,
) -> RunArtifacts:
    """Create and return the per-invocation artifact directory."""
    root = Path(logs_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    base_run_id = f"{safe_slug(agent_name or 'no-agent')}-{stamp}"
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


def write_manifest(
    artifacts: RunArtifacts,
    *,
    agent: str | None,
    games: list[str],
    tags: list[str],
    status: str,
    card_id: str | None = None,
    bootstrap_memory: str | Path | None = None,
    bootstrap_skills: str | Path | None = None,
    bootstrap_subagents: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the current run manifest atomically."""
    payload: dict[str, Any] = {
        "run_id": artifacts.run_id,
        "status": status,
        "agent": agent,
        "games": games,
        "tags": tags,
        "card_id": card_id,
        "bootstrap_memory": str(bootstrap_memory) if bootstrap_memory else None,
        "bootstrap_skills": str(bootstrap_skills) if bootstrap_skills else None,
        "bootstrap_subagents": (
            str(bootstrap_subagents) if bootstrap_subagents else None
        ),
        "paths": {
            "run_dir": str(artifacts.run_dir),
            "log": str(artifacts.log_path),
            "recordings": str(artifacts.recordings_dir),
            "artifacts": str(artifacts.artifacts_dir),
            "memory": str(bootstrap_memory or artifacts.memory_path),
            "memory_initial": str(artifacts.memory_initial_path),
            "memory_final": str(artifacts.memory_final_path),
            "skills": str(bootstrap_skills or artifacts.skills_path),
            "skills_initial": str(artifacts.skills_initial_path),
            "skills_final": str(artifacts.skills_final_path),
            "subagents": str(bootstrap_subagents or artifacts.subagents_path),
            "subagents_initial": str(artifacts.subagents_initial_path),
            "subagents_final": str(artifacts.subagents_final_path),
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

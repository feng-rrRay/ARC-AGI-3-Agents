import json
import os
from pathlib import Path

import pytest

from agents.run_artifacts import (
    RUN_ARTIFACTS_DIR_ENV,
    RUN_DIR_ENV,
    RUN_ID_ENV,
    RUN_LOG_PATH_ENV,
    RUN_MEMORY_PATH_ENV,
    RUN_RECORDINGS_DIR_ENV,
    create_run_artifacts,
    export_run_env,
    snapshot_memory,
    write_manifest,
    write_scorecard,
)


@pytest.mark.unit
def test_create_run_artifacts_builds_expected_layout(tmp_path: Path) -> None:
    artifacts = create_run_artifacts(
        "agent/name:with spaces",
        logs_dir=tmp_path / "logs",
        timestamp="20260518-120000",
    )

    assert artifacts.run_id == "agent-name-with-spaces-20260518-120000"
    assert artifacts.run_dir.is_dir()
    assert artifacts.log_path == artifacts.run_dir / "run.log"
    assert artifacts.recordings_dir.is_dir()
    assert artifacts.artifacts_dir.is_dir()


@pytest.mark.unit
def test_export_run_env_sets_internal_run_paths(tmp_path: Path) -> None:
    artifacts = create_run_artifacts("agent", logs_dir=tmp_path / "logs")
    keys = [
        RUN_ID_ENV,
        RUN_DIR_ENV,
        RUN_LOG_PATH_ENV,
        RUN_RECORDINGS_DIR_ENV,
        RUN_ARTIFACTS_DIR_ENV,
        RUN_MEMORY_PATH_ENV,
    ]
    original = {key: os.environ.get(key) for key in keys}

    try:
        export_run_env(artifacts)

        assert os.environ[RUN_ID_ENV] == artifacts.run_id
        assert os.environ[RUN_DIR_ENV] == str(artifacts.run_dir)
        assert os.environ[RUN_LOG_PATH_ENV] == str(artifacts.log_path)
        assert os.environ[RUN_RECORDINGS_DIR_ENV] == str(artifacts.recordings_dir)
        assert os.environ[RUN_ARTIFACTS_DIR_ENV] == str(artifacts.artifacts_dir)
        assert os.environ[RUN_MEMORY_PATH_ENV] == str(artifacts.memory_path)
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.mark.unit
def test_manifest_and_scorecard_are_written(tmp_path: Path) -> None:
    artifacts = create_run_artifacts("agent", logs_dir=tmp_path / "logs")

    write_manifest(
        artifacts,
        agent="agent",
        games=["game-1"],
        tags=["tag"],
        status="running",
        card_id="card-1",
        bootstrap_memory=tmp_path / "memory.json",
    )
    write_scorecard(artifacts, {"score": 1})

    manifest = json.loads(artifacts.manifest_path.read_text())
    scorecard = json.loads(artifacts.scorecard_path.read_text())
    assert manifest["run_id"] == artifacts.run_id
    assert manifest["status"] == "running"
    assert manifest["paths"]["recordings"] == str(artifacts.recordings_dir)
    assert manifest["paths"]["memory"] == str(tmp_path / "memory.json")
    assert manifest["paths"]["memory_initial"] == str(artifacts.memory_initial_path)
    assert manifest["paths"]["memory_final"] == str(artifacts.memory_final_path)
    assert scorecard == {"score": 1}


@pytest.mark.unit
def test_snapshot_memory_copies_existing_or_writes_empty(tmp_path: Path) -> None:
    artifacts = create_run_artifacts("agent", logs_dir=tmp_path / "logs")
    source = tmp_path / "memory.json"
    source.write_text('{"next_id": 2, "entries": [{"id": "mem_001"}]}')

    snapshot_memory(source, artifacts.memory_initial_path)
    snapshot_memory(tmp_path / "missing.json", artifacts.memory_final_path)

    assert json.loads(artifacts.memory_initial_path.read_text())["next_id"] == 2
    assert json.loads(artifacts.memory_final_path.read_text()) == {
        "next_id": 1,
        "entries": [],
    }

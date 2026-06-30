from __future__ import annotations

import builtins
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .run_artifacts import RUN_RECORDINGS_DIR_ENV

RECORDING_SUFFIX = ".recording.jsonl"


def get_recordings_dir() -> str:
    """Get the current recordings directory from environment variable."""
    return os.environ.get("RECORDINGS_DIR", "")


class Recorder:
    def __init__(
        self,
        prefix: str,
        filename: Optional[str] = None,
        guid: Optional[str] = None,
        directory: Optional[str] = None,
    ) -> None:
        self.guid = self.get_guid(filename) if filename else (guid or str(uuid.uuid4()))
        self.prefix: str = prefix
        if filename:
            path = self._resolve_existing_recording_path(filename)
        else:
            recordings_dir = (
                directory
                or os.environ.get(RUN_RECORDINGS_DIR_ENV)
                or get_recordings_dir()
            )
            basename = f"{self.prefix}.{self.guid}{RECORDING_SUFFIX}"
            path = Path(recordings_dir) / basename if recordings_dir else Path(basename)

        path.parent.mkdir(parents=True, exist_ok=True)
        self.filename = str(path)

    def record(self, data: dict[str, Any]) -> None:
        """
        Records an event to the file.
        `data` should be a dictionary (JSON-serializable) or a JSON string.
        """
        event: dict[str, Any] = {}
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        event["data"] = data

        with open(self.filename, "a", encoding="utf-8") as f:
            json.dump(event, f)
            f.write("\n")

    def get(self) -> list[dict[str, Any]]:
        """
        Loads all recorded events and returns them as a list of dictionaries.
        """
        if not os.path.isfile(self.filename):
            return []

        events: list[dict[str, Any]] = []
        with open(self.filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    def __repr__(self) -> str:
        return f"<Recorder guid={self.guid} file={self.filename}>"

    @classmethod
    def list(cls) -> list[str]:
        recordings: list[str] = []
        recordings_dir = get_recordings_dir()
        if recordings_dir:
            root = Path(recordings_dir)
            root.mkdir(parents=True, exist_ok=True)
            recordings.extend(
                path.name for path in sorted(root.glob(f"*{RECORDING_SUFFIX}"))
            )

        run_recordings_dir = os.environ.get(RUN_RECORDINGS_DIR_ENV)
        if run_recordings_dir:
            recordings.extend(cls._relative_recording_paths(Path(run_recordings_dir)))

        recordings.extend(cls._relative_recording_paths(Path("logs")))
        return list(dict.fromkeys(recordings))

    @classmethod
    def get_prefix(cls, filename: str) -> str:
        """
        Example filename: locksmith.random.50.81329339-1951-487c-8bed-e9d4780320f2.recording.jsonl
        Returns: locksmith.random.50
        """
        filename = Path(filename).name
        if "." in filename:
            parts = filename.split(".")
            return ".".join(parts[:-3])
        else:
            return filename

    @classmethod
    def get_prefix_one(cls, filename: str) -> str:
        """
        Example filename: locksmith.random.50.81329339-1951-487c-8bed-e9d4780320f2.recording.jsonl
        Returns: locksmith
        """
        filename = Path(filename).name
        if "." in filename:
            parts = filename.split(".")
            return parts[0]
        else:
            return filename

    @classmethod
    def get_guid(cls, filename: str) -> str:
        """
        Example filename: locksmith.random.50.81329339-1951-487c-8bed-e9d4780320f2.recording.jsonl
        Returns: 81329339-1951-487c-8bed-e9d4780320f2
        """
        filename = Path(filename).name
        if "." in filename:
            parts = filename.split(".")
            return parts[-3]
        else:
            return filename

    @staticmethod
    def _resolve_existing_recording_path(filename: str) -> Path:
        path = Path(filename)
        if path.is_absolute() or path.parent != Path("."):
            return path

        recordings_dir = get_recordings_dir()
        if recordings_dir:
            return Path(recordings_dir) / filename
        return path

    @staticmethod
    def _relative_recording_paths(root: Path) -> builtins.list[str]:
        if not root.exists():
            return []
        paths: builtins.list[str] = []
        for path in sorted(root.glob(f"**/*{RECORDING_SUFFIX}")):
            if path.is_file():
                try:
                    paths.append(str(path.relative_to(Path.cwd())))
                except ValueError:
                    paths.append(str(path))
        return paths

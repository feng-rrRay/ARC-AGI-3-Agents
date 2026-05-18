from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from ...run_artifacts import RUN_ARTIFACTS_DIR_ENV, RUN_LOG_PATH_ENV, safe_slug


class TraceWriter:
    """Append-only thread-safe JSONL writer for per-VLM-call records."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        record.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
        line = json.dumps(record, default=str)
        with self._lock, self.path.open("a") as f:
            f.write(line + "\n")


def default_trace_path(prefix: str | None = None, guid: str | None = None) -> Path:
    """Trace path for a VLM call stream."""
    artifacts_dir = os.getenv(RUN_ARTIFACTS_DIR_ENV)
    if artifacts_dir and prefix and guid:
        return (
            Path(artifacts_dir) / f"{safe_slug(prefix)}.{safe_slug(guid)}.trace.jsonl"
        )

    run_log = os.getenv(RUN_LOG_PATH_ENV)
    if run_log:
        return Path(run_log).with_suffix(".trace.jsonl")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir / f"trace-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"


def serialize_response(response: Any) -> dict[str, Any]:
    """Convert a Gemini response into a JSON-serialisable {text, function_calls, finish_reason}."""
    if isinstance(response, str):
        return {"text": response, "function_calls": [], "finish_reason": None}
    out: dict[str, Any] = {"text": "", "function_calls": []}
    for cand in getattr(response, "candidates", None) or []:
        for part in getattr(getattr(cand, "content", None), "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                out["text"] += str(text)
            fc = getattr(part, "function_call", None)
            if fc and getattr(fc, "name", None):
                out["function_calls"].append(
                    {
                        "name": str(fc.name),
                        "args": dict(getattr(fc, "args", None) or {}),
                    }
                )
    cands = getattr(response, "candidates", None) or []
    out["finish_reason"] = getattr(cands[0], "finish_reason", None) if cands else None
    return out

from __future__ import annotations

import threading
from pathlib import Path

_LOCKS_LOCK = threading.Lock()
_LOCKS_BY_PATH: dict[Path, threading.Lock] = {}


def lock_for_path(path: Path) -> threading.Lock:
    """Return a process-wide threading.Lock keyed on the resolved path.

    Used by MemoryStore and SkillStore to serialize concurrent writes from
    different Swarm threads that happen to share the same backing file.
    """
    key = path.resolve()
    with _LOCKS_LOCK:
        lock = _LOCKS_BY_PATH.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS_BY_PATH[key] = lock
        return lock

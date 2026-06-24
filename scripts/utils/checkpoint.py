"""Checkpoint helpers for robust incremental processing.

Provides:

* ``fsync_write_line(f, line)`` — flush + fsync a single line to an open file
  (used by ingestion scripts for ``success_records.txt``).
* ``atomic_json_dump(obj, path, **kw)`` — write JSON via a temp file and
  ``os.replace`` so readers never see a half-written file.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from typing import IO, Any

_LOCKS_GUARD = threading.Lock()
_FILE_LOCKS: dict[int, threading.Lock] = {}


def _lock_for_file(f: IO[str]) -> threading.Lock:
    fd = f.fileno()
    with _LOCKS_GUARD:
        lock = _FILE_LOCKS.get(fd)
        if lock is None:
            lock = threading.Lock()
            _FILE_LOCKS[fd] = lock
        return lock


def fsync_write_line(f: IO[str], line: str) -> None:
    """Write *line* (with trailing newline) and guarantee it hits disk."""
    with _lock_for_file(f):
        f.write(f"{line}\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_json_dump(obj: Any, path: str | os.PathLike, **json_kw: Any) -> None:
    """Write *obj* as JSON to *path* atomically.

    Writes to a temporary file in the same directory, then uses
    ``os.replace`` to atomically move it into place.  This prevents
    readers from ever seeing a truncated or partially-written JSON file
    (e.g. after a crash during ``json.dump``).
    """
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, **json_kw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

"""Shared helpers for streaming checkpoints, cleanup, and long add logs.

Streaming benchmarks process one independent user/conversation at a time and
then delete that user's memories. These helpers keep LongMemEval streaming
checkpoint, cleanup, single-user routing behavior, and long-call progress logs
in one place.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

from utils.checkpoint import fsync_write_line

_T = TypeVar("_T")


def configure_single_user_streaming(frame: str) -> None:
    """Apply process-local env defaults for single-user streaming datasets."""
    if frame == "everos":
        os.environ["EVEROS_USE_GROUP"] = "false"


def resolve_max_batch_chars(frame: str, default: int = 0) -> int:
    """Resolve the standard character budget for one memory add request.

    Priority follows the existing client convention:
    ``{CLIENT}_MAX_BATCH_CHARS`` -> ``MAX_BATCH_CHARS`` -> *default*.
    """
    for key in (f"{frame.upper()}_MAX_BATCH_CHARS", "MAX_BATCH_CHARS"):
        value = os.getenv(key, "").strip()
        if value:
            try:
                max_chars = int(value)
            except ValueError as exc:
                raise ValueError(f"{key} must be an integer, got {value!r}") from exc
            if max_chars < 0:
                raise ValueError(f"{key} must be >= 0, got {value!r}")
            return max_chars
    return default


def resolve_add_heartbeat_seconds(default: float = 30.0) -> float:
    """Resolve the heartbeat interval for long memory add calls.

    Set ``MEMEVAL_ADD_HEARTBEAT_SECONDS=0`` to keep start/done logs but disable
    periodic heartbeat lines while a single client call is blocked.
    """
    for key in ("MEMEVAL_ADD_HEARTBEAT_SECONDS", "ADD_HEARTBEAT_SECONDS"):
        value = os.getenv(key, "").strip()
        if not value:
            continue
        try:
            return max(0.0, float(value))
        except ValueError:
            print(
                f"WARNING: ignoring invalid {key}={value!r}; using {default}s",
                flush=True,
            )
            return default
    return default


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    return f"{minutes}m{sec:02d}s"


class LongCallLogger:
    """Log start/done/heartbeat lines around a potentially long blocking call."""

    def __init__(
        self,
        label: str,
        *,
        heartbeat_seconds: float | None = None,
    ) -> None:
        self.label = label
        self.heartbeat_seconds = (
            resolve_add_heartbeat_seconds()
            if heartbeat_seconds is None
            else max(0.0, heartbeat_seconds)
        )
        self.started_at = 0.0
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "LongCallLogger":
        self.started_at = time.time()
        timestamp = datetime.now().isoformat(timespec="seconds")
        print(f"{timestamp} [ADD START] {self.label}", flush=True)
        if self.heartbeat_seconds > 0:
            self._stop_event = threading.Event()
            self._thread = threading.Thread(
                target=self._heartbeat_loop,
                name="memeval-add-heartbeat",
                daemon=True,
            )
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)

        elapsed = time.time() - self.started_at
        timestamp = datetime.now().isoformat(timespec="seconds")
        if exc_type is None:
            print(
                f"{timestamp} [ADD DONE] {self.label} in {_fmt_elapsed(elapsed)}",
                flush=True,
            )
        else:
            print(
                f"{timestamp} [ADD FAILED] {self.label} after {_fmt_elapsed(elapsed)}: "
                f"{exc_type.__name__}: {exc}",
                flush=True,
            )
        return False

    def _heartbeat_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.wait(self.heartbeat_seconds):
            elapsed = time.time() - self.started_at
            timestamp = datetime.now().isoformat(timespec="seconds")
            print(
                f"{timestamp} [ADD RUNNING] {self.label} elapsed={_fmt_elapsed(elapsed)}",
                flush=True,
            )


def load_marker_set(path: Path, *, cast: Callable[[str], _T] = str) -> set[_T]:
    if not path.exists():
        return set()
    markers: set[_T] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                markers.add(cast(line))
    return markers


def mark_marker(path: Path, markers: set[_T], marker: _T) -> None:
    if marker in markers:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fsync_write_line(f, str(marker))
    markers.add(marker)


def log_event(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event": event,
    }
    record.update(fields)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def is_timeout_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{type(current).__name__}: {current}".lower()
        if "timed out" in text or "timeout" in text:
            return True
        current = current.__cause__ or current.__context__
    return False


def delete_user_data(
    frame: str,
    client: Any,
    user_id: str,
    *,
    skip_timeout: bool = False,
    skip_errors: bool = False,
) -> tuple[bool, str | None]:
    """Delete one benchmark user's memory data using the product's adapter API."""
    try:
        if "mem0" in frame and callable(getattr(client, "delete_all", None)):
            client.delete_all(user_id=user_id)
            return True, None
        if callable(getattr(client, "delete", None)):
            client.delete(user_id)
            return True, None
        if callable(getattr(client, "delete_user", None)):
            client.delete_user(user_id)
            return True, None
        raise RuntimeError(f"{frame} client has no delete/delete_user method")
    except Exception as exc:
        if skip_errors or (skip_timeout and is_timeout_error(exc)):
            print(f"WARNING: skipping delete failure for {user_id}: {exc}", flush=True)
            return False, str(exc)
        raise


def timed_delete_user_data(
    frame: str,
    client: Any,
    user_id: str,
    *,
    phase: str,
    skip_timeout: bool = False,
    skip_errors: bool = False,
) -> tuple[bool, str | None, float]:
    start = time.time()
    try:
        ok, error = delete_user_data(
            frame,
            client,
            user_id,
            skip_timeout=skip_timeout,
            skip_errors=skip_errors,
        )
    except Exception:
        duration_ms = (time.time() - start) * 1000
        print(f"{phase} delete failed after {duration_ms:.1f} ms", flush=True)
        raise

    duration_ms = (time.time() - start) * 1000
    status = "ok" if ok else "error_skipped"
    print(f"{phase} delete {status} in {duration_ms:.1f} ms", flush=True)
    return ok, error, duration_ms


def prepare_user_after_delete(frame: str, client: Any, user_id: str) -> None:
    if frame == "zep" and callable(getattr(client, "add_user", None)):
        client.add_user(user_id)

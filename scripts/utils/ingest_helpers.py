"""Unified time-injection and session-kwarg helpers for ingestion scripts.

These helpers centralise the per-lib branching that was previously duplicated
across all five benchmark ingestion scripts.  Client ``add()`` interfaces are
left unchanged — the helpers only decide *how* to pass time information and
which session-identification kwarg each client expects.
"""

from __future__ import annotations

import os
import time as _time
from typing import Any

from utils.time import to_iso, to_unix, to_readable


# ── Per-call add timing ──────────────────────────────────────────────────────

_ADD_METHOD_NAMES = ("add", "add_group", "sdk_graph_add")


class AddCallTimer:
    """Monkey-patch a client to record wall-clock duration of every add call.

    Usage::

        timer = AddCallTimer(client)
        # … use client.add() / client.add_group() / client.sdk_graph_add() …
        per_call_ms = timer.durations_ms   # list[float]

    Only methods that actually exist on *client* are wrapped.
    """

    def __init__(self, client: Any) -> None:
        self.durations_ms: list[float] = []
        self._client = client
        for name in _ADD_METHOD_NAMES:
            original = getattr(client, name, None)
            if original is not None:
                setattr(client, name, self._wrap(original))

    def _wrap(self, fn):
        durations = self.durations_ms

        def timed(*args: Any, **kwargs: Any):
            start = _time.time()
            result = fn(*args, **kwargs)
            durations.append((_time.time() - start) * 1000)
            return result

        return timed

# ── Libs grouped by time-injection strategy ──────────────────────────────────

_CHAT_TIME_LIBS = frozenset({
    "memos", "everos", "mem0", "cognee", "backboard",
    "memorylake", "viking", "memmachine", "mem9",
})

# ── Libs grouped by session-identification kwarg ─────────────────────────────

_CONV_ID_LIBS = frozenset({"memos", "everos"})
_SESSION_KEY_LIBS = frozenset({
    "cognee", "backboard", "memorylake", "letta", "hindsight", "graphiti",
})


def inject_time(messages, dt, lib_name):
    """Inject time information into *messages* and return extra kwargs.

    Modifies *messages* **in-place** for libs that embed time in the message
    dict (``chat_time`` field or ``[iso] content`` prefix).

    Returns a dict of keyword arguments to forward to ``client.add()``.

    Args:
        messages: List of ``{"role": str, "content": str, ...}`` dicts.
        dt: Parsed ``datetime`` (UTC) or ``None`` when no time is available.
        lib_name: Library name from ``SUPPORTED_LIBS``.

    Returns:
        Dict of extra kwargs (may be empty).
    """
    if dt is None:
        return {}

    iso = to_iso(dt)

    if lib_name in _CHAT_TIME_LIBS:
        for m in messages:
            m["chat_time"] = iso
        return {}

    if lib_name == "zep":
        return {"timestamp": to_unix(dt)}

    if lib_name == "supermemory":
        return {"session_date": to_readable(dt)}

    if lib_name == "letta":
        return {"timestamp": to_unix(dt)}

    if lib_name == "hindsight":
        return {"timestamp": iso}

    if lib_name == "graphiti":
        return {"timestamp": iso}

    if lib_name == "memori":
        include = os.environ.get(
            "MEMORI_INCLUDE_SESSION_TIME", "true"
        ).lower() in ("true", "1", "yes")
        if include:
            for m in messages:
                m["content"] = f"[{iso}] {m['content']}"
        return {}

    # Unknown lib — fall back to chat_time
    for m in messages:
        m["chat_time"] = iso
    return {}


def session_id_kwargs(lib_name, session_id):
    """Return the session-identification kwargs for ``client.add()``.

    Different products use different parameter names:

    * ``conv_id``     — memos, everos
    * ``session_key`` — cognee, backboard, memorylake, letta, hindsight
    * *(nothing)*     — all others
    """
    if session_id is None:
        return {}
    if lib_name in _CONV_ID_LIBS:
        return {"conv_id": session_id}
    if lib_name in _SESSION_KEY_LIBS:
        return {"session_key": session_id}
    return {}

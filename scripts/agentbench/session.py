from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class SessionSpec:
    """Session identifiers used by an agent run.

    ``cli_session_id`` is constrained by the agent CLI.  Semantic identifiers
    can preserve richer source information for plugins and result analysis.
    """

    cli_session_id: str
    semantic_session_id: str
    source_ref: str
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_session_ref: str | None = None
    agent_session_metadata: dict[str, Any] = field(default_factory=dict)
    openclaw_session_key: str | None = None
    openclaw_gateway_session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "cli_session_id": self.cli_session_id,
            "semantic_session_id": self.semantic_session_id,
            "source_ref": self.source_ref,
            "metadata": self.metadata,
        }
        if self.agent_session_metadata:
            data["agent_session_metadata"] = self.agent_session_metadata
        for key in (
            "agent_session_ref",
            "openclaw_session_key",
            "openclaw_gateway_session_id",
        ):
            value = getattr(self, key)
            if value:
                data[key] = value
        return data


def sanitize_session_id(value: str, *, max_len: int = 180) -> str:
    """Return a CLI-safe session id using characters accepted by common CLIs."""

    safe = _SAFE_ID_RE.sub("-", value).strip("-")
    safe = re.sub(r"-{2,}", "-", safe)
    if not safe:
        safe = "session"
    if len(safe) > max_len:
        safe = safe[:max_len].rstrip("-")
    return safe


def make_base_session_id(
    *,
    phase: str,
    domain: str,
    split: str,
    task_name: str,
    trial: int,
    prefix: str = "omnimemeval",
) -> str:
    raw = f"{prefix}-{phase}-{domain}-{split}-{task_name}-t{trial}-{uuid.uuid4().hex[:6]}"
    return sanitize_session_id(raw)

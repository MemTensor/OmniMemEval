"""Evaluation-only read-only wrapper for the Hermes MemOS provider.

This module is linked into an isolated ``HERMES_HOME/plugins`` only for test
trials.  It reuses MemOS retrieval and tool formatting while suppressing all
conversation, session, delegation, and tool-invocation persistence.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

from plugins.memory.memtensor import (  # type: ignore[import-not-found]
    MemosBridgeClient,
    MemTensorProvider,
    ensure_bridge_running,
)


class OmniMemEvalReadOnlyMemOSProvider(MemTensorProvider):
    @property
    def name(self) -> str:
        return "omnimemeval_memos_readonly"

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Connect to the bridge without opening a writable MemOS session."""

        self._session_id = session_id or self._session_id
        self._hermes_home = str(kwargs.get("hermes_home") or "")
        self._platform = str(kwargs.get("platform") or "cli")
        self._agent_identity = str(kwargs.get("agent_identity") or "hermes")
        ensure_bridge_running()
        bridge = MemosBridgeClient()
        bridge.register_host_handler("host.llm.complete", self._handle_host_llm_complete)
        self._bridge = bridge
        self._start_bridge_keepalive()

    def _open_session(self, session_id: str = "", *, timeout: float = 30.0) -> None:
        # Reconnects must remain read-only too; never issue session.open.
        del timeout
        self._session_id = session_id or self._session_id

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Retrieve directly with memory.search, avoiding turn.start writes."""

        del session_id
        text = (query or "").strip()
        if not text or not self._ensure_bridge(self._session_id, timeout=10.0):
            return ""
        try:
            response = self._bridge.request(
                "memory.search",
                {
                    "agent": "hermes",
                    "namespace": self._runtime_namespace(),
                    "query": text,
                    "topK": {"tier1": 10, "tier2": 10, "tier3": 10},
                },
            )
        except Exception:
            return ""
        hits = response.get("hits", []) if isinstance(response, dict) else []
        lines = []
        for hit in hits[:20]:
            if not isinstance(hit, dict):
                continue
            snippet = str(hit.get("snippet") or hit.get("body") or "").strip()
            if snippet:
                lines.append(f"- {snippet}")
        return "## Recalled Memories\n" + "\n".join(lines) if lines else ""

    def sync_turn(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        del messages

    def on_delegation(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def on_memory_write(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        del messages
        return self.prefetch(self._last_user_text)

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        if tool_name != "memos_skill_get":
            return super().handle_tool_call(tool_name, args, **kwargs)
        if not self._bridge:
            return json.dumps({"error": "bridge not connected"})
        skill_id = str(args.get("id") or "").strip()
        if not skill_id:
            return json.dumps({"error": "missing id"})
        try:
            skill = self._bridge.request(
                "skill.get",
                {
                    "id": skill_id,
                    "namespace": self._runtime_namespace(),
                    "recordTrial": False,
                },
            )
            return json.dumps({"found": bool(skill), "skill": skill})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def shutdown(self) -> None:
        self._bridge_keepalive_stop.set()
        if self._bridge_keepalive_thread and self._bridge_keepalive_thread.is_alive():
            self._bridge_keepalive_thread.join(timeout=12.0)
        if self._bridge:
            with contextlib.suppress(Exception):
                self._bridge.close()
            self._bridge = None


def register(ctx: Any) -> None:
    ctx.register_memory_provider(OmniMemEvalReadOnlyMemOSProvider())


__all__ = ["OmniMemEvalReadOnlyMemOSProvider", "register"]

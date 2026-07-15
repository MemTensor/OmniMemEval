"""Evaluation wrapper that decouples Hermes retrieval timeout from capture.

The installed provider treats ``turn.start`` as one JSON-RPC request containing
both fast episode routing and slow model-backed retrieval.  If the client-side
30-second wait expires, the bridge may already have opened the episode even
though the provider never receives its id.  Upstream ``sync_turn`` then starts a
second retrieval before capture and can lose the completed turn entirely.

This wrapper keeps the 30-second retrieval limit.  On timeout it marks the
episode id unresolved, skips that duplicate retrieval, and sends ``turn.end``
with an empty id so the core resolves the canonical open episode by session.
Only evaluation-scoped Hermes homes load this provider; installed MemOS code is
not modified.
"""

from __future__ import annotations

from typing import Any

from plugins.memory.memtensor import MemTensorProvider  # type: ignore[import-not-found]


class OmniMemEvalMemOSProvider(MemTensorProvider):
    _UNRESOLVED_EPISODE_PREFIX = "omnimemeval:unresolved:"

    @property
    def name(self) -> str:
        return "omnimemeval_memos"

    @staticmethod
    def _is_rpc_timeout(exc: Exception) -> bool:
        if str(getattr(exc, "code", "")).strip().lower() == "timeout":
            return True
        message = str(exc).lower()
        return "did not respond within" in message or "rpc timeout" in message

    def _has_unresolved_episode(self) -> bool:
        return str(self._episode_id or "").startswith(self._UNRESOLVED_EPISODE_PREFIX)

    def _turn_start(self, query: str, *, session_id: str = "") -> str:
        try:
            return super()._turn_start(query, session_id=session_id)
        except Exception as exc:
            if self._is_rpc_timeout(exc):
                # openEpisodeIfNeeded runs before retrieval inside MemOS.  A
                # timeout therefore usually means the real episode exists but
                # its id is trapped in the late response.  A truthy sentinel
                # prevents upstream sync_turn from launching turn.start again.
                resolved_session = session_id or self._session_id or "unknown"
                self._episode_id = self._UNRESOLVED_EPISODE_PREFIX + resolved_session
            raise

    def _turn_end(
        self,
        user_content: str,
        assistant_content: str,
        tool_calls: list[dict[str, Any]],
        ts_ms: int,
        *,
        agent_thinking: str = "",
    ) -> str:
        if not self._bridge:
            return ""
        clean_tool_calls = [
            {key: value for key, value in call.items() if key not in {"_id", "_ids"}}
            for call in tool_calls
        ]
        unresolved = self._has_unresolved_episode()
        payload: dict[str, Any] = {
            "agent": "hermes",
            "namespace": self._runtime_namespace(),
            "sessionId": self._session_id,
            # Empty is intentional after a retrieval timeout.  MemOS
            # onTurnEnd resolves the canonical open episode by session.
            "episodeId": "" if unresolved else self._episode_id,
            "agentText": assistant_content,
            "userText": user_content,
            "toolCalls": clean_tool_calls,
            "contextHints": {
                "agentIdentity": self._agent_identity,
                "namespace": self._runtime_namespace(),
                **self._host_runtime_context(),
            },
            "ts": ts_ms,
        }
        if agent_thinking:
            payload["agentThinking"] = agent_thinking
        result = self._bridge.request("turn.end", payload)
        if isinstance(result, dict):
            episode_id = str(result.get("episodeId") or "").strip()
            if episode_id:
                self._episode_id = episode_id
            trace_ids = result.get("traceIds")
            if not isinstance(trace_ids, list):
                trace_id = str(result.get("traceId") or "").strip()
                trace_ids = [trace_id] if trace_id else []
            if trace_ids:
                trace_id = str(trace_ids[-1])
                self._last_trace_id = trace_id
                return trace_id
        return ""


def register(ctx: Any) -> None:
    ctx.register_memory_provider(OmniMemEvalMemOSProvider())


__all__ = ["OmniMemEvalMemOSProvider", "register"]

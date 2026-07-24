"""Evaluation wrapper that decouples Hermes retrieval timeout from capture.

The installed provider treats ``turn.start`` as one JSON-RPC request containing
both fast episode routing and slow model-backed retrieval.  If the client-side
30-second wait expires, the bridge may already have opened the episode even
though the provider never receives its id.  Upstream ``sync_turn`` then starts a
second retrieval before capture and can lose the completed turn entirely.

This wrapper keeps the 30-second RPC limit.  On retrieval timeout it marks the
episode id unresolved, skips that duplicate retrieval, and sends ``turn.end``
with an empty id so the core resolves the canonical open episode by session.
Oversized tool transcripts are compacted before ``turn.end``; if an otherwise
normal request times out, capture is retried for the same episode without the
tool details.  The degraded trace remains explicit in ``contextHints``.  Only
evaluation-scoped Hermes homes load this provider; installed MemOS code is not
modified.
"""

from __future__ import annotations

import json
import sys
from typing import Any


try:
    from plugins.memory.memtensor import MemTensorProvider  # type: ignore[import-not-found]
except ImportError as exc:
    # @memtensor/memos-local-plugin 2.0.10 imports MemosHttpClient from its
    # bridge_client module, but that release does not package the class.  Patch
    # only the already-loaded module object and retry; no installed MemOS file
    # is changed.  The placeholder makes the provider's optional HTTP probe
    # fail normally, after which upstream falls back to its stdio bridge.
    if "cannot import name 'MemosHttpClient' from 'bridge_client'" not in str(exc):
        raise
    bridge_client = sys.modules.get("bridge_client")
    if bridge_client is None or hasattr(bridge_client, "MemosHttpClient"):
        raise

    class _UnavailableMemosHttpClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError(
                "MemosHttpClient is unavailable in memos-local-plugin 2.0.10"
            )

    bridge_client.MemosHttpClient = _UnavailableMemosHttpClient
    from plugins.memory.memtensor import MemTensorProvider  # type: ignore[import-not-found,no-redef]


class OmniMemEvalMemOSProvider(MemTensorProvider):
    _UNRESOLVED_EPISODE_PREFIX = "omnimemeval:unresolved:"
    _TOOL_CALL_COUNT_LIMIT = 12
    _TOOL_PAYLOAD_CHAR_LIMIT = 64_000

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

    @classmethod
    def _tool_payload_is_oversized(cls, tool_calls: list[dict[str, Any]]) -> bool:
        if len(tool_calls) > cls._TOOL_CALL_COUNT_LIMIT:
            return True
        try:
            payload_chars = len(
                json.dumps(tool_calls, ensure_ascii=False, default=str)
            )
        except (TypeError, ValueError):
            return True
        return payload_chars > cls._TOOL_PAYLOAD_CHAR_LIMIT

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
        if self._tool_payload_is_oversized(clean_tool_calls):
            # The bridge handles requests serially.  Waiting for an oversized
            # request to time out before retrying leaves the compact retry
            # queued behind the still-running first request.  Compact known
            # oversized transcripts before the first request instead.
            payload = {
                **payload,
                "toolCalls": [],
                "contextHints": {
                    **payload["contextHints"],
                    "omnimemevalCaptureFallback": "oversized_tool_transcript",
                    "omnimemevalDroppedToolCalls": len(clean_tool_calls),
                },
            }
        try:
            result = self._bridge.request("turn.end", payload)
        except Exception as exc:
            if not self._is_rpc_timeout(exc):
                raise
            if not payload["toolCalls"]:
                # This request was already compact.  A duplicate would only
                # queue behind the still-running request and cannot reduce its
                # payload further.
                raise
            # Long tool-heavy Hermes sessions can make MemOS reflection exceed
            # the upstream 30-second JSON-RPC wait.  Retrying the full payload
            # repeats the same failure.  Preserve the task and final answer on
            # the canonical episode, but omit tool details so training can
            # record an honest degraded trace instead of an empty episode.
            fallback_payload = {
                **payload,
                "toolCalls": [],
                "contextHints": {
                    **payload["contextHints"],
                    "omnimemevalCaptureFallback": "turn_end_rpc_timeout",
                    "omnimemevalDroppedToolCalls": len(clean_tool_calls),
                },
            }
            result = self._bridge.request("turn.end", fallback_payload)
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

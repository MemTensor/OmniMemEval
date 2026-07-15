"""Evaluation wrapper that makes Hermes MemOS RPC latency configurable.

The installed MemOS provider defaults omitted JSON-RPC timeouts to 30 seconds.
Retrieval performs model-backed ranking and can exceed that under concurrent
evaluation, even though the bridge later completes successfully.  This wrapper
changes only omitted timeouts; explicit health, reconnect, and feedback limits
from the upstream provider keep their original values.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from plugins.memory.memtensor import MemTensorProvider  # type: ignore[import-not-found]


class OmniMemEvalMemOSProvider(MemTensorProvider):
    @property
    def name(self) -> str:
        return "omnimemeval_memos"

    @staticmethod
    def _default_rpc_timeout() -> float:
        raw = os.environ.get("MEMOS_HERMES_RPC_TIMEOUT_SECONDS", "180")
        try:
            return max(1.0, float(raw))
        except (TypeError, ValueError):
            return 180.0

    def _wrap_bridge_request(self) -> None:
        bridge = self._bridge
        if bridge is None or getattr(bridge, "_omnimemeval_timeout_wrapped", False):
            return
        original: Callable[..., dict[str, Any]] = bridge.request
        default_timeout = self._default_rpc_timeout()

        def request(
            method: str,
            params: Any = None,
            *,
            timeout: float | None = None,
        ) -> dict[str, Any]:
            effective_timeout = default_timeout if timeout is None else timeout
            return original(method, params, timeout=effective_timeout)

        bridge.request = request
        bridge._omnimemeval_timeout_wrapped = True

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        super().initialize(session_id, **kwargs)
        self._wrap_bridge_request()

    def _reconnect_bridge(self, session_id: str = "", *, timeout: float = 30.0) -> None:
        super()._reconnect_bridge(session_id, timeout=timeout)
        self._wrap_bridge_request()


def register(ctx: Any) -> None:
    ctx.register_memory_provider(OmniMemEvalMemOSProvider())


__all__ = ["OmniMemEvalMemOSProvider", "register"]

import time

from .base_client import (
    BaseApiClient,
    env_bool,
    env_float,
    env_int,
    env_max_batch_chars,
    env_str,
    require_env,
    iter_batches,
)


class Mem0Client(BaseApiClient):
    """Mem0 platform client (REST API, V3).

    Reference: https://docs.mem0.ai/api-reference/memory/search-memories
    Migration: https://docs.mem0.ai/migration/platform-v2-to-v3

    V3 uses ADD-only extraction with hybrid retrieval (semantic + BM25 +
    entity matching) and built-in entity linking.  The ``add`` endpoint is
    asynchronous — we poll ``/v1/event/{event_id}/`` until completion.

    V3 ``add`` accepts ``custom_instructions`` per-call so extraction
    guidance is passed inline — no need for the deprecated ``update_project``.

    Configurable env vars (all optional, official V3 defaults used):
        MEM0_API_KEY                  – API key (required)
        MEM0_BASE_URL                 – base URL (default: https://api.mem0.ai)
        MEM0_SEARCH_THRESHOLD         – min relevance score 0~1 (V3 default: 0.1; 0.0 disables)
        MEM0_SEARCH_RERANK            – enable managed reranker (V3 default: false; adds ~200ms)
        MEM0_EMBED_CHAT_TIME          – embed [chat_time] in content (default: true)
        MEM0_CUSTOM_INSTRUCTIONS      – "true" to send extraction instructions per-call (default: true)
    """

    _EVENT_POLL_INTERVAL = 1
    _EVENT_POLL_MAX_WAIT = 120

    def __init__(self):
        api_key = require_env("MEM0_API_KEY")
        base_url = env_str("MEM0_BASE_URL", "https://api.mem0.ai")
        super().__init__(
            base_url=base_url,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Token {api_key}",
            },
        )
        self._batch_size = env_int("MEM0_BATCH_SIZE", 20, min_value=1)
        self._max_batch_chars = env_max_batch_chars("MEM0_MAX_BATCH_CHARS")
        self._search_threshold = env_float(
            "MEM0_SEARCH_THRESHOLD", 0.1, min_value=0
        )
        self._search_rerank = env_bool("MEM0_SEARCH_RERANK", False)
        self._embed_chat_time = env_bool("MEM0_EMBED_CHAT_TIME", True)
        self._custom_instructions_enabled = env_bool(
            "MEM0_CUSTOM_INSTRUCTIONS", True
        )
        self._custom_instructions_text = None

    @staticmethod
    def _fmt_content(msg):
        """Embed ``chat_time`` into content so the extraction pipeline can
        capture temporal context (V3 add API has no dedicated time field)."""
        content = msg.get("content", "")
        chat_time = msg.get("chat_time")
        if chat_time:
            return f"[{chat_time}] {content}"
        return content

    def set_custom_instructions(self, text):
        """Cache instructions text to be sent with every ``add`` call."""
        self._custom_instructions_text = text

    def add(self, messages, user_id, batch_size=None):
        if self._embed_chat_time:
            messages = [
                {**m, "content": self._fmt_content(m)}
                for m in messages
            ]

        formatted = [
            {"role": m.get("role", "user"), "content": m["content"]}
            for m in messages
        ]

        event_ids = []
        for batch in iter_batches(formatted, batch_size or self._batch_size,
                                  max_chars=self._max_batch_chars):
            payload = {
                "messages": batch,
                "user_id": user_id,
            }
            if self._custom_instructions_enabled and self._custom_instructions_text:
                payload["custom_instructions"] = self._custom_instructions_text

            def _do(p=payload):
                resp = self._post("/v3/memories/add/", json=p)
                resp.raise_for_status()
                return resp.json()

            result = self._retry(_do)

            event_id = result.get("event_id")
            if event_id:
                event_ids.append(event_id)

        if event_ids:
            self._wait_for_events(event_ids)

    def search(self, query, user_id, top_k):
        """Search memories. Returns plain-text string of formatted results."""
        payload = {
            "query": query,
            "top_k": top_k,
            "threshold": self._search_threshold,
            "rerank": self._search_rerank,
            "filters": {"user_id": user_id},
        }

        def _do():
            resp = self._post("/v3/memories/search/", json=payload)
            resp.raise_for_status()
            return resp.json()

        res = self._retry(_do)

        if isinstance(res, dict):
            memories = res.get("results") or res.get("memories") or []
        elif isinstance(res, list):
            memories = res
        else:
            memories = []

        return "\n".join(m.get("memory", "") for m in memories)

    def delete_all(self, user_id):
        resp = self._delete("/v1/memories", params={"user_id": user_id})
        if resp.status_code not in (200, 204):
            resp.raise_for_status()

    def _wait_for_events(self, event_ids):
        """Poll all *event_ids* concurrently until every one reaches
        SUCCEEDED / FAILED or the global timeout expires.

        All batches were already submitted, so their server-side extraction
        runs in parallel.  We only need to wait for the slowest one.
        """
        pending = set(event_ids)
        deadline = time.time() + self._EVENT_POLL_MAX_WAIT
        while pending and time.time() < deadline:
            for eid in list(pending):
                try:
                    resp = self._get(f"/v1/event/{eid}/")
                    if resp.status_code == 200:
                        status = resp.json().get("status", "")
                        if status in ("SUCCEEDED", "FAILED"):
                            pending.discard(eid)
                except Exception as exc:
                    print(f"[WARN] Polling event {eid} failed: {exc}")
            if pending:
                time.sleep(self._EVENT_POLL_INTERVAL)

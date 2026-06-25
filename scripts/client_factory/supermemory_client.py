import json
import time

from contextlib import suppress

from .base_client import (
    BaseApiClient,
    RateLimitError,
    env_bool,
    env_float,
    env_int,
    env_str,
    require_env,
)
# ── Supermemory ──────────────────────────────────────────────────────────────


class SupermemoryClient(BaseApiClient):
    """Supermemory client (REST API).

    Reference: https://supermemory.ai/docs/search/examples/document-search

    Aligned with official memorybench (supermemoryai/memorybench):
      - Ingestion via POST /v3/documents (document-level, matching SDK ``client.add``)
      - Search  via POST /v4/search with ``searchMode: "hybrid"``
      - Indexing poll via GET /v3/documents/{id} + GET /v4/memories/{id}
      - DELETE  /v3/documents/bulk — cleanup by container tag

    Rate limiting: defaults to 3 QPS (configurable via SUPERMEMORY_QPS env var).
    """

    _RETRYABLE_STATUS_CODES = frozenset({401, 429, 500, 502, 503, 504})

    def __init__(self):
        api_key = require_env("SUPERMEMORY_API_KEY")
        base_url = env_str("SUPERMEMORY_BASE_URL", "https://api.supermemory.ai")
        qps = env_float("SUPERMEMORY_QPS", 3, min_value=0)
        super().__init__(
            base_url=base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            qps=qps,
        )
        self._batch_size = env_int("SUPERMEMORY_BATCH_SIZE", 20, min_value=1)
        self._pending_doc_ids: list[str] = []

    def delete(self, user_id):
        """Bulk-delete all documents tagged with *user_id* (permanent)."""
        def _do():
            resp = self._delete("/v3/documents/bulk", json={"containerTags": [user_id]})
            if resp.status_code in (401, 429):
                raise RateLimitError(
                    retry_after=self._parse_retry_after(resp) or 5, response=resp
                )
            resp.raise_for_status()

        with suppress(Exception):
            self._retry(_do)
            print(f"Deleted supermemory documents for {user_id}")
        self._pending_doc_ids.clear()

    def add(self, messages, user_id, *, session_date=None, session_id=None):
        """Ingest a session as a single document via POST /v3/documents.

        Aligned with memorybench: the entire session is serialised as
        stringified JSON inside the ``content`` field (matching the SDK
        ``client.add()`` behaviour), with optional date prefix.
        """
        if not messages:
            return

        session_str = json.dumps(messages).replace("<", "&lt;").replace(">", "&gt;")
        if session_date:
            content = (
                f"Here is the date the following session took place: {session_date}\n\n"
                f"Here is the session as a stringified JSON:\n{session_str}"
            )
        else:
            content = f"Here is the session as a stringified JSON:\n{session_str}"

        metadata = {"source": "omnimemeval", "user_id": user_id}
        if session_id:
            metadata["sessionId"] = session_id
        if session_date:
            metadata["date"] = session_date

        payload = {
            "content": content,
            "containerTag": user_id,
            "metadata": metadata,
        }

        def _do():
            resp = self._post("/v3/documents", json=payload)
            if resp.status_code in (401, 429):
                raise RateLimitError(
                    retry_after=self._parse_retry_after(resp) or 5, response=resp
                )
            if resp.status_code not in (200, 201, 202):
                resp.raise_for_status()
            return resp.json()

        data = self._retry(_do)
        doc_id = None
        if isinstance(data, dict):
            doc_id = data.get("id") or data.get("documentId")
        if doc_id:
            self._pending_doc_ids.append(doc_id)

    def await_indexing(self, timeout=600):
        """Poll document + memory status until all pending docs are done.

        Aligned with memorybench ``awaitIndexing``: checks both
        ``/v3/documents/{id}`` (doc status) and ``/v4/memories/{id}``
        (memory extraction status).
        """
        if not self._pending_doc_ids:
            return

        pending = set(self._pending_doc_ids)
        completed, failed = [], []
        backoff_ms = 1000
        start = time.time()

        print(f"  ⏳ Waiting for {len(pending)} documents to finish indexing...")
        while pending and (time.time() - start) < timeout:
            still_pending = set()
            for doc_id in pending:
                try:
                    doc_resp = self._retry(
                        lambda did=doc_id: self._get_json(f"/v3/documents/{did}"),
                        max_retries=3,
                    )
                    doc_status = doc_resp.get("status", "pending")
                    if doc_status == "failed":
                        failed.append(doc_id)
                        continue
                    if doc_status != "done":
                        still_pending.add(doc_id)
                        continue
                    mem_resp = self._retry(
                        lambda did=doc_id: self._get_json(f"/v4/memories/{did}"),
                        max_retries=3,
                    )
                    mem_status = mem_resp.get("status", "pending")
                    if mem_status == "failed":
                        failed.append(doc_id)
                    elif mem_status == "done":
                        completed.append(doc_id)
                    else:
                        still_pending.add(doc_id)
                except Exception:
                    still_pending.add(doc_id)

            pending = still_pending
            if pending:
                elapsed = int(time.time() - start)
                if elapsed > 0 and elapsed % 30 < (backoff_ms / 1000 + 1):
                    print(f"    … {len(completed)} done, {len(failed)} failed, "
                          f"{len(pending)} pending ({elapsed}s elapsed)")
                time.sleep(backoff_ms / 1000)
                backoff_ms = min(backoff_ms * 1.2, 5000)

        if failed:
            print(f"  ⚠ {len(failed)} documents failed indexing")
        if pending:
            print(f"  ⚠ {len(pending)} documents still pending after {timeout}s timeout")
        else:
            print(f"  ✓ All {len(completed)} documents indexed successfully")

        self._pending_doc_ids.clear()

    def _get_json(self, path):
        """GET *path* and return parsed JSON (with rate-limit handling)."""
        resp = self._get(path)
        if resp.status_code in (401, 429):
            raise RateLimitError(
                retry_after=self._parse_retry_after(resp) or 5, response=resp
            )
        resp.raise_for_status()
        return resp.json()

    def search(self, query, user_id, top_k):
        """Search memories via POST /v4/search.

        Aligned with official docs and memorybench: ``searchMode: "hybrid"``
        searches both memories and document chunks.  ``include`` is deprecated
        in favour of the hybrid search mode.
        """
        threshold = env_float("SUPERMEMORY_SEARCH_THRESHOLD", 0.3, min_value=0)
        rerank = env_bool("SUPERMEMORY_SEARCH_RERANK", False)
        search_mode = env_str("SUPERMEMORY_SEARCH_MODE", "hybrid")
        payload = {
            "q": query,
            "containerTag": user_id,
            "threshold": threshold,
            "rerank": rerank,
            "limit": top_k,
            "searchMode": search_mode,
        }

        def _do():
            resp = self._post("/v4/search", json=payload)
            if resp.status_code in (401, 429):
                raise RateLimitError(
                    retry_after=self._parse_retry_after(resp) or 5, response=resp
                )
            resp.raise_for_status()
            return resp.json()

        data = self._retry(_do, max_retries=10)
        results = data.get("results", [])
        return self._format_context(results)

    @staticmethod
    def _format_context(results):
        """Build structured context from search results.

        Handles both legacy (``chunks`` array) and current (``chunk`` string)
        response formats.  Temporal context is extracted from ``metadata``.
        """
        memory_parts = []
        all_chunks = []

        for i, result in enumerate(results):
            if not isinstance(result, dict):
                continue

            memory = result.get("memory", "")
            parts = [f"Result {i + 1}:", memory] if memory else []

            metadata = result.get("metadata") or {}
            temporal = metadata.get("temporalContext")
            if temporal:
                info = []
                if temporal.get("documentDate"):
                    info.append(f"documentDate: {temporal['documentDate']}")
                event_date = temporal.get("eventDate")
                if event_date:
                    dates = event_date if isinstance(event_date, list) else [event_date]
                    info.append(f"eventDate: {', '.join(str(d) for d in dates)}")
                if info:
                    parts.append(f"Temporal Context: {' | '.join(info)}")

            updated_at = result.get("updatedAt")
            if updated_at and not temporal:
                parts.append(f"updatedAt: {updated_at}")

            if parts:
                memory_parts.append("\n".join(parts))

            # Legacy format: chunks array
            for chunk in result.get("chunks", []):
                if isinstance(chunk, dict) and chunk.get("content"):
                    all_chunks.append((chunk.get("position", i), chunk["content"]))

            # Current format: single chunk string
            standalone_chunk = result.get("chunk")
            if standalone_chunk and isinstance(standalone_chunk, str) and standalone_chunk.strip():
                all_chunks.append((i, standalone_chunk))

        seen = set()
        unique_chunks = []
        for pos, content in all_chunks:
            if content not in seen:
                seen.add(content)
                unique_chunks.append((pos, content))
        unique_chunks.sort(key=lambda x: x[0])

        sections = "\n\n---\n\n".join(memory_parts)
        if unique_chunks:
            chunks_text = "\n\n---\n\n".join(c for _, c in unique_chunks)
            sections += f"\n\n=== DEDUPLICATED CHUNKS ===\n{chunks_text}"

        return sections

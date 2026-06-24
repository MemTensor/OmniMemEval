import time

from .base_client import (
    BaseApiClient,
    env_bool,
    env_float,
    env_int,
    env_max_batch_chars,
    env_str,
    iter_batches,
)


class GraphitiClient(BaseApiClient):
    """Graphiti (getzep/graphiti) REST API client for MemEval.

    Reference: https://github.com/getzep/graphiti
    Endpoints:
        POST   /messages              – ingest messages (async, 202)
        POST   /messages/sync         – ingest messages (sync, 200; waits for completion)
        POST   /search                – search facts
        DELETE /group/{group_id}      – delete a group
        POST   /clear                 – clear entire graph
        GET    /episodes/{group_id}   – list episodes
        GET    /healthcheck           – health check

    Set GRAPHITI_SYNC_ADD=1 to use the synchronous endpoint so each add()
    blocks until all episodes are fully processed (LLM + Neo4j).
    """

    def __init__(self):
        base_url = env_str("GRAPHITI_BASE_URL", "http://localhost:8000")
        qps = env_float("GRAPHITI_QPS", 10, min_value=0)
        self._sync_add = env_bool("GRAPHITI_SYNC_ADD", False)
        default_timeout = 3600 if self._sync_add else 120
        timeout = env_int("GRAPHITI_TIMEOUT", default_timeout, min_value=1)
        self._batch_size = env_int("GRAPHITI_BATCH_SIZE", 20, min_value=1)
        self._sync_add_retries = env_int("GRAPHITI_SYNC_ADD_RETRIES", 1, min_value=1)
        self._max_batch_chars = env_max_batch_chars("GRAPHITI_MAX_BATCH_CHARS")
        super().__init__(
            base_url=base_url,
            headers={"Content-Type": "application/json"},
            qps=qps,
            timeout=timeout,
        )

    def _group_id(self, user_id: str) -> str:
        return user_id

    def add(self, messages, user_id, **kwargs):
        """Ingest messages into a Graphiti group.

        Args:
            messages: List of {"role": str, "content": str, "name": str, ...}
            user_id: Maps to Graphiti group_id.
            **kwargs:
                session_key: Used for episode naming.
                timestamp: ISO-8601 string for message reference time.
                raw_content: If provided, send as a single message.
                uuid: Ignored. Current Graphiti treats episode UUIDs as
                    existing-node lookups during add, not create-time IDs.
                max_batch_chars: Per-call character budget override. Set 0
                    to disable client-side splitting for this add call.
        """
        group_id = self._group_id(user_id)
        raw_content = kwargs.get("raw_content")
        timestamp = kwargs.get("timestamp")
        session_key = kwargs.get("session_key", "")
        source_description = kwargs.get("source_description", "")
        max_batch_chars = kwargs.get("max_batch_chars", self._max_batch_chars)

        if raw_content is not None:
            item = {
                "content": raw_content,
                "role_type": "user",
                "role": kwargs.get("role", "conversation"),
                "name": f"episode_{session_key}" if session_key else "episode",
                "timestamp": timestamp or self._now_iso(),
                "source_description": source_description,
            }
            graphiti_messages = [item]
        elif messages:
            graphiti_messages = []
            for i, msg in enumerate(messages):
                role = msg.get("role", "user")
                role_type = msg.get("role_type") or ("assistant" if role == "assistant" else "user")
                name = msg.get("name", role)
                content = msg.get("content", "")
                ts = msg.get("chat_time") or msg.get("timestamp") or timestamp
                item = {
                    "content": content,
                    "role_type": role_type,
                    "role": name,
                    "name": f"ep_{session_key}_{i}" if session_key else f"ep_{i}",
                    "timestamp": ts or self._now_iso(),
                    "source_description": msg.get("source_description", source_description),
                }
                graphiti_messages.append(item)
        else:
            return

        endpoint = "/messages/sync" if self._sync_add else "/messages"
        max_retries = self._sync_add_retries if self._sync_add else None

        for batch in iter_batches(
            graphiti_messages,
            self._batch_size,
            max_chars=max_batch_chars,
        ):
            payload = {
                "group_id": group_id,
                "messages": batch,
            }

            def _do(p=payload):
                resp = self._post(endpoint, json=p)
                resp.raise_for_status()
                return resp

            self._retry(_do, max_retries=max_retries)

    def search(self, query, user_id, top_k=10, **kwargs):
        """Search facts from the knowledge graph.

        Returns a formatted text string of facts for LLM consumption.
        """
        group_id = self._group_id(user_id)
        payload = {
            "group_ids": [group_id],
            "query": query,
            "max_facts": top_k,
        }

        def _do():
            resp = self._post("/search", json=payload)
            resp.raise_for_status()
            return resp

        resp = self._retry(_do)
        data = resp.json()
        return self._format_search_results(data)

    def delete(self, user_id):
        """Delete all data for a group."""
        group_id = self._group_id(user_id)

        def _do():
            resp = self._delete(f"/group/{group_id}")
            if resp.status_code == 404:
                return resp
            resp.raise_for_status()
            return resp

        self._retry(_do)

    def clear(self):
        """Clear the entire graph database."""
        def _do():
            resp = self._post("/clear")
            resp.raise_for_status()
            return resp

        self._retry(_do)

    def healthcheck(self):
        resp = self._get("/healthcheck")
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _format_search_results(data):
        """Format Graphiti facts, entities, and supporting episodes."""
        facts = data.get("facts", [])
        nodes = data.get("nodes", [])
        episodes = data.get("episodes", [])
        lines = []
        if facts:
            lines.append("Facts:")
            for f in facts:
                fact_text = f.get("fact", "")
                name = f.get("name", "")
                valid_at = f.get("valid_at")
                invalid_at = f.get("invalid_at")

                parts = []
                if name:
                    parts.append(f"[{name}] {fact_text}")
                else:
                    parts.append(fact_text)

                meta = []
                if valid_at:
                    meta.append(f"valid: {valid_at[:10]}")
                if invalid_at:
                    meta.append(f"invalid: {invalid_at[:10]}")
                if meta:
                    parts.append(f"({' | '.join(meta)})")

                lines.append("- " + " ".join(parts))

        if nodes:
            if lines:
                lines.append("")
            lines.append("Relevant entities:")
            for node in nodes:
                name = node.get("name", "")
                summary = node.get("summary", "")
                if summary:
                    lines.append(f"- {name}: {summary}")
                elif name:
                    lines.append(f"- {name}")

        if episodes:
            if lines:
                lines.append("")
            lines.append("Supporting conversation snippets:")
            for episode in episodes:
                content = (episode.get("content") or "").strip()
                valid_at = episode.get("valid_at")
                if not content:
                    continue
                prefix = f"- ({valid_at[:10]}) " if valid_at else "- "
                lines.append(prefix + content.replace("\n", "\n  "))
        return "\n".join(lines)

    @staticmethod
    def _now_iso():
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

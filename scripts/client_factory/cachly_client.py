import hashlib
import json
import os
from datetime import datetime, timezone

from .base_client import BaseApiClient, env_int, env_str


class CachlyClient(BaseApiClient):
    """cachly AI Brain client (self-hosted add/search service, AML memory-api-v1.1).

    Talks to a service in front of a cachly Brain: POST /v1/memory/add,
    /v1/memory/search and /v1/memory/delete with a bearer key. ``add`` is
    synchronous: it returns once the stored memories are searchable. Every
    ``user_id`` has its own isolated store.

    Environment variables:
        CACHLY_AML_URL            - base URL of the service (required)
        CACHLY_AML_KEY            - bearer key of the service (required; environment only)
        CACHLY_ADD_TIMEOUT        - seconds per add call (default: 3600; add waits for embeddings)
        CACHLY_SEARCH_TIMEOUT     - seconds per search call (default: 120)
    """

    def __init__(self):
        url = env_str("CACHLY_AML_URL", "")
        if not url:
            raise ValueError("CACHLY_AML_URL environment variable is not set")
        key = env_str("CACHLY_AML_KEY", "")
        if not key:
            raise ValueError("CACHLY_AML_KEY environment variable is not set")
        super().__init__(
            base_url=url,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {key}",
            },
        )
        self._add_timeout = env_int("CACHLY_ADD_TIMEOUT", 3600, min_value=1)
        self._search_timeout = env_int("CACHLY_SEARCH_TIMEOUT", 120, min_value=1)

    @staticmethod
    def _timestamp(msg):
        return msg.get("chat_time") or msg.get("timestamp") or ""

    @staticmethod
    def _unix_ms(value):
        """LoCoMo '1:56 pm on 8 May, 2023', ISO strings or numbers -> Unix ms (UTC), else None."""
        if value in (None, ""):
            return None
        if isinstance(value, (int, float)):
            return int(value if value > 1e11 else value * 1000)
        text = str(value).strip()
        for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y", "%Y/%m/%d (%a) %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(text, fmt)
                return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
            except ValueError:
                pass
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            return None

    def add(self, messages, user_id, batch_size=None):
        msgs = []
        for m in messages:
            name = m.get("name") or ""
            content = m["content"]
            # LoCoMo already sends "Speaker: text"; only prefix when the name is missing.
            if name and not content.startswith(f"{name}:"):
                content = f"{name}: {content}"
            entry = {
                "role": "assistant" if m.get("role") == "assistant" else "user",
                "content": content,
            }
            ms = self._unix_ms(self._timestamp(m))
            if ms is not None:
                entry["timestamp"] = ms
            msgs.append(entry)
        digest = hashlib.sha256(json.dumps([user_id, msgs], sort_keys=True).encode("utf-8")).hexdigest()[:32]
        payload = {"request_id": f"omni:{digest}", "messages": msgs, "user_id": user_id, "session_id": user_id}

        def _do():
            resp = self._post("/v1/memory/add", json=payload, timeout=self._add_timeout)
            resp.raise_for_status()
            return resp.json()

        result = self._retry(_do)
        if not result.get("success") or result.get("request_id") != payload["request_id"]:
            raise RuntimeError(f"cachly add incomplete: {result}")

    def search(self, query, user_id, top_k):
        """Return plain text: one stored memory (full lesson text) per line, in recall order."""
        payload = {"query": query, "user_id": user_id, "top_k": top_k}

        def _do():
            resp = self._post("/v1/memory/search", json=payload, timeout=self._search_timeout)
            resp.raise_for_status()
            return resp.json()

        res = self._retry(_do)
        return "\n".join(d.get("content", "") for d in res.get("data", []))

    def delete(self, user_id):
        """Remove the store of ``user_id``. Unknown users are not an error."""
        resp = self._post("/v1/memory/delete", json={"user_id": user_id})
        resp.raise_for_status()

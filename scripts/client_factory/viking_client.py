import time
import uuid

from contextlib import suppress

from .base_client import (
    BaseApiClient,
    env_csv,
    env_int,
    env_max_batch_chars,
    env_str,
    require_env,
    iter_batches,
)

# ── Viking memory client ─────────────────────────────────────────────────────


class VikingClient:
    """Viking memory client (Volcengine VikingDB Memory).

    Reference: https://www.volcengine.com/docs/84313/1946665
    SDK: pip install vikingdb-python-sdk  (import module: vikingdb)

    Uses the official ``vikingdb`` Python SDK (v2, ≥0.1.18) for auth and
    collection operations.  One VikingDB *collection* maps to one evaluation
    run; ``user_id`` + ``assistant_id`` isolate per-user data.

    **Data model**: Viking is **per-user_id** — each user_id has its own
    events and profiles.  This is analogous to mem0/memos, NOT to
    Letta/Cognee (which are per-conversation).  For LoCoMo we therefore
    use **dual-perspective ingestion**: speaker A's messages are ingested
    under speaker_a_user_id (A=user, B=assistant), speaker B's under
    speaker_b_user_id (B=user, A=assistant).  This ensures:
    - Viking's profile extraction captures each speaker's individual traits.
    - Search by user_id returns that speaker's perspective.

    **Ingestion** uses ``AddSession`` — the only interface that triggers
    Viking's built-in LLM extraction pipeline to auto-generate events +
    profiles from raw conversation.  ``AddEvent`` / ``AddProfile`` bypass
    the extraction pipeline and are not used here.

    **Search** defaults to split mode — ``SearchEventMemory`` +
    ``SearchProfileMemory`` — because the specialised event endpoint
    exposes ``time_decay_config`` (time-aware re-ranking) unavailable in
    the generic ``SearchMemory``.  Set ``VIKING_SEARCH_MODE=unified`` to
    fall back to a single ``SearchMemory`` call.

    Env vars (all optional unless noted):
        VIKING_API_KEY          – API key (**required**)
        VIKING_HOST             – endpoint (default: api-knowledgebase.mlp.cn-beijing.volces.com)
        VIKING_REGION           – region (default: cn-beijing)
        VIKING_SCHEME           – http or https (default: https)
        VIKING_COLLECTION       – collection name (**required**)
        VIKING_PROJECT          – project name (default: default)
        VIKING_ASSISTANT_ID     – assistant id for isolation (default: memeval)
        VIKING_EVENT_TYPES      – event types to search (default: event_v1)
        VIKING_PROFILE_TYPES    – profile types to search (default: profile_v1)
        VIKING_PROFILE_LIMIT    – max profile results per search (default: 10)
        VIKING_SEARCH_MODE      – "split" (default) or "unified"
        VIKING_TIMEOUT          – SDK request timeout in seconds (default: 120)
    """

    def __init__(self):
        api_key = require_env("VIKING_API_KEY")
        host = env_str("VIKING_HOST", "api-knowledgebase.mlp.cn-beijing.volces.com")
        region = env_str("VIKING_REGION", "cn-beijing")
        scheme = env_str("VIKING_SCHEME", "https")
        collection_name = require_env("VIKING_COLLECTION")
        project_name = env_str("VIKING_PROJECT", "default")

        from vikingdb import APIKey
        from vikingdb.memory import VikingMem

        auth = APIKey(api_key=api_key)
        self._client = VikingMem(
            host=host, region=region, auth=auth, scheme=scheme,
        )
        self._collection = self._client.get_collection(
            collection_name=collection_name, project_name=project_name,
        )
        self._assistant_id = env_str("VIKING_ASSISTANT_ID", "memeval")
        self._event_types = env_csv("VIKING_EVENT_TYPES") or ["event_v1"]
        self._profile_types = env_csv("VIKING_PROFILE_TYPES") or ["profile_v1"]
        self._profile_limit = env_int("VIKING_PROFILE_LIMIT", 10, min_value=0)
        self._search_mode = env_str("VIKING_SEARCH_MODE", "split").lower()
        self._timeout = env_int("VIKING_TIMEOUT", 120, min_value=1)
        self._batch_size = env_int("VIKING_BATCH_SIZE", 20, min_value=1)
        self._max_batch_chars = env_max_batch_chars("VIKING_MAX_BATCH_CHARS")

    def _add_session_batch(self, session_messages, user_id, metadata):
        """Send one batch of messages via ``AddSession``."""
        session_id = f"{user_id}_{uuid.uuid4().hex[:8]}"

        def _do():
            self._collection.add_session(
                session_id=session_id,
                messages=session_messages,
                metadata=metadata,
                timeout=self._timeout,
            )

        BaseApiClient.sdk_retry(_do)

    def add(self, messages, user_id, **kwargs):
        """Ingest messages via ``AddSession`` under the given user_id.

        Called once per speaker per session (dual-perspective mode).
        Viking's LLM pipeline extracts events and profiles scoped to
        this user_id, capturing that speaker's perspective.

        Messages are automatically chunked by ``VIKING_BATCH_SIZE``
        (default 20) to stay within the AddSession API limit.
        """
        all_msgs = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
        ]

        chat_time = None
        for m in messages:
            ct = m.get("chat_time") or m.get("timestamp")
            if ct:
                chat_time = ct
                break

        metadata = {
            "default_user_id": user_id,
            "default_assistant_id": self._assistant_id,
        }
        if chat_time:
            try:
                from datetime import datetime as _dt
                dt = _dt.fromisoformat(str(chat_time))
                metadata["time"] = int(dt.timestamp() * 1000)
            except (ValueError, TypeError):
                pass
        if "time" not in metadata:
            metadata["time"] = int(time.time() * 1000)

        for batch in iter_batches(all_msgs, self._batch_size, max_chars=self._max_batch_chars):
            self._add_session_batch(batch, user_id, metadata)

    @staticmethod
    def _extract_result_text(result_list):
        """Extract readable text from search result_list entries."""
        parts = []
        for r in result_list:
            info = r.get("memory_info", {})
            if not isinstance(info, dict):
                if info:
                    parts.append(str(info))
                continue
            text = info.get("user_profile", "")
            if not text:
                text = info.get("summary", "")
            if not text:
                text = info.get("original_messages", "")
            if not text:
                text = " | ".join(
                    f"{k}: {v}" for k, v in info.items() if v
                ) if info else ""
            if text:
                parts.append(text)
        return parts

    def _search_split(self, query, user_id, top_k):
        """Split-mode: SearchEventMemory + SearchProfileMemory separately.

        Advantages over unified SearchMemory:
        - SearchEventMemory supports ``time_decay_config`` for time-aware
          re-ranking (important for LoCoMo's temporally-spread dialogues).
        - Guarantees both event and profile results are included regardless
          of relative score distribution.
        """
        event_limit = top_k
        profile_limit = self._profile_limit
        base_filter = {"user_id": user_id, "assistant_id": self._assistant_id}

        event_result = self._collection.search_event_memory(
            query=query,
            filter={**base_filter, "memory_type": self._event_types},
            limit=event_limit,
            timeout=self._timeout,
        )
        event_data = event_result.get("data", {}) if isinstance(event_result, dict) else {}
        event_list = event_data.get("result_list", [])

        profile_list = []
        if self._profile_types and profile_limit > 0:
            profile_result = self._collection.search_profile_memory(
                query=query,
                filter={**base_filter, "memory_type": self._profile_types},
                limit=profile_limit,
                timeout=self._timeout,
            )
            profile_data = profile_result.get("data", {}) if isinstance(profile_result, dict) else {}
            profile_list = profile_data.get("result_list", [])

        parts = self._extract_result_text(profile_list) + self._extract_result_text(event_list)
        return "\n\n".join(parts) if parts else ""

    def _search_unified(self, query, user_id, top_k):
        """Unified-mode: single SearchMemory call (event+profile mixed)."""
        all_types = self._event_types + self._profile_types
        result = self._collection.search_memory(
            query=query,
            filter={
                "memory_type": all_types,
                "user_id": user_id,
                "assistant_id": self._assistant_id,
            },
            limit=top_k,
            timeout=self._timeout,
        )
        data = result.get("data", {}) if isinstance(result, dict) else {}
        result_list = data.get("result_list", [])
        parts = self._extract_result_text(result_list)
        return "\n\n".join(parts) if parts else ""

    def search(self, query, user_id, top_k):
        """Search memories — dispatches to split or unified mode."""
        def _do():
            if self._search_mode == "unified":
                return self._search_unified(query, user_id, top_k)
            return self._search_split(query, user_id, top_k)

        return BaseApiClient.sdk_retry(_do)

    def delete_user(self, user_id):
        with suppress(Exception):
            self._collection.batch_delete_event(
                filter={"user_id": user_id, "assistant_id": self._assistant_id},
            )
        with suppress(Exception):
            self._collection.batch_delete_profile(
                filter={"user_id": user_id, "assistant_id": self._assistant_id},
            )

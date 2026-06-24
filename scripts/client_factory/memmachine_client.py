import re

from contextlib import suppress

from .base_client import (
    BaseApiClient,
    env_bool,
    env_csv,
    env_float,
    env_int,
    env_max_batch_chars,
    env_str,
    iter_batches,
)
# ── MemMachine ────────────────────────────────────────────────────────────────


class MemMachineClient(BaseApiClient):
    """MemMachine universal memory layer client (REST API v0.3.5+).

    Reference: https://docs.memmachine.ai/api-reference/memories/
    Paper: arXiv:2604.04853

    Architecture (paper §3):
        - **Episodic memory**: short-term (recent turns) + long-term (vector-indexed
          sentences with contextualized retrieval expanding nucleus episodes).
        - **Semantic / Profile memory**: structured knowledge graph of Sets →
          Categories → Tags, auto-extracted by LLM from ingested episodes.
        - **Retrieval Agent** (``agent_mode``): LLM-orchestrated multi-hop search
          that routes queries to direct / parallel / chain-of-query strategies.

    User isolation:
        - **Cloud** (``/v2`` prefix): the ``producer`` field is a first-class
          filter dimension; searches use ``producer=<user_id>``.
        - **Local** (``/api/v2`` prefix): semantic memory only supports
          ``metadata.*`` filters.  Messages carry ``metadata.producer`` and
          searches filter with ``metadata.producer=<user_id>``.

    Cloud/local routing is controlled only by ``MEMMACHINE_MODE``.

    Key env vars (all optional, sensible defaults from official docs):
        MEMMACHINE_API_KEY          Bearer token for cloud / auth-enabled servers
        MEMMACHINE_MODE            cloud|local  (default: cloud)
        MEMMACHINE_BASE_URL         Server root (cloud default:
                                    https://api.memmachine.ai; local default:
                                    http://localhost:8080)
        MEMMACHINE_ORG_ID           Organization namespace (default: universal)
        MEMMACHINE_PROJECT_ID       Project namespace (default: universal)
        MEMMACHINE_TYPES            Comma-separated memory types (default: episodic,semantic)
        MEMMACHINE_AGENT_MODE       Enable retrieval-agent orchestration (default: true)
        MEMMACHINE_EXPAND_CONTEXT   Neighbour episodes to include per match (default: 3)
        MEMMACHINE_SCORE_THRESHOLD  Min score filter; empty = disabled (default: empty)
        MEMMACHINE_QPS              Client-side rate limit (default: none)
    """

    def __init__(self):
        mode = self._resolve_mode()
        default_base_url = (
            "http://localhost:8080"
            if mode == "local"
            else "https://api.memmachine.ai"
        )
        api_key = env_str("MEMMACHINE_API_KEY", "")
        base_url = env_str("MEMMACHINE_BASE_URL", default_base_url)
        qps = env_float("MEMMACHINE_QPS", None, min_value=0)
        super().__init__(
            base_url=base_url,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
            },
            qps=qps,
        )
        self._mode = mode
        self._prefix = "/api/v2" if mode == "local" else "/v2"
        self._org_id = env_str("MEMMACHINE_ORG_ID", "universal")
        self._project_id = env_str("MEMMACHINE_PROJECT_ID", "universal")
        self._types = env_csv("MEMMACHINE_TYPES") or ["episodic", "semantic"]
        self._agent_mode = env_bool("MEMMACHINE_AGENT_MODE", True)
        self._expand_context = env_int(
            "MEMMACHINE_EXPAND_CONTEXT", 3, min_value=0
        )
        self._score_threshold = env_float(
            "MEMMACHINE_SCORE_THRESHOLD", None, min_value=0
        )
        self._batch_size = env_int("MEMMACHINE_BATCH_SIZE", 20, min_value=1)
        self._max_batch_chars = env_max_batch_chars("MEMMACHINE_MAX_BATCH_CHARS")
        self._local_mode = mode == "local"

    @staticmethod
    def _resolve_mode():
        mode = env_str("MEMMACHINE_MODE", "").lower()
        mode = mode or "cloud"
        if mode not in ("cloud", "local"):
            raise ValueError("MEMMACHINE_MODE must be 'cloud' or 'local'")
        return mode

    def _scope(self):
        """Return the org/project scope dict shared by all requests."""
        return {"org_id": self._org_id, "project_id": self._project_id}

    @staticmethod
    def _filter_literal(value):
        value = str(value)
        if re.fullmatch(r"[A-Za-z0-9_.]+", value):
            return value
        if "'" in value:
            raise ValueError("MemMachine filter values cannot contain single quotes")
        return f"'{value}'"

    def _producer_filter(self, user_id):
        producer = self._filter_literal(user_id)
        if self._local_mode:
            return f"metadata.producer={producer}"
        return f"producer={producer}"

    def add(self, messages, user_id, **kwargs):
        all_mm = []
        for msg in messages:
            ts = msg.get("chat_time") or msg.get("timestamp")
            mm_msg = {
                "content": msg["content"],
                "role": msg.get("role", "user"),
                "producer": str(user_id),
            }
            if self._local_mode:
                mm_msg["metadata"] = {"producer": str(user_id)}
            if ts:
                mm_msg["timestamp"] = str(ts)
            all_mm.append(mm_msg)

        for batch in iter_batches(all_mm, self._batch_size,
                                  max_chars=self._max_batch_chars):
            payload = {
                **self._scope(),
                "messages": batch,
                "types": self._types,
            }

            def _do(p=payload):
                resp = self._post(f"{self._prefix}/memories", json=p)
                resp.raise_for_status()

            self._retry(_do)

    def search(self, query, user_id, top_k):
        payload = {
            **self._scope(),
            "query": query,
            "top_k": top_k,
            "filter": self._producer_filter(user_id),
            "types": self._types,
            "agent_mode": self._agent_mode,
            "expand_context": self._expand_context,
        }
        if self._score_threshold is not None:
            payload["score_threshold"] = self._score_threshold

        def _do():
            resp = self._post(f"{self._prefix}/memories/search", json=payload)
            resp.raise_for_status()
            return resp.json()

        result = self._retry(_do)
        content = result.get("content", {}) or {}

        parts = []
        episodic = content.get("episodic_memory") or {}
        for section in ("long_term_memory", "short_term_memory"):
            mem_section = episodic.get(section) or {}
            for ep in mem_section.get("episodes") or []:
                text = ep.get("content", "")
                if text:
                    ts = ep.get("created_at", "")
                    role = ep.get("producer_role", "")
                    prefix = f"[{ts}] " if ts else ""
                    prefix += f"({role}) " if role else ""
                    parts.append(f"{prefix}{text}")
            for s in mem_section.get("episode_summary") or []:
                if s:
                    parts.append(s)

        for sem in content.get("semantic_memory") or []:
            val = sem.get("value", "")
            tag = sem.get("tag", "")
            name = sem.get("feature_name", "")
            cat = sem.get("category", "")
            if val:
                label = " - ".join(p for p in (cat, tag, name) if p)
                parts.append(f"{label}: {val}" if label else val)

        return "\n\n".join(parts) if parts else ""

    def delete_user(self, user_id):
        scope = self._scope()
        prefix = self._prefix

        def _list(memory_type):
            resp = self._post(
                f"{prefix}/memories/list",
                json={
                    **scope,
                    "filter": self._producer_filter(user_id),
                    "page_num": 0,
                    "page_size": 100,
                    "type": memory_type,
                },
            )
            resp.raise_for_status()
            return resp.json()

        with suppress(Exception):
            data = self._retry(lambda: _list("episodic"))
            episodic = (data.get("content", {}) or {}).get("episodic_memory", [])
            episodic_ids = [
                item.get("uid")
                for item in episodic
                if isinstance(item, dict) and item.get("uid")
            ]
            if episodic_ids:
                resp = self._post(
                    f"{prefix}/memories/episodic/delete",
                    json={**scope, "episodic_ids": episodic_ids},
                )
                resp.raise_for_status()

        with suppress(Exception):
            data = self._retry(lambda: _list("semantic"))
            semantic = (data.get("content", {}) or {}).get("semantic_memory", [])
            semantic_ids = [
                item.get("set_id") or (item.get("metadata") or {}).get("id")
                for item in semantic
                if isinstance(item, dict)
            ]
            semantic_ids = [sid for sid in semantic_ids if sid]
            if semantic_ids:
                resp = self._post(
                    f"{prefix}/memories/semantic/delete",
                    json={**scope, "semantic_ids": semantic_ids},
                )
                resp.raise_for_status()

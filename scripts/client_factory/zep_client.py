import time

from contextlib import suppress
from datetime import datetime, timezone

from .base_client import (
    BaseApiClient,
    RateLimitError,
    _AttrDict,
    env_bool,
    env_csv,
    env_float,
    env_int,
    env_json,
    env_max_batch_chars,
    env_optional_bool,
    env_str,
    require_env,
    iter_batches,
)

class ZepClient(BaseApiClient):
    """Zep Cloud graph memory client — current REST API.

    References:
        https://help.getzep.com/v3/adding-messages
        https://help.getzep.com/sdk-reference/thread/add-messages-to-a-thread
        https://help.getzep.com/sdk-reference/graph/search

    Endpoints:
        POST /api/v2/users                       – create user
        POST /api/v2/threads                     – create thread
        POST /api/v2/threads/{id}/messages       – add messages to thread
        POST /api/v2/graph/search                – search graph
        GET  /api/v2/tasks/{id}                  – poll async ingestion task
        DELETE /api/v2/users/{id}                – delete user

    Ingestion uses the current thread API for chat history. Zep stores thread
    messages and builds the user-level knowledge graph from them.

    Configurable via env:
        ZEP_QPS                       – local rate limit (default 5)
        ZEP_BATCH_SIZE                – add_messages chunk size (default 20, max 30)
        ZEP_MESSAGE_MAX_CHARS         – max chars per message (default/max 4096)
        ZEP_WAIT_FOR_INGESTION        – poll returned task_id (default true)
        ZEP_INGEST_TIMEOUT_SECONDS    – task polling timeout (default 300)
        ZEP_SEARCH_SCOPES             – comma-separated scopes (default edges)
        ZEP_RERANKER                  – global search reranker (optional)
        ZEP_EDGES_RERANKER            – edges scope reranker (overrides global)
        ZEP_NODES_RERANKER            – nodes scope reranker (overrides global)
        ZEP_MMR_LAMBDA                – required when reranker=mmr
        ZEP_SEARCH_FILTERS_JSON       – JSON search_filters object
        ZEP_MAX_CHARACTERS            – for scope=auto (unset = official 2000)
        ZEP_RETURN_RAW_RESULTS        – for scope=auto
        ZEP_DISABLE_DEFAULT_ONTOLOGY  – optional user creation flag
    """

    _LIST_RESULT_KEYS = ("edges", "nodes", "episodes", "observations", "thread_summaries")
    _TASK_DONE = {"completed", "succeeded", "success"}
    _TASK_FAILED = {"failed", "cancelled", "canceled", "error"}

    def __init__(self):
        api_key = require_env("ZEP_API_KEY")
        base_url = env_str("ZEP_BASE_URL", "https://api.getzep.com")
        qps = env_float("ZEP_QPS", 5, min_value=0)
        self._thread_counter = 0
        self._sdk = None
        try:
            from zep_cloud.client import Zep as _ZepSDK
            self._sdk = _ZepSDK(api_key=api_key)
        except ImportError:
            pass
        self._message_batch_size = min(
            max(env_int("ZEP_BATCH_SIZE", 20), 1),
            30,
        )
        self._max_batch_chars = env_max_batch_chars("ZEP_MAX_BATCH_CHARS")
        self._message_max_chars = min(
            max(env_int("ZEP_MESSAGE_MAX_CHARS", 4096), 1),
            4096,
        )
        self._wait_for_ingestion = env_bool("ZEP_WAIT_FOR_INGESTION", True)
        self._ingest_timeout = env_float(
            "ZEP_INGEST_TIMEOUT_SECONDS", 300, min_value=1
        )
        self._ingest_poll_interval = env_float(
            "ZEP_INGEST_POLL_INTERVAL", 1, min_value=0
        )
        self._ignore_roles = env_csv("ZEP_IGNORE_ROLES")
        self._return_context = env_optional_bool("ZEP_RETURN_CONTEXT")
        self._search_scopes = env_csv("ZEP_SEARCH_SCOPES") or ["edges"]
        self._reranker = env_str("ZEP_RERANKER", "") or None
        self._scope_rerankers = {
            "edges": env_str("ZEP_EDGES_RERANKER", "") or self._reranker,
            "nodes": env_str("ZEP_NODES_RERANKER", "") or self._reranker,
            "episodes": env_str("ZEP_EPISODES_RERANKER", "") or self._reranker,
        }
        self._mmr_lambda = env_float("ZEP_MMR_LAMBDA")
        self._query_max_chars = env_int("ZEP_QUERY_MAX_CHARS", 400, min_value=1)
        self._max_characters = env_int("ZEP_MAX_CHARACTERS")
        self._return_raw_results = env_optional_bool("ZEP_RETURN_RAW_RESULTS")
        self._search_filters = env_json("ZEP_SEARCH_FILTERS_JSON")
        self._disable_default_ontology = env_optional_bool(
            "ZEP_DISABLE_DEFAULT_ONTOLOGY"
        )
        super().__init__(
            base_url=base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Api-Key {api_key}",
            },
            qps=qps,
        )

    def _check_rate_limit(self, resp):
        if resp.status_code == 429:
            retry_after = self._parse_retry_after(resp)
            raise RateLimitError(retry_after=retry_after, response=resp)

    def _create_thread(self, thread_id, user_id):
        """Create a thread linked to a user."""
        payload = {"thread_id": thread_id, "user_id": user_id}

        def _do():
            resp = self._post("/api/v2/threads", json=payload)
            if resp.status_code == 409:
                return
            self._check_rate_limit(resp)
            resp.raise_for_status()

        self._retry(_do)

    @staticmethod
    def _to_rfc3339(value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        return str(value)

    def _wait_for_task(self, task_id):
        deadline = time.time() + self._ingest_timeout
        last_status = None
        while time.time() < deadline:
            def _do():
                resp = self._get(f"/api/v2/tasks/{task_id}")
                self._check_rate_limit(resp)
                resp.raise_for_status()
                return resp.json()

            task = self._retry(_do)
            last_status = str(task.get("status", "")).lower()
            if last_status in self._TASK_DONE:
                return
            if last_status in self._TASK_FAILED:
                raise RuntimeError(f"Zep ingestion task failed: {task}")
            time.sleep(self._ingest_poll_interval)
        raise TimeoutError(
            f"Zep ingestion task {task_id} did not finish within "
            f"{self._ingest_timeout:.0f}s (last_status={last_status})"
        )

    def add(self, messages, user_id, timestamp=None, thread_id=None, **kwargs):
        """Add conversation messages via the current thread API."""
        thread_id = thread_id or kwargs.get("session_id")
        if not thread_id:
            self._thread_counter += 1
            thread_id = f"{user_id}_thread_{self._thread_counter}"
        self._create_thread(thread_id, user_id)

        iso_date = self._to_rfc3339(timestamp)

        zep_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))[: self._message_max_chars]
            m = {
                "content": content,
                "role": role,
            }
            name = msg.get("name") or msg.get("speaker")
            if name:
                m["name"] = str(name)
            if iso_date:
                m["created_at"] = iso_date
            zep_messages.append(m)

        for chunk in iter_batches(zep_messages, self._message_batch_size,
                                  max_chars=self._max_batch_chars):
            payload = {"messages": chunk}
            if self._ignore_roles:
                payload["ignore_roles"] = self._ignore_roles
            if self._return_context is not None:
                payload["return_context"] = self._return_context

            def _do():
                resp = self._post(
                    f"/api/v2/threads/{thread_id}/messages",
                    json=payload,
                )
                self._check_rate_limit(resp)
                resp.raise_for_status()
                return resp.json()

            result = self._retry(_do)
            task_id = result.get("task_id") if isinstance(result, dict) else None
            if self._wait_for_ingestion and task_id:
                self._wait_for_task(task_id)

    _GRAPH_TEXT_TEMPLATE = (
        "FACTS and ENTITIES represent relevant context to the current conversation.\n\n"
        "# These are the most relevant facts for the conversation along with the "
        "datetime of the event that the fact refers to.\n"
        "If a fact mentions something happening a week ago, then the datetime will be "
        "the date time of last week and not the datetime of when the fact was stated.\n"
        "Timestamps in memories represent the actual time the event occurred, not the "
        "time the event was mentioned in a message.\n\n"
        "<FACTS>\n{facts}\n</FACTS>\n\n"
        "# These are the most relevant entities\n"
        "# ENTITY_NAME: entity summary\n"
        "<ENTITIES>\n{entities}\n</ENTITIES>"
    )

    @staticmethod
    def _format_graph_results(results):
        """Convert raw graph search results dict into formatted plain text."""
        facts = [
            f"  - {e.get('fact', '')} "
            f"(event_time: {e.get('valid_at') or e.get('created_at') or 'unknown'})"
            for e in results.get("edges", [])
            if e.get("fact")
        ]
        contexts = [
            "  - " + str(c).replace("\n", "\n    ")
            for c in results.get("contexts", [])
            if c
        ]
        entities = [
            f"  - {n.get('name', '')}: {n.get('summary', '')}"
            for n in results.get("nodes", [])
            if n.get("name") or n.get("summary")
        ]
        all_facts = facts + contexts
        return ZepClient._GRAPH_TEXT_TEMPLATE.format(
            facts="\n".join(all_facts) if all_facts else "  - No results",
            entities="\n".join(entities) if entities else "  - No results",
        )

    def search(self, query, user_id=None, top_k=20, group_id=None):
        """Search a user or group knowledge graph.

        Returns plain-text string with formatted facts, entities, and contexts.
        """
        query = str(query)[: self._query_max_chars]

        def _search_scope(scope):
            body = {
                "query": query,
                "scope": scope,
                "limit": top_k,
            }
            if group_id:
                body["group_id"] = group_id
            elif user_id:
                body["user_id"] = user_id
            reranker = self._scope_rerankers.get(scope) or self._reranker
            if reranker:
                body["reranker"] = reranker
            if self._mmr_lambda is not None:
                body["mmr_lambda"] = self._mmr_lambda
            if self._max_characters is not None:
                body["max_characters"] = self._max_characters
            if self._return_raw_results is not None:
                body["return_raw_results"] = self._return_raw_results
            if self._search_filters is not None:
                body["search_filters"] = self._search_filters

            def _do():
                resp = self._post("/api/v2/graph/search", json=body)
                self._check_rate_limit(resp)
                resp.raise_for_status()
                return resp.json()

            return self._retry(_do)

        results = {key: [] for key in self._LIST_RESULT_KEYS}
        results["contexts"] = []
        for scope in self._search_scopes:
            data = _search_scope(scope)
            context = data.get("context")
            if context:
                results["contexts"].append(context)
            for key in self._LIST_RESULT_KEYS:
                results[key].extend(_AttrDict(item) for item in (data.get(key) or []))

        return self._format_graph_results(results)

    def add_user(self, user_id):
        payload = {"user_id": user_id}
        if self._disable_default_ontology is not None:
            payload["disable_default_ontology"] = self._disable_default_ontology

        def _do():
            resp = self._post("/api/v2/users", json=payload)
            if resp.status_code in (200, 201, 409):
                return
            self._check_rate_limit(resp)
            resp.raise_for_status()

        self._retry(_do)

    def graph_add(self, user_id=None, data="", data_type="text", created_at=None, group_id=None):
        """Add data directly to a user/group graph via POST /api/v2/graph.

        Useful for supplementing thread-based ingestion with explicit text
        or JSON so the graph captures details that entity extraction may miss.
        The API returns 202 with an episode object; processing is async.

        Args:
            user_id: Target user graph (mutually exclusive with group_id).
            data: The text or JSON string to add (max 10,000 chars).
            data_type: One of "text", "json", "message".
            created_at: Optional RFC3339 timestamp for temporal context.
            group_id: Target group graph (mutually exclusive with user_id).
        """
        payload = {
            "data": str(data)[:10000],
            "type": data_type,
        }
        if user_id:
            payload["user_id"] = user_id
        if group_id:
            payload["group_id"] = group_id
        if created_at:
            payload["created_at"] = (
                self._to_rfc3339(created_at)
                if isinstance(created_at, (int, float))
                else str(created_at)
            )

        def _do():
            resp = self._post("/api/v2/graph", json=payload)
            self._check_rate_limit(resp)
            resp.raise_for_status()
            return resp.json()

        return self._retry(_do)

    def delete_user(self, user_id):
        with suppress(Exception):
            self._delete(f"/api/v2/users/{user_id}")

    def add_group(self, group_id):
        """Create a group for multi-party conversations.

        Tolerates 404 (endpoint unavailable on some plans) since
        ``graph.add`` with ``group_id`` may auto-create the group.
        """
        payload = {"group_id": group_id}

        def _do():
            resp = self._post("/api/v2/groups", json=payload)
            if resp.status_code in (200, 201, 409):
                return
            if resp.status_code == 404:
                return
            self._check_rate_limit(resp)
            resp.raise_for_status()

        self._retry(_do)

    def delete_group(self, group_id):
        """Delete a group and its associated graph data."""
        with suppress(Exception):
            self._delete(f"/api/v2/groups/{group_id}")

    # ── SDK-based graph methods (graph_id mode for LoCoMo) ──────────────

    def sdk_graph_create(self, graph_id):
        """Create a standalone graph via zep-cloud SDK."""
        if not self._sdk:
            raise RuntimeError("zep-cloud SDK not installed")
        with suppress(Exception):
            BaseApiClient.sdk_retry(lambda: self._sdk.graph.delete(graph_id))
        BaseApiClient.sdk_retry(lambda: self._sdk.graph.create(graph_id=graph_id))

    def sdk_graph_add(self, graph_id, data, data_type="message", created_at=None):
        """Add data to a standalone graph via zep-cloud SDK."""
        if not self._sdk:
            raise RuntimeError("zep-cloud SDK not installed")
        kwargs = {"data": str(data)[:10000], "graph_id": graph_id, "type": data_type}
        if created_at:
            kwargs["created_at"] = (
                self._to_rfc3339(created_at)
                if isinstance(created_at, (int, float))
                else str(created_at)
            )
        BaseApiClient.sdk_retry(lambda: self._sdk.graph.add(**kwargs))

    def sdk_graph_search(self, graph_id, query, top_k=20):
        """Search a standalone graph via zep-cloud SDK.

        Returns plain-text string with formatted facts, entities, and contexts.
        """
        if not self._sdk:
            raise RuntimeError("zep-cloud SDK not installed")
        query = str(query)[: self._query_max_chars]

        combined = {key: [] for key in self._LIST_RESULT_KEYS}
        combined["contexts"] = []

        for scope in self._search_scopes:
            kwargs = {
                "graph_id": graph_id,
                "query": query,
                "limit": top_k,
                "scope": scope,
            }
            reranker = self._scope_rerankers.get(scope) or self._reranker
            if reranker:
                kwargs["reranker"] = reranker
            if self._mmr_lambda is not None:
                kwargs["mmr_lambda"] = self._mmr_lambda
            if self._max_characters is not None:
                kwargs["max_characters"] = self._max_characters
            if self._return_raw_results is not None:
                kwargs["return_raw_results"] = self._return_raw_results
            if self._search_filters is not None:
                from zep_cloud.types import SearchFilters
                kwargs["search_filters"] = SearchFilters(**self._search_filters)

            result = BaseApiClient.sdk_retry(lambda: self._sdk.graph.search(**kwargs))

            for edge in (result.edges or []):
                combined["edges"].append(_AttrDict({
                    "fact": edge.fact,
                    "valid_at": getattr(edge, "valid_at", None),
                    "created_at": getattr(edge, "created_at", None),
                    "name": getattr(edge, "name", None),
                }))
            for node in (result.nodes or []):
                combined["nodes"].append(_AttrDict({
                    "name": node.name,
                    "summary": getattr(node, "summary", None),
                }))
            for ep in (result.episodes or []):
                combined["episodes"].append(_AttrDict({
                    "content": getattr(ep, "content", None),
                    "created_at": getattr(ep, "created_at", None),
                }))

        return self._format_graph_results(combined)

    def sdk_graph_delete(self, graph_id):
        """Delete a standalone graph via zep-cloud SDK."""
        if not self._sdk:
            return
        with suppress(Exception):
            BaseApiClient.sdk_retry(lambda: self._sdk.graph.delete(graph_id))

import time

import requests

from .base_client import (
    BaseApiClient,
    env_bool,
    env_float,
    env_int,
    env_str,
)

# ── Cognee ───────────────────────────────────────────────────────────────────


class CogneeClient(BaseApiClient):
    """Cognee knowledge-graph memory client — v1.0 HTTP API.

    Reference: https://docs.cognee.ai/api-reference/introduction

    **Authentication:** Cognee Cloud uses ``X-Api-Key``.  Self-hosted
    instances with ``REQUIRE_AUTHENTICATION=true`` use ``Authorization: Bearer``
    after login — set ``COGNEE_BEARER_TOKEN`` for that mode.

    **Workflow (v1.0):**
        ``POST /api/v1/remember`` — ingest + cognify in one call
        ``POST /api/v1/search``  — search with explicit strategy control
        ``POST /api/v1/recall``  — alias for search (auto-routing)
        ``POST /api/v1/forget``  — delete dataset data

    Dataset isolation: ``datasetName`` equals the conversation-level user key.

    For LoCoMo the two speakers in a conversation share identical dialogue
    text, so we map **one dataset per conversation** (not per speaker).
    Both speaker user_ids resolve to the same dataset name, avoiding
    duplicate ingestion and redundant recall calls.

    Env vars (all optional except auth):
        COGNEE_API_KEY                      – X-Api-Key for Cognee Cloud
        COGNEE_BEARER_TOKEN                 – JWT for self-hosted auth
        COGNEE_BASE_URL                     – default https://api.cognee.ai
        COGNEE_QPS                          – rate limit (default 5)

        COGNEE_REMEMBER_RUN_IN_BACKGROUND   – async cognify (default false)
        COGNEE_REMEMBER_CUSTOM_PROMPT       – entity extraction prompt override
        COGNEE_REMEMBER_CHUNKS_PER_BATCH    – chunks per cognify batch (default 10)
        COGNEE_REMEMBER_NODE_SET            – comma-separated node set tags

        COGNEE_SEARCH_TYPE                  – SearchType enum (default GRAPH_COMPLETION)
        COGNEE_SEARCH_ENDPOINT              – "search" or "recall" (default search)
        COGNEE_ONLY_CONTEXT                 – skip Cognee LLM, return raw context
                                              (default true; only for completion types)
        COGNEE_SYSTEM_PROMPT                – custom system prompt for completions
        COGNEE_VERBOSE                      – verbose output (default false)
    """

    VALID_SEARCH_TYPES = frozenset({
        "SUMMARIES", "CHUNKS", "RAG_COMPLETION", "TRIPLET_COMPLETION",
        "GRAPH_COMPLETION", "GRAPH_COMPLETION_DECOMPOSITION",
        "GRAPH_SUMMARY_COMPLETION", "CYPHER", "NATURAL_LANGUAGE",
        "GRAPH_COMPLETION_COT", "GRAPH_COMPLETION_CONTEXT_EXTENSION",
        "FEELING_LUCKY", "TEMPORAL", "CODING_RULES", "CHUNKS_LEXICAL",
    })

    def __init__(self):
        base_url = env_str("COGNEE_BASE_URL", "https://api.cognee.ai")
        bearer = env_str("COGNEE_BEARER_TOKEN", "")
        api_key = env_str("COGNEE_API_KEY", "")
        qps = env_int("COGNEE_QPS", 5, min_value=0)
        timeout = env_int("COGNEE_TIMEOUT", 600, min_value=1)

        if bearer:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {bearer}",
            }
        elif api_key:
            headers = {
                "Content-Type": "application/json",
                "X-Api-Key": api_key,
            }
        else:
            raise ValueError(
                "Set COGNEE_API_KEY (Cognee Cloud) or COGNEE_BEARER_TOKEN "
                "(self-hosted JWT)"
            )

        super().__init__(base_url=base_url, headers=headers, qps=qps, timeout=timeout)
        self._batch_size = env_int("COGNEE_BATCH_SIZE", 20, min_value=1)
        self._ingested_datasets: set = set()

        # ── remember parameters ──
        self._remember_bg = env_bool("COGNEE_REMEMBER_RUN_IN_BACKGROUND", False)
        self._remember_custom_prompt = env_str(
            "COGNEE_REMEMBER_CUSTOM_PROMPT", ""
        ) or None
        self._remember_chunks_per_batch = env_int(
            "COGNEE_REMEMBER_CHUNKS_PER_BATCH", None, min_value=1
        )
        _ns = env_str("COGNEE_REMEMBER_NODE_SET", "")
        self._remember_node_set = (
            [t.strip() for t in _ns.split(",") if t.strip()] if _ns else None
        )

        # ── search parameters ──
        self._search_type = env_str(
            "COGNEE_SEARCH_TYPE", "GRAPH_COMPLETION",
        ).upper()
        if self._search_type not in self.VALID_SEARCH_TYPES:
            raise ValueError(
                f"Invalid COGNEE_SEARCH_TYPE={self._search_type!r}. "
                f"Valid: {sorted(self.VALID_SEARCH_TYPES)}"
            )
        self._search_endpoint = env_str(
            "COGNEE_SEARCH_ENDPOINT", "search",
        ).lower()
        self._only_context = env_bool("COGNEE_ONLY_CONTEXT", True)
        self._system_prompt = env_str("COGNEE_SYSTEM_PROMPT", "") or None
        self._verbose = env_bool("COGNEE_VERBOSE", False)

    @staticmethod
    def _conv_key(user_id):
        """Strip the ``_speaker_{a,b}_<version>`` suffix to get a
        conversation-level key shared by both speakers."""
        for tag in ("_speaker_a_", "_speaker_b_"):
            idx = user_id.find(tag)
            if idx != -1:
                return user_id[:idx]
        return user_id

    def _headers_no_json_content_type(self):
        """Headers for multipart/form-data uploads (drop JSON Content-Type)."""
        return {k: v for k, v in self.headers.items() if k.lower() != "content-type"}

    # ── add (remember) ────────────────────────────────────────────────────

    def add(self, messages, user_id, **kwargs):
        """Ingest messages via ``POST /api/v1/remember`` (add + cognify).

        Both speakers resolve to the same dataset, so only the first call
        per session actually sends data; the second call is a no-op.

        Messages are formatted as ``"Name (timestamp): text"`` so the
        extraction LLM can attribute facts to speakers and resolve
        temporal references.
        """
        dataset_name = self._conv_key(user_id)
        session_key = kwargs.get("session_key")
        if session_key is not None:
            dedup_key = f"{dataset_name}_{session_key}"
            if dedup_key in self._ingested_datasets:
                return
            self._ingested_datasets.add(dedup_key)

        lines: list[str] = []
        for msg in messages:
            text = msg.get("content", "")
            ts = msg.get("chat_time", "")
            lines.append(f"{ts} {text}".strip() if ts else text)
        content = "\n".join(lines)

        files = [("data", ("input.txt", content.encode("utf-8"), "text/plain"))]
        form_fields: list[tuple[str, str]] = [("datasetName", dataset_name)]
        if self._remember_bg:
            form_fields.append(("run_in_background", "true"))
        if self._remember_custom_prompt:
            form_fields.append(("custom_prompt", self._remember_custom_prompt))
        if self._remember_chunks_per_batch is not None:
            form_fields.append(
                ("chunks_per_batch", str(self._remember_chunks_per_batch))
            )
        if self._remember_node_set:
            for ns in self._remember_node_set:
                form_fields.append(("node_set", ns))

        def _remember():
            self._throttle()
            with requests.Session() as session:
                session.trust_env = False
                resp = session.post(
                    self._url("/api/v1/remember"),
                    headers=self._headers_no_json_content_type(),
                    files=files,
                    data=form_fields,
                    timeout=self._timeout,
                )
                resp.raise_for_status()

        self._retry(_remember)

    COMPLETION_SEARCH_TYPES = frozenset({
        "GRAPH_COMPLETION", "GRAPH_COMPLETION_COT",
        "GRAPH_COMPLETION_CONTEXT_EXTENSION", "GRAPH_COMPLETION_DECOMPOSITION",
        "GRAPH_SUMMARY_COMPLETION", "RAG_COMPLETION", "TRIPLET_COMPLETION",
        "FEELING_LUCKY",
    })

    @property
    def is_completion_search(self):
        """Whether the configured search type returns an LLM-generated answer."""
        return self._search_type in self.COMPLETION_SEARCH_TYPES

    # ── search ────────────────────────────────────────────────────────────

    def search(self, query, user_id, top_k):
        """Search memories via ``POST /api/v1/search`` (or ``/recall``).

        For completion-type searches (e.g. GRAPH_COMPLETION), Cognee returns
        an LLM-generated answer.  When ``onlyContext=false`` (default), that
        answer is available as the direct result.

        For retrieval-only types (CHUNKS, SUMMARIES, …), raw text is returned.
        """
        dataset_name = self._conv_key(user_id)
        endpoint = f"/api/v1/{self._search_endpoint}"

        payload: dict = {
            "query": query,
            "searchType": self._search_type,
            "datasets": [dataset_name],
            "topK": top_k,
            "onlyContext": self._only_context,
            "verbose": self._verbose,
        }
        if self._system_prompt:
            payload["systemPrompt"] = self._system_prompt

        def _do():
            resp = self._post(endpoint, json=payload)
            resp.raise_for_status()
            return resp.json()

        result = self._retry(_do)
        return self._format_results(result, top_k)

    @staticmethod
    def _format_results(result, top_k):
        """Normalise search API response into a context string."""
        if isinstance(result, str):
            return result
        if not isinstance(result, list):
            return str(result) if result else ""

        parts: list[str] = []
        for item in result[:top_k]:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            sr = item.get("search_result", item.get("searchResult"))
            if sr is None:
                continue
            if isinstance(sr, dict):
                text = sr.get("text") or sr.get("content") or ""
                if text:
                    parts.append(str(text))
                else:
                    parts.append(str(sr))
            elif isinstance(sr, list):
                parts.extend(str(s) for s in sr if s)
            else:
                parts.append(str(sr))
        return "\n\n".join(parts)

    # ── delete ────────────────────────────────────────────────────────────

    def delete_user(self, user_id):
        """Remove user data and fail loudly when cleanup cannot be verified.

        Streaming evaluations rely on delete being a hard boundary between
        independent users/conversations.  A silent cleanup failure would leave
        data resident in Cognee and contaminate later units, so this method
        tries both ``forget`` and dataset deletion, then verifies that the
        dataset name no longer appears in ``/datasets``.
        """
        dataset_name = self._conv_key(user_id)
        errors: list[str] = []

        def _forget():
            resp = self._post(
                "/api/v1/forget",
                json={"dataset": dataset_name, "everything": False},
            )
            if resp.status_code in (200, 202, 204):
                return True
            if resp.status_code == 404:
                return False
            resp.raise_for_status()
            return True

        delete_retries = env_int("COGNEE_DELETE_RETRIES", 3, min_value=1)
        forget_retries = env_int("COGNEE_FORGET_RETRIES", 2, min_value=1)

        try:
            self._retry(
                lambda: self._delete_dataset_by_name(dataset_name),
                max_retries=max(1, delete_retries),
            )
            self._verify_dataset_absent(dataset_name)
            return
        except Exception as exc:
            errors.append(f"dataset delete/verify failed: {exc}")

        try:
            self._retry(_forget, max_retries=max(1, forget_retries))
            self._retry(
                lambda: self._delete_dataset_by_name(dataset_name),
                max_retries=max(1, delete_retries),
            )
            self._verify_dataset_absent(dataset_name)
            return
        except Exception as exc:
            errors.append(f"forget fallback failed: {exc}")

        detail = "; ".join(errors) if errors else "unknown cleanup error"
        raise RuntimeError(
            f"Cognee delete_user({user_id!r}) failed for dataset "
            f"{dataset_name!r}: {detail}"
        )

    def _list_datasets(self):
        resp = self._get("/api/v1/datasets")
        resp.raise_for_status()
        items = resp.json()
        if not isinstance(items, list):
            raise RuntimeError(f"Unexpected Cognee datasets response: {items!r}")
        return items

    @staticmethod
    def _dataset_name(item):
        if not isinstance(item, dict):
            return None
        return item.get("name") or item.get("datasetName")

    @staticmethod
    def _dataset_id(item):
        if not isinstance(item, dict):
            return None
        return item.get("id") or item.get("datasetId")

    def _delete_dataset_by_name(self, dataset_name):
        deleted = False
        delete_timeout = env_float("COGNEE_DELETE_TIMEOUT", 600, min_value=1)
        for ds in self._list_datasets():
            if self._dataset_name(ds) != dataset_name:
                continue
            dataset_id = self._dataset_id(ds)
            if not dataset_id:
                raise RuntimeError(f"Cognee dataset has no id: {ds!r}")
            print(
                f"  cognee delete dataset {dataset_name} ({dataset_id})",
                flush=True,
            )
            resp = self._delete(
                f"/api/v1/datasets/{dataset_id}",
                timeout=delete_timeout,
            )
            if resp.status_code == 404:
                continue
            if resp.status_code not in (200, 202, 204):
                resp.raise_for_status()
            deleted = True
        return deleted

    def _verify_dataset_absent(self, dataset_name):
        attempts = env_int("COGNEE_DELETE_VERIFY_RETRIES", 3, min_value=1)
        delay = env_float("COGNEE_DELETE_VERIFY_DELAY", 1, min_value=0)
        attempts = max(1, attempts)
        for attempt in range(attempts):
            remaining = [
                ds for ds in self._list_datasets()
                if self._dataset_name(ds) == dataset_name
            ]
            if not remaining:
                return
            if attempt < attempts - 1:
                time.sleep(delay)
        raise RuntimeError(f"Cognee dataset still exists after delete: {dataset_name}")

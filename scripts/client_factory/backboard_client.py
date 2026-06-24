import time

from contextlib import suppress

from .base_client import (
    BaseApiClient,
    env_bool,
    env_float,
    env_int,
    env_str,
    require_env,
)

# ── Backboard ─────────────────────────────────────────────────────────────────


class BackboardClient(BaseApiClient):
    """Backboard.io — Messages + Auto Memory end-to-end mode (REST API).

    Reference: https://docs.backboard.io/
    Endpoints:
        POST   /assistants                              – create assistant
        POST   /assistants/{id}/threads                 – create thread
        POST   /threads/{thread_id}/messages            – send message (form-data)
        POST   /assistants/{id}/memories/search         – search memories
        GET    /assistants/memories/operations/{id}      – poll memory operation
        DELETE /assistants/{id}                          – delete assistant + memories

    Aligned with the official Backboard-Locomo-Benchmark:
        https://github.com/Backboard-io/Backboard-Locomo-Benchmark

    Key design choices (matching official benchmark):
        - One assistant per conversation (shared memory for both speakers)
        - Each session → separate thread; each turn → separate message
        - Messages sent as form-data (not JSON) with ``memory="auto"``
        - Timestamps passed via ``metadata.custom_timestamp`` (ISO 8601)
        - LLM provider/model configured at assistant level for reflect answers

    Configurable via env:
        BACKBOARD_QPS                          – local rate limit (default 5)
        BACKBOARD_MEMORY_MODE                  – "lite" or "pro" (default lite)
        BACKBOARD_SEND_TO_LLM                  – skip LLM response on ingest (default false)
        BACKBOARD_WAIT_FOR_MEMORY              – poll memory_operation_id (default true)
        BACKBOARD_MEMORY_TIMEOUT               – operation poll timeout secs (default 300)
        BACKBOARD_MEMORY_POLL_INTERVAL         – poll interval secs (default 2)
        BACKBOARD_CUSTOM_FACT_EXTRACTION_PROMPT – custom fact extraction prompt
        BACKBOARD_CUSTOM_UPDATE_MEMORY_PROMPT   – custom memory update prompt
        BACKBOARD_SEARCH_MODE                  – "api" or "message" (default message)
        BACKBOARD_LLM_PROVIDER                 – LLM provider for assistant (default openai)
        BACKBOARD_LLM_MODEL_NAME               – LLM model for assistant (default gpt-4.1-mini)
        BACKBOARD_INGEST_MODE                  – "per_turn" or "batch" (default per_turn)
    """

    _OP_DONE = {"completed", "complete", "success"}
    _OP_FAILED = {"error", "failed"}

    def __init__(self):
        api_key = require_env("BACKBOARD_API_KEY")
        base_url = env_str("BACKBOARD_BASE_URL", "https://app.backboard.io/api")
        qps = env_float("BACKBOARD_QPS", 5, min_value=0)

        self._memory_mode = env_str("BACKBOARD_MEMORY_MODE", "lite").lower()
        self._send_to_llm = str(env_bool("BACKBOARD_SEND_TO_LLM", False)).lower()
        self._wait_for_memory = env_bool("BACKBOARD_WAIT_FOR_MEMORY", True)
        self._memory_timeout = env_float(
            "BACKBOARD_MEMORY_TIMEOUT", 300, min_value=1
        )
        self._memory_poll_interval = env_float(
            "BACKBOARD_MEMORY_POLL_INTERVAL", 2, min_value=0
        )
        self._custom_fact_prompt = env_str(
            "BACKBOARD_CUSTOM_FACT_EXTRACTION_PROMPT", ""
        ) or None
        self._custom_update_prompt = env_str(
            "BACKBOARD_CUSTOM_UPDATE_MEMORY_PROMPT", ""
        ) or None
        self._search_mode = env_str(
            "BACKBOARD_SEARCH_MODE", "message"
        ).lower()
        self._llm_provider = env_str("BACKBOARD_LLM_PROVIDER", "openai")
        self._llm_model_name = env_str(
            "BACKBOARD_LLM_MODEL_NAME", "gpt-4.1-mini"
        )
        self._ingest_mode = env_str(
            "BACKBOARD_INGEST_MODE", "per_turn"
        ).lower()
        self._batch_size = env_int("BACKBOARD_BATCH_SIZE", 20, min_value=1)

        http_timeout = env_int("BACKBOARD_HTTP_TIMEOUT", 180, min_value=1)

        super().__init__(
            base_url=base_url,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
            },
            qps=qps,
            timeout=http_timeout,
        )
        self._api_key = api_key
        self._assistant_map = {}
        self._thread_map = {}
        self._search_thread_map = {}
        self._ingested_assistants = set()

    @staticmethod
    def _conv_key(user_id):
        """Strip ``_speaker_{a,b}_<version>`` suffix for conversation-level
        deduplication in LoCoMo."""
        for tag in ("_speaker_a_", "_speaker_b_"):
            idx = user_id.find(tag)
            if idx != -1:
                return user_id[:idx]
        return user_id

    def _get_or_create_assistant(self, user_id):
        key = self._conv_key(user_id)
        if key in self._assistant_map:
            return self._assistant_map[key]

        name = f"memeval_{key}"
        aid = self._find_assistant_by_name(name)
        if aid:
            self._assistant_map[key] = aid
            return aid

        payload = {
            "name": name,
            "system_prompt": "You are a conversation memory assistant.",
            "llm_provider": self._llm_provider,
            "llm_model_name": self._llm_model_name,
            "tools": [],
        }
        if self._custom_fact_prompt:
            payload["custom_fact_extraction_prompt"] = self._custom_fact_prompt
        if self._custom_update_prompt:
            payload["custom_update_memory_prompt"] = self._custom_update_prompt

        resp = self._post("/assistants", json=payload)
        resp.raise_for_status()
        aid = resp.json().get("assistant_id", resp.json().get("id"))
        self._assistant_map[key] = aid
        return aid

    def _find_assistant_by_name(self, name):
        resp = self._get("/assistants")
        if resp.status_code == 200:
            data = resp.json()
            assistants = data.get("assistants", data) if isinstance(data, dict) else data
            if isinstance(assistants, list):
                for a in assistants:
                    if a.get("name") == name:
                        return a.get("assistant_id", a.get("id"))
        return None

    def _create_thread(self, assistant_id):
        """Create a new thread for the given assistant."""
        resp = self._post(f"/assistants/{assistant_id}/threads", json={})
        resp.raise_for_status()
        return resp.json().get("thread_id", resp.json().get("id"))

    def _get_or_create_thread(self, assistant_id):
        if assistant_id in self._thread_map:
            return self._thread_map[assistant_id]

        tid = self._create_thread(assistant_id)
        self._thread_map[assistant_id] = tid
        return tid

    def _get_or_create_search_thread(self, assistant_id):
        """Dedicated thread for message-mode search to avoid polluting the
        ingestion thread with query messages."""
        if assistant_id in self._search_thread_map:
            return self._search_thread_map[assistant_id]

        resp = self._post(f"/assistants/{assistant_id}/threads", json={})
        resp.raise_for_status()
        tid = resp.json().get("thread_id", resp.json().get("id"))
        self._search_thread_map[assistant_id] = tid
        return tid

    def _poll_memory_operation(self, operation_id):
        """Poll memory operation status until completion or timeout."""
        start = time.monotonic()
        while time.monotonic() - start < self._memory_timeout:
            try:
                resp = self._get(
                    f"/assistants/memories/operations/{operation_id}"
                )
                if resp.status_code == 200:
                    status = resp.json().get("status", "").lower()
                    if status in self._OP_DONE:
                        return
                    if status in self._OP_FAILED:
                        print(f"  ⚠ Memory operation {operation_id} failed: {resp.text}")
                        return
            except Exception:
                pass
            time.sleep(self._memory_poll_interval)
        print(
            f"  ⚠ Memory operation {operation_id} timed out "
            f"after {self._memory_timeout}s"
        )

    def _send_message(self, thread_id, payload):
        """POST a single message as JSON and return the parsed response.

        The Backboard Send Message API accepts ``application/json``
        (see OpenAPI spec at docs.backboard.io).
        """
        result = {}

        def _do():
            nonlocal result
            resp = self._post(
                f"/threads/{thread_id}/messages",
                json=payload,
            )
            resp.raise_for_status()
            try:
                result = resp.json()
            except Exception:
                result = {}

        self._retry(_do)
        return result

    def add(self, messages, user_id, **kwargs):
        """Send messages via Thread with automatic memory extraction.

        Both LoCoMo speakers resolve to the same assistant; only the first
        call per session actually sends data when ``session_key`` is provided.

        Each session creates a new thread (matching the official benchmark
        which maps one thread per conversation session).

        Supports two ingestion modes (``BACKBOARD_INGEST_MODE``):
        - ``per_turn`` (default, matches official benchmark): each message
          sent as a separate API call with form-data, ISO-8601 timestamp
          passed via ``metadata.custom_timestamp``.
        - ``batch``: all messages concatenated into one API call.
        """
        assistant_id = self._get_or_create_assistant(user_id)

        session_key = kwargs.get("session_key")
        if session_key is not None:
            key = self._conv_key(user_id)
            dedup_key = f"{key}_{session_key}"
            if dedup_key in self._ingested_assistants:
                return
            self._ingested_assistants.add(dedup_key)

        thread_id = self._create_thread(assistant_id)

        memory_field = "memory"
        memory_value = "Auto"
        if self._memory_mode == "pro":
            memory_field = "memory_pro"

        if self._ingest_mode == "per_turn":
            self._add_per_turn(
                thread_id, messages, memory_field, memory_value,
            )
        else:
            self._add_batch(
                thread_id, messages, memory_field, memory_value,
            )

    def _add_per_turn(self, thread_id, messages, memory_field, memory_value):
        """Send each message individually (official benchmark approach)."""
        import json as _json

        send_to_llm = "true" if self._send_to_llm in ("true", "1", "yes") else "false"

        for msg in messages:
            name = msg.get("name", msg.get("role", "user"))
            content_text = msg.get("content", "")
            message_content = f"{name}: {content_text}"

            payload = {
                "content": message_content,
                "stream": False,
                memory_field: memory_value,
                "send_to_llm": send_to_llm,
            }

            chat_time = msg.get("chat_time")
            if chat_time:
                iso_ts = str(chat_time)
                if iso_ts.endswith("+00:00"):
                    iso_ts = iso_ts[:-6] + "Z"
                elif not iso_ts.endswith("Z") and "+" not in iso_ts:
                    iso_ts = iso_ts + "Z"
                payload["metadata"] = _json.dumps(
                    {"custom_timestamp": iso_ts}
                )

            result = self._send_message(thread_id, payload)

            if self._wait_for_memory:
                op_id = result.get("memory_operation_id")
                if op_id:
                    self._poll_memory_operation(op_id)

    def _add_batch(self, thread_id, messages, memory_field, memory_value):
        """Concatenate all messages into one API call."""
        send_to_llm = "true" if self._send_to_llm in ("true", "1", "yes") else "false"

        parts = []
        for msg in messages:
            name = msg.get("name", msg.get("role", "user"))
            chat_time = msg.get("chat_time")
            prefix = f"[{chat_time}] {name}" if chat_time else name
            parts.append(f"{prefix}: {msg['content']}")
        conversation_text = "\n".join(parts)

        payload = {
            "content": conversation_text,
            "stream": False,
            memory_field: memory_value,
            "send_to_llm": send_to_llm,
        }

        result = self._send_message(thread_id, payload)

        if self._wait_for_memory:
            op_id = result.get("memory_operation_id")
            if op_id:
                self._poll_memory_operation(op_id)

    # ── search: dual-mode ─────────────────────────────────────────

    def search(self, query, user_id, top_k):
        """Search memories using the configured search mode.

        - ``api``     — direct ``POST /memories/search`` (basic semantic search)
        - ``message`` — send query via ``memory_pro="Readonly"`` to leverage
                        Backboard's Pro retrieval pipeline (higher accuracy)
        """
        if self._search_mode == "message":
            return self._search_via_message(query, user_id, top_k)
        return self._search_via_api(query, user_id, top_k)

    def _search_via_api(self, query, user_id, top_k):
        """POST /assistants/{id}/memories/search — basic semantic search.

        This endpoint accepts JSON (it's a search endpoint, not a message).
        """
        assistant_id = self._get_or_create_assistant(user_id)
        payload = {"query": query, "limit": top_k}

        def _do():
            resp = self._post(
                f"/assistants/{assistant_id}/memories/search", json=payload
            )
            resp.raise_for_status()
            return resp.json()

        result = self._retry(_do)
        memories = result.get("memories", [])
        return "\n\n".join(
            m.get("content", "") if isinstance(m, dict) else str(m)
            for m in memories[:top_k]
        )

    def _search_via_message(self, query, user_id, top_k):
        """Send query as a message with ``memory="Readonly"`` to tap into
        Backboard's retrieval pipeline, then extract the
        ``retrieved_memories`` from the response.

        ``send_to_llm`` is explicitly set to ``"false"`` so that Backboard
        only performs memory retrieval without generating an LLM answer,
        saving cost and latency in RAG mode.
        """
        assistant_id = self._get_or_create_assistant(user_id)
        thread_id = self._get_or_create_search_thread(assistant_id)

        memory_field = "memory"
        memory_value = "Readonly"
        if self._memory_mode == "pro":
            memory_field = "memory_pro"
            memory_value = "Readonly"

        payload = {
            "content": query,
            "stream": False,
            memory_field: memory_value,
            "send_to_llm": False,
        }

        result = self._send_message(thread_id, payload)

        retrieved = result.get("retrieved_memories") or []
        if retrieved:
            return "\n\n".join(
                m.get("memory", m.get("content", ""))
                if isinstance(m, dict) else str(m)
                for m in retrieved[:top_k]
            )
        memories = result.get("memories") or []
        if memories:
            return "\n\n".join(
                m.get("content", "") if isinstance(m, dict) else str(m)
                for m in memories[:top_k]
            )
        return ""

    # ── reflect: Backboard LLM answers directly ────────────────────

    def reflect(self, query, user_id, top_k=20):
        """Send *query* with ``memory="auto"`` + ``send_to_llm=true`` so
        Backboard's built-in LLM (configured at assistant level) generates
        the answer using retrieved memories as context.

        Uses form-data to match the official benchmark.
        Returns ``(answer_text, retrieved_memories_text)``.
        """
        assistant_id = self._get_or_create_assistant(user_id)
        thread_id = self._get_or_create_search_thread(assistant_id)

        payload = {
            "content": query,
            "stream": False,
            "memory": "auto",
            "send_to_llm": True,
        }

        result = self._send_message(thread_id, payload)

        answer = result.get("content", "")

        retrieved = result.get("retrieved_memories") or []
        if retrieved:
            mem_text = "\n\n".join(
                m.get("memory", m.get("content", ""))
                if isinstance(m, dict) else str(m)
                for m in retrieved[:top_k]
            )
        else:
            memories = result.get("memories") or []
            mem_text = "\n\n".join(
                m.get("content", "") if isinstance(m, dict) else str(m)
                for m in memories[:top_k]
            )

        return answer, mem_text

    def delete_user(self, user_id):
        key = self._conv_key(user_id)
        name = f"memeval_{key}"
        assistant_id = self._assistant_map.pop(key, None) or self._find_assistant_by_name(name)
        if assistant_id:
            self._thread_map.pop(assistant_id, None)
            self._search_thread_map.pop(assistant_id, None)
            with suppress(Exception):
                self._delete(f"/assistants/{assistant_id}")

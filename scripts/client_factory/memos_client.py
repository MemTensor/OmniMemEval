from .base_client import (
    BaseApiClient,
    RateLimitError,
    env_bool,
    env_float,
    env_int,
    env_max_batch_chars,
    env_str,
    iter_batches,
)


class MemosClient(BaseApiClient):
    """MemOS memory client.

    Supports both the hosted OpenMem API and the self-hosted MemOS product API:

    - ``MEMOS_MODE=cloud``: ``/add/message`` / ``/search/memory`` / ``/delete/memory``
    - ``MEMOS_MODE=local``: ``/product/add`` / ``/product/search`` / ``/product/delete_memory``
    """

    def __init__(self):
        mode = env_str("MEMOS_MODE", "cloud").lower()
        if mode not in ("cloud", "local"):
            raise ValueError("MEMOS_MODE must be 'cloud' or 'local'")

        api_key = env_str("MEMOS_API_KEY", "")
        if mode == "cloud" and not api_key:
            raise ValueError("MEMOS_API_KEY environment variable is not set")
        default_base_url = (
            "http://localhost:8000"
            if mode == "local"
            else "https://memos.memtensor.cn/api/openmem/v1"
        )
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Token {api_key}"
        super().__init__(
            base_url=env_str("MEMOS_BASE_URL", default_base_url),
            headers=headers,
            qps=env_float("MEMOS_QPS", 0, min_value=0) or None,
            timeout=env_int("MEMOS_TIMEOUT", 300, min_value=1),
        )
        self._mode = mode
        self._batch_size = env_int("MEMOS_BATCH_SIZE", 20, min_value=1)
        self._max_batch_chars = env_max_batch_chars("MEMOS_MAX_BATCH_CHARS")
        self._local_mem_cube_id_mode = env_str(
            "MEMOS_LOCAL_MEM_CUBE_ID_MODE", "user"
        ).lower()

    @staticmethod
    def _is_rate_limited(body):
        return body.get("code") == 40309 or "rate limit" in body.get("message", "").lower()

    def _check_ok(self, resp, body, *, messages):
        if self._is_rate_limited(body):
            raise RateLimitError(retry_after=2, response=resp)
        assert resp.status_code == 200, resp.text
        assert body.get("message") in messages, resp.text

    def _local_mem_cube_id(self, user_id):
        if self._local_mem_cube_id_mode == "constant":
            return env_str("MEMOS_LOCAL_MEM_CUBE_ID", "omnimemeval")
        return user_id

    @staticmethod
    def _cloud_async_mode():
        return env_str("MEMOS_ASYNC_MODE", "false").lower() in (
            "true",
            "1",
            "yes",
            "async",
        )

    @staticmethod
    def _local_async_mode():
        value = env_str("MEMOS_ASYNC_MODE", "sync").lower()
        if value in ("async", "sync"):
            return value
        raise ValueError("MEMOS_ASYNC_MODE must be 'sync' or 'async' in local mode")

    def delete(self, user_id):
        """Delete all memories for a given user_id."""
        if self._mode == "local":
            return self._delete_local(user_id)

        resp = self._post("/delete/memory", json={"user_id": user_id})
        result = resp.json()
        if result.get("message") == "ok":
            print(f"Deleted memories for {user_id}")
        else:
            print(f"Delete failed for {user_id}: {result}")

    def _delete_local(self, user_id):
        payload = {
            "user_id": user_id,
            "writable_cube_ids": [self._local_mem_cube_id(user_id)],
        }
        resp = self._post("/product/delete_memory", json=payload)
        body = resp.json()
        self._check_ok(
            resp,
            body,
            messages={
                "Successfully",
                "Called Successfully",
                "Memory deleted successfully",
                "Memories deleted successfully",
            },
        )
        print(f"Deleted MemOS local memories for {user_id}")

    def add(self, messages, user_id, conv_id=None, batch_size: int = None):
        if self._mode == "local":
            return self._add_local(messages, user_id, conv_id=conv_id, batch_size=batch_size)

        async_mode = self._cloud_async_mode()
        for batch in iter_batches(messages, batch_size or self._batch_size,
                                  max_chars=self._max_batch_chars):
            payload = {
                "messages": batch,
                "user_id": user_id,
                "conversation_id": conv_id,
                "async_mode": async_mode,
            }

            def _do():
                resp = self._post("/add/message", json=payload)
                body = resp.json()
                self._check_ok(resp, body, messages={"ok"})

            self._retry(_do)

    def _add_local(self, messages, user_id, conv_id=None, batch_size=None):
        async_mode = self._local_async_mode()
        mem_cube_id = self._local_mem_cube_id(user_id)
        add_mode = env_str("MEMOS_LOCAL_ADD_MODE", "").lower()
        for batch in iter_batches(messages, batch_size or self._batch_size,
                                  max_chars=self._max_batch_chars):
            payload = {
                "messages": batch,
                "user_id": user_id,
                "mem_cube_id": mem_cube_id,
                "writable_cube_ids": [mem_cube_id],
                "async_mode": async_mode,
            }
            if conv_id is not None:
                payload["session_id"] = str(conv_id)
                payload["conversation_id"] = str(conv_id)
            if add_mode in ("fast", "fine"):
                payload["mode"] = add_mode

            def _do():
                resp = self._post("/product/add", json=payload)
                body = resp.json()
                self._check_ok(resp, body, messages={"Memory added successfully"})

            self._retry(_do)

    def search(self, query, user_id, top_k):
        """Search memories."""
        if self._mode == "local":
            return self._search_local(query, user_id, top_k)

        include_pref = env_bool("MEMOS_INCLUDE_PREFERENCE", True)
        pref_limit = env_int("MEMOS_PREFERENCE_LIMIT", 9, min_value=0)
        relativity = env_float("MEMOS_RELATIVITY", None)
        context_format = env_str("MEMOS_CONTEXT_FORMAT", "")
        search_tool_memory = env_bool("MEMOS_SEARCH_TOOL_MEMORY", False)
        tool_mem_limit = env_int("MEMOS_TOOL_MEMORY_LIMIT", top_k, min_value=0)

        payload = {
            "query": query,
            "user_id": user_id,
            "memory_limit_number": top_k,
            "include_preference": include_pref,
            "preference_limit_number": pref_limit,
        }
        if context_format:
            payload["context_format"] = context_format
        if search_tool_memory:
            payload["include_tool_memory"] = True
            payload["tool_memory_limit_number"] = tool_mem_limit
        if relativity is not None:
            payload["relativity"] = relativity

        def _do():
            resp = self._post("/search/memory", json=payload)
            body = resp.json()
            code = body.get("code")
            if code == 40309 or "rate limit" in body.get("message", "").lower():
                raise RateLimitError(retry_after=2, response=resp)
            assert resp.status_code == 200, resp.text
            assert code == 0, resp.text
            return body

        body = self._retry(_do)
        data = body["data"]
        return self._format_cloud_search_data(data)

    @staticmethod
    def _format_cloud_search_data(data):
        text_mem_res = data.get("memory_detail_list") or []
        memory_lines = [
            m.get("memory_value", "") for m in text_mem_res
        ]
        tool_mem_res = data.get("tool_memory_detail_list") or []
        tool_lines = [
            f"Tool Memory: {text}"
            for text in (MemosClient._memory_text(m) for m in tool_mem_res)
            if text
        ]

        pref_mem_res = data.get("preference_detail_list") or []
        preference_note = data.get("preference_note") or ""

        pref_parts = []
        explicit_prefs = []
        implicit_prefs = []
        for pref in pref_mem_res:
            if pref["preference_type"] == "explicit_preference":
                explicit_prefs.append(pref["preference"])
            elif pref["preference_type"] == "implicit_preference":
                implicit_prefs.append(pref["preference"])

        if explicit_prefs:
            pref_parts.append(
                "Explicit Preference:\n"
                + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(explicit_prefs))
            )
        if implicit_prefs:
            pref_parts.append(
                "Implicit Preference:\n"
                + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(implicit_prefs))
            )
        if preference_note:
            pref_parts.append(preference_note)

        parts = memory_lines + tool_lines + pref_parts
        return "\n".join(parts)

    def _search_local(self, query, user_id, top_k):
        include_pref = env_bool("MEMOS_INCLUDE_PREFERENCE", True)
        pref_limit = env_int("MEMOS_PREFERENCE_LIMIT", 6, min_value=0)
        relativity = env_float("MEMOS_RELATIVITY", None)
        payload = {
            "query": query,
            "user_id": user_id,
            "mem_cube_id": self._local_mem_cube_id(user_id),
            "readable_cube_ids": [self._local_mem_cube_id(user_id)],
            "top_k": top_k,
            "mode": env_str("MEMOS_SEARCH_MODE", "fast"),
            "include_preference": include_pref,
            "pref_top_k": pref_limit,
            "context_format": env_str(
                "MEMOS_CONTEXT_FORMAT",
                env_str("MEMOS_SEARCH_CONTEXT_FORMAT", "memory"),
            ),
            "search_tool_memory": False,
            "tool_mem_top_k": 0,
            "include_skill_memory": False,
            "skill_mem_top_k": 0,
        }
        if relativity is not None:
            payload["relativity"] = relativity
        dedup = env_str("MEMOS_DEDUP", "")
        if dedup:
            payload["dedup"] = dedup
        rerank = env_str("MEMOS_RERANK", "")
        if rerank:
            payload["rerank"] = rerank.strip().lower() not in ("0", "false", "no", "off")

        def _do():
            resp = self._post("/product/search", json=payload)
            body = resp.json()
            self._check_ok(resp, body, messages={"Search completed successfully"})
            return body

        body = self._retry(_do)
        data = body.get("data") or {}
        return self._format_local_search_data(data)

    @classmethod
    def _format_local_search_data(cls, data):
        if "memory_detail_list" in data:
            return cls._format_cloud_search_data(data)

        parts = []
        parts.extend(cls._format_local_bucket_list(data.get("text_mem"), label=None))
        parts.extend(cls._format_local_bucket_list(data.get("pref_mem"), label="Preference"))
        preference_note = data.get("preference_note") or data.get("pref_note")
        if preference_note:
            parts.append(str(preference_note))
        return "\n".join(p for p in parts if p)

    @classmethod
    def _format_local_bucket_list(cls, buckets, *, label=None):
        if not buckets:
            return []
        if isinstance(buckets, dict):
            buckets = [buckets]
        lines = []
        for bucket in buckets:
            memories = bucket.get("memories", bucket) if isinstance(bucket, dict) else bucket
            if isinstance(memories, dict):
                memories = [memories]
            if not isinstance(memories, list):
                continue
            for memory in memories:
                text = cls._memory_text(memory)
                if text:
                    lines.append(f"{label}: {text}" if label else text)
        return lines

    @staticmethod
    def _memory_text(memory):
        if isinstance(memory, str):
            return memory
        if not isinstance(memory, dict):
            return str(memory) if memory is not None else ""
        for key in ("memory", "memory_value", "experience", "value", "content", "text", "tool_value"):
            value = memory.get(key)
            if value:
                return str(value)
        metadata = memory.get("metadata")
        if isinstance(metadata, dict):
            for key in ("memory", "memory_value", "experience", "value", "content", "text", "tool_value"):
                value = metadata.get(key)
                if value:
                    return str(value)
        return ""

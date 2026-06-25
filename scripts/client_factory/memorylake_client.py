import threading

from contextlib import suppress

import requests

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

# ── MemoryLake ────────────────────────────────────────────────────────────────


class MemoryLakeClient(BaseApiClient):
    """MemoryLake (Powerdrill) project-centric memory client (REST API V2).

    Docs: https://docs.memorylake.ai/features/memorylake/api-reference/overview
    GitHub: https://github.com/powerdrillai/memorylake-client

    Architecture:
        - **Project-scoped**: all memories live inside a project (``proj-{uuid}``).
        - **User isolation**: ``user_id`` provides second-level namespace within
          a project, filtering both add and search operations.
        - **Async extraction**: ``POST .../memories`` returns ``event_id``
          (status ``PENDING``); the platform asynchronously extracts structured
          memories from the submitted conversation when ``infer=true``.

    Auth: ``Authorization: Bearer <MEMORYLAKE_API_KEY>``.
    Base URL: ``https://app.memorylake.ai/openapi/memorylake``

    Rate limits (official):
        - 60 requests / minute per IP
        - 600 requests / minute per user account
    """

    _project_cache: dict = {}
    _project_lock = threading.Lock()

    def __init__(self):
        api_key = require_env("MEMORYLAKE_API_KEY")
        base_url = env_str(
            "MEMORYLAKE_BASE_URL",
            "https://app.memorylake.ai/openapi/memorylake",
        )
        qps = env_float("MEMORYLAKE_QPS", 1, min_value=0)

        self._configured_project_id = env_str("MEMORYLAKE_PROJECT_ID", "")
        self._project_name = env_str("MEMORYLAKE_PROJECT_NAME", "omnimemeval")
        self._infer = env_bool("MEMORYLAKE_INFER", True)
        self._search_threshold = env_float(
            "MEMORYLAKE_SEARCH_THRESHOLD", None, min_value=0
        )
        self._search_rerank = env_bool("MEMORYLAKE_SEARCH_RERANK", True)
        timeout = env_int("MEMORYLAKE_TIMEOUT", 120, min_value=1)
        self._search_timeout = env_int(
            "MEMORYLAKE_SEARCH_TIMEOUT", 60, min_value=1
        )
        self._search_max_retries = env_int(
            "MEMORYLAKE_SEARCH_MAX_RETRIES", 3, min_value=1
        )
        self._batch_size = env_int("MEMORYLAKE_BATCH_SIZE", 20, min_value=1)
        self._max_batch_chars = env_max_batch_chars("MEMORYLAKE_MAX_BATCH_CHARS")

        super().__init__(
            base_url=base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            qps=qps,
            timeout=timeout,
        )

    @property
    def _project_id(self):
        if self._configured_project_id:
            return self._configured_project_id
        cache_key = (self.base_url, self._project_name)
        if cache_key not in MemoryLakeClient._project_cache:
            with MemoryLakeClient._project_lock:
                if cache_key not in MemoryLakeClient._project_cache:
                    pid = self._get_or_create_project()
                    MemoryLakeClient._project_cache[cache_key] = pid
                    print(f"  MemoryLake project resolved: {pid}")
        return MemoryLakeClient._project_cache[cache_key]

    def _get_or_create_project(self):
        resp = self._get(
            "/api/v1/projects",
            params={"name": self._project_name, "size": 100},
        )
        if resp.status_code == 200:
            items = resp.json().get("data", {}).get("items", [])
            for item in items:
                if item.get("name") == self._project_name:
                    return item["id"]
        payload = {
            "name": self._project_name,
            "description": "OmniMemEval evaluation project (auto-created)",
        }
        resp = self._post("/api/v1/projects", json=payload)
        resp.raise_for_status()
        return resp.json()["data"]["id"]

    def add(self, messages, user_id, **kwargs):
        all_msgs = []
        for m in messages:
            content = m["content"]
            chat_time = m.get("chat_time")
            if chat_time:
                content = f"[{chat_time}] {content}"
            all_msgs.append({"role": m.get("role", "user"), "content": content})

        session_id = kwargs.get("session_key")
        metadata = kwargs.get("metadata")

        for batch in iter_batches(all_msgs, self._batch_size,
                                  max_chars=self._max_batch_chars):

            payload = {
                "messages": batch,
                "user_id": user_id,
                "infer": self._infer,
            }
            if session_id:
                payload["chat_session_id"] = session_id
            if metadata and isinstance(metadata, dict):
                payload["metadata"] = metadata

            def _do(p=payload):
                resp = self._post(
                    f"/api/v2/projects/{self._project_id}/memories", json=p,
                )
                resp.raise_for_status()

            self._retry(_do)

    def search(self, query, user_id, top_k):
        payload = {
            "query": query,
            "user_id": user_id,
            "top_k": top_k,
        }
        if self._search_threshold is not None:
            payload["threshold"] = self._search_threshold
        if self._search_rerank:
            payload["rerank"] = True

        def _do():
            resp = self._post(
                f"/api/v2/projects/{self._project_id}/memories/search",
                json=payload,
                timeout=self._search_timeout,
            )
            resp.raise_for_status()
            return resp.json()

        try:
            result = self._retry(_do, max_retries=self._search_max_retries)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            raise RuntimeError(
                f"MemoryLake search failed after {self._search_max_retries} retries: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"MemoryLake search error: {e}") from e

        memories = result.get("data", {}).get("results", [])
        if isinstance(memories, list):
            return "\n\n".join(
                m.get("content", "") if isinstance(m, dict) else str(m)
                for m in memories[:top_k]
            )
        return str(result)

    def delete_user(self, user_id):
        """Forget all memories belonging to *user_id* within the project."""
        max_passes = env_int("MEMORYLAKE_DELETE_MAX_PASSES", 20, min_value=1)
        max_passes = max(1, max_passes)

        for _ in range(max_passes):
            def _list():
                resp = self._get(
                    f"/api/v2/projects/{self._project_id}/memories",
                    params={"user_id": user_id, "page": 1, "size": 100},
                )
                resp.raise_for_status()
                return resp.json()

            data = self._retry(_list).get("data", {})
            items = data.get("items", [])
            if not items:
                return

            deleted = 0
            for mem in items:
                if not isinstance(mem, dict):
                    continue
                mid = mem.get("id")
                if not mid:
                    continue

                def _forget(memory_id=mid):
                    resp = self._post(
                        f"/api/v2/projects/{self._project_id}"
                        f"/memories/{memory_id}/forget",
                    )
                    if resp.status_code not in (200, 202, 204, 404):
                        resp.raise_for_status()
                    return resp

                self._retry(_forget)
                deleted += 1

            if deleted == 0:
                raise RuntimeError(
                    f"MemoryLake delete_user({user_id}) could not find memory ids "
                    f"in {len(items)} returned memories"
                )

        raise RuntimeError(
            f"MemoryLake delete_user({user_id}) still returned memories after "
            f"{max_passes} delete passes"
        )

    def delete_project(self):
        """Delete the entire project and all associated data."""
        with suppress(Exception):
            self._delete(f"/api/v1/projects/{self._project_id}")
            cache_key = (self.base_url, self._project_name)
            MemoryLakeClient._project_cache.pop(cache_key, None)

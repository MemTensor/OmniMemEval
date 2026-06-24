import time

from datetime import datetime, timezone

from .base_client import (
    BaseApiClient,
    RateLimitError,
    env_bool,
    env_csv,
    env_float,
    env_int,
    env_max_batch_chars,
    env_str,
    iter_batches,
)

# ── EverOS ───────────────────────────────────────────────────────────────────


class EverosClient(BaseApiClient):
    """EverOS Cloud memory client (REST API, v1).

    Reference: https://docs.evermind.ai/api-reference/introduction
               https://docs.evermind.ai/llms.txt
               https://github.com/EverMind-AI/EverOS

    Supports two modes controlled by ``EVEROS_USE_GROUP``:

    **Group mode** (recommended for multi-party benchmarks like LoCoMo):
      - POST /api/v1/memories/group       — add group memories
      - POST /api/v1/memories/group/flush  — force group extraction
      - POST /api/v1/memories/search       — search by group_id
      - POST /api/v1/memories/delete       — delete group memories

    **Personal mode** (1 human + AI assistant):
      - POST /api/v1/memories             — add personal memories
      - POST /api/v1/memories/flush        — force personal extraction
      - POST /api/v1/memories/search       — search by user_id

    Configurable env vars:
      EVEROS_MODE             — cloud|local  (default: cloud)
      EVEROS_API_KEY          — required in cloud mode, optional in local mode
      EVEROS_BASE_URL         — https://api.evermind.ai or http://localhost:1995
      EVEROS_QPS              — 5
      EVEROS_SEARCH_METHOD    — hybrid  (keyword|vector|hybrid|agentic)
      EVEROS_MEMORY_TYPES     — episodic_memory,profile  (comma-separated)
      EVEROS_FLUSH_AFTER_ADD  — true
      EVEROS_FLUSH_POLICY     — always|accumulated|never  (default: always)
      EVEROS_ASYNC_MODE       — false  (whether to request async add processing)
      EVEROS_FETCH_PROFILE    — true  (fetch profile via /get and merge)
      EVEROS_USE_GROUP        — true  (use group memory for multi-party data)
    """

    TASK_POLL_INTERVAL = 2.0
    TASK_POLL_MAX_WAIT = 120

    def __init__(self):
        mode = env_str("EVEROS_MODE", "cloud").lower()
        if mode not in ("cloud", "local"):
            raise ValueError("EVEROS_MODE must be 'cloud' or 'local'")
        api_key = env_str("EVEROS_API_KEY", "")
        if mode == "cloud" and not api_key:
            raise ValueError("EVEROS_API_KEY environment variable is not set")
        default_base_url = (
            "http://localhost:1995" if mode == "local" else "https://api.evermind.ai"
        )
        base_url = env_str("EVEROS_BASE_URL", default_base_url)
        qps = env_float("EVEROS_QPS", 5, min_value=0)
        timeout = env_int("EVEROS_TIMEOUT", 300, min_value=1)
        self._batch_size = env_int("EVEROS_BATCH_SIZE", 20, min_value=1)
        self._max_batch_chars = env_max_batch_chars("EVEROS_MAX_BATCH_CHARS")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        super().__init__(
            base_url=base_url,
            headers=headers,
            qps=qps,
            timeout=timeout,
        )
        self._search_method = env_str("EVEROS_SEARCH_METHOD", "hybrid")
        self._memory_types = env_csv("EVEROS_MEMORY_TYPES") or [
            "episodic_memory",
            "profile",
        ]
        self._flush_after_add = env_bool("EVEROS_FLUSH_AFTER_ADD", True)
        flush_policy = env_str("EVEROS_FLUSH_POLICY", "").lower()
        if not flush_policy:
            flush_policy = "always" if self._flush_after_add else "never"
        if flush_policy not in ("always", "accumulated", "never"):
            raise ValueError(
                "EVEROS_FLUSH_POLICY must be 'always', 'accumulated', or 'never'"
            )
        self._flush_policy = flush_policy
        self._async_mode = env_bool("EVEROS_ASYNC_MODE", False)
        self._fetch_profile = env_bool("EVEROS_FETCH_PROFILE", True)
        self._use_group = env_bool("EVEROS_USE_GROUP", True)

    @staticmethod
    def _to_unix_ms(iso_string):
        """Convert an ISO-8601 date string to unix milliseconds."""
        if iso_string is None:
            return int(time.time() * 1000)
        try:
            dt = datetime.fromisoformat(iso_string)
            return int(dt.timestamp() * 1000)
        except (ValueError, TypeError):
            return int(time.time() * 1000)

    def _check_rate_limit(self, resp):
        """Raise RateLimitError on 429/403 so ``_retry`` backs off.

        EverOS may return **403 Forbidden** with body ``{"message":"Forbidden"}``
        when per-key call-frequency limits are exceeded (same as throttling).
        """
        if resp.status_code == 429:
            retry_after = self._parse_retry_after(resp)
            raise RateLimitError(retry_after=retry_after, response=resp)
        if resp.status_code == 403:
            # Same backoff as 429: ``_retry`` uses ``retry_after or 1`` × 2^attempt
            retry_after = self._parse_retry_after(resp)
            raise RateLimitError(retry_after=retry_after, response=resp)

    @staticmethod
    def _clean_latex_delimiters_for_everos(content):
        """Remove LaTeX math delimiters that destabilize EverOS JSON parsing."""
        if not isinstance(content, str):
            return content
        return (
            content.replace(r"\(", "(")
            .replace(r"\)", ")")
            .replace(r"\[", "[")
            .replace(r"\]", "]")
        )

    # ── Task polling ─────────────────────────────────────────────────────

    def _poll_task(self, task_id):
        """Poll ``GET /api/v1/tasks/{task_id}`` until success/failed."""
        if not task_id:
            return
        deadline = time.time() + self.TASK_POLL_MAX_WAIT
        while time.time() < deadline:
            try:
                resp = self._get(f"/api/v1/tasks/{task_id}")
                if resp.status_code == 200:
                    status = resp.json().get("data", {}).get("status", "")
                    if status == "success":
                        return
                    if status == "failed":
                        print(f"  ⚠ Task {task_id} failed")
                        return
                elif resp.status_code == 404:
                    return
            except Exception:
                pass
            time.sleep(self.TASK_POLL_INTERVAL)
        print(f"  ⚠ Task {task_id} poll timeout after {self.TASK_POLL_MAX_WAIT}s")

    # ── Delete ───────────────────────────────────────────────────────────

    def delete(self, user_id):
        """Delete all memories for *user_id* using the v1 delete endpoint."""
        if self._use_group:
            return self.delete_group(user_id)

        def _do():
            resp = self._post(
                "/api/v1/memories/delete",
                json={"user_id": user_id},
            )
            self._check_rate_limit(resp)
            if resp.status_code not in (200, 204):
                resp.raise_for_status()

        self._retry(_do)

    def delete_group(self, group_id):
        """Delete all memories for *group_id*."""
        def _do():
            resp = self._post(
                "/api/v1/memories/delete",
                json={"group_id": group_id},
            )
            self._check_rate_limit(resp)
            if resp.status_code not in (200, 204):
                resp.raise_for_status()

        self._retry(_do)

    # ── Flush ────────────────────────────────────────────────────────────

    def flush(self, user_id, session_id=None):
        """Force personal memory extraction via ``POST /api/v1/memories/flush``."""
        payload = {"user_id": user_id}
        if session_id is not None:
            payload["session_id"] = str(session_id)

        def _do():
            resp = self._post("/api/v1/memories/flush", json=payload)
            self._check_rate_limit(resp)
            if resp.status_code not in (200, 202):
                resp.raise_for_status()
            return resp.json()

        return self._retry(_do)

    def flush_group(self, group_id):
        """Force group memory extraction via ``POST /api/v1/memories/group/flush``."""
        payload = {"group_id": group_id}

        def _do():
            resp = self._post("/api/v1/memories/group/flush", json=payload)
            self._check_rate_limit(resp)
            if resp.status_code not in (200, 202):
                resp.raise_for_status()
            return resp.json()

        result = self._retry(_do)
        status = (result.get("data") or {}).get("status", "")
        if status == "extracted":
            print(f"  ✓ Flush group {group_id}: extracted")
        return result

    # ── Add (personal) ───────────────────────────────────────────────────

    def _should_flush_after_add(self, add_statuses):
        if self._flush_policy == "never":
            return False
        if self._flush_policy == "always":
            return True
        return any(status != "extracted" for status in add_statuses)

    def add(self, messages, user_id, conv_id=None, batch_size=None, flush=None):
        """Add messages to EverOS personal memory (``POST /api/v1/memories``).

        By default, requests synchronous processing and flushes after add so
        session tails are converted into memories.  ``EVEROS_FLUSH_POLICY`` can
        reduce the expensive flush calls for large offline benchmarks.
        """
        if self._use_group:
            return self.add_group(messages, user_id, batch_size=batch_size, flush=flush)

        session_id = str(conv_id) if conv_id is not None else None
        add_statuses = []

        for batch in iter_batches(messages, batch_size or self._batch_size,
                                  max_chars=self._max_batch_chars):
            everos_msgs = []
            for msg in batch:
                role = msg.get("role", "user")
                everos_msgs.append({
                    "role": role,
                    "content": self._clean_latex_delimiters_for_everos(
                        msg["content"]
                    ),
                    "timestamp": self._to_unix_ms(msg.get("chat_time")),
                })
            payload = {
                "user_id": user_id,
                "messages": everos_msgs,
                "async_mode": self._async_mode,
            }
            if session_id is not None:
                payload["session_id"] = session_id

            def _do(p=payload):
                resp = self._post("/api/v1/memories", json=p)
                self._check_rate_limit(resp)
                if resp.status_code not in (200, 202):
                    resp.raise_for_status()
                return resp.json()

            result = self._retry(_do)
            add_statuses.append((result.get("data") or {}).get("status", ""))

        should_flush = self._should_flush_after_add(add_statuses) if flush is None else flush
        if should_flush:
            self.flush(user_id, session_id=session_id)

    # ── Add (group) ──────────────────────────────────────────────────────

    def add_group(self, messages, group_id, batch_size=None, flush=None):
        """Add messages to EverOS group memory (``POST /api/v1/memories/group``).

        Each message must have a ``name`` key used as ``sender_id`` /
        ``sender_name`` for per-participant attribution.
        """
        add_statuses = []
        for batch in iter_batches(messages, batch_size or self._batch_size,
                                  max_chars=self._max_batch_chars):
            everos_msgs = []
            for msg in batch:
                sender = msg.get("name") or msg.get("sender_id") or "unknown"
                everos_msgs.append({
                    "role": "user",
                    "sender_id": sender,
                    "sender_name": sender,
                    "content": self._clean_latex_delimiters_for_everos(
                        msg["content"]
                    ),
                    "timestamp": self._to_unix_ms(msg.get("chat_time")),
                })
            payload = {
                "group_id": group_id,
                "messages": everos_msgs,
                "async_mode": self._async_mode,
            }

            def _do(p=payload):
                resp = self._post("/api/v1/memories/group", json=p)
                self._check_rate_limit(resp)
                if resp.status_code not in (200, 202):
                    print(f"  ⚠ Group add failed ({resp.status_code}): {resp.text[:300]}")
                    resp.raise_for_status()
                return resp.json()

            result = self._retry(_do)
            add_statuses.append((result.get("data") or {}).get("status", ""))

        should_flush = self._should_flush_after_add(add_statuses) if flush is None else flush
        if should_flush:
            self.flush_group(group_id)

    # ── Profile (GET endpoint) ───────────────────────────────────────────

    def _get_profile(self, user_id=None, group_id=None):
        """Fetch profile memories via ``POST /api/v1/memories/get``."""
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if group_id:
            filters["group_id"] = group_id
        payload = {
            "filters": filters,
            "memory_type": "profile",
        }

        def _do():
            resp = self._post("/api/v1/memories/get", json=payload)
            self._check_rate_limit(resp)
            if resp.status_code == 200:
                return resp.json()
            return {}

        return self._retry(_do)

    def _get_user_profile(self, user_id):
        """Fetch user profile memories.

        The v1 API documents profile ``/get`` as user-scoped.  In group mode,
        search still returns group-filtered profile hits when requested via
        ``memory_types``; this helper avoids sending group_id to ``/get``.
        """
        return self._get_profile(user_id=user_id)

    # ── Search ───────────────────────────────────────────────────────────

    def search(self, query, user_id, top_k):
        """Search memories by user_id or group_id. Returns plain text.

        When ``_use_group`` is True, *user_id* is treated as a group_id
        and the filter is ``{group_id: user_id}``.

        When ``_use_group`` is False, filters by ``{user_id: user_id}``.
        """
        if self._use_group:
            filters = {"group_id": user_id}
        else:
            filters = {"user_id": user_id}

        search_payload = {
            "query": query,
            "method": self._search_method,
            "top_k": top_k,
            "filters": filters,
            "memory_types": self._memory_types,
        }

        def _search():
            resp = self._post("/api/v1/memories/search", json=search_payload)
            self._check_rate_limit(resp)
            if resp.status_code != 200:
                resp.raise_for_status()
            return resp.json()

        result = self._retry(_search)
        data = result.get("data", {})

        profile_data = {}
        if self._fetch_profile:
            try:
                profile_data = self._get_user_profile(user_id).get("data", {})
            except Exception:
                pass

        return self._format_as_text(data, profile_data)

    @staticmethod
    def _append_unique(parts, text):
        """Append non-empty text once, preserving order."""
        if not isinstance(text, str):
            return
        text = text.strip()
        if text and text not in parts:
            parts.append(text)

    @classmethod
    def _profile_item_to_text(cls, item, default_label=None):
        """Normalize one EverOS profile item from cloud or local v1 shapes."""
        if isinstance(item, str):
            return item.strip()
        if not isinstance(item, dict):
            return ""

        profile_inner = item.get("profile_data")
        if isinstance(profile_inner, dict):
            embed_text = profile_inner.get("embed_text")
            if embed_text:
                label = (
                    profile_inner.get("item_type")
                    or item.get("item_type")
                    or default_label
                )
                return f"[{label}] {embed_text}" if label else str(embed_text)

        desc = (
            item.get("description") or item.get("content") or item.get("summary")
            or item.get("episode") or item.get("embed_text")
        )
        label = (
            item.get("category") or item.get("trait") or item.get("trait_name")
            or item.get("item_type") or default_label
        )
        if desc:
            text = f"[{label}] {desc}" if label else str(desc)
            evidence = item.get("evidence") or item.get("basis")
            if evidence:
                text = f"{text} Evidence: {evidence}"
            return text
        return ""

    @classmethod
    def _extend_profile_group(cls, profile_parts, items, default_label=None):
        """Append profile entries from string, dict, or list containers."""
        if isinstance(items, list):
            for item in items:
                cls._append_unique(
                    profile_parts,
                    cls._profile_item_to_text(item, default_label=default_label),
                )
            return
        if isinstance(items, dict):
            text = cls._profile_item_to_text(items, default_label=default_label)
            if text:
                cls._append_unique(profile_parts, text)
            else:
                for key, value in items.items():
                    if value:
                        cls._append_unique(profile_parts, f"{key}: {value}")
            return
        if isinstance(items, str):
            cls._append_unique(profile_parts, items)

    @classmethod
    def _extend_profile_parts(cls, profile_parts, profiles):
        """Append normalized profile text from EverOS search/get responses."""
        if not isinstance(profiles, list):
            return
        for profile in profiles:
            if not isinstance(profile, dict):
                cls._append_unique(profile_parts, str(profile))
                continue

            text = cls._profile_item_to_text(profile)
            if text:
                cls._append_unique(profile_parts, text)

            profile_inner = profile.get("profile_data") or {}
            if not isinstance(profile_inner, dict):
                continue
            for group_key, default_label in (
                ("explicit_info", "explicit_info"),
                ("implicit_traits", "implicit_trait"),
            ):
                cls._extend_profile_group(
                    profile_parts,
                    profile_inner.get(group_key),
                    default_label=default_label,
                )

    @classmethod
    def _extend_raw_messages(cls, parts, raw_messages):
        if not isinstance(raw_messages, list):
            return
        raw_parts = []
        for msg in raw_messages:
            if isinstance(msg, str):
                cls._append_unique(raw_parts, msg)
                continue
            if not isinstance(msg, dict):
                continue
            content = msg.get("content") or msg.get("message") or msg.get("text")
            if not content:
                continue
            role = msg.get("role") or msg.get("sender_name") or msg.get("sender_id")
            prefix = f"{role}: " if role else ""
            cls._append_unique(raw_parts, f"{prefix}{content}")
        if raw_parts:
            parts.append("Raw Messages:")
            parts.extend(f"  {idx}. {text}" for idx, text in enumerate(raw_parts, 1))

    @classmethod
    def _extend_agent_memory(cls, parts, agent_memory):
        if not isinstance(agent_memory, dict):
            return
        agent_parts = []
        for key in ("cases", "skills", "agent_cases", "agent_skills"):
            items = agent_memory.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, str):
                    cls._append_unique(agent_parts, item)
                    continue
                if not isinstance(item, dict):
                    continue
                text = (
                    item.get("description") or item.get("content")
                    or item.get("summary") or item.get("name")
                )
                if text:
                    label = item.get("title") or item.get("skill_name") or key
                    cls._append_unique(agent_parts, f"[{label}] {text}")
        if agent_parts:
            parts.append("Agent Memory:")
            parts.extend(f"  {idx}. {text}" for idx, text in enumerate(agent_parts, 1))

    @classmethod
    def _format_as_text(cls, data, profile_data=None):
        """Format EverOS search results into a plain-text context string.

        Includes episode narratives, atomic facts, and profile information.
        """
        parts = []

        for ep in data.get("episodes", []):
            text = ep.get("episode") or ep.get("summary") or ""
            if text:
                cls._append_unique(parts, text)
            for af in ep.get("atomic_facts", []):
                fact = af.get("atomic_fact") or ""
                if fact and fact not in text:
                    cls._append_unique(parts, f"  - {fact}")

        profile_parts = []
        cls._extend_profile_parts(profile_parts, data.get("profiles", []))

        if profile_data:
            profiles = profile_data.get("profiles", profile_data.get("memories", []))
            cls._extend_profile_parts(profile_parts, profiles)

        gp = data.get("global_profile")
        if isinstance(gp, dict):
            for key, value in gp.items():
                if value:
                    cls._append_unique(profile_parts, f"{key}: {value}")
        elif isinstance(gp, str) and gp:
            cls._append_unique(profile_parts, gp)

        if profile_parts:
            parts.append("User Profile:")
            for idx, p in enumerate(profile_parts, 1):
                parts.append(f"  {idx}. {p}")

        cls._extend_raw_messages(parts, data.get("raw_messages", []))
        cls._extend_agent_memory(parts, data.get("agent_memory", {}))

        return "\n".join(parts) if parts else ""

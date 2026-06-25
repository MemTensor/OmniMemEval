import json
import threading
import uuid

from contextlib import suppress
from datetime import datetime, timezone

import requests

from .base_client import (
    BaseApiClient,
    _split_text,
    env_csv,
    env_float,
    env_int,
    env_json,
    env_max_batch_chars,
    env_optional_bool,
    env_str,
    require_env,
)


# ── Letta (MemGPT) ──────────────────────────────────────────────────────────


class LettaClient(BaseApiClient):
    """Letta (MemGPT) stateful agent memory client (REST API).

    References:
        https://docs.letta.com/api/resources/agents/methods/create/
        https://docs.letta.com/api/resources/agents/subresources/messages/methods/create/
        https://docs.letta.com/api/resources/agents/subresources/passages/methods/create/
        https://docs.letta.com/api/resources/agents/subresources/passages/methods/search/
        https://docs.letta.com/api/resources/passages/methods/search/
        https://docs.letta.com/api/resources/folders/subresources/files/methods/upload/
        https://docs.letta.com/guides/agents/architectures/sleeptime

    Letta is agent-centric.  For LoCoMo the two speakers in a conversation
    share identical dialogue text, so we map **one agent per conversation**
    (not per speaker).  Both speaker user_ids resolve to the same agent,
    avoiding duplicate ingestion and redundant search calls.

    Supported ingest modes (LETTA_INGEST_MODE):
      - files    : upload session transcripts to a folder → auto chunk/embed.
      - archival : directly insert text into agent archival memory via
                   POST /v1/agents/{id}/archival-memory (fastest, deterministic).
      - messages : send sessions through the messages endpoint and let the
                   agent autonomously write archival memory (slowest, LLM-based).

    Supported eval modes (LETTA_EVAL_MODE):
      - direct : ask the Letta agent to answer; the pipeline stores it as
                 reflect_answer and skips external ANSWER LLM.
      - rag    : retrieve passages/archival memory and use the external
                 ANSWER LLM as with other memory products.

    Supported search backends (LETTA_SEARCH_BACKEND):
      - passages : POST /v1/passages/search (global, returns relevance scores).
      - archival : GET  /v1/agents/{id}/archival-memory/search (agent-scoped,
                   supports temporal filtering).
    """

    _REQUEST_TIMEOUT = 120

    def __init__(self):
        api_key = require_env("LETTA_API_KEY")
        base_url = env_str("LETTA_BASE_URL", "https://api.letta.com")
        qps = env_float("LETTA_QPS", 2, min_value=0)
        super().__init__(
            base_url=base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            qps=qps,
        )
        self._model = env_str("LETTA_MODEL", "") or None
        self._embedding = env_str("LETTA_EMBEDDING_MODEL", "") or None
        self._agent_type = env_str("LETTA_AGENT_TYPE", "") or None
        self._ingest_mode = env_str("LETTA_INGEST_MODE", "files").lower()
        self._eval_mode = env_str("LETTA_EVAL_MODE", "direct").lower()
        self._search_backend = env_str("LETTA_SEARCH_BACKEND", "passages").lower()

        self._message_buffer_autoclear = env_optional_bool(
            "LETTA_MESSAGE_BUFFER_AUTOCLEAR"
        )
        self._include_base_tools = env_optional_bool("LETTA_INCLUDE_BASE_TOOLS")
        self._include_base_tool_rules = env_optional_bool(
            "LETTA_INCLUDE_BASE_TOOL_RULES"
        )
        self._enable_sleeptime = env_optional_bool("LETTA_ENABLE_SLEEPTIME")
        self._sleeptime_agent_frequency = env_int(
            "LETTA_SLEEPTIME_AGENT_FREQUENCY"
        )
        self._max_files_open = env_int("LETTA_MAX_FILES_OPEN")
        self._per_file_view_window_char_limit = env_int(
            "LETTA_PER_FILE_VIEW_WINDOW_CHAR_LIMIT"
        )

        self._description = env_str("LETTA_AGENT_DESCRIPTION", "") or None
        self._system = env_str("LETTA_SYSTEM", "") or None
        self._context_window_limit = env_int("LETTA_CONTEXT_WINDOW_LIMIT")
        self._timezone = env_str("LETTA_TIMEZONE", "") or None
        self._memory_blocks = env_json("LETTA_MEMORY_BLOCKS_JSON")
        self._model_settings = env_json("LETTA_MODEL_SETTINGS_JSON")
        self._compaction_settings = env_json("LETTA_COMPACTION_SETTINGS_JSON")
        self._initial_message_sequence = env_json(
            "LETTA_INITIAL_MESSAGE_SEQUENCE_JSON"
        )
        self._tool_rules = env_json("LETTA_TOOL_RULES_JSON")
        self._secrets = env_json("LETTA_SECRETS_JSON")
        self._tool_ids = env_csv("LETTA_TOOL_IDS")
        self._tools = env_csv("LETTA_TOOLS")
        self._identity_ids = env_csv("LETTA_IDENTITY_IDS")
        self._agent_folder_ids = env_csv("LETTA_FOLDER_IDS")
        self._tags = env_csv("LETTA_TAGS")
        self._create_metadata = env_json("LETTA_AGENT_METADATA_JSON")

        self._folder_embedding = env_str("LETTA_FOLDER_EMBEDDING_MODEL", "") or None
        self._folder_embedding_config = env_json(
            "LETTA_FOLDER_EMBEDDING_CONFIG_JSON"
        )
        self._folder_instructions = env_str("LETTA_FOLDER_INSTRUCTIONS", "") or None
        self._folder_embedding_chunk_size = env_int(
            "LETTA_FOLDER_EMBEDDING_CHUNK_SIZE"
        )
        self._folder_duplicate_handling = (
            env_str("LETTA_FILE_DUPLICATE_HANDLING", "") or None
        )

        self._batch_size = env_int("LETTA_BATCH_SIZE", 20, min_value=1)
        self._archival_max_chars = env_max_batch_chars(
            "LETTA_ARCHIVAL_MAX_CHARS", default=12000
        )
        self._ingest_max_steps = env_int("LETTA_INGEST_MAX_STEPS")
        self._answer_max_steps = env_int("LETTA_ANSWER_MAX_STEPS")
        self._message_timeout = env_float(
            "LETTA_MESSAGE_TIMEOUT", self._REQUEST_TIMEOUT, min_value=1
        )
        self._enable_thinking = env_optional_bool("LETTA_ENABLE_THINKING")
        self._include_return_message_types = env_csv(
            "LETTA_INCLUDE_RETURN_MESSAGE_TYPES"
        )

        self._search_tags = env_csv("LETTA_SEARCH_TAGS")
        self._search_tag_match_mode = (
            env_str("LETTA_SEARCH_TAG_MATCH_MODE", "") or None
        )
        self._answer_prompt = env_str("LETTA_ANSWER_PROMPT", "") or (
            "Answer the following LoCoMo memory question using your available "
            "memories and attached files. If files are available, search them "
            "before answering. Return only the shortest possible answer, "
            "preferably under 5-6 words.\n\nQuestion: {question}"
        )

        self._agent_map = {}      # conversation key -> agent_id
        self._folder_map = {}     # conversation key -> folder_id
        self._ingested_sessions = set()
        self._agent_locks = {}
        self._agent_locks_guard = threading.Lock()

        self._validate_config()

    def _validate_config(self):
        def _check(name, value, allowed):
            if value not in allowed:
                raise ValueError(
                    f"{name}={value!r} is invalid; expected one of {sorted(allowed)}"
                )

        _check(
            "LETTA_INGEST_MODE",
            self._ingest_mode,
            {"files", "archival", "messages"},
        )
        _check("LETTA_EVAL_MODE", self._eval_mode, {"direct", "rag"})
        _check("LETTA_SEARCH_BACKEND", self._search_backend, {"passages", "archival"})
        if self._folder_duplicate_handling:
            _check(
                "LETTA_FILE_DUPLICATE_HANDLING",
                self._folder_duplicate_handling,
                {"skip", "error", "suffix", "replace"},
            )
        if self._search_tag_match_mode:
            _check(
                "LETTA_SEARCH_TAG_MATCH_MODE",
                self._search_tag_match_mode,
                {"any", "all"},
            )

    @staticmethod
    def _page_items(data):
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("items", "data", "results"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
        return []

    @staticmethod
    def _conv_key(user_id):
        """Strip the ``_speaker_{a,b}_<version>`` suffix to get a
        conversation-level key shared by both speakers."""
        for tag in ("_speaker_a_", "_speaker_b_"):
            idx = user_id.find(tag)
            if idx != -1:
                return user_id[:idx]
        return user_id

    def _lock_for_agent(self, agent_id):
        with self._agent_locks_guard:
            if agent_id not in self._agent_locks:
                self._agent_locks[agent_id] = threading.Lock()
            return self._agent_locks[agent_id]

    def _find_agent(self, conv_key):
        agent_name = f"omnimemeval_{conv_key}"
        resp = self._get("/v1/agents/", params={"name": agent_name, "limit": 1})
        if resp.status_code != 200:
            return None
        agents = self._page_items(resp.json())
        if not agents:
            return None
        return agents[0]["id"] if isinstance(agents[0], dict) else agents[0]

    def _get_or_create_agent(self, user_id):
        conv_key = self._conv_key(user_id)
        if conv_key in self._agent_map:
            return self._agent_map[conv_key]
        aid = self._find_agent(conv_key)
        if aid:
            self._agent_map[conv_key] = aid
            return aid
        agent_name = f"omnimemeval_{conv_key}"
        payload = {
            "name": agent_name,
        }
        optional_fields = {
            "description": self._description,
            "system": self._system,
            "model": self._model,
            "embedding": self._embedding,
            "agent_type": self._agent_type,
            "context_window_limit": self._context_window_limit,
            "timezone": self._timezone,
            "message_buffer_autoclear": self._message_buffer_autoclear,
            "include_base_tools": self._include_base_tools,
            "include_base_tool_rules": self._include_base_tool_rules,
            "enable_sleeptime": self._enable_sleeptime,
            "max_files_open": self._max_files_open,
            "per_file_view_window_char_limit": self._per_file_view_window_char_limit,
            "model_settings": self._model_settings,
            "compaction_settings": self._compaction_settings,
            "initial_message_sequence": self._initial_message_sequence,
            "tool_ids": self._tool_ids or None,
            "tools": self._tools or None,
            "tool_rules": self._tool_rules,
            "secrets": self._secrets,
            "identity_ids": self._identity_ids or None,
            "folder_ids": self._agent_folder_ids or None,
            "tags": self._tags or None,
            "metadata": self._create_metadata,
            "memory_blocks": self._memory_blocks,
        }
        for key, value in optional_fields.items():
            if value is not None:
                payload[key] = value
        resp = self._post("/v1/agents/", json=payload)
        resp.raise_for_status()
        data = resp.json()
        aid = data["id"]
        self._agent_map[conv_key] = aid

        if self._sleeptime_agent_frequency is not None:
            group = data.get("managed_group") or data.get("multi_agent_group")
            if group and group.get("id"):
                self._update_sleeptime_frequency(group["id"])

        return aid

    def _update_sleeptime_frequency(self, group_id):
        payload = {
            "manager_config": {
                "sleeptime_agent_frequency": self._sleeptime_agent_frequency,
            }
        }
        try:
            resp = self._patch(f"/v1/groups/{group_id}", json=payload)
            if resp.status_code not in (200, 201, 204):
                print(f"  ⚠ Failed to set sleeptime frequency: {resp.status_code}")
        except Exception as e:
            print(f"  ⚠ Failed to set sleeptime frequency: {e}")

    def _find_folder(self, conv_key):
        folder_name = f"omnimemeval_{conv_key}"
        resp = self._get("/v1/folders/", params={"name": folder_name, "limit": 1})
        if resp.status_code != 200:
            return None
        folders = self._page_items(resp.json())
        for folder in folders:
            if not isinstance(folder, dict):
                continue
            if folder.get("name") == folder_name and folder.get("id"):
                return folder["id"]
        if folders:
            folder = folders[0]
            if (
                isinstance(folder, dict)
                and folder.get("id")
                and folder.get("name") is None
            ):
                return folder["id"]
        return self._scan_folder_by_name(folder_name)

    def _scan_folder_by_name(self, folder_name, page_limit=100, max_pages=20):
        """Fallback for resuming after partial runs or API name-filter quirks."""
        params = {"limit": page_limit, "order": "desc", "order_by": "created_at"}
        for _ in range(max_pages):
            resp = self._get("/v1/folders/", params=params)
            if resp.status_code != 200:
                return None
            folders = self._page_items(resp.json())
            for folder in folders:
                if (
                    isinstance(folder, dict)
                    and folder.get("name") == folder_name
                    and folder.get("id")
                ):
                    return folder["id"]
            if len(folders) < page_limit:
                break
            last = folders[-1]
            if not isinstance(last, dict) or not last.get("id"):
                break
            params["after"] = last["id"]
        return None

    @staticmethod
    def _response_error(resp, action):
        body = (resp.text or "").strip()
        if len(body) > 1000:
            body = body[:1000] + "..."
        return requests.exceptions.HTTPError(
            f"{action} failed: {resp.status_code} {resp.reason}; body={body}",
            response=resp,
        )

    def _get_or_create_folder(self, conv_key):
        if conv_key in self._folder_map:
            return self._folder_map[conv_key]
        folder_id = self._find_folder(conv_key)
        if folder_id:
            self._folder_map[conv_key] = folder_id
            return folder_id
        payload = {
            "name": f"omnimemeval_{conv_key}",
            "metadata": {"source": "omnimemeval", "conv_key": conv_key},
        }
        if self._folder_embedding:
            payload["embedding"] = self._folder_embedding
        if self._folder_embedding_config:
            payload["embedding_config"] = self._folder_embedding_config
        if self._folder_embedding_chunk_size is not None:
            payload["embedding_chunk_size"] = self._folder_embedding_chunk_size
        if self._folder_instructions:
            payload["instructions"] = self._folder_instructions
        resp = self._post("/v1/folders/", json=payload)
        if resp.status_code in (400, 409):
            folder_id = self._scan_folder_by_name(payload["name"])
            if folder_id:
                self._folder_map[conv_key] = folder_id
                return folder_id
        if resp.status_code >= 400:
            raise self._response_error(resp, "Letta create folder")
        folder_id = resp.json()["id"]
        self._folder_map[conv_key] = folder_id
        return folder_id

    def _attach_folder(self, agent_id, folder_id):
        resp = self._patch(f"/v1/agents/{agent_id}/folders/attach/{folder_id}")
        if resp.status_code not in (200, 201, 204, 409):
            resp.raise_for_status()

    def _upload_session_file(self, folder_id, name, content):
        self._throttle()
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() != "content-type"
        }
        files = {"file": (name, content.encode("utf-8"), "text/plain")}
        params = {"name": name}
        if self._folder_duplicate_handling:
            params["duplicate_handling"] = self._folder_duplicate_handling
        with requests.Session() as session:
            session.trust_env = False
            resp = session.post(
                self._url(f"/v1/folders/{folder_id}/upload"),
                headers=headers,
                params=params,
                files=files,
                timeout=self._message_timeout,
            )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _session_text(messages, timestamp=None):
        lines = []
        if timestamp:
            if isinstance(timestamp, (int, float)):
                readable = datetime.fromtimestamp(
                    timestamp, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S UTC")
            else:
                readable = str(timestamp)
            lines.append(f"Session time: {readable}")
        for msg in messages:
            speaker = msg.get("name") or msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{speaker}: {content}")
        return "\n".join(lines)

    def add(self, messages, user_id, **kwargs):
        """Ingest a session into Letta using the configured ingestion mode.

        For LoCoMo both speakers resolve to the same agent; the caller passes
        ``session_key`` so only the first call per session sends data.
        When no ``session_key`` is given, every call sends data (no dedup).
        """
        conv_key = self._conv_key(user_id)
        agent_id = self._get_or_create_agent(user_id)
        session_key = kwargs.get("session_key")
        if session_key is not None:
            dedup_key = f"{conv_key}_{session_key}"
            if dedup_key in self._ingested_sessions:
                return
            self._ingested_sessions.add(dedup_key)

        session_text = self._session_text(messages, kwargs.get("timestamp"))
        timestamp = kwargs.get("timestamp")

        if self._ingest_mode == "files":
            folder_id = self._get_or_create_folder(conv_key)
            self._attach_folder(agent_id, folder_id)
            file_name = f"{session_key or uuid.uuid4().hex}.txt"
            self._retry(
                lambda: self._upload_session_file(folder_id, file_name, session_text)
            )
            return

        if self._ingest_mode == "archival":
            self._insert_archival(
                agent_id,
                session_text,
                timestamp=timestamp,
                tags=self._tags or None,
            )
            return

        payload = {
            "input": session_text,
            "streaming": False,
        }
        if self._ingest_max_steps is not None:
            payload["max_steps"] = self._ingest_max_steps
        if self._enable_thinking is not None:
            payload["enable_thinking"] = self._enable_thinking
        if self._include_return_message_types:
            payload["include_return_message_types"] = self._include_return_message_types

        def _do():
            resp = self._post(
                f"/v1/agents/{agent_id}/messages",
                json=payload,
                timeout=self._message_timeout,
            )
            resp.raise_for_status()

        with self._lock_for_agent(agent_id):
            self._retry(_do)

    def _insert_archival(self, agent_id, text, timestamp=None, tags=None):
        """Insert text directly into agent archival memory.

        POST /v1/agents/{agent_id}/archival-memory

        If *text* exceeds ``_archival_max_chars`` it is split at natural
        boundaries (paragraph → line → sentence → hard cut) and each chunk
        is inserted as a separate archival entry.
        """
        created_at = None
        if timestamp:
            if isinstance(timestamp, (int, float)):
                created_at = datetime.fromtimestamp(
                    timestamp, tz=timezone.utc
                ).isoformat()
            else:
                created_at = str(timestamp)

        chunks = _split_text(text, self._archival_max_chars)

        results = []
        for idx, chunk in enumerate(chunks):
            if len(chunks) > 1:
                chunk = f"[part {idx + 1}/{len(chunks)}] {chunk}"
            payload = {"text": chunk}
            if created_at:
                payload["created_at"] = created_at
            if tags:
                payload["tags"] = tags

            def _do(p=payload):
                resp = self._post(
                    f"/v1/agents/{agent_id}/archival-memory", json=p
                )
                if resp.status_code >= 400:
                    raise self._response_error(resp, "Letta archival insert")
                return resp.json()

            results.append(self._retry(_do))
        return results

    @staticmethod
    def _message_content_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text))
                elif item:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content or "")

    def _extract_answer(self, data):
        turns = data.get("turns") or []
        for turn in reversed(turns):
            if turn.get("role") == "assistant" and turn.get("content"):
                return self._message_content_text(turn.get("content"))
        messages = data.get("messages") or []
        for msg in reversed(messages):
            if msg.get("message_type") == "assistant_message" and msg.get("content"):
                return self._message_content_text(msg.get("content"))
        return ""

    def answer(self, query, user_id):
        agent_id = self._get_or_create_agent(user_id)
        payload = {
            "input": self._answer_prompt.format(question=query),
            "streaming": False,
        }
        if self._answer_max_steps is not None:
            payload["max_steps"] = self._answer_max_steps
        if self._enable_thinking is not None:
            payload["enable_thinking"] = self._enable_thinking
        if self._include_return_message_types:
            payload["include_return_message_types"] = self._include_return_message_types

        def _do():
            resp = self._post(
                f"/v1/agents/{agent_id}/messages",
                json=payload,
                timeout=self._message_timeout,
            )
            resp.raise_for_status()
            return resp.json()

        with self._lock_for_agent(agent_id):
            data = self._retry(_do)
        return self._extract_answer(data), data

    def search(self, query, user_id, top_k):
        """Search agent memory.

        Returns:
          - eval_mode=direct → dict {"answer": str, "context": str}
          - eval_mode=rag    → str (concatenated passage texts)
        """
        agent_id = self._get_or_create_agent(user_id)

        if self._eval_mode == "direct":
            answer, data = self.answer(query, user_id)
            usage = data.get("usage") if isinstance(data, dict) else None
            context = ""
            if usage:
                context = f"Letta usage: {json.dumps(usage, ensure_ascii=False)}"
            return {"answer": answer, "context": context}

        if self._search_backend == "passages":
            return self._search_passages(agent_id, query, top_k)

        return self._search_archival(agent_id, query, top_k)

    def _search_passages(self, agent_id, query, top_k):
        """POST /v1/passages/search — global search with relevance scores."""
        payload = {
            "query": query,
            "agent_id": agent_id,
            "limit": top_k,
        }
        if self._search_tags:
            payload["tags"] = self._search_tags
        if self._search_tag_match_mode:
            payload["tag_match_mode"] = self._search_tag_match_mode

        def _do():
            resp = self._post("/v1/passages/search", json=payload)
            resp.raise_for_status()
            return resp.json()

        results = self._retry(_do)
        if not isinstance(results, list):
            results = results.get("results", []) if isinstance(results, dict) else []

        parts = []
        for item in results[:top_k]:
            if isinstance(item, dict):
                passage = item.get("passage", item)
                text = passage.get("text") or passage.get("content") or ""
            else:
                text = str(item)
            if text:
                parts.append(str(text))
        return "\n\n".join(parts)

    def _search_archival(self, agent_id, query, top_k):
        """GET /v1/agents/{id}/archival-memory/search — agent-scoped semantic search."""
        params = {"query": query, "top_k": top_k}
        if self._search_tags:
            params["tags"] = self._search_tags
        if self._search_tag_match_mode:
            params["tag_match_mode"] = self._search_tag_match_mode

        def _do():
            resp = self._get(
                f"/v1/agents/{agent_id}/archival-memory/search", params=params
            )
            resp.raise_for_status()
            return resp.json()

        result = self._retry(_do)
        if isinstance(result, dict):
            results = result.get("results", [])
        elif isinstance(result, list):
            results = result
        else:
            results = []

        return "\n\n".join(
            r.get("content", "") if isinstance(r, dict) else str(r)
            for r in results[:top_k]
            if (r.get("content") if isinstance(r, dict) else r)
        )

    def delete_user(self, user_id):
        conv_key = self._conv_key(user_id)
        agent_id = self._agent_map.pop(conv_key, None) or self._find_agent(conv_key)
        if agent_id:
            with suppress(Exception):
                self._delete(f"/v1/agents/{agent_id}")
        folder_id = self._folder_map.pop(conv_key, None) or self._find_folder(conv_key)
        if folder_id:
            with suppress(Exception):
                self._delete(f"/v1/folders/{folder_id}")

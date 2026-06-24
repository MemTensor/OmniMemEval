import json
import re

from contextlib import suppress

from .base_client import (
    BaseApiClient,
    RateLimitError,
    env_bool,
    env_int,
    env_max_batch_chars,
    env_str,
    require_env,
    iter_batches,
)

_SPECIAL_TOKEN_RE = re.compile(r"<\|[a-z_]+\|>")

# ── Hindsight ────────────────────────────────────────────────────────────────


class HindsightClient(BaseApiClient):
    """Hindsight (Vectorize) agent memory client — v1 REST API (v0.5.6+).

    Reference: https://docs.hindsight.vectorize.io/api-reference
    Developer docs: https://docs.hindsight.vectorize.io/
    Endpoints:
        POST   /v1/default/banks/{bank_id}/memories         – retain (store)
        POST   /v1/default/banks/{bank_id}/memories/recall   – recall (search)
        POST   /v1/default/banks/{bank_id}/reflect           – reflect (reason + answer)
        DELETE /v1/default/banks/{bank_id}/memories          – clear memories
        PUT    /v1/default/banks/{bank_id}                   – create/update bank
        PATCH  /v1/default/banks/{bank_id}/config            – update bank config
        GET    /v1/default/banks/{bank_id}/config            – get bank config

    Uses *memory banks* (one per user_id).  ``retain`` stores content with
    automatic fact extraction; ``recall`` retrieves via TEMPR multi-strategy
    retrieval (semantic + BM25 + graph + temporal); ``reflect`` performs
    AI-powered reasoning over memories and returns a synthesized answer.

    For LoCoMo the two speakers in a conversation share identical dialogue
    text, so we map **one bank per conversation** (not per speaker).

    Key optimizations over naive usage:
    - Retains entire conversations as a single document (not per-message items)
      so the extraction LLM can reason across the full dialogue.
    - Sets ``context`` and ``timestamp`` on each retain for better extraction.
    - Configures bank with ``retain_mission``, ``retain_extraction_mode``,
      and ``observations_mission`` for richer fact graphs.
    - Recalls all three memory types: world, experience, and observation.
    """

    def __init__(self):
        api_key = require_env("HINDSIGHT_API_KEY")
        base_url = env_str(
            "HINDSIGHT_BASE_URL", "https://api.hindsight.vectorize.io"
        )
        qps = env_int("HINDSIGHT_QPS", 5, min_value=0)
        super().__init__(
            base_url=base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            qps=qps,
        )
        self._ingested_banks = set()
        self._configured_banks = set()

        self._retain_extraction_mode = env_str(
            "HINDSIGHT_RETAIN_EXTRACTION_MODE", ""
        )
        self._retain_mission = env_str("HINDSIGHT_RETAIN_MISSION", "")
        self._observations_mission = env_str(
            "HINDSIGHT_OBSERVATIONS_MISSION", ""
        )
        self._retain_async = env_bool("HINDSIGHT_RETAIN_ASYNC", True)
        self._batch_size = env_int("HINDSIGHT_BATCH_SIZE", 20, min_value=1)
        self._retain_batch_size = env_int(
            "HINDSIGHT_RETAIN_BATCH_SIZE", self._batch_size, min_value=1
        )
        self._max_batch_chars = env_max_batch_chars("HINDSIGHT_MAX_BATCH_CHARS")
        self._recall_budget = env_str("HINDSIGHT_RECALL_BUDGET", "high")
        self._recall_max_tokens = env_int(
            "HINDSIGHT_RECALL_MAX_TOKENS", 32768, min_value=1
        )
        self._recall_include_chunks = env_bool(
            "HINDSIGHT_RECALL_INCLUDE_CHUNKS", True
        )
        self._recall_max_chunk_tokens = env_int(
            "HINDSIGHT_RECALL_MAX_CHUNK_TOKENS", 16384, min_value=1
        )
        self._reflect_budget = env_str("HINDSIGHT_REFLECT_BUDGET", "high")
        self._reflect_max_tokens = env_int(
            "HINDSIGHT_REFLECT_MAX_TOKENS", 4096, min_value=1
        )
        self._reflect_include_facts = env_bool(
            "HINDSIGHT_REFLECT_INCLUDE_FACTS", True
        )

    @staticmethod
    def _conv_key(user_id):
        """Strip the ``_speaker_{a,b}_<version>`` suffix to get a
        conversation-level key shared by both speakers."""
        for tag in ("_speaker_a_", "_speaker_b_"):
            idx = user_id.find(tag)
            if idx != -1:
                return user_id[:idx]
        return user_id

    @staticmethod
    def _normalize_timestamp(ts):
        """Convert various date formats to ISO-8601 for the Hindsight API.

        Handles:
          - '2023/05/30 (Tue) 23:40' (LongMemEval question_date format)
          - '2023-05-30T23:40:00' (already ISO)
          - '2023-05-30 23:40:00' (common datetime)
        """
        if not ts:
            return None
        ts = ts.strip()
        if "T" in ts and "/" not in ts:
            return ts
        import re
        m = re.match(r"(\d{4})/(\d{2})/(\d{2})\s*\(\w+\)\s*(\d{2}):(\d{2})", ts)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}:{m.group(5)}:00"
        ts_clean = ts.replace("/", "-")
        if " " in ts_clean and "T" not in ts_clean:
            ts_clean = ts_clean.replace(" ", "T", 1)
        return ts_clean

    def _check_rate_limit(self, resp):
        if resp.status_code == 402:
            raise RuntimeError(
                "Hindsight 402 Payment Required: account balance is "
                "insufficient. Please top up at "
                "https://ui.hindsight.vectorize.io/ and retry."
            )
        if resp.status_code == 429:
            raise RateLimitError(
                retry_after=self._parse_retry_after(resp), response=resp
            )

    def _ensure_bank(self, bank_id):
        def _do():
            resp = self._put(
                f"/v1/default/banks/{bank_id}",
                json={"bank_id": bank_id, "enable_observations": False},
            )
            if resp.status_code not in (200, 201, 409):
                self._check_rate_limit(resp)
                resp.raise_for_status()
        self._retry(_do, max_retries=5)

    def _configure_bank(self, bank_id, retain_mission_override=None):
        """Apply bank-level config overrides (retain_mission, extraction_mode,
        observations_mission) once per bank per client lifetime."""
        if bank_id in self._configured_banks:
            return
        config = {}
        mission = retain_mission_override or self._retain_mission
        if mission:
            config["retain_mission"] = mission
        if self._retain_extraction_mode:
            config["retain_extraction_mode"] = self._retain_extraction_mode
        if self._observations_mission:
            config["observations_mission"] = self._observations_mission
        if not config:
            self._configured_banks.add(bank_id)
            return

        def _do():
            resp = self._patch(
                f"/v1/default/banks/{bank_id}/config", json=config
            )
            if resp.status_code in (200, 204):
                return
            if resp.status_code == 404:
                return
            self._check_rate_limit(resp)
            resp.raise_for_status()

        with suppress(Exception):
            self._retry(_do, max_retries=3)
        self._configured_banks.add(bank_id)

    @staticmethod
    def _try_parse_conversation(raw: str):
        """Try to convert a JSON conversation array into readable plain text.

        Supports two common formats:
        1. LoCoMo: [{"speaker": "X", "text": "...", "blip_caption": "..."}, ...]
        2. Standard role/content: [{"role": "user", "content": "..."}, ...]

        Returns the plain-text string on success, or None if the input is not
        a recognised conversation array (in which case the caller should fall
        back to the raw JSON string).
        """
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, list) or not data:
            return None

        lines = []
        for msg in data:
            if not isinstance(msg, dict):
                return None
            if "speaker" in msg and "text" in msg:
                speaker = msg["speaker"]
                text = msg["text"]
                extra = ""
                if msg.get("blip_caption"):
                    extra = f" [image: {msg['blip_caption']}]"
                lines.append(f"{speaker}: {text}{extra}")
            elif "role" in msg and "content" in msg:
                name = msg.get("name", msg["role"])
                lines.append(f"{name}: {msg['content']}")
            else:
                return None
        return "\n".join(lines)

    def _build_item(self, messages, bank_id, **kwargs):
        """Build a single retain item from messages or raw_content."""
        raw_content = kwargs.get("raw_content")
        context = kwargs.get("context")
        explicit_ts = kwargs.get("timestamp")
        session_key = kwargs.get("session_key")

        if raw_content is not None:
            parsed = self._try_parse_conversation(raw_content)
            conversation_text = parsed if parsed else raw_content
            last_ts = explicit_ts
        else:
            lines = []
            last_ts = explicit_ts
            for msg in messages:
                ts = msg.get("chat_time") or msg.get("timestamp") or ""
                name = msg.get("name", msg.get("role", "user"))
                content = msg.get("content", "")
                if ts:
                    lines.append(f"{name} ({ts}): {content}")
                    if not last_ts:
                        last_ts = str(ts)
                else:
                    lines.append(f"{name}: {content}")
            conversation_text = "\n".join(lines)

        document_id = f"{bank_id}_{session_key}" if session_key else None

        conversation_text = _SPECIAL_TOKEN_RE.sub("", conversation_text)

        item = {"content": conversation_text}
        if last_ts:
            item["timestamp"] = last_ts
        if document_id:
            item["document_id"] = document_id
            item["metadata"] = {"doc_id": document_id}
        if context:
            item["context"] = context
        if kwargs.get("tags"):
            item["tags"] = kwargs["tags"]
        return item

    def add(self, messages, user_id, **kwargs):
        """Retain messages into a memory bank as a single conversation document.

        Aligned with AMB benchmark: supports async retain for faster ingestion,
        document_id deduplication, and per-bank retain_mission config.

        Both speakers resolve to the same bank, so only the first call
        per session actually sends data when ``session_key`` is provided.

        Kwargs:
            session_key: Dedup key — only the first call per bank+session is sent.
            raw_content: If provided, used as the item content directly.
            context: Descriptive context string for the item.
            timestamp: Explicit ISO-8601 timestamp for the item.
            retain_mission: Per-bank retain_mission override.
            tags: List of tag strings for user isolation within shared banks.
        """
        bank_id = self._conv_key(user_id)
        session_key = kwargs.get("session_key")
        if session_key is not None:
            dedup_key = f"{bank_id}_{session_key}"
            if dedup_key in self._ingested_banks:
                return
            self._ingested_banks.add(dedup_key)

        self._ensure_bank(bank_id)
        self._configure_bank(bank_id, retain_mission_override=kwargs.get("retain_mission"))

        item = self._build_item(messages, bank_id, **kwargs)
        use_async = self._retain_async
        payload = {"items": [item], "async": use_async}

        def _do():
            resp = self._post(
                f"/v1/default/banks/{bank_id}/memories", json=payload
            )
            self._check_rate_limit(resp)
            if resp.status_code == 409:
                return resp
            resp.raise_for_status()
            return resp

        self._retry(_do)

    def add_batch(self, items_with_meta, user_id, **kwargs):
        """Retain multiple items in batches (aligned with AMB retain_batch).

        Args:
            items_with_meta: List of dicts, each containing kwargs for _build_item
                (messages, raw_content, context, timestamp, session_key, tags).
            user_id: User/bank identifier.
            **kwargs: Shared kwargs (retain_mission, etc).
        """
        bank_id = self._conv_key(user_id)
        self._ensure_bank(bank_id)
        self._configure_bank(bank_id, retain_mission_override=kwargs.get("retain_mission"))

        all_items = []
        for meta in items_with_meta:
            session_key = meta.get("session_key")
            if session_key is not None:
                dedup_key = f"{bank_id}_{session_key}"
                if dedup_key in self._ingested_banks:
                    continue
                self._ingested_banks.add(dedup_key)
            merged = {**kwargs, **meta}
            item = self._build_item(meta.get("messages", []), bank_id, **merged)
            all_items.append(item)

        if not all_items:
            return

        use_async = self._retain_async
        for batch in iter_batches(all_items, self._retain_batch_size,
                                  max_chars=self._max_batch_chars):
            payload = {"items": batch, "async": use_async}

            def _do(p=payload):
                resp = self._post(
                    f"/v1/default/banks/{bank_id}/memories", json=p
                )
                self._check_rate_limit(resp)
                if resp.status_code == 409:
                    return resp
                resp.raise_for_status()
                return resp

            self._retry(_do)

    def await_extraction(self, user_id, max_wait_s=3600, poll_interval=10):
        """Poll until all async retain operations complete for a bank.

        Aligned with AMB: checks for pending operations and waits until
        extraction is done before running recall/search.
        """
        bank_id = self._conv_key(user_id)
        import time as _time
        start = _time.monotonic()
        while _time.monotonic() - start < max_wait_s:
            try:
                resp = self._get(
                    f"/v1/default/banks/{bank_id}/operations",
                    params={"status": "pending", "limit": 1},
                )
                if resp.status_code == 200:
                    pending = resp.json().get("total", 0)
                    if pending == 0:
                        return True
                elif resp.status_code == 404:
                    return True
            except Exception:
                pass
            _time.sleep(poll_interval)
        return False

    @staticmethod
    def _dedup_results(results):
        """Remove duplicate results by fact id, preserving multiple facts from the same chunk."""
        seen = set()
        out = []
        for r in results:
            key = r.get("id")
            if key and key not in seen:
                seen.add(key)
                out.append(r)
            elif not key:
                out.append(r)
        return out

    @staticmethod
    def _format_fact(fact, chunks, seen_chunk_ids):
        """Format a single recall result, inlining chunk text on first appearance."""
        text = fact.get("text", "") or fact.get("content", "")
        if not text:
            return ""
        fact_type = fact.get("type", "")
        lines = []
        lines.append(f"[{fact_type}] {text}" if fact_type else text)

        meta = []
        date_start = fact.get("occurred_start")
        date_end = fact.get("occurred_end")
        if date_start:
            ds = date_start[:10]
            de = date_end[:10] if date_end else ""
            if de and de != ds:
                meta.append(f"occurred: {ds} to {de}")
            else:
                meta.append(f"occurred: {ds}")
        mentioned = fact.get("mentioned_at")
        if mentioned:
            meta.append(f"mentioned: {mentioned[:10]}")
        if meta:
            lines.append("(" + " | ".join(meta) + ")")

        chunk_id = fact.get("chunk_id")
        if chunks and chunk_id and chunk_id in chunks:
            if chunk_id not in seen_chunk_ids:
                chunk_text = chunks[chunk_id].get("text", "")
                if chunk_text:
                    lines.append(f"> {chunk_text}")
                seen_chunk_ids.add(chunk_id)

        return "\n".join(lines)

    def search(self, query, user_id, top_k, **kwargs):
        """Recall memories from a bank using TEMPR multi-strategy retrieval.

        Aligned with AMB official benchmark (v0.5.6+):
        - No explicit `types` filter (server defaults to all: world+experience+observation)
        - `include_entities=False` to avoid wasting tokens on entity data
        - `include_chunks` with generous token budget for full conversation context
        - `query_timestamp` for temporal reasoning
        - Query truncated to 1900 chars (server limit)

        Optional kwargs for per-dataset tuning:
            max_tokens: Override recall max_tokens budget (default 32768).
            max_chunk_tokens: Override chunk max_tokens budget (default 16384).
            query_timestamp: ISO date string for temporal reasoning.
            tags: List of tag filters for user isolation within shared banks.
            tags_match: Tag matching strategy (default None).
        """
        bank_id = self._conv_key(user_id)
        max_tokens = kwargs.get("max_tokens", self._recall_max_tokens)
        max_chunk_tokens = kwargs.get("max_chunk_tokens", self._recall_max_chunk_tokens)
        query_timestamp = kwargs.get("query_timestamp")
        payload = {
            "query": query[:1900],
            "max_tokens": max_tokens,
            "budget": self._recall_budget,
        }
        if query_timestamp:
            payload["query_timestamp"] = self._normalize_timestamp(query_timestamp)
        include_opts = {"entities": None}
        if self._recall_include_chunks:
            include_opts["chunks"] = {"max_tokens": max_chunk_tokens}
        payload["include"] = include_opts
        if kwargs.get("tags"):
            payload["tags"] = kwargs["tags"]
        if kwargs.get("tags_match"):
            payload["tags_match"] = kwargs["tags_match"]

        def _do():
            resp = self._post(
                f"/v1/default/banks/{bank_id}/memories/recall", json=payload
            )
            self._check_rate_limit(resp)
            resp.raise_for_status()
            return resp.json()

        result = self._retry(_do)
        facts = self._dedup_results(result.get("results", []))
        chunks = result.get("chunks") or {}

        seen_chunk_ids = set()
        parts = []
        for f in facts:
            formatted = self._format_fact(f, chunks, seen_chunk_ids)
            if formatted:
                parts.append(formatted)
        return "\n\n".join(parts)

    def reflect(self, query, user_id, budget=None, context=None, query_timestamp=None):
        """Reflect over memories — agentic reasoning that returns a
        synthesized answer directly.

        The ``context`` parameter is **deprecated** upstream (v0.5.6).
        New call sites should concatenate extra context into ``query`` directly;
        if context is provided here, it is appended before the request.

        Returns ``(answer_text, based_on_sources)``.
        """
        bank_id = self._conv_key(user_id)
        if budget is None:
            budget = self._reflect_budget
        effective_query = f"{query}\n\nAdditional context:\n{context}" if context else query
        payload = {
            "query": effective_query,
            "budget": budget,
            "max_tokens": self._reflect_max_tokens,
        }
        if query_timestamp:
            payload["query_timestamp"] = self._normalize_timestamp(query_timestamp)
        if self._reflect_include_facts:
            payload["include"] = {"facts": True}

        def _do():
            resp = self._post(
                f"/v1/default/banks/{bank_id}/reflect", json=payload
            )
            self._check_rate_limit(resp)
            resp.raise_for_status()
            return resp.json()

        result = self._retry(_do)
        answer = result.get("text", "")
        based_on = result.get("based_on", [])
        return answer, based_on

    def delete_user(self, user_id):
        """Delete all memories in the bank (clear the bank)."""
        bank_id = self._conv_key(user_id)
        with suppress(Exception):
            self._delete(f"/v1/default/banks/{bank_id}/memories")

    def delete(self, user_id):
        """Alias for delete_user — full bank reset."""
        self.delete_user(user_id)

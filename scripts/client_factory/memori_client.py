from contextlib import suppress
from datetime import datetime, timezone

from .base_client import (
    BaseApiClient,
    RateLimitError,
    env_int,
    env_max_batch_chars,
    env_str,
    require_env,
    iter_batches,
)
# ── Memori ────────────────────────────────────────────────────────────────────


class MemoriClient(BaseApiClient):
    """Memori Cloud REST client — aligned with SDK v3.3.2 internals.

    Product docs: https://memorilabs.ai/docs/
    Dashboard / keys: https://app.memorilabs.ai/

    The SDK's cloud flow uses **three** REST endpoints (confirmed via
    SDK v3.3.2 source ``memori/_network.py``, ``memory/_manager.py``,
    ``memory/recall.py``, ``memory/augmentation/_handler.py``):

    ┌──────────────────────────────────────────────────────────────────┐
    │ Endpoint                          │ Host       │ Purpose        │
    ├──────────────────────────────────────────────────────────────────┤
    │ POST /v1/cloud/conversation/messages │ api.*     │ Persistence   │
    │ POST /v1/cloud/augmentation          │ collector.*│ Fact extract │
    │ POST /v1/cloud/recall                │ api.*     │ Semantic recall│
    └──────────────────────────────────────────────────────────────────┘

    Auth headers:
      - ``X-Memori-API-Key``: SDK-level key supplied via ``MEMORI_SDK_API_KEY``
      - ``Authorization: Bearer {MEMORI_API_KEY}``: user's API key

    Persistence only stores raw messages; the SDK explicitly calls
    ``cloud/augmentation`` to trigger Advanced Augmentation (async fact
    extraction).  Without the augmentation call, recall returns nothing.
    """

    def __init__(self):
        api_key = require_env("MEMORI_API_KEY")
        sdk_api_key = require_env("MEMORI_SDK_API_KEY")
        base_url = env_str("MEMORI_BASE_URL", "https://api.memorilabs.ai")
        self._collector_url = env_str(
            "MEMORI_COLLECTOR_URL", "https://collector.memorilabs.ai"
        )
        qps = env_int("MEMORI_QPS", 5, min_value=0)
        super().__init__(
            base_url=base_url,
            headers={
                "Content-Type": "application/json",
                "X-Memori-API-Key": sdk_api_key,
                "Authorization": f"Bearer {api_key}",
            },
            qps=qps,
        )
        self._process_id = env_str("MEMORI_PROCESS_ID", "omnimemeval")
        self._session_tpl = env_str("MEMORI_SESSION_ID", "")
        self._batch_size = env_int("MEMORI_BATCH_SIZE", 20, min_value=1)
        self._max_batch_chars = env_max_batch_chars("MEMORI_MAX_BATCH_CHARS")

    def _check_rate_limit(self, resp):
        if resp.status_code == 429:
            retry_after = self._parse_retry_after(resp)
            raise RateLimitError(retry_after=retry_after, response=resp)

    def _session_id(self, user_id):
        if self._session_tpl:
            return self._session_tpl.replace("{user_id}", user_id)
        return f"omnimemeval_{user_id}"

    def _attribution(self, user_id):
        return {
            "entity": {"id": user_id},
            "process": {"id": self._process_id},
        }

    @staticmethod
    def _fmt_text(msg):
        """Extract plain message text for augmentation.

        Aligned with official Memori benchmark: content is the plain message
        text only.  Speaker identity is conveyed via the ``role`` field;
        temporal context is handled by the Augmentation engine internally.
        """
        return msg.get("content", "")

    @staticmethod
    def _memory_entity_external_id(memory):
        if not isinstance(memory, dict):
            return None
        ent = memory.get("entity") or {}
        ext = ent.get("external") or {}
        return ext.get("id")

    def _post_collector(self, path, **kwargs):
        """POST to the collector host (used for augmentation)."""
        self._throttle()
        self._apply_timeout(kwargs)
        url = f"{self._collector_url.rstrip('/')}{path}"
        return self._session.post(url, **kwargs)

    def add(self, messages, user_id, **kwargs):
        session = {"id": self._session_id(user_id)}
        attribution = self._attribution(user_id)

        for batch in iter_batches(messages, self._batch_size,
                                  max_chars=self._max_batch_chars):

            persist_msgs = [
                {
                    "role": m.get("role", "user"),
                    "text": self._fmt_text(m),
                }
                for m in batch
            ]
            aug_msgs = [
                {
                    "role": m.get("role", "user"),
                    "content": self._fmt_text(m),
                }
                for m in batch
            ]

            persist_payload = {
                "attribution": attribution,
                "session": session,
                "messages": persist_msgs,
            }

            def _persist(p=persist_payload):
                resp = self._post("/v1/cloud/conversation/messages", json=p)
                self._check_rate_limit(resp)
                if resp.status_code not in (200, 201):
                    resp.raise_for_status()

            self._retry(_persist)

            aug_payload = {
                "conversation": {"messages": aug_msgs, "summary": None},
                "meta": {
                    "attribution": attribution,
                    "sdk": {"lang": "python", "version": "3.3.2"},
                },
                "session": session,
            }

            def _augment(p=aug_payload):
                resp = self._post_collector("/v1/cloud/augmentation", json=p)
                self._check_rate_limit(resp)
                if resp.status_code not in (200, 201, 202, 204):
                    resp.raise_for_status()

            with suppress(Exception):
                self._retry(_augment)

    @staticmethod
    def _format_date_created(value):
        """Format date_created to 'YYYY-MM-DD HH:MM' — mirrors SDK _utils.format_date_created."""
        if not value:
            return None
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            try:
                normalized = s[:-1] + "+00:00" if s.endswith("Z") else s
                if "T" not in normalized and " " in normalized:
                    normalized = normalized.replace(" ", "T", 1)
                dt = datetime.fromisoformat(normalized)
                return dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                if len(s) >= 16 and s[4] == "-" and s[7] == "-":
                    return s[:16].replace("T", " ")
                return None
        return None

    def search(self, query, user_id, top_k):
        payload = {
            "attribution": self._attribution(user_id),
            "query": query,
            "session": {"id": self._session_id(user_id)},
            "limit": top_k,
        }

        def _do():
            resp = self._post("/v1/cloud/recall", json=payload)
            self._check_rate_limit(resp)
            resp.raise_for_status()
            return resp.json()

        result = self._retry(_do)
        facts = result.get("facts", [])
        if not isinstance(facts, list):
            return str(result)

        # --- Conversation messages (absolute temporal anchors) ---
        conv_lines = []
        conv_data = result.get("conversation") or {}
        raw_msgs = conv_data.get("messages") or []
        for msg in raw_msgs:
            if not isinstance(msg, dict):
                continue
            text = msg.get("text") or msg.get("content") or ""
            if not text:
                continue
            conv_lines.append(text)

        # --- Facts ---
        parts = []
        for f in facts[:top_k]:
            if isinstance(f, dict):
                text = f.get("content") or f.get("fact") or ""
                if not text:
                    continue
                parts.append(f"- {text}")
            elif f:
                parts.append(f"- {f}")

        # --- Summaries (deduplicated) ---
        seen_summaries = set()
        summary_lines = []
        for f in facts[:top_k]:
            if not isinstance(f, dict):
                continue
            for s in (f.get("summaries") or []):
                if not isinstance(s, dict):
                    continue
                content = s.get("content", "").strip()
                if not content or content in seen_summaries:
                    continue
                seen_summaries.add(content)
                summary_lines.append(content)

        # --- Assemble output: Conversation Context → Facts → Summaries ---
        sections = []
        if conv_lines:
            sections.append("## Conversation Context\n\n" + "\n".join(conv_lines))
        if parts:
            sections.append("## Facts\n\n" + "\n".join(parts))
        if summary_lines:
            sections.append("## Summaries\n\n" + "\n\n".join(summary_lines))

        return "\n\n".join(sections)

    def delete_user(self, user_id):
        """Memori Cloud does not expose a per-entity delete API (the
        ``GET /v1/cloud/memories`` endpoint requires admin-level auth
        not available through the SDK key).  Use versioned user_id
        strings to avoid stale data collisions across evaluation runs.
        """
        pass

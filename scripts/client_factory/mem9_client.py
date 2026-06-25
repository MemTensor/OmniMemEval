import re
import time

import requests

from .base_client import (
    BaseApiClient,
    env_bool,
    env_int,
    env_max_batch_chars,
    env_str,
    require_env,
    iter_batches,
)
# ── mem9 ──────────────────────────────────────────────────────────────────────

_URL_RE = re.compile(
    r"\b(?:https?://)?(?P<host>"
    r"localhost|127(?:\.\d{1,3}){3}|\[::1\]|"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}"
    r")(?P<port>:\d+)?(?P<path>/[^\s\"'<>`]*)?",
    re.IGNORECASE,
)
_HTTP_SCHEME_FALLBACK_RE = re.compile(r"https?://[^\s\"'<>`)]*", re.IGNORECASE)
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_CODE_FENCE_RE = re.compile(r"`{3,}")
_HTML_TAG_RE = re.compile(
    r"<\s*/?\s*[A-Za-z][A-Za-z0-9:_-]*(?:\s+[^<>]*)?/?>"
)
_HTML_TAG_CHARS_RE = re.compile(r"[<>]")
_WAF_TOKEN_RE = re.compile(
    r"\b(javascript|script|iframe|svg|onerror|onclick|onload|eval)\b",
    re.IGNORECASE,
)
_SPACE_RUN_RE = re.compile(r"[ \t]{2,}")


class Mem9Client(BaseApiClient):
    """mem9 (PingCAP) TiDB-backed memory client (REST API v1alpha2).

    Reference: https://github.com/mem9-ai/mem9
    Docs: https://mem9.ai/docs/
    API:  https://mem9.ai/api/

    Uses ``X-API-Key`` for auth and ``X-Mnemo-Agent-Id`` for agent identity.
    Ingestion via ``messages`` mode triggers mem9's refinement pipeline which
    automatically extracts facts/insights from conversation messages.
    User isolation is achieved through ``agent_id`` per speaker.
    """

    _API_PREFIX = "/v1alpha2/mem9s"

    @staticmethod
    def _sanitize_content(content):
        """Make message text safer for the hosted mem9 gateway.

        Some LME coding sessions contain raw HTML, SVG, code fences, and
        URLs. The hosted mem9 gateway can reject those request bodies before
        they reach the JSON API. Keep the useful text while neutralizing the
        byte patterns commonly handled by WAF rules.
        """
        if content is None:
            return ""

        text = str(content)
        text = _CODE_FENCE_RE.sub(" code block ", text)
        text = text.replace("`", "'")
        text = _HTML_TAG_RE.sub(" html markup ", text)
        text = _HTML_TAG_CHARS_RE.sub(" ", text)
        text = _EMAIL_RE.sub("email address", text)
        text = _URL_RE.sub(Mem9Client._replace_url, text)
        text = _HTTP_SCHEME_FALLBACK_RE.sub("url", text)
        text = _WAF_TOKEN_RE.sub(Mem9Client._replace_waf_token, text)
        text = _SPACE_RUN_RE.sub(" ", text)
        return text

    @staticmethod
    def _replace_url(match):
        value = match.group(0)
        trailing = ""
        while value and value[-1] in ".,;:":
            trailing = value[-1] + trailing
            value = value[:-1]

        host = (match.group("host") or "").lower()
        if host in ("localhost", "[::1]") or host.startswith("127."):
            return f"local address{trailing}"
        return f"url{trailing}"

    @staticmethod
    def _replace_waf_token(match):
        token = match.group(1).lower()
        replacements = {
            "javascript": "js term",
            "script": "code term",
            "iframe": "embedded frame",
            "svg": "vector graphic",
            "onerror": "event handler",
            "onclick": "event handler",
            "onload": "event handler",
            "eval": "evaluate call",
        }
        return replacements[token]

    def __init__(self):
        api_key = require_env("MEM9_API_KEY")
        base_url = env_str("MEM9_BASE_URL", "https://api.mem9.ai")
        header_agent_id = env_str("MEM9_AGENT_ID", "omnimemeval")
        self._ingest_mode = env_str("MEM9_INGEST_MODE", "smart")
        self._sync = env_bool("MEM9_SYNC", False)
        self._batch_size = env_int("MEM9_BATCH_SIZE", 20, min_value=1)
        self._max_batch_chars = env_max_batch_chars("MEM9_MAX_BATCH_CHARS")
        timeout = env_int("MEM9_TIMEOUT", 180, min_value=1)
        super().__init__(
            base_url=base_url,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
                "X-Mnemo-Agent-Id": header_agent_id,
            },
            timeout=timeout,
        )

    @staticmethod
    def _fmt_content(msg):
        """Embed ``chat_time`` into content so the refinement pipeline can
        extract temporal context (mem9 API has no dedicated time field)."""
        content = Mem9Client._sanitize_content(msg.get("content", ""))
        chat_time = msg.get("chat_time")
        if chat_time:
            return f"[{chat_time}] {content}"
        return content

    def add(self, messages, user_id, **kwargs):
        all_msgs = [
            {"role": m.get("role", "user"), "content": self._fmt_content(m)}
            for m in messages
        ]

        for batch in iter_batches(all_msgs, self._batch_size,
                                  max_chars=self._max_batch_chars):
            payload = {
                "messages": batch,
                "agent_id": user_id,
                "mode": self._ingest_mode,
                "sync": self._sync,
            }

            def _do(p=payload):
                resp = self._post(f"{self._API_PREFIX}/memories", json=p)
                if resp.status_code not in (200, 201, 202):
                    body = (resp.text or "")[:1000]
                    is_server_timeout = (
                        resp.status_code == 400
                        and "i/o timeout" in body
                    )
                    exc = requests.exceptions.HTTPError(
                        f"mem9 add failed: {resp.status_code} {resp.reason}; body={body}",
                        response=resp,
                    )
                    if is_server_timeout:
                        from .base_client import RateLimitError
                        raise RateLimitError(retry_after=5, response=resp) from exc
                    raise exc

            self._retry(_do)

    def search(self, query, user_id, top_k):
        params = {
            "q": query,
            "agent_id": user_id,
            "limit": top_k,
        }

        def _do():
            resp = self._get(f"{self._API_PREFIX}/memories", params=params)
            resp.raise_for_status()
            return resp.json()

        result = self._retry(_do)
        memories = result.get("memories", result.get("results", result.get("data", [])))
        if isinstance(memories, list):
            return "\n\n".join(
                m.get("content", m.get("text", "")) if isinstance(m, dict) else str(m)
                for m in memories[:top_k]
            )
        return str(result)

    def delete_user(self, user_id):
        # mem9 caps list requests at 200 items. Always fetch offset 0 after
        # deletion because the result set shrinks; incrementing offset can skip
        # records.
        page_limit = 200
        while True:
            resp = self._retry(
                lambda: self._get(
                    f"{self._API_PREFIX}/memories",
                    params={"agent_id": user_id, "limit": page_limit, "offset": 0},
                )
            )
            resp.raise_for_status()
            result = resp.json()
            memories = result.get("memories", result.get("results", result.get("data", [])))
            if not memories:
                return

            deleted = 0
            for memory in list(memories):
                if not isinstance(memory, dict):
                    continue
                memory_id = memory.get("id") or memory.get("memory_id")
                if memory_id:
                    delete_resp = self._retry(
                        lambda mid=memory_id: self._delete(f"{self._API_PREFIX}/memories/{mid}")
                    )
                    if delete_resp.status_code not in (200, 202, 204, 404):
                        delete_resp.raise_for_status()
                    deleted += 1

            if deleted == 0:
                raise RuntimeError(
                    f"mem9 delete_user({user_id}) could not find deletable ids "
                    f"in {len(memories)} returned memories"
                )

            if len(memories) < page_limit:
                return

            time.sleep(0.2)

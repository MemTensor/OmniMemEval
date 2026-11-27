"""CPersona, self-hosted: a CPersona server reached over MCP streamable HTTP.

CPersona calls no generative model when it stores or recalls, so ``Memory service model``
does not apply. Each ``user_id`` is a CPersona agent, and every call names it, so one user's
search never reads another user's records.

- ``add`` stores one session as one record: a date line, then one paragraph per turn prefixed
  by its role, at the session's time. A session can run past CPersona's default write bound
  (16,000 characters); run the server with ``CPERSONA_MAX_CONTENT_LENGTH`` raised so that a
  session is stored whole. A truncated or rejected write raises here.
- ``search`` is CPersona's ``reconstruct``. Each item is rendered as its record's time
  followed by the text the item quotes (the head quote, then any excerpts). The harness's
  ``top_k`` is not passed: ``reconstruct`` decides how many items to return from its mode.
- ``delete`` removes the user's records and episodes on the server.

Configurable env vars:
    CPERSONA_MCP_URL          - the server's MCP endpoint, e.g. http://127.0.0.1:8501/mcp (required)
    CPERSONA_AUTH_TOKEN       - the server's bearer token (required)
    CPERSONA_SEARCH_MODE      - ``reconstruct``'s ``mode``: ``pro`` or ``lite``; unset, the server's default
    CPERSONA_SEARCH_COUNT     - ``reconstruct``'s ``count``; unset, the mode's default
    CPERSONA_TIMEOUT_SECONDS  - per call (default 600)

Requires the ``mcp`` package (2.x) in the harness's environment.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading

from .base_client import env_float, require_env


class CpersonaClient:
    def __init__(self):
        self.url = require_env("CPERSONA_MCP_URL")
        self._headers = {"Authorization": "Bearer " + require_env("CPERSONA_AUTH_TOKEN")}
        self._timeout = env_float("CPERSONA_TIMEOUT_SECONDS", 600.0)
        # One event loop on its own thread: the harness calls the client from worker threads,
        # and every call is a short-lived MCP session against a stateless server.
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()

    @staticmethod
    def parse_result(tool: str, result) -> dict:
        """The tool's JSON answer, or a RuntimeError when the server reports the call failed."""
        text = result.content[0].text if result.content else ""
        if getattr(result, "is_error", None) or getattr(result, "isError", None):
            raise RuntimeError(f"cpersona {tool} failed: {text[:300]}")
        return json.loads(text)

    async def _acall(self, tool: str, args: dict) -> dict:
        import httpx2
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        async with httpx2.AsyncClient(headers=self._headers, timeout=self._timeout) as http:
            async with streamable_http_client(self.url, http_client=http) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    result = await session.call_tool(tool, args)
        return self.parse_result(tool, result)

    def call(self, tool: str, args: dict) -> dict:
        return asyncio.run_coroutine_threadsafe(self._acall(tool, args), self._loop).result(self._timeout + 30)

    def add(self, messages, user_id, session_key=None, **_kw):
        when = next((m.get("chat_time") for m in messages if m.get("chat_time")), None)
        turns = "\n\n".join(f"{m.get('role', 'user')}: {str(m.get('content', '')).strip()}" for m in messages)
        message = {
            "content": (f"Session date: {when}\n\n{turns}" if when else turns),
            "source": {"type": "User", "id": user_id, "name": ""},
        }
        if when:
            message["timestamp"] = when
        if session_key:
            message["id"] = str(session_key)
        res = self.call("store", {"agent_id": user_id, "message": message})
        if res.get("result") == "rejected":
            raise RuntimeError(f"cpersona store rejected a session: {res.get('reason')}")
        if res.get("truncated"):
            raise RuntimeError("cpersona store truncated a session: raise CPERSONA_MAX_CONTENT_LENGTH on the server")
        return res

    @staticmethod
    def render(response: dict) -> list[str]:
        """One block per item: the record's time, then the quoted text."""
        blocks = []
        for item in response.get("items", []):
            claims = item.get("claims", [])
            when = next((c.get("as_of") for c in claims if c.get("ref") == item.get("head_ref")), None)
            if when is None and claims:
                when = claims[0].get("as_of")
            if when is None:
                when = item.get("as_of")  # a lite item carries its lone claim's time on the item itself
            parts = [item.get("content") or ""]
            parts += [x.get("content", "") for x in item.get("excerpts", []) if isinstance(x, dict)]
            body = "\n…\n".join(p for p in parts if p)
            blocks.append(f"[{when}]\n{body}\n" if when else f"{body}\n")
        return blocks

    @staticmethod
    def search_args(query: str, user_id: str, env=os.environ) -> dict:
        """``reconstruct``'s arguments: the server's defaults, or the mode and count the environment names."""
        args = {"agent_id": user_id, "query": query}
        mode = env.get("CPERSONA_SEARCH_MODE", "").strip()
        if mode:
            args["mode"] = mode
        count = env.get("CPERSONA_SEARCH_COUNT", "").strip()
        if count:
            args["count"] = int(count)
        return args

    def search(self, query, user_id, top_k=20, **_kw):
        return self.render(self.call("reconstruct", self.search_args(query, user_id)))

    def delete(self, user_id):
        return self.call("delete_agent_data", {"agent_id": user_id})

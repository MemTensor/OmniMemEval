from .base_client import BaseApiClient, env_float, env_str


class MaasvClient(BaseApiClient):
    """Maasv memory service (self-hosted), over HTTP.

    ``add`` sends each message with its ``chat_time``; ``search`` returns one
    line per retrieved memory, prefixed with its time; ``delete_user`` removes
    one evaluation user's memories. Users are isolated by ``user_id``.

    LoCoMo uses one store per conversation (all turns, under the speaker A
    user id) and one search per question, like the other shared-store
    backends.

    Key env vars:
        MAASV_BASE_URL   Service root (default: http://127.0.0.1:8765)
        MAASV_QPS        Client-side rate limit (default: none)
    """

    def __init__(self):
        super().__init__(
            base_url=env_str("MAASV_BASE_URL", "http://127.0.0.1:8765"),
            headers={"Content-Type": "application/json"},
            qps=env_float("MAASV_QPS", None, min_value=0),
        )

    def _call(self, path, payload):
        def _do():
            resp = self._post(path, json=payload)
            resp.raise_for_status()
            return resp.json()

        return self._retry(_do)

    def add(self, messages, user_id, **kwargs):
        msgs = [
            {"content": m["content"], **({"chat_time": m["chat_time"]} if m.get("chat_time") else {})}
            for m in messages
        ]
        return self._call("/add", {"user_id": str(user_id), "messages": msgs})

    def search(self, query, user_id, top_k):
        data = self._call("/search", {"user_id": str(user_id), "query": query, "top_k": int(top_k)})
        return data["memories"]

    def delete_user(self, user_id):
        return self._call("/delete_user", {"user_id": str(user_id)})

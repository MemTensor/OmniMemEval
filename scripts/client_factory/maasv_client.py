from .base_client import BaseApiClient, env_float, env_int, env_max_batch_chars, env_str, iter_batches


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
        MAASV_BATCH_SIZE Messages per add request (default: 50)
        MAASV_MAX_BATCH_CHARS  Character cap per add request (default: none)
    """

    def __init__(self):
        super().__init__(
            base_url=env_str("MAASV_BASE_URL", "http://127.0.0.1:8765"),
            headers={"Content-Type": "application/json"},
            qps=env_float("MAASV_QPS", None, min_value=0),
        )
        self._batch_size = env_int("MAASV_BATCH_SIZE", 50, min_value=1)
        self._max_batch_chars = env_max_batch_chars("MAASV_MAX_BATCH_CHARS")

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
        totals = {"stored": 0, "deduped": 0}
        for batch in iter_batches(msgs, self._batch_size, max_chars=self._max_batch_chars):
            res = self._call("/add", {"user_id": str(user_id), "messages": batch})
            for key in totals:
                totals[key] += res.get(key, 0)
        return totals

    def search(self, query, user_id, top_k):
        data = self._call("/search", {"user_id": str(user_id), "query": query, "top_k": int(top_k)})
        return data["memories"]

    def delete_user(self, user_id):
        return self._call("/delete_user", {"user_id": str(user_id)})

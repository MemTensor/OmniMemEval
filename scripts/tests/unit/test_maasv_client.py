"""Unit tests for the Maasv client: configuration, registration and request shapes."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from client_factory.maasv_client import MaasvClient
from client_factory.registry import SUPPORTED_LIBS, create_client
from utils.ingest_helpers import inject_time
from utils.search_helpers import DEFAULT_SEARCH_DISPATCH
from locomo.locomo_search import shared_conv_search


class _Resp:
    def __init__(self, body, status=200):
        self._body = body
        self.status_code = status

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class TestMaasvClient(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.copy()
        for key in list(os.environ):
            if key.startswith("MAASV_"):
                os.environ.pop(key)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_registered(self):
        self.assertIn("maasv", SUPPORTED_LIBS)
        self.assertIsInstance(create_client("maasv"), MaasvClient)
        self.assertIn("maasv", DEFAULT_SEARCH_DISPATCH)

    def test_default_and_override_base_url(self):
        self.assertEqual(MaasvClient().base_url, "http://127.0.0.1:8765")
        os.environ["MAASV_BASE_URL"] = "http://maasv.internal:9000"
        self.assertEqual(MaasvClient().base_url, "http://maasv.internal:9000")

    def test_inject_time_sets_chat_time(self):
        from datetime import datetime, timezone

        msgs = [{"role": "user", "content": "Caroline: hi"}]
        self.assertEqual(inject_time(msgs, datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc), "maasv"), {})
        self.assertTrue(msgs[0]["chat_time"].startswith("2023-05-08T13:56"))

    def test_add_sends_content_and_chat_time_per_user(self):
        client = MaasvClient()
        with mock.patch.object(client, "_post", return_value=_Resp({"stored": 2, "deduped": 0})) as post:
            client.add([{"role": "user", "name": "A", "content": "A: x", "chat_time": "2023-05-08T13:56:00+00:00"},
                        {"role": "assistant", "content": "B: y"}], "u1")
        path, kwargs = post.call_args.args[0], post.call_args.kwargs
        self.assertEqual(path, "/add")
        self.assertEqual(kwargs["json"], {"user_id": "u1", "messages": [
            {"content": "A: x", "chat_time": "2023-05-08T13:56:00+00:00"}, {"content": "B: y"}]})

    def test_add_splits_large_sessions_into_ordered_batches(self):
        os.environ["MAASV_BATCH_SIZE"] = "50"
        client = MaasvClient()
        msgs = [{"role": "user", "content": f"m{i}", "chat_time": "2024-01-01T00:00:00+00:00"} for i in range(120)]
        with mock.patch.object(client, "_post", return_value=_Resp({"stored": 1, "deduped": 0})) as post:
            totals = client.add(msgs, "u1")
        sent = [c.kwargs["json"]["messages"] for c in post.call_args_list]
        self.assertEqual([len(b) for b in sent], [50, 50, 20])
        self.assertEqual([m["content"] for b in sent for m in b], [f"m{i}" for i in range(120)])
        self.assertEqual(totals, {"stored": 3, "deduped": 0})

    def test_default_batch_keeps_small_sessions_in_one_request(self):
        client = MaasvClient()
        with mock.patch.object(client, "_post", return_value=_Resp({"stored": 1, "deduped": 0})) as post:
            client.add([{"content": f"m{i}"} for i in range(50)], "u1")
        self.assertEqual(post.call_count, 1)

    def test_search_returns_memory_lines(self):
        client = MaasvClient()
        lines = ["[2023-05-08 13:56] A: x"]
        with mock.patch.object(client, "_post", return_value=_Resp({"memories": lines})) as post:
            self.assertEqual(client.search("q", "u1", 20), lines)
        self.assertEqual(post.call_args.kwargs["json"], {"user_id": "u1", "query": "q", "top_k": 20})

    def test_users_are_separate_requests(self):
        client = MaasvClient()
        with mock.patch.object(client, "_post", return_value=_Resp({"memories": []})) as post:
            client.search("q", "u1", 5)
            client.search("q", "u2", 5)
        self.assertEqual([c.kwargs["json"]["user_id"] for c in post.call_args_list], ["u1", "u2"])

    def test_delete_user(self):
        client = MaasvClient()
        with mock.patch.object(client, "_post", return_value=_Resp({"deleted": 3, "found": 3})) as post:
            self.assertEqual(client.delete_user("u1"), {"deleted": 3, "found": 3})
        self.assertEqual(post.call_args.args[0], "/delete_user")

    def test_server_error_raises_after_retries(self):
        client = MaasvClient()
        with mock.patch.object(client, "_post", return_value=_Resp({"error": "boom"}, status=500)), \
                mock.patch("time.sleep"):
            with self.assertRaises(Exception):
                client._retry(lambda: client._post("/search", json={}).raise_for_status(), max_retries=1)

    def test_locomo_shared_store_search_queries_speaker_a_once(self):
        client = mock.Mock()
        client.search.return_value = ["[2023-05-08 13:56] A: x"]
        context, _, answer, raw = shared_conv_search(client, "q", "ua", "ub", 20, "A", "B")
        client.search.assert_called_once_with("q", "ua", 20)
        self.assertIsNone(answer)
        self.assertEqual(raw, "[2023-05-08 13:56] A: x")
        self.assertIn("Memories for user A", context)


if __name__ == "__main__":
    unittest.main()

"""Offline cachly adapter contract tests; no service needed (requests is patched)."""

import hashlib
import json
import os
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from dotenv import dotenv_values

from scripts.client_factory import registry
from scripts.client_factory.cachly_client import CachlyClient

URL = "http://cachly.test:8780"


def _response(status=200, body=None):
    resp = requests.Response()
    resp.status_code = status
    resp._content = json.dumps(body if body is not None else {}).encode("utf-8")
    resp.url = URL
    return resp


def _request_id(user_id, msgs):
    digest = hashlib.sha256(json.dumps([user_id, msgs], sort_keys=True).encode("utf-8")).hexdigest()[:32]
    return f"omni:{digest}"


class TestCachlyClient(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"CACHLY_AML_URL": URL + "/", "CACHLY_AML_KEY": "test-key"}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        sleep = patch("time.sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)
        self.client = CachlyClient()
        self.post = Mock()
        self.client._session.post = self.post

    def _add_ok(self, *_args, **kwargs):
        return _response(200, {"success": True, "request_id": kwargs["json"]["request_id"]})

    # configuration ----------------------------------------------------
    def test_missing_url_or_key_raises(self):
        with patch.dict(os.environ, {"CACHLY_AML_URL": ""}):
            with self.assertRaisesRegex(ValueError, "CACHLY_AML_URL"):
                CachlyClient()
        with patch.dict(os.environ, {"CACHLY_AML_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "CACHLY_AML_KEY"):
                CachlyClient()

    def test_defaults_and_headers(self):
        self.assertEqual(self.client.base_url, URL)
        self.assertEqual(self.client._add_timeout, 3600)
        self.assertEqual(self.client._search_timeout, 120)
        self.assertEqual(self.client.headers["Authorization"], "Bearer test-key")

    def test_timeouts_from_environment(self):
        with patch.dict(os.environ, {"CACHLY_ADD_TIMEOUT": "7", "CACHLY_SEARCH_TIMEOUT": "9"}):
            client = CachlyClient()
        self.assertEqual((client._add_timeout, client._search_timeout), (7, 9))

    def test_env_template_has_placeholders_only(self):
        template = Path(__file__).resolve().parents[3] / "env_examples/.env.cachly"
        values = dict(dotenv_values(template, interpolate=False))
        self.assertTrue(values["CACHLY_AML_URL"].startswith("<"))
        self.assertTrue(values["CACHLY_AML_KEY"].startswith("<"))
        self.assertEqual(values["TOPK"], "20")

    def test_registered(self):
        self.assertIn("cachly", registry.SUPPORTED_LIBS)
        self.assertIsInstance(registry.create_client("cachly"), CachlyClient)

    # add --------------------------------------------------------------
    def test_add_payload(self):
        self.post.side_effect = self._add_ok
        self.client.add(
            [
                {"role": "user", "name": "Caroline", "content": "I like tea.",
                 "chat_time": "1:56 pm on 8 May, 2023"},
                {"role": "user", "name": "Melanie", "content": "Melanie: Me too."},
                {"role": "assistant", "content": "Noted.", "chat_time": "2023-05-08"},
                {"role": "tool", "content": "no time"},
            ],
            "u1",
        )
        (url,), kwargs = self.post.call_args
        self.assertEqual(url, f"{URL}/v1/memory/add")
        self.assertEqual(kwargs["timeout"], 3600)
        payload = kwargs["json"]
        ms = int(datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc).timestamp() * 1000)
        day = int(datetime(2023, 5, 8, tzinfo=timezone.utc).timestamp() * 1000)
        self.assertEqual(payload["messages"], [
            {"role": "user", "content": "Caroline: I like tea.", "timestamp": ms},
            {"role": "user", "content": "Melanie: Me too."},
            {"role": "assistant", "content": "Noted.", "timestamp": day},
            {"role": "user", "content": "no time"},
        ])
        self.assertEqual(payload["user_id"], "u1")
        self.assertEqual(payload["session_id"], "u1")

    def test_request_id_is_deterministic(self):
        self.post.side_effect = self._add_ok
        messages = [{"role": "user", "content": "hello"}]
        self.client.add(messages, "u1")
        first = self.post.call_args.kwargs["json"]
        self.client.add(messages, "u1")
        second = self.post.call_args.kwargs["json"]
        self.client.add(messages, "u2")
        other = self.post.call_args.kwargs["json"]
        self.assertEqual(first["request_id"], _request_id("u1", first["messages"]))
        self.assertEqual(first["request_id"], second["request_id"])
        self.assertNotEqual(first["request_id"], other["request_id"])

    def test_add_rejects_incomplete_answer(self):
        self.post.return_value = _response(200, {"success": False})
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.client.add([{"role": "user", "content": "x"}], "u1")
        self.post.return_value = _response(200, {"success": True, "request_id": "other"})
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.client.add([{"role": "user", "content": "x"}], "u1")

    def test_unix_ms_formats(self):
        f = CachlyClient._unix_ms
        self.assertIsNone(f(""))
        self.assertIsNone(f("not a date"))
        self.assertEqual(f(1_700_000_000), 1_700_000_000_000)
        self.assertEqual(f(1_700_000_000_000), 1_700_000_000_000)
        self.assertEqual(f("2023-05-08T13:56:00Z"), f("2023-05-08 13:56:00"))

    # search -----------------------------------------------------------
    def test_search_payload_and_text(self):
        self.post.return_value = _response(200, {"data": [{"content": "first"}, {"content": "second"}]})
        text = self.client.search("what tea?", "u1", 20)
        (url,), kwargs = self.post.call_args
        self.assertEqual(url, f"{URL}/v1/memory/search")
        self.assertEqual(kwargs["json"], {"query": "what tea?", "user_id": "u1", "top_k": 20})
        self.assertEqual(kwargs["timeout"], 120)
        self.assertEqual(text, "first\nsecond")

    def test_search_empty(self):
        self.post.return_value = _response(200, {"data": []})
        self.assertEqual(self.client.search("q", "u1", 5), "")

    def test_users_are_isolated(self):
        self.post.return_value = _response(200, {"data": []})
        self.client.search("q", "user-a", 5)
        sent = json.dumps(self.post.call_args.kwargs["json"])
        self.assertIn("user-a", sent)
        self.assertNotIn("user-b", sent)
        self.client.search("q", "user-b", 5)
        self.assertEqual(self.post.call_args.kwargs["json"]["user_id"], "user-b")

    # delete -----------------------------------------------------------
    def test_delete_route_and_payload(self):
        self.post.return_value = _response(200, {"deleted": True})
        self.client.delete("u1")
        (url,), kwargs = self.post.call_args
        self.assertEqual(url, f"{URL}/v1/memory/delete")
        self.assertEqual(kwargs["json"], {"user_id": "u1"})

    def test_delete_unknown_user_is_ok(self):
        self.post.return_value = _response(200, {"deleted": False})
        self.client.delete("nobody")

    def test_delete_error_raises(self):
        self.post.return_value = _response(500, {"error": "boom"})
        with self.assertRaises(requests.exceptions.HTTPError):
            self.client.delete("u1")

    # errors -----------------------------------------------------------
    def test_4xx_raises_without_retry(self):
        self.post.return_value = _response(401, {"error": "bad key"})
        with self.assertRaises(requests.exceptions.HTTPError):
            self.client.search("q", "u1", 5)
        self.assertEqual(self.post.call_count, 1)
        self.sleep.assert_not_called()

    def test_5xx_is_retried(self):
        self.post.side_effect = [_response(503), _response(502), _response(200, {"data": [{"content": "ok"}]})]
        self.assertEqual(self.client.search("q", "u1", 5), "ok")
        self.assertEqual(self.post.call_count, 3)
        self.assertEqual(self.sleep.call_count, 2)

    def test_connection_error_is_retried(self):
        msgs = [{"role": "user", "content": "x"}]
        ok = _response(200, {"success": True, "request_id": _request_id("u1", msgs)})
        self.post.side_effect = [requests.exceptions.ConnectionError("down"), ok]
        self.client.add(msgs, "u1")
        self.assertEqual(self.post.call_count, 2)
        self.assertEqual(self.sleep.call_count, 1)

    def test_persistent_5xx_gives_up(self):
        self.client._max_retries = 3
        self.post.return_value = _response(500)
        with self.assertRaises(requests.exceptions.HTTPError):
            self.client.search("q", "u1", 5)
        self.assertEqual(self.post.call_count, 3)


if __name__ == "__main__":
    unittest.main()

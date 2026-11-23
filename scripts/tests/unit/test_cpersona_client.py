"""Unit tests for the CPersona client: configuration, registration, add/search/delete, user isolation
and failures.

The MCP transport is replaced by a small in-memory stand-in for the three CPersona tools the client
calls, so these tests run offline and need neither a server nor the ``mcp`` package.
"""

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from client_factory.cpersona_client import CpersonaClient
from client_factory.registry import SUPPORTED_LIBS, create_client


class StandIn:
    """store / reconstruct / delete_agent_data, keyed by agent_id, answering as CPersona does."""

    def __init__(self):
        self.records = {}
        self.calls = []

    def __call__(self, tool, args):
        self.calls.append((tool, args))
        agent = args["agent_id"]
        if tool == "store":
            message = args["message"]
            if "FAIL" in message["content"]:
                raise RuntimeError("cpersona store failed: connection refused")
            if "REJECT" in message["content"]:
                return {"ok": False, "result": "rejected", "reason": "empty content"}
            rows = self.records.setdefault(agent, [])
            rows.append(message)
            return {"result": "stored", "id": len(rows), "truncated": "TRUNCATE" in message["content"]}
        if tool == "reconstruct":
            items = []
            for i, message in enumerate(self.records.get(agent, [])):
                if args["query"] in message["content"]:
                    ref = f"mem:{i}"
                    items.append({
                        "head_ref": ref,
                        "content": message["content"],
                        "claims": [{"ref": "mem:other", "as_of": "1999-01-01T00:00:00"},
                                   {"ref": ref, "as_of": message.get("timestamp")}],
                    })
            return {"items": items}
        if tool == "delete_agent_data":
            self.records.pop(agent, None)
            return {"ok": True, "agent_id": agent}
        raise AssertionError(f"unexpected tool {tool}")


def session(text, when="2023-05-20T02:38:00+00:00"):
    return [{"role": "user", "content": text, "chat_time": when},
            {"role": "assistant", "content": "  noted  "}]


class TestCpersonaClient(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       ("CPERSONA_MCP_URL", "CPERSONA_AUTH_TOKEN", "CPERSONA_SEARCH_MODE", "CPERSONA_SEARCH_COUNT")}
        os.environ["CPERSONA_MCP_URL"] = "http://127.0.0.1:1/mcp"
        os.environ["CPERSONA_AUTH_TOKEN"] = "test-token"
        os.environ.pop("CPERSONA_SEARCH_MODE", None)
        os.environ.pop("CPERSONA_SEARCH_COUNT", None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def client(self):
        client = create_client("cpersona")
        client.call = StandIn()
        return client

    def test_registered(self):
        self.assertIn("cpersona", SUPPORTED_LIBS)
        self.assertIsInstance(create_client("cpersona"), CpersonaClient)

    def test_harness_passes_the_session_key_and_time_and_searches_text(self):
        from datetime import datetime, timezone

        from utils.ingest_helpers import inject_time, session_id_kwargs
        from utils.search_helpers import DEFAULT_SEARCH_DISPATCH, generic_text_search

        self.assertEqual(session_id_kwargs("cpersona", "u1_s3"), {"session_key": "u1_s3"})
        msgs = [{"role": "user", "content": "hi"}]
        self.assertEqual(inject_time(msgs, datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc), "cpersona"), {})
        self.assertTrue(msgs[0]["chat_time"].startswith("2023-05-08T13:56"))
        self.assertIs(DEFAULT_SEARCH_DISPATCH["cpersona"], generic_text_search)

    def test_missing_configuration_fails_clearly(self):
        for name in ("CPERSONA_MCP_URL", "CPERSONA_AUTH_TOKEN"):
            saved = os.environ.pop(name)
            with self.assertRaisesRegex(ValueError, name):
                create_client("cpersona")
            os.environ[name] = saved

    def test_bearer_token_is_sent(self):
        self.assertEqual(create_client("cpersona")._headers, {"Authorization": "Bearer test-token"})

    def test_add_stores_one_dated_record_per_session(self):
        client = self.client()
        client.add(session("I moved to Denver"), "u1", session_key="u1:session_3")
        (tool, args), = client.call.calls
        self.assertEqual(tool, "store")
        self.assertEqual(args["agent_id"], "u1")
        message = args["message"]
        self.assertEqual(message["content"],
                         "Session date: 2023-05-20T02:38:00+00:00\n\nuser: I moved to Denver\n\nassistant: noted")
        self.assertEqual(message["timestamp"], "2023-05-20T02:38:00+00:00")
        self.assertEqual(message["id"], "u1:session_3")
        self.assertEqual(message["source"], {"type": "User", "id": "u1", "name": ""})

    def test_add_without_time_or_key_sends_neither(self):
        client = self.client()
        client.add([{"role": "user", "content": "no date here"}], "u1")
        message = client.call.calls[0][1]["message"]
        self.assertEqual(message["content"], "user: no date here")
        self.assertNotIn("timestamp", message)
        self.assertNotIn("id", message)

    def test_search_renders_the_head_claims_time_and_quotes(self):
        client = self.client()
        client.add(session("I moved to Denver"), "u1")
        blocks = client.search("Denver", "u1", 20)
        self.assertEqual(len(blocks), 1)
        self.assertTrue(blocks[0].startswith("[2023-05-20T02:38:00+00:00]\nSession date:"))

    def test_render_joins_excerpts_and_reads_a_lite_items_time(self):
        response = {"items": [
            {"head_ref": "mem:1", "content": "head", "excerpts": [{"content": "more"}, "skip"],
             "claims": [{"ref": "mem:1", "as_of": "2023-01-02"}]},
            {"head_ref": "mem:2", "content": "lite", "as_of": "2023-03-04"},
            {"head_ref": "mem:3", "content": "undated"},
        ]}
        self.assertEqual(CpersonaClient.render(response),
                         ["[2023-01-02]\nhead\n…\nmore\n", "[2023-03-04]\nlite\n", "undated\n"])

    def test_search_does_not_pass_top_k(self):
        client = self.client()
        client.search("Denver", "u1", 7)
        self.assertEqual(client.call.calls[-1], ("reconstruct", {"agent_id": "u1", "query": "Denver"}))

    def test_search_mode_and_count_come_from_the_environment(self):
        os.environ["CPERSONA_SEARCH_MODE"] = "pro"
        os.environ["CPERSONA_SEARCH_COUNT"] = "12"
        client = self.client()
        client.search("Denver", "u1", 20)
        self.assertEqual(client.call.calls[-1],
                         ("reconstruct", {"agent_id": "u1", "query": "Denver", "mode": "pro", "count": 12}))

    def test_users_are_isolated_and_delete_removes_only_one(self):
        client = self.client()
        client.add(session("my dog is Rex"), "a")
        client.add(session("my dog is Max"), "b")
        self.assertEqual(client.search("Max", "a", 20), [])
        client.delete("a")
        self.assertEqual(client.call.calls[-1], ("delete_agent_data", {"agent_id": "a"}))
        self.assertEqual(client.search("dog", "a", 20), [])
        self.assertEqual(len(client.search("Max", "b", 20)), 1)

    def test_rejected_and_truncated_writes_raise(self):
        client = self.client()
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            client.add(session("REJECT"), "u1")
        with self.assertRaisesRegex(RuntimeError, "CPERSONA_MAX_CONTENT_LENGTH"):
            client.add(session("TRUNCATE"), "u1")

    def test_transport_failure_propagates(self):
        client = self.client()
        with self.assertRaisesRegex(RuntimeError, "connection refused"):
            client.add(session("FAIL"), "u1")

    def test_a_tool_error_raises_and_a_result_parses(self):
        def result(text, error):
            return types.SimpleNamespace(content=[types.SimpleNamespace(text=text)], isError=error)
        with self.assertRaisesRegex(RuntimeError, "cpersona reconstruct failed: agent_id is required"):
            CpersonaClient.parse_result("reconstruct", result("agent_id is required", True))
        self.assertEqual(CpersonaClient.parse_result("store", result('{"result": "stored"}', False)),
                         {"result": "stored"})


class TestCpersonaLocomoSharedStore(unittest.TestCase):
    """LoCoMo: each session stored once in the conversation's speaker-A store, searched once."""

    META = {
        "session_date": "1:56 pm on 8 May, 2023", "conv_idx": 0, "session_key": "session_1",
        "speaker_a": "Caroline", "speaker_b": "Melanie",
        "speaker_a_user_id": "u_a", "speaker_b_user_id": "u_b",
    }
    SESSION = [{"speaker": "Caroline", "text": "Hi Mel!"}, {"speaker": "Melanie", "text": "Hey Caroline."}]

    def test_session_is_added_once_to_speaker_a_under_a_conversation_unique_key(self):
        from unittest.mock import Mock

        from scripts.locomo.locomo_ingestion import ingest_session

        client = Mock()
        ingest_session(client, self.SESSION, "cpersona", dict(self.META))
        client.add.assert_called_once()
        msgs, user_id = client.add.call_args.args
        self.assertEqual(user_id, "u_a")
        self.assertEqual(client.add.call_args.kwargs, {"session_key": "u_a:session_1"})
        self.assertEqual([m["content"] for m in msgs], ["Caroline: Hi Mel!", "Melanie: Hey Caroline."])
        self.assertTrue(all(m["chat_time"].startswith("2023-05-08T13:56") for m in msgs))

    def test_search_reads_the_one_store(self):
        from unittest.mock import Mock

        from scripts.locomo.locomo_search import search_query

        client = Mock()
        client.search.return_value = ["[2023-05-08]\nCaroline: Hi Mel!\n"]
        search_query(client, "who said hi?", dict(self.META), "cpersona", top_k=20)
        client.search.assert_called_once_with("who said hi?", "u_a", 20)


if __name__ == "__main__":
    unittest.main()

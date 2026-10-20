"""Offline LCM-X adapter contract tests; no external checkout or model needed."""

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dotenv import dotenv_values

from scripts.client_factory import lcmx_client, registry


class TestLcmxClient(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        template = Path(__file__).resolve().parents[3] / "env_examples/.env.lcmx"
        self.settings = dict(dotenv_values(template, interpolate=False))
        self.settings.update(LCMX_REPO=str(self.root), LCMX_DB_ROOT=str(self.root / "stores"))
        env = patch.dict(os.environ, self.settings, clear=True)
        env.start()
        self.addCleanup(env.stop)
        paths = patch.object(sys, "path", list(sys.path))
        paths.start()
        self.addCleanup(paths.stop)
        provider = SimpleNamespace(
            model_id="BAAI/bge-small-en-v1.5", dim=2,
            embed_documents=Mock(side_effect=lambda texts: [[1.0, 0.0] for _ in texts]),
        )
        self.resolve = Mock(return_value=provider)
        self.chunker = Mock(return_value=[SimpleNamespace(
            text="remember tea", chunk_id="c1", store_id=1, chunk_index=0,
            char_start=0, char_end=12, token_estimate=3,
        )])
        self.recall = Mock(side_effect=self._recall)
        modules = {}
        for name in ("benchmarking", "benchmarking.longmemeval", "hermes_lcm",
                     "hermes_lcm.chunking", "hermes_lcm.dag", "hermes_lcm.vector_store",
                     "hermes_lcm.tools"):
            modules[name] = types.ModuleType(name)
        modules["benchmarking.longmemeval"].__dict__.update(
            _ensure_hermes_lcm_package=Mock(),
            deterministic_session_summary=lambda rows: "session summary",
            resolve_harness_provider=self.resolve,
        )
        modules["hermes_lcm.chunking"].iter_message_chunks = self.chunker
        modules["hermes_lcm.dag"].SummaryNode = SimpleNamespace
        modules["hermes_lcm.vector_store"].EmbeddingIdentity = Mock()
        modules["hermes_lcm.tools"].lcm_recall = self.recall
        modules["hermes_lcm"].tools = modules["hermes_lcm.tools"]
        fake_modules = patch.dict(sys.modules, modules)
        fake_modules.start()
        self.addCleanup(fake_modules.stop)
        for name, value in (("_EMBEDDER", None), ("_USER_LOCKS", {}), ("_WARNED", set())):
            state = patch.object(lcmx_client, name, value)
            state.start()
            self.addCleanup(state.stop)
        self.client = lcmx_client.LcmxClient()
        self.backends = {}
        opened = patch.object(self.client, "_open", side_effect=self._open)
        opened.start()
        self.addCleanup(opened.stop)

    def _open(self, user_id):
        directory = self.client._user_dir(user_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "lcm.db").touch()
        if user_id not in self.backends:
            rows = {}
            store = Mock()

            def append(session_id, incoming, **kwargs):
                ids = []
                for row in incoming:
                    sid = len(rows) + 1
                    rows[sid] = dict(row, session_id=session_id)
                    ids.append(sid)
                return ids

            store.append_batch.side_effect = append
            store.get_batch.side_effect = lambda ids: {sid: rows[sid] for sid in ids}
            self.backends[user_id] = (SimpleNamespace(), store, Mock(), Mock(), rows)
        return self.backends[user_id][:4]

    @staticmethod
    def _recall(args, engine):
        rows = engine._store.get_batch([1])
        return json.dumps({"hits": [{
            "store_id": sid, "kind": "message", "snippet": ">>>short<<<",
            "event_time": row.get("timestamp"), "session_id": row["session_id"],
        } for sid, row in rows.items()]})

    def test_registration(self):
        self.assertIn("lcmx", registry.SUPPORTED_LIBS)
        self.assertIs(registry._load_client_class("lcmx"), lcmx_client.LcmxClient)
        self.assertIsInstance(registry.create_client("lcmx"), lcmx_client.LcmxClient)

    def test_template_configuration_matches_run(self):
        expected = {
            "LCMX_CONFIG": "fleet", "LCMX_CHUNK_POLICY": "full", "LCMX_DELIVERY": "full",
            "LCMX_EMBED_PROVIDER": "fastembed", "LCMX_EMBED_MODEL": "BAAI/bge-small-en-v1.5",
            "HF_HUB_OFFLINE": "1", "TOPK": "20", "LLM_WORKERS": "10",
            "ANSWER_MODEL": "gpt-4.1-mini-2025-04-14", "EVAL_MODEL": "gpt-4o-mini-2024-07-18",
        }
        self.assertEqual(self.client._chunk_policy, "full")
        self.assertEqual(self.client._delivery, "full")
        self.assertEqual({key: self.settings.get(key) for key in expected}, expected)
        self.assertTrue(self.client._embeddings)
        self.resolve.assert_called_once_with("fastembed", "BAAI/bge-small-en-v1.5")

    def test_defaults_remain_conversational_and_full(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.update(LCMX_REPO=str(self.root), LCMX_DB_ROOT=str(self.root / "defaults"))
            client = lcmx_client.LcmxClient()
        self.assertEqual((client._chunk_policy, client._delivery), ("conversational", "full"))

    def _reject_invalid(self, action):
        for user_id in ("a/b", "a:b", "../x", ".", "", "a" * 201, "a\n", None, 7):
            with self.subTest(user_id=user_id):
                with patch.object(lcmx_client, "_user_lock") as lock:
                    with (
                        patch.object(Path, "exists", side_effect=AssertionError("filesystem access")),
                        self.assertRaisesRegex(ValueError, "user_id must fullmatch"),
                    ):
                        action(user_id)
                    lock.assert_not_called()
                self.assertEqual(list(self.client._db_root.iterdir()), [])

    def test_invalid_add(self):
        self._reject_invalid(lambda uid: self.client.add([{"content": "tea"}], uid))

    def test_invalid_empty_add(self):
        self._reject_invalid(lambda uid: self.client.add([], uid))

    def test_invalid_search(self):
        self._reject_invalid(lambda uid: self.client.search("tea", uid, 20))

    def test_invalid_delete(self):
        self._reject_invalid(self.client.delete)

    def test_invalid_delete_all(self):
        self._reject_invalid(self.client.delete_all)

    def test_distinct_accepted_ids_and_delete_isolation(self):
        ids = ("a-b", "a_b", "a.b", "A", "a" * 200)
        for uid in ids:
            self.client.add([{"role": "user", "content": "remember tea"}], uid)
            key = self.client._user_dir(uid).name
            self.assertRegex(key, r"^[a-z0-9][a-z0-9_.-]{0,23}-[0-9a-f]{32}$")
            self.assertEqual(self.client._user_dir(uid), self.client._db_root / key)
            self.assertIs(lcmx_client._user_lock(uid), lcmx_client._USER_LOCKS[key])
        self.assertEqual(len({self.client._user_dir(uid) for uid in ids}), len(ids))
        self.assertEqual(len({lcmx_client._user_lock(uid) for uid in ids}), len(ids))
        self.client.delete(ids[0])
        self.assertFalse(self.client._user_dir(ids[0]).exists())
        for uid in ids[1:]:
            self.assertTrue(self.client._db_path(uid).exists())
            self.assertEqual(self.client.search("tea", uid, 20), "remember tea")

    def test_case_variant_directory_keys_are_distinct_and_stable(self):
        upper = self.client._user_dir("CaseUser")
        lower = self.client._user_dir("caseuser")
        # This also catches case-insensitive filesystem collisions on Linux CI.
        self.assertNotEqual(upper.name.casefold(), lower.name.casefold())
        self.assertEqual(upper, self.client._user_dir("CaseUser"))
        self.assertEqual(lower, self.client._user_dir("caseuser"))

    def test_case_variant_marker_and_delete_isolation(self):
        upper = self.client._user_dir("CaseUser")
        lower = self.client._user_dir("caseuser")
        upper.mkdir()
        (upper / "marker").write_text("upper", encoding="utf-8")
        self.assertFalse((lower / "marker").exists())
        lower.mkdir()
        (lower / "marker").write_text("lower", encoding="utf-8")
        self.client.delete("caseuser")
        self.assertFalse(lower.exists())
        self.assertEqual((upper / "marker").read_text(encoding="utf-8"), "upper")

    def test_lock_uses_storage_key(self):
        for uid in ("CaseUser", "caseuser"):
            key = self.client._user_dir(uid).name
            lock = lcmx_client._user_lock(uid)
            self.assertIs(lock, lcmx_client._USER_LOCKS[key])
            self.assertIs(lock, lcmx_client._user_lock(uid))
            self.assertNotIn(uid, lcmx_client._USER_LOCKS)
        self.assertIsNot(lcmx_client._user_lock("CaseUser"), lcmx_client._user_lock("caseuser"))

    def test_add_search_delete_round_trip(self):
        self.assertEqual(self.client.search("tea", "user-1", 20), "")
        self.client.add([{"role": "user", "content": "remember tea", "chat_time": "2026-10-18"}],
                        "user-1", session_key="s1")
        self.chunker.assert_called_once()
        self.assertEqual(self.chunker.call_args.kwargs["policy"], "full")
        self.assertEqual(self.client.search("tea", "user-1", 20), "[2026-10-18] remember tea")
        self.assertEqual(self.recall.call_args.args[0], {"query": "tea", "limit": 20})
        for resource in self.backends["user-1"][1:4]:
            self.assertEqual(resource.close.call_count, 2)
        self.client.delete_all("user-1")
        self.assertEqual(self.client.search("tea", "user-1", 20), "")

    def test_add_backend_exception_propagates_and_closes(self):
        _, store, dag, vectors = self._open("user-1")
        store.append_batch.side_effect = RuntimeError("append failed")
        with self.assertRaisesRegex(RuntimeError, "append failed"):
            self.client.add([{"content": "tea"}], "user-1")
        for resource in (store, dag, vectors):
            resource.close.assert_called_once()

    def test_search_backend_exception_propagates_and_closes(self):
        self.client.add([{"content": "tea"}], "user-1")
        self.recall.side_effect = RuntimeError("recall failed")
        with self.assertRaisesRegex(RuntimeError, "recall failed"):
            self.client.search("tea", "user-1", 20)
        for resource in self.backends["user-1"][1:4]:
            self.assertEqual(resource.close.call_count, 2)

    def test_search_error_payload_propagates(self):
        self.client.add([{"content": "tea"}], "user-1")
        self.recall.side_effect = None
        self.recall.return_value = json.dumps({"error": "backend rejected query"})
        with self.assertRaisesRegex(RuntimeError, "lcm_recall error: backend rejected query"):
            self.client.search("tea", "user-1", 20)


if __name__ == "__main__":
    unittest.main()

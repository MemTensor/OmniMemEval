"""Synthetic BEAM retry regressions; no model, network, or external store."""

import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from scripts.beam.beam_ingestion import ingest_session
from scripts.tests.unit.test_lcmx_client import TestLcmxClient as _Fixture
from scripts.utils.ingest_helpers import session_id_kwargs


class TestBeamRetry(unittest.TestCase):
    setUp = _Fixture.setUp
    _recall = staticmethod(_Fixture._recall)

    def _open(self, user_id):
        directory = self.client._user_dir(user_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "lcm.db").touch()
        if user_id not in self.backends:
            rows, nodes, chunks, summaries = {}, [], {}, {}
            store, dag, vectors = Mock(), Mock(), Mock()

            def append(session_id, incoming, **kwargs):
                ids = []
                for row in incoming:
                    sid = len(rows) + 1
                    rows[sid] = dict(row, session_id=session_id, store_id=sid)
                    ids.append(sid)
                return ids

            def add_node(node):
                node.node_id = len(nodes) + 1
                nodes.append(node)
                return node.node_id

            store.append_batch.side_effect = append
            store.get_batch.side_effect = lambda ids: {sid: rows[sid] for sid in ids}
            store.get_session_messages.side_effect = lambda session_id, **kw: [
                row for row in rows.values() if row["session_id"] == session_id
            ]
            dag.add_node.side_effect = add_node
            dag.get_session_nodes.side_effect = lambda session_id, **kw: [
                node for node in nodes if node.session_id == session_id and node.depth == 0
            ][:kw.get("limit", 1000)]
            vectors.record_chunk_embedding.side_effect = lambda cid, model, vec, **kw: chunks.update({cid: vec})
            vectors.record_embedding.side_effect = lambda nid, kind, model, vec, **kw: summaries.update({nid: vec})
            self.backends[user_id] = (SimpleNamespace(), store, dag, vectors, rows, nodes, chunks, summaries)
        return self.backends[user_id][:4]

    @staticmethod
    def messages(index, count=3):
        return [{"role": "user" if i % 2 else "assistant", "content": f"synthetic {index} message {i}"}
                for i in range(count)]

    @staticmethod
    def digest(rows):
        return hashlib.sha256(json.dumps([(r["role"], r["content"]) for r in rows]).encode()).hexdigest()

    def configure_chunks(self):
        self.chunker.side_effect = lambda rows, **kw: [SimpleNamespace(
            text=row["content"], chunk_id=f"chunk-{row['store_id']}", store_id=row["store_id"],
            chunk_index=0, char_start=0, char_end=len(row["content"]), token_estimate=4,
        ) for row in rows]

    def beam_add(self, index, count=3):
        ingest_session(self.client, self.messages(index, count), "lcmx", "synthetic-user",
                       f"synthetic-session-{index}", None)

    def assert_session(self, index, count=3):
        backend = self.backends["synthetic-user"]
        session_id = f"synthetic-session-{index}"
        rows = [r for r in backend[4].values() if r["session_id"] == session_id]
        self.assertEqual(self.digest(rows), self.digest(self.messages(index, count)))
        self.assertEqual(sum(n.session_id == session_id for n in backend[5]), 1)

    def test_a_bounded_batches(self):
        self.configure_chunks()
        self.beam_add(1, 600)
        calls = self.client._embedder._provider.embed_documents.call_args_list
        self.assertEqual([len(call.args[0]) for call in calls], [256, 256, 88, 1])
        self.assertEqual(len(self.backends["synthetic-user"][6]), 600)

    def test_b_failed_second_then_third_fourth_and_retry(self):
        self.configure_chunks()
        provider = self.client._embedder._provider
        self.beam_add(1)
        provider.embed_documents.side_effect = RuntimeError("synthetic embed failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic embed failure"):
            self.beam_add(2)
        provider.embed_documents.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
        self.beam_add(3)
        self.beam_add(4)
        self.beam_add(2)
        for index in range(1, 5):
            self.assert_session(index)
        backend = self.backends["synthetic-user"]
        self.assertEqual((len(backend[4]), len(backend[5]), len(backend[6]), len(backend[7])), (12, 4, 12, 4))

    def test_c_retry_is_idempotent_after_partial_embeddings(self):
        self.configure_chunks()
        provider = self.client._embedder._provider
        calls = 0

        def fail_summary(texts):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic summary failure")
            return [[1.0, 0.0] for _ in texts]

        provider.embed_documents.side_effect = fail_summary
        with self.assertRaisesRegex(RuntimeError, "synthetic summary failure"):
            self.beam_add(1)
        provider.embed_documents.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
        self.beam_add(1)
        self.beam_add(1)
        self.assert_session(1)
        backend = self.backends["synthetic-user"]
        self.assertEqual((len(backend[4]), len(backend[5]), len(backend[6]), len(backend[7])), (3, 1, 3, 1))

    def test_d_partial_different_or_duplicate_node_fails_without_writes(self):
        self.configure_chunks()
        self.beam_add(1)
        backend = self.backends["synthetic-user"]
        original = dict(backend[4])
        mutations = ("partial", "different", "duplicate_node")
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                backend[4].clear()
                backend[4].update({k: dict(v) for k, v in original.items()})
                if mutation == "partial":
                    backend[4].pop(3)
                elif mutation == "different":
                    backend[4][1]["content"] = "synthetic mismatch"
                else:
                    backend[5].append(backend[5][0])
                before = (self.digest(backend[4].values()), len(backend[5]), dict(backend[6]), dict(backend[7]))
                for resource in backend[1:4]:
                    resource.reset_mock()
                with self.assertRaisesRegex(ValueError, "session.*mismatch"):
                    self.beam_add(1)
                after = (self.digest(backend[4].values()), len(backend[5]), dict(backend[6]), dict(backend[7]))
                self.assertEqual(before, after)
                backend[1].append_batch.assert_not_called()
                backend[2].add_node.assert_not_called()
                backend[3].record_chunk_embedding.assert_not_called()
                backend[3].record_embedding.assert_not_called()

    def test_e_non_beam_no_id_characterization(self):
        self.configure_chunks()
        for suite in ("locomo", "longmemeval", "halumem", "personamem_v2", "smoke"):
            with self.subTest(suite=suite):
                uid = f"synthetic-{suite}"
                kwargs = {} if suite == "locomo" else session_id_kwargs("lcmx", "dataset-id")
                self.assertEqual(kwargs, {})
                for i in range(1, 4):
                    self.client.add(self.messages(i), uid, **kwargs)
                # Characterize the historical count/suffix collision path too.
                sessions = self.client._load_sessions(uid)
                sessions.pop(f"{uid}-s2")
                self.client._sessions_path(uid).write_text(json.dumps(sessions))
                self.client.add(self.messages(4), uid, **kwargs)
                backend = self.backends[uid]
                expected = [f"{uid}-s1"] * 3 + [f"{uid}-s2"] * 3 + [f"{uid}-s3"] * 3 + [f"{uid}-s3#3"] * 3
                self.assertEqual([r["session_id"] for r in backend[4].values()], expected)
                self.assertEqual(self.digest(backend[4].values()), self.digest(sum([self.messages(i) for i in range(1, 5)], [])))


if __name__ == "__main__":
    unittest.main()

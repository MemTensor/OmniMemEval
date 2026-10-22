"""Opt-in acceptance of offline real-model BEAM storage proof artifacts.

Set LCMX_BEAM_DIFFERENTIAL_EVIDENCE to a directory containing old-storage.json
and candidate-storage.json from the metadata-only offline differential runner.
This test reads those real-model outputs; it never ingests, downloads, contacts
a model service, or prints corpus text. Ordinary test runs skip it.
"""
import hashlib
import json
import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.environ.get("LCMX_BEAM_DIFFERENTIAL_EVIDENCE"),
                     "requires opt-in offline real-model differential artifacts")
class TestLcmxBeamStorageDifferential(unittest.TestCase):
    def test_three_clean_sessions_match_old_storage(self):
        root = Path(os.environ["LCMX_BEAM_DIFFERENTIAL_EVIDENCE"])
        old = json.loads((root / "old-storage.json").read_text())
        new = json.loads((root / "candidate-storage.json").read_text())
        adapter = Path(__file__).resolve().parents[2] / "client_factory/lcmx_client.py"
        self.assertEqual(new["imported_adapter_sha256"],
                         hashlib.sha256(adapter.read_bytes()).hexdigest())
        self.assertEqual(old["adapter_sha"], "f7476c5502115542eca9971195bf8c5ca97b6256")
        self.assertEqual(old["selected_conversation_id"], new["selected_conversation_id"])
        self.assertEqual(old["user_message_counts"], new["user_message_counts"])
        self.assertEqual(old["externalization_clock_control"], new["externalization_clock_control"])
        for proof in (old, new):
            self.assertEqual(proof["status"], "COMPLETE")
            self.assertEqual(proof["network_connect"], "denied")
            self.assertTrue(proof["local_files_only"])
            self.assertEqual(len(proof["sessions"]), 3)
        fields = (
            "row_count", "ordered_role_content_sha256", "ordered_role_content_sequence",
            "depth0_node_count", "nodes", "incoming_chunks", "stored_chunks",
            "stored_recomputed_chunks", "stored_recomputed_vectorless_chunks",
            "incoming_chunks_with_one_vector", "chunk_vector_rows_cumulative",
            "summary_vector_rows_cumulative",
        )
        for index, (before, after) in enumerate(zip(old["sessions"], new["sessions"])):
            with self.subTest(session_index=index):
                for field in fields:
                    # A short assertion message prevents a failure dumping the
                    # full per-row metadata sequences into ordinary test logs.
                    self.assertTrue(before[field] == after[field], field)
                self.assertEqual(after["session_index"], index)
                self.assertTrue(after["session_id"].endswith(f"_session_{index}"))
                self.assertTrue(before["session_id"].endswith(f"-s{index + 1}"))
                self.assertEqual(after["depth0_node_count"], 1)
                self.assertGreater(len(after["incoming_chunks"]), 0)
                self.assertEqual(after["incoming_chunks_with_one_vector"], len(after["incoming_chunks"]))
                self.assertTrue(all(call["texts"] <= 256 for call in after["calls"]))


if __name__ == "__main__":
    unittest.main()

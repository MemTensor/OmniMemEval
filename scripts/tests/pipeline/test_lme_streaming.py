import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from longmemeval.lme_streaming import (
    add_conversation,
    add_conversation_batched_graphiti,
    build_batched_cognee_messages,
    build_batched_graphiti_content,
    build_graphiti_content_chunks,
    graphiti_lme_chunk_id,
    load_added_chunks,
    per_session_checkpoint_id,
)
from utils.streaming import delete_user_data, is_timeout_error


class _FakeILoc:
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, idx):
        return self._rows[idx]


class _FakeDataFrame:
    def __init__(self, rows):
        self._rows = rows
        self.iloc = _FakeILoc(rows)


class TestLmeStreaming(unittest.TestCase):
    def test_build_batched_cognee_messages_preserves_session_boundaries(self):
        sessions = [
            [
                {"role": "user", "content": "I started a chemistry degree."},
                {"role": "assistant", "content": "Noted."},
            ],
            [{"role": "user", "content": "I now commute by train."}],
        ]
        dates = ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 03:24"]

        messages = build_batched_cognee_messages(sessions, dates)

        self.assertEqual(len(messages), 5)
        self.assertIn("SESSION 0 START", messages[0]["content"])
        self.assertIn("SESSION 1 START", messages[3]["content"])
        self.assertIn("SESSION 0 TURN 0 user:", messages[1]["content"])
        self.assertIn("chemistry degree", messages[1]["content"])
        self.assertIn("SESSION 1 TURN 0 user:", messages[4]["content"])
        self.assertEqual(messages[1]["chat_time"], "2023-05-20T02:21:00+00:00")
        self.assertEqual(messages[4]["chat_time"], "2023-05-21T03:24:00+00:00")

    def test_batched_builders_sanitize_special_tokens_and_skip_empty_turns(self):
        sessions = [
            [
                {"role": "user", "content": "alpha<|endoftext|>"},
                {"role": "assistant", "content": "   "},
                {"role": "user", "content": None},
            ]
        ]
        dates = ["2023/05/20 (Sat) 02:21"]

        messages = build_batched_cognee_messages(sessions, dates)
        content = build_batched_graphiti_content(sessions, dates)

        self.assertEqual(len(messages), 2)
        self.assertIn("alpha", messages[1]["content"])
        self.assertNotIn("<|endoftext|>", messages[1]["content"])
        self.assertIn("SESSION 0 TURN 0 user: alpha", content)
        self.assertNotIn("<|endoftext|>", content)
        self.assertNotIn("TURN 1", content)
        self.assertNotIn("TURN 2", content)

    def test_build_batched_graphiti_content_preserves_session_boundaries(self):
        sessions = [
            [
                {"role": "user", "content": "I started a chemistry degree."},
                {"role": "assistant", "content": "Noted."},
            ],
            [{"role": "user", "content": "I now commute by train."}],
        ]
        dates = ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 03:24"]

        content = build_batched_graphiti_content(sessions, dates)

        self.assertIn("SESSION 0 START", content)
        self.assertIn("SESSION 1 START", content)
        self.assertIn("SESSION 0 TIMESTAMP: 2023-05-20T02:21:00+00:00 UTC", content)
        self.assertIn("SESSION 1 TIMESTAMP: 2023-05-21T03:24:00+00:00 UTC", content)
        self.assertIn("SESSION 0 TURN 0 user: I started a chemistry degree.", content)
        self.assertIn("SESSION 1 TURN 0 user: I now commute by train.", content)

    def test_build_graphiti_content_chunks_preserves_global_session_indices(self):
        sessions = [
            [{"role": "user", "content": "alpha " * 80}],
            [{"role": "user", "content": "beta " * 80}],
            [{"role": "user", "content": "gamma " * 80}],
        ]
        dates = [
            "2023/05/20 (Sat) 02:21",
            "2023/05/21 (Sun) 03:24",
            "2023/05/22 (Mon) 04:25",
        ]

        chunks = build_graphiti_content_chunks(sessions, dates, max_chars=800)

        self.assertGreater(len(chunks), 1)
        combined = "\n".join(chunk.content for chunk in chunks)
        self.assertIn("SESSION 0 START", combined)
        self.assertIn("SESSION 1 START", combined)
        self.assertIn("SESSION 2 START", combined)
        self.assertIn("SESSION 2 TURN 0 user:", combined)
        self.assertEqual(chunks[0].start_session_idx, 0)
        self.assertEqual(chunks[-1].end_session_idx, 2)

    def test_add_conversation_batched_graphiti_chunks_raw_episodes(self):
        class FakeGraphitiClient:
            def __init__(self):
                self.calls = []

            def add(self, messages, user_id, **kwargs):
                self.calls.append((messages, user_id, kwargs))

        client = FakeGraphitiClient()
        df = _FakeDataFrame(
            [
                {
                    "haystack_sessions": [
                        [{"role": "user", "content": "I started a chemistry degree."}],
                        [{"role": "assistant", "content": "Noted."}],
                    ],
                    "haystack_dates": [
                        "2023/05/20 (Sat) 02:21",
                        "2023/05/21 (Sun) 03:24",
                    ],
                }
            ]
        )

        durations = add_conversation_batched_graphiti(
            df,
            0,
            "v1",
            client,
            max_chars=350,
        )

        self.assertGreater(len(client.calls), 1)
        messages, user_id, kwargs = client.calls[0]
        self.assertEqual(messages, [])
        self.assertEqual(user_id, "lme_exper_user_v1_0")
        self.assertIn("_lme_exper_batch_000_sessions_0-0", kwargs["session_key"])
        self.assertIn("SESSION 0 START", kwargs["raw_content"])
        self.assertEqual(kwargs["timestamp"], "2023-05-20T02:21:00+00:00")
        combined = "\n".join(call[2]["raw_content"] for call in client.calls)
        self.assertIn("SESSION 1 START", combined)
        self.assertIn("longmemeval_conversation_chunk", kwargs["role"])
        self.assertEqual(len(durations), len(client.calls))

    def test_add_conversation_batched_graphiti_skips_recorded_chunks(self):
        class FakeGraphitiClient:
            def __init__(self):
                self.calls = []

            def add(self, messages, user_id, **kwargs):
                self.calls.append((messages, user_id, kwargs))

        df = _FakeDataFrame(
            [
                {
                    "haystack_sessions": [
                        [{"role": "user", "content": "alpha " * 40}],
                        [{"role": "user", "content": "beta " * 40}],
                    ],
                    "haystack_dates": [
                        "2023/05/20 (Sat) 02:21",
                        "2023/05/21 (Sun) 03:24",
                    ],
                }
            ]
        )
        chunks = build_graphiti_content_chunks(
            df.iloc[0]["haystack_sessions"],
            df.iloc[0]["haystack_dates"],
            max_chars=350,
        )
        first_chunk_id = graphiti_lme_chunk_id(0, chunks[0])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "added.txt"
            path.write_text(first_chunk_id + "\n")
            added = load_added_chunks(path)
            client = FakeGraphitiClient()

            add_conversation_batched_graphiti(
                df,
                0,
                "v1",
                client,
                max_chars=350,
                added_chunks_path=path,
                added_chunks=added,
            )

            self.assertEqual(len(client.calls), len(chunks) - 1)
            self.assertEqual(len(load_added_chunks(path)), len(chunks))

    def test_add_conversation_per_session_skips_recorded_sessions(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            def add(self, messages, user_id, **kwargs):
                self.calls.append((messages, user_id, kwargs))

        df = _FakeDataFrame(
            [
                {
                    "haystack_sessions": [
                        [{"role": "user", "content": "alpha"}],
                        [{"role": "user", "content": "beta"}],
                    ],
                    "haystack_dates": [
                        "2023/05/20 (Sat) 02:21",
                        "2023/05/21 (Sun) 03:24",
                    ],
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "added.txt"
            path.write_text(per_session_checkpoint_id(0, 0) + "\n")
            added = load_added_chunks(path)
            client = FakeClient()

            add_conversation(
                df,
                0,
                "memos",
                "v1",
                client,
                "per-session",
                added_parts_path=path,
                added_parts=added,
            )

            self.assertEqual(len(client.calls), 1)
            self.assertEqual(client.calls[0][0][0]["content"], "beta")
            self.assertEqual(
                load_added_chunks(path),
                {per_session_checkpoint_id(0, 0), per_session_checkpoint_id(0, 1)},
            )

    def test_delete_timeout_can_be_skipped(self):
        class TimeoutDeleteClient:
            def delete_user(self, user_id):
                raise RuntimeError("Read timed out. (read timeout=600.0)")

        self.assertTrue(
            is_timeout_error(
                RuntimeError("HTTPConnectionPool Read timed out.")
            )
        )
        ok, error = delete_user_data(
            "cognee",
            TimeoutDeleteClient(),
            "lme_user_1",
            skip_timeout=True,
        )
        self.assertFalse(ok)
        self.assertIn("timed out", error)

    def test_delete_error_can_be_skipped_after_search(self):
        class ServerErrorDeleteClient:
            def delete_user(self, user_id):
                raise RuntimeError("500 Server Error: Internal Server Error")

        ok, error = delete_user_data(
            "cognee",
            ServerErrorDeleteClient(),
            "lme_user_1",
            skip_errors=True,
        )
        self.assertFalse(ok)
        self.assertIn("500 Server Error", error)

    def test_delete_non_timeout_still_fails_when_skip_enabled(self):
        class BrokenDeleteClient:
            def delete_user(self, user_id):
                raise RuntimeError("Cognee dataset still exists after delete")

        with self.assertRaises(RuntimeError):
            delete_user_data(
                "cognee",
                BrokenDeleteClient(),
                "lme_user_1",
                skip_timeout=True,
            )


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the ThreadAtlas client: configuration, registration, add/search/delete, user isolation
and failures.

The ThreadAtlas package is replaced by a small stand-in with the same three calls, so these tests run
offline and need no store.
"""

import os
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from client_factory.registry import SUPPORTED_LIBS, create_client

STAND_IN = textwrap.dedent('''
    MEMORY = {}

    def add(turns, user_id, minute):
        if turns[0]["content"] == "FAIL":
            raise RuntimeError("reading failed")
        MEMORY.setdefault(user_id, []).append((minute, turns))

    def search(query, user_id, top_k):
        if query == "FAIL":
            raise RuntimeError("search failed")
        found = ["HEADER"]
        for minute, turns in MEMORY.get(user_id, []):
            for turn in turns:
                if query in turn["content"]:
                    found.append(minute + " | " + turn["content"])
        return found[:top_k]

    def delete(user_id):
        MEMORY.pop(user_id, None)
''')


class TestThreadAtlasClient(unittest.TestCase):
    def setUp(self):
        self.repo = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.repo, "threadatlas_memory"))
        with open(os.path.join(self.repo, "threadatlas_memory", "__init__.py"), "w", encoding="utf-8") as f:
            f.write(STAND_IN)
        sys.modules.pop("threadatlas_memory", None)
        os.environ["THREADATLAS_REPO"] = self.repo
        os.environ["THREADATLAS_STATE_DIR"] = tempfile.mkdtemp()

    def tearDown(self):
        sys.modules.pop("threadatlas_memory", None)
        if self.repo in sys.path:
            sys.path.remove(self.repo)

    def test_registered(self):
        self.assertIn("threadatlas", SUPPORTED_LIBS)

    def test_missing_configuration_fails_clearly(self):
        del os.environ["THREADATLAS_REPO"]
        with self.assertRaises(ValueError):
            create_client("threadatlas")

    def test_add_passes_the_session_minute(self):
        client = create_client("threadatlas")
        client.add([{"role": "user", "content": "I moved to Denver", "chat_time": "2023-05-20T02:38:00+00:00"},
                    {"role": "assistant", "content": "Welcome to Denver"}], "u1")
        packet = client.search("Denver", "u1", 20)
        self.assertEqual(packet[0], "HEADER")
        self.assertEqual(packet[1], "2023-05-20T02:38 | I moved to Denver")

    def test_empty_session_is_ignored(self):
        client = create_client("threadatlas")
        client.add([], "u1")
        self.assertEqual(client.search("anything", "u1", 20), ["HEADER"])

    def test_top_k_bounds_the_packet(self):
        client = create_client("threadatlas")
        turns = []
        for i in range(30):
            turns.append({"role": "user", "content": "note " + str(i), "chat_time": "2023-05-20T02:38:00"})
        client.add(turns, "u1")
        self.assertEqual(len(client.search("note", "u1", 20)), 20)

    def test_users_are_isolated_and_delete_removes_only_one(self):
        client = create_client("threadatlas")
        client.add([{"role": "user", "content": "my dog is Rex", "chat_time": "2023-05-20T02:38:00"}], "a")
        client.add([{"role": "user", "content": "my dog is Max", "chat_time": "2023-05-21T09:00:00"}], "b")
        self.assertEqual(len(client.search("Max", "a", 20)), 1)
        client.delete("a")
        self.assertEqual(client.search("dog", "a", 20), ["HEADER"])
        self.assertEqual(len(client.search("Max", "b", 20)), 2)

    def test_add_failure_propagates_and_stores_nothing(self):
        client = create_client("threadatlas")
        with self.assertRaises(RuntimeError):
            client.add([{"role": "user", "content": "FAIL", "chat_time": "2023-05-20T02:38:00"}], "u1")
        self.assertEqual(client.search("anything", "u1", 20), ["HEADER"])

    def test_search_failure_propagates(self):
        client = create_client("threadatlas")
        client.add([{"role": "user", "content": "my dog is Rex", "chat_time": "2023-05-20T02:38:00"}], "u1")
        with self.assertRaises(RuntimeError):
            client.search("FAIL", "u1", 20)


if __name__ == "__main__":
    unittest.main()

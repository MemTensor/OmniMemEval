import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from client_factory.memos_client import MemosClient


class TestMemosConfig(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.copy()
        self._clear_memos_env()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)

    def _clear_memos_env(self):
        for key in list(os.environ):
            if key.startswith("MEMOS_"):
                os.environ.pop(key)

    def test_cloud_mode_requires_api_key_and_uses_cloud_default(self):
        os.environ["MEMOS_MODE"] = "cloud"
        os.environ["MEMOS_API_KEY"] = "test-key"

        client = MemosClient()

        self.assertEqual(client.base_url, "https://memos.memtensor.cn/api/openmem/v1")
        self.assertEqual(client.headers["Authorization"], "Token test-key")

    def test_cloud_mode_rejects_missing_api_key(self):
        os.environ["MEMOS_MODE"] = "cloud"

        with self.assertRaises(ValueError):
            MemosClient()

    def test_local_mode_does_not_require_api_key(self):
        os.environ["MEMOS_MODE"] = "local"

        client = MemosClient()

        self.assertEqual(client.base_url, "http://localhost:8000")
        self.assertNotIn("Authorization", client.headers)

    def test_base_url_override_applies_after_mode_default(self):
        os.environ["MEMOS_MODE"] = "local"
        os.environ["MEMOS_BASE_URL"] = "http://memos.internal:8000"

        client = MemosClient()

        self.assertEqual(client.base_url, "http://memos.internal:8000")

    def test_local_async_mode_accepts_explicit_modes(self):
        os.environ["MEMOS_ASYNC_MODE"] = "async"
        self.assertEqual(MemosClient._local_async_mode(), "async")

        os.environ["MEMOS_ASYNC_MODE"] = "sync"
        self.assertEqual(MemosClient._local_async_mode(), "sync")

    def test_local_async_mode_rejects_boolean_strings(self):
        os.environ["MEMOS_ASYNC_MODE"] = "true"

        with self.assertRaisesRegex(ValueError, "MEMOS_ASYNC_MODE"):
            MemosClient._local_async_mode()

    def test_invalid_mode_rejected(self):
        os.environ["MEMOS_MODE"] = "self-hosted"

        with self.assertRaises(ValueError):
            MemosClient()


class TestMemosLocalFormatting(unittest.TestCase):
    def test_format_local_search_data_extracts_text_and_preference_memories_only(self):
        data = {
            "text_mem": [
                {"memories": [{"memory": "Alice likes tea."}, {"value": "Bob moved."}]}
            ],
            "pref_mem": [
                {"memories": [{"memory": "Alice prefers quiet rooms."}]}
            ],
            "tool_mem": [
                {"memories": [{"content": "Use calendar for appointments."}]}
            ],
            "skill_mem": [
                {"memories": [{"content": "Use a domain-specific skill."}]}
            ],
            "preference_note": "Preference summary",
        }

        text = MemosClient._format_local_search_data(data)

        self.assertIn("Alice likes tea.", text)
        self.assertIn("Bob moved.", text)
        self.assertIn("Preference: Alice prefers quiet rooms.", text)
        self.assertIn("Preference summary", text)
        self.assertNotIn("Use calendar for appointments.", text)
        self.assertNotIn("Use a domain-specific skill.", text)

    def test_format_local_search_data_accepts_cloud_shape(self):
        data = {
            "memory_detail_list": [{"memory_value": "Cloud-shaped memory."}],
            "preference_detail_list": [],
        }

        self.assertEqual(
            MemosClient._format_local_search_data(data),
            "Cloud-shaped memory.",
        )

    def test_format_local_search_data_includes_local_pref_note(self):
        data = {
            "text_mem": [{"memories": [{"memory": "Alice likes tea."}]}],
            "pref_mem": [],
            "pref_note": "Local preference note",
        }

        text = MemosClient._format_local_search_data(data)

        self.assertIn("Alice likes tea.", text)
        self.assertIn("Local preference note", text)


class TestMemosCloudSearch(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.copy()
        for key in list(os.environ):
            if key.startswith("MEMOS_"):
                os.environ.pop(key)
        os.environ["MEMOS_MODE"] = "cloud"
        os.environ["MEMOS_API_KEY"] = "test-key"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_search_cloud_sends_context_format_and_tool_memory_options(self):
        class Response:
            status_code = 200
            text = '{"code":0}'

            @staticmethod
            def json():
                return {
                    "code": 0,
                    "message": "ok",
                    "data": {
                        "memory_detail_list": [
                            {"memory_value": "Alice likes tea."}
                        ],
                        "tool_memory_detail_list": [
                            {"experience": "Use the calendar tool."}
                        ],
                    },
                }

        os.environ["MEMOS_CONTEXT_FORMAT"] = "memory"
        os.environ["MEMOS_SEARCH_TOOL_MEMORY"] = "true"
        os.environ["MEMOS_TOOL_MEMORY_LIMIT"] = "3"
        client = MemosClient()
        client._retry = lambda fn: fn()
        requests = []

        def fake_post(path, json):
            requests.append((path, json))
            return Response()

        client._post = fake_post
        text = client.search("tea", "user-1", 5)

        self.assertEqual(requests[0][0], "/search/memory")
        payload = requests[0][1]
        self.assertEqual(payload["context_format"], "memory")
        self.assertTrue(payload["include_tool_memory"])
        self.assertEqual(payload["tool_memory_limit_number"], 3)
        self.assertIn("Alice likes tea.", text)
        self.assertIn("Tool Memory: Use the calendar tool.", text)


class TestMemosLocalDelete(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.copy()
        for key in list(os.environ):
            if key.startswith("MEMOS_"):
                os.environ.pop(key)
        os.environ["MEMOS_MODE"] = "local"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_delete_local_accepts_plural_success_message(self):
        class Response:
            status_code = 200
            text = '{"message":"Memories deleted successfully"}'

            @staticmethod
            def json():
                return {
                    "code": 200,
                    "message": "Memories deleted successfully",
                    "data": {"status": "success"},
                }

        client = MemosClient()
        requests = []

        def fake_post(path, json):
            requests.append((path, json))
            return Response()

        client._post = fake_post
        client._delete_local("user-1")

        self.assertEqual(requests[0][0], "/product/delete_memory")
        self.assertEqual(requests[0][1]["writable_cube_ids"], ["user-1"])


class TestMemosLocalAdd(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.copy()
        for key in list(os.environ):
            if key.startswith("MEMOS_"):
                os.environ.pop(key)
        os.environ["MEMOS_MODE"] = "local"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_add_local_sends_fine_mode_for_async(self):
        class Response:
            status_code = 200
            text = '{"message":"Memory added successfully"}'

            @staticmethod
            def json():
                return {
                    "code": 200,
                    "message": "Memory added successfully",
                    "data": [],
                }

        os.environ["MEMOS_ASYNC_MODE"] = "async"
        os.environ["MEMOS_LOCAL_ADD_MODE"] = "fine"
        client = MemosClient()
        client._retry = lambda fn: fn()
        requests = []

        def fake_post(path, json):
            requests.append((path, json))
            return Response()

        client._post = fake_post
        client.add([{"role": "user", "content": "I prefer tea."}], "user-1")

        payload = requests[0][1]
        self.assertEqual(payload["async_mode"], "async")
        self.assertEqual(payload["mode"], "fine")


class TestMemosLocalSearch(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.copy()
        for key in list(os.environ):
            if key.startswith("MEMOS_"):
                os.environ.pop(key)
        os.environ["MEMOS_MODE"] = "local"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_search_local_disables_tool_and_skill_memory(self):
        class Response:
            status_code = 200
            text = '{"message":"Search completed successfully"}'

            @staticmethod
            def json():
                return {
                    "code": 200,
                    "message": "Search completed successfully",
                    "data": {
                        "text_mem": [{"memories": [{"memory": "Alice likes tea."}]}],
                        "tool_mem": [{"memories": [{"memory": "tool memory"}]}],
                        "skill_mem": [{"memories": [{"memory": "skill memory"}]}],
                    },
                }

        client = MemosClient()
        requests = []
        client._retry = lambda fn: fn()

        def fake_post(path, json):
            requests.append((path, json))
            return Response()

        client._post = fake_post
        text = client._search_local("tea", "user-1", 3)

        payload = requests[0][1]
        self.assertFalse(payload["search_tool_memory"])
        self.assertEqual(payload["tool_mem_top_k"], 0)
        self.assertFalse(payload["include_skill_memory"])
        self.assertEqual(payload["skill_mem_top_k"], 0)
        self.assertEqual(text, "Alice likes tea.")


if __name__ == "__main__":
    unittest.main()

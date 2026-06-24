"""Unit tests for the EverOS client routing and formatting helpers."""

import os
import sys
import unittest
from unittest.mock import patch

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from client_factory.everos_client import EverosClient


class TestEverosFormatting(unittest.TestCase):
    def test_format_as_text_includes_local_response_sections(self):
        data = {
            "episodes": [
                {
                    "episode": "Alice booked a Paris trip.",
                    "atomic_facts": [{"atomic_fact": "Alice travels in June."}],
                }
            ],
            "profiles": [
                {
                    "profile_data": {
                        "item_type": "explicit_info",
                        "embed_text": "Alice prefers aisle seats.",
                    }
                }
            ],
            "raw_messages": [
                {"role": "user", "content": "I need a flight to Paris."}
            ],
            "agent_memory": {
                "skills": [{"skill_name": "travel", "description": "Plan trips."}]
            },
        }

        text = EverosClient._format_as_text(data)

        self.assertIn("Alice booked a Paris trip.", text)
        self.assertIn("Alice travels in June.", text)
        self.assertIn("[explicit_info] Alice prefers aisle seats.", text)
        self.assertIn("Raw Messages:", text)
        self.assertIn("user: I need a flight to Paris.", text)
        self.assertIn("[travel] Plan trips.", text)


class TestEverosConfig(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.copy()
        self._clear_everos_env()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)

    def _clear_everos_env(self):
        for key in list(os.environ):
            if key.startswith("EVEROS_"):
                os.environ.pop(key)

    def test_cloud_mode_requires_api_key_and_uses_cloud_default(self):
        os.environ["EVEROS_MODE"] = "cloud"
        os.environ["EVEROS_API_KEY"] = "test-key"

        client = EverosClient()

        self.assertEqual(client.base_url, "https://api.evermind.ai")
        self.assertEqual(client.headers["Authorization"], "Bearer test-key")

    def test_cloud_mode_rejects_missing_api_key(self):
        os.environ["EVEROS_MODE"] = "cloud"

        with self.assertRaises(ValueError):
            EverosClient()

    def test_local_mode_does_not_require_api_key(self):
        os.environ["EVEROS_MODE"] = "local"

        client = EverosClient()

        self.assertEqual(client.base_url, "http://localhost:1995")
        self.assertNotIn("Authorization", client.headers)

    def test_base_url_override_applies_after_mode_default(self):
        os.environ["EVEROS_MODE"] = "local"
        os.environ["EVEROS_BASE_URL"] = "http://everos.internal:1995"

        client = EverosClient()

        self.assertEqual(client.base_url, "http://everos.internal:1995")

    def test_flush_policy_accumulated(self):
        os.environ["EVEROS_MODE"] = "local"
        os.environ["EVEROS_FLUSH_POLICY"] = "accumulated"
        client = EverosClient()

        self.assertFalse(client._should_flush_after_add(["extracted"]))
        self.assertTrue(client._should_flush_after_add(["accumulated"]))
        self.assertTrue(client._should_flush_after_add(["extracted", "accumulated"]))

    def test_invalid_mode_rejected(self):
        os.environ["EVEROS_MODE"] = "self-hosted"

        with self.assertRaises(ValueError):
            EverosClient()


class TestEverosAddSanitization(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.copy()
        for key in list(os.environ):
            if key.startswith("EVEROS_"):
                os.environ.pop(key)
        os.environ["EVEROS_MODE"] = "local"
        os.environ["EVEROS_FLUSH_POLICY"] = "never"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)

    def _response(self):
        class FakeResponse:
            status_code = 200
            text = "{}"
            headers = {}

            def json(self):
                return {"data": {"status": "extracted"}}

        return FakeResponse()

    def test_personal_add_cleans_latex_math_delimiters(self):
        os.environ["EVEROS_USE_GROUP"] = "false"
        client = EverosClient()
        captured = []

        def fake_post(path, json):
            captured.append((path, json))
            return self._response()

        client._post = fake_post

        client.add(
            [
                {
                    "role": "user",
                    "content": r"Solve \( u_t = u_{xx} \) and \[ x=1 \].",
                }
            ],
            "user-1",
            conv_id="session-1",
            flush=False,
        )

        self.assertEqual(captured[0][0], "/api/v1/memories")
        self.assertEqual(
            captured[0][1]["messages"][0]["content"],
            r"Solve ( u_t = u_{xx} ) and [ x=1 ].",
        )

    def test_group_add_cleans_latex_math_delimiters(self):
        os.environ["EVEROS_USE_GROUP"] = "true"
        client = EverosClient()
        captured = []

        def fake_post(path, json):
            captured.append((path, json))
            return self._response()

        client._post = fake_post

        client.add_group(
            [
                {
                    "name": "alice",
                    "content": r"Boundary is \[ a,b \], not \( c,d \).",
                }
            ],
            "group-1",
            flush=False,
        )

        self.assertEqual(captured[0][0], "/api/v1/memories/group")
        self.assertEqual(captured[0][1]["messages"][0]["sender_name"], "alice")
        self.assertEqual(
            captured[0][1]["messages"][0]["content"],
            r"Boundary is [ a,b ], not ( c,d ).",
        )


class TestEverosSearch(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.copy()
        for key in list(os.environ):
            if key.startswith("EVEROS_"):
                os.environ.pop(key)
        os.environ["EVEROS_MODE"] = "local"
        os.environ["EVEROS_FETCH_PROFILE"] = "false"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_search_retries_transient_server_error(self):
        class FakeResponse:
            def __init__(self, status_code, body):
                self.status_code = status_code
                self._body = body
                self.text = str(body)
                self.headers = {}

            def json(self):
                return self._body

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise requests.exceptions.HTTPError(
                        f"{self.status_code} Server Error",
                        response=self,
                    )

        client = EverosClient()
        responses = [
            FakeResponse(500, {"detail": "temporary search failure"}),
            FakeResponse(
                200,
                {
                    "data": {
                        "episodes": [{"episode": "Recovered search result."}],
                        "profiles": [],
                    }
                },
            ),
        ]

        def fake_post(path, json):
            self.assertEqual(path, "/api/v1/memories/search")
            return responses.pop(0)

        client._post = fake_post

        with patch("client_factory.base_client.time.sleep", return_value=None):
            text = client.search("question", "user-1", 20)

        self.assertIn("Recovered search result.", text)
        self.assertEqual(responses, [])


if __name__ == "__main__":
    unittest.main()

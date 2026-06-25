"""Unit tests for MemMachine cloud/local routing."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from client_factory.memmachine_client import MemMachineClient


class TestMemMachineConfig(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.copy()
        self._clear_memmachine_env()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)

    def _clear_memmachine_env(self):
        for key in list(os.environ):
            if key.startswith("MEMMACHINE_"):
                os.environ.pop(key)

    def test_cloud_default_uses_v2_and_producer_filter(self):
        client = MemMachineClient()

        self.assertEqual(client._mode, "cloud")
        self.assertEqual(client.base_url, "https://api.memmachine.ai")
        self.assertEqual(client._prefix, "/v2")
        self.assertEqual(client._producer_filter("u1"), "producer=u1")

    def test_cloud_mode_uses_optional_bearer_header(self):
        os.environ["MEMMACHINE_MODE"] = "cloud"
        os.environ["MEMMACHINE_API_KEY"] = "test-key"

        client = MemMachineClient()

        self.assertEqual(client.headers["Authorization"], "Bearer test-key")

    def test_local_mode_uses_api_v2_and_metadata_filter(self):
        os.environ["MEMMACHINE_MODE"] = "local"
        client = MemMachineClient()

        self.assertEqual(client._mode, "local")
        self.assertEqual(client.base_url, "http://localhost:8080")
        self.assertEqual(client._prefix, "/api/v2")
        self.assertEqual(client._producer_filter("u1"), "metadata.producer=u1")

    def test_base_url_override_applies_after_mode_default(self):
        os.environ["MEMMACHINE_MODE"] = "local"
        os.environ["MEMMACHINE_BASE_URL"] = "http://memmachine.internal:8080"

        client = MemMachineClient()

        self.assertEqual(client.base_url, "http://memmachine.internal:8080")
        self.assertEqual(client._prefix, "/api/v2")

    def test_local_mode_quotes_uuid_filter_value(self):
        os.environ["MEMMACHINE_MODE"] = "local"
        client = MemMachineClient()

        self.assertEqual(
            client._producer_filter(
                "hm_exp_user_omnimemeval_20260516_2f1f897e-d67f-dbc5"
            ),
            "metadata.producer='hm_exp_user_omnimemeval_20260516_2f1f897e-d67f-dbc5'",
        )

    def test_cloud_quotes_uuid_filter_value(self):
        client = MemMachineClient()

        self.assertEqual(
            client._producer_filter("user-with-dashes"),
            "producer='user-with-dashes'",
        )

    def test_invalid_mode_rejected(self):
        os.environ["MEMMACHINE_MODE"] = "self-hosted"

        with self.assertRaises(ValueError):
            MemMachineClient()


if __name__ == "__main__":
    unittest.main()

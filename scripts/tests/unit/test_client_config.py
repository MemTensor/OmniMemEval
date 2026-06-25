"""Configuration validation tests for API-backed clients."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from client_factory.backboard_client import BackboardClient
from client_factory.cognee_client import CogneeClient
from client_factory.everos_client import EverosClient
from client_factory.hindsight_client import HindsightClient
from client_factory.letta_client import LettaClient
from client_factory.mem0_client import Mem0Client
from client_factory.mem9_client import Mem9Client
from client_factory.memori_client import MemoriClient
from client_factory.memorylake_client import MemoryLakeClient
from client_factory.supermemory_client import SupermemoryClient
from client_factory.viking_client import VikingClient
from client_factory.zep_client import ZepClient
from client_factory.base_client import (
    BaseApiClient,
    env_bool,
    env_csv,
    env_float,
    env_int,
    env_json,
    env_max_batch_chars,
    require_env,
)


class TestEnvConfigHelpers(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.copy()
        for key in list(os.environ):
            if key.startswith("OMNIMEMEVAL_TEST_") or key == "MAX_BATCH_CHARS":
                os.environ.pop(key)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_require_env_strips_and_rejects_blank_values(self):
        with self.assertRaisesRegex(ValueError, "OMNIMEMEVAL_TEST_KEY"):
            require_env("OMNIMEMEVAL_TEST_KEY")

        os.environ["OMNIMEMEVAL_TEST_KEY"] = "  value  "
        self.assertEqual(require_env("OMNIMEMEVAL_TEST_KEY"), "value")

        os.environ["OMNIMEMEVAL_TEST_KEY"] = "  "
        with self.assertRaisesRegex(ValueError, "OMNIMEMEVAL_TEST_KEY"):
            require_env("OMNIMEMEVAL_TEST_KEY")

    def test_env_bool_accepts_documented_values_and_rejects_invalid_values(self):
        for value in ("1", "true", "yes", "y", "on"):
            with self.subTest(value=value):
                os.environ["OMNIMEMEVAL_TEST_BOOL"] = value
                self.assertTrue(env_bool("OMNIMEMEVAL_TEST_BOOL"))

        for value in ("0", "false", "no", "n", "off"):
            with self.subTest(value=value):
                os.environ["OMNIMEMEVAL_TEST_BOOL"] = value
                self.assertFalse(env_bool("OMNIMEMEVAL_TEST_BOOL", True))

        os.environ["OMNIMEMEVAL_TEST_BOOL"] = "maybe"
        with self.assertRaisesRegex(ValueError, "OMNIMEMEVAL_TEST_BOOL"):
            env_bool("OMNIMEMEVAL_TEST_BOOL")

    def test_env_numeric_helpers_validate_type_and_bounds(self):
        os.environ["OMNIMEMEVAL_TEST_INT"] = "3"
        os.environ["OMNIMEMEVAL_TEST_FLOAT"] = "0.25"
        self.assertEqual(env_int("OMNIMEMEVAL_TEST_INT", min_value=1), 3)
        self.assertEqual(env_float("OMNIMEMEVAL_TEST_FLOAT", min_value=0), 0.25)

        os.environ["OMNIMEMEVAL_TEST_INT"] = "0"
        with self.assertRaisesRegex(ValueError, "OMNIMEMEVAL_TEST_INT"):
            env_int("OMNIMEMEVAL_TEST_INT", min_value=1)

        os.environ["OMNIMEMEVAL_TEST_FLOAT"] = "bad"
        with self.assertRaisesRegex(ValueError, "OMNIMEMEVAL_TEST_FLOAT"):
            env_float("OMNIMEMEVAL_TEST_FLOAT")

    def test_env_csv_json_and_batch_char_fallback(self):
        os.environ["OMNIMEMEVAL_TEST_CSV"] = " a, ,b ,, c "
        self.assertEqual(env_csv("OMNIMEMEVAL_TEST_CSV"), ["a", "b", "c"])

        os.environ["OMNIMEMEVAL_TEST_JSON"] = '{"a": 1}'
        self.assertEqual(env_json("OMNIMEMEVAL_TEST_JSON"), {"a": 1})

        os.environ["MAX_BATCH_CHARS"] = "100"
        self.assertEqual(env_max_batch_chars("OMNIMEMEVAL_TEST_MAX_CHARS"), 100)

        os.environ["OMNIMEMEVAL_TEST_MAX_CHARS"] = "50"
        self.assertEqual(env_max_batch_chars("OMNIMEMEVAL_TEST_MAX_CHARS"), 50)

        os.environ["OMNIMEMEVAL_TEST_MAX_CHARS"] = "-1"
        with self.assertRaisesRegex(ValueError, "OMNIMEMEVAL_TEST_MAX_CHARS"):
            env_max_batch_chars("OMNIMEMEVAL_TEST_MAX_CHARS")

    def test_base_api_client_uses_configurable_memory_retry_count(self):
        os.environ["OMNIMEMEVAL_MEMORY_MAX_RETRIES"] = "3"
        client = BaseApiClient("https://example.test", headers={})

        self.assertEqual(client._max_retries, 3)

        os.environ["OMNIMEMEVAL_MEMORY_SDK_MAX_RETRIES"] = "2"
        calls = {"count": 0}

        def always_transient():
            calls["count"] += 1
            raise RuntimeError("500 transient")

        with self.assertRaises(RuntimeError):
            BaseApiClient.sdk_retry(always_transient, base_wait=0)
        self.assertEqual(calls["count"], 2)


class TestRequiredApiKeys(unittest.TestCase):
    _PREFIXES = (
        "BACKBOARD_",
        "COGNEE_",
        "EVEROS_",
        "HINDSIGHT_",
        "LETTA_",
        "MEM0_",
        "MEM9_",
        "MEMORI_",
        "MEMORYLAKE_",
        "SUPERMEMORY_",
        "VIKING_",
        "ZEP_",
    )

    def setUp(self):
        self._old_env = os.environ.copy()
        for key in list(os.environ):
            if key.startswith(self._PREFIXES):
                os.environ.pop(key)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_single_key_clients_fail_fast_when_api_key_missing(self):
        cases = [
            (BackboardClient, "BACKBOARD_API_KEY"),
            (EverosClient, "EVEROS_API_KEY"),
            (HindsightClient, "HINDSIGHT_API_KEY"),
            (LettaClient, "LETTA_API_KEY"),
            (Mem0Client, "MEM0_API_KEY"),
            (Mem9Client, "MEM9_API_KEY"),
            (MemoryLakeClient, "MEMORYLAKE_API_KEY"),
            (SupermemoryClient, "SUPERMEMORY_API_KEY"),
            (VikingClient, "VIKING_API_KEY"),
            (ZepClient, "ZEP_API_KEY"),
        ]
        for client_cls, env_name in cases:
            with self.subTest(client=client_cls.__name__):
                with self.assertRaisesRegex(ValueError, env_name):
                    client_cls()

    def test_cognee_requires_one_supported_auth_env(self):
        with self.assertRaisesRegex(ValueError, "COGNEE_API_KEY"):
            CogneeClient()

    def test_memori_requires_user_and_sdk_keys(self):
        with self.assertRaisesRegex(ValueError, "MEMORI_API_KEY"):
            MemoriClient()

        os.environ["MEMORI_API_KEY"] = "user-key"
        with self.assertRaisesRegex(ValueError, "MEMORI_SDK_API_KEY"):
            MemoriClient()

    def test_memori_uses_env_keys_for_both_auth_headers(self):
        os.environ["MEMORI_API_KEY"] = "user-key"
        os.environ["MEMORI_SDK_API_KEY"] = "sdk-key"

        client = MemoriClient()

        self.assertEqual(client.headers["X-Memori-API-Key"], "sdk-key")
        self.assertEqual(client.headers["Authorization"], "Bearer user-key")

    def test_memorylake_uses_env_key_for_bearer_header(self):
        os.environ["MEMORYLAKE_API_KEY"] = "lake-key"

        client = MemoryLakeClient()

        self.assertEqual(client.headers["Authorization"], "Bearer lake-key")

    def test_memorylake_delete_user_raises_when_list_fails(self):
        os.environ["MEMORYLAKE_API_KEY"] = "lake-key"
        os.environ["MEMORYLAKE_PROJECT_ID"] = "project-id"

        class Response:
            status_code = 500
            text = "server error"

            @staticmethod
            def raise_for_status():
                raise RuntimeError("list failed")

        client = MemoryLakeClient()
        client._get = lambda *args, **kwargs: Response()

        with self.assertRaisesRegex(RuntimeError, "list failed"):
            client.delete_user("user-1")

    def test_memorylake_delete_user_raises_when_forget_fails(self):
        os.environ["MEMORYLAKE_API_KEY"] = "lake-key"
        os.environ["MEMORYLAKE_PROJECT_ID"] = "project-id"

        class ListResponse:
            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {"data": {"items": [{"id": "memory-1"}]}}

        class ForgetResponse:
            status_code = 500
            text = "server error"

            @staticmethod
            def raise_for_status():
                raise RuntimeError("forget failed")

        client = MemoryLakeClient()
        client._get = lambda *args, **kwargs: ListResponse()
        client._post = lambda *args, **kwargs: ForgetResponse()

        with self.assertRaisesRegex(RuntimeError, "forget failed"):
            client.delete_user("user-1")


if __name__ == "__main__":
    unittest.main()

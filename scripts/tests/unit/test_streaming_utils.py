import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.streaming import (  # noqa: E402
    configure_single_user_streaming,
    delete_user_data,
    is_timeout_error,
    load_marker_set,
    LongCallLogger,
    mark_marker,
    prepare_user_after_delete,
    resolve_max_batch_chars,
)


class TestStreamingUtils(unittest.TestCase):
    def test_markers_roundtrip_with_cast(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "markers.txt"
            markers = set()

            mark_marker(path, markers, 3)
            mark_marker(path, markers, 3)

            self.assertEqual(markers, {3})
            self.assertEqual(load_marker_set(path, cast=int), {3})
            self.assertEqual(path.read_text().splitlines(), ["3"])

    def test_mem0_delete_prefers_delete_all(self):
        class Client:
            def __init__(self):
                self.calls = []

            def delete_all(self, user_id):
                self.calls.append(("delete_all", user_id))

            def delete(self, user_id):
                self.calls.append(("delete", user_id))

        client = Client()

        ok, error = delete_user_data("mem0", client, "u1")

        self.assertTrue(ok)
        self.assertIsNone(error)
        self.assertEqual(client.calls, [("delete_all", "u1")])

    def test_prepare_zep_user(self):
        class Client:
            def __init__(self):
                self.added = []

            def add_user(self, user_id):
                self.added.append(user_id)

        client = Client()

        prepare_user_after_delete("zep", client, "u2")
        prepare_user_after_delete("memos", client, "u3")

        self.assertEqual(client.added, ["u2"])

    def test_timeout_detection_checks_exception_chain(self):
        try:
            try:
                raise TimeoutError("request timed out")
            except TimeoutError as exc:
                raise RuntimeError("wrapped") from exc
        except RuntimeError as exc:
            self.assertTrue(is_timeout_error(exc))

    def test_everos_streaming_forces_personal_mode(self):
        old_value = os.environ.get("EVEROS_USE_GROUP")
        try:
            os.environ["EVEROS_USE_GROUP"] = "true"
            configure_single_user_streaming("everos")
            self.assertEqual(os.environ["EVEROS_USE_GROUP"], "false")
        finally:
            if old_value is None:
                os.environ.pop("EVEROS_USE_GROUP", None)
            else:
                os.environ["EVEROS_USE_GROUP"] = old_value

    def test_resolve_max_batch_chars_uses_existing_client_env_convention(self):
        old_graphiti = os.environ.get("GRAPHITI_MAX_BATCH_CHARS")
        old_global = os.environ.get("MAX_BATCH_CHARS")
        try:
            os.environ.pop("GRAPHITI_MAX_BATCH_CHARS", None)
            os.environ["MAX_BATCH_CHARS"] = "12000"
            self.assertEqual(resolve_max_batch_chars("graphiti"), 12000)

            os.environ["GRAPHITI_MAX_BATCH_CHARS"] = "34000"
            self.assertEqual(resolve_max_batch_chars("graphiti"), 34000)
        finally:
            if old_graphiti is None:
                os.environ.pop("GRAPHITI_MAX_BATCH_CHARS", None)
            else:
                os.environ["GRAPHITI_MAX_BATCH_CHARS"] = old_graphiti
            if old_global is None:
                os.environ.pop("MAX_BATCH_CHARS", None)
            else:
                os.environ["MAX_BATCH_CHARS"] = old_global

    def test_resolve_max_batch_chars_rejects_invalid_values(self):
        old_graphiti = os.environ.get("GRAPHITI_MAX_BATCH_CHARS")
        old_global = os.environ.get("MAX_BATCH_CHARS")
        try:
            os.environ.pop("GRAPHITI_MAX_BATCH_CHARS", None)
            os.environ["MAX_BATCH_CHARS"] = "bad"
            with self.assertRaisesRegex(ValueError, "MAX_BATCH_CHARS must be an integer"):
                resolve_max_batch_chars("graphiti")

            os.environ["MAX_BATCH_CHARS"] = "-1"
            with self.assertRaisesRegex(ValueError, "MAX_BATCH_CHARS must be >= 0"):
                resolve_max_batch_chars("graphiti")
        finally:
            if old_graphiti is None:
                os.environ.pop("GRAPHITI_MAX_BATCH_CHARS", None)
            else:
                os.environ["GRAPHITI_MAX_BATCH_CHARS"] = old_graphiti
            if old_global is None:
                os.environ.pop("MAX_BATCH_CHARS", None)
            else:
                os.environ["MAX_BATCH_CHARS"] = old_global

    def test_long_call_logger_prints_start_and_done(self):
        buf = StringIO()

        with redirect_stdout(buf):
            with LongCallLogger("test add call", heartbeat_seconds=0):
                pass

        output = buf.getvalue()
        self.assertIn("[ADD START] test add call", output)
        self.assertIn("[ADD DONE] test add call", output)


if __name__ == "__main__":
    unittest.main()

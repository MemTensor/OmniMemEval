"""Unit tests for MemEval core utility functions."""

import json
import os
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from client_factory.base_client import iter_batches, _split_text, _TokenBucketLimiter
from utils.checkpoint import fsync_write_line, atomic_json_dump


# ── extract_label_json (conditionally imported due to heavy deps) ─────────────

_extract_label_json = None
try:
    from utils.nlp_metrics import extract_label_json as _elj

    _extract_label_json = _elj
except ImportError:
    import re

    def _extract_label_json_fallback(text: str):
        pattern = r'\{\s*"label"\s*:\s*["\']([^"\']*)["\']\s*\}'
        match = re.search(pattern, text)
        if match:
            return match.group(0)
        return None

    _extract_label_json = _extract_label_json_fallback


# ═══════════════════════════════════════════════════════════════════════════════
# iter_batches
# ═══════════════════════════════════════════════════════════════════════════════


class TestIterBatches(unittest.TestCase):
    def test_count_based_basic(self):
        result = list(iter_batches([1, 2, 3, 4, 5], batch_size=2))
        self.assertEqual(result, [[1, 2], [3, 4], [5]])

    def test_empty_input(self):
        result = list(iter_batches([], batch_size=2))
        self.assertEqual(result, [])

    def test_single_item(self):
        result = list(iter_batches([1], batch_size=5))
        self.assertEqual(result, [[1]])

    def test_batch_size_larger_than_items(self):
        result = list(iter_batches([1, 2], batch_size=10))
        self.assertEqual(result, [[1, 2]])

    def test_char_budget_mode(self):
        msgs = [
            {"content": "hello"},      # 5 chars
            {"content": "world"},      # 5 chars
            {"content": "foobar"},     # 6 chars
        ]
        batches = list(iter_batches(msgs, max_chars=12))
        self.assertTrue(len(batches) >= 2)
        total_items = sum(len(b) for b in batches)
        self.assertEqual(total_items, 3)

    def test_oversized_message_split(self):
        big_msg = {"content": "A" * 100}
        batches = list(iter_batches([big_msg], max_chars=30))
        self.assertTrue(len(batches) > 1)
        for batch in batches:
            self.assertEqual(len(batch), 1)
            self.assertIn("[part", batch[0]["content"])


# ═══════════════════════════════════════════════════════════════════════════════
# _split_text
# ═══════════════════════════════════════════════════════════════════════════════


class TestSplitText(unittest.TestCase):
    def test_short_text_no_split(self):
        result = _split_text("hello world", max_chars=100)
        self.assertEqual(result, ["hello world"])

    def test_split_at_paragraph(self):
        text = "Paragraph one.\n\nParagraph two."
        result = _split_text(text, max_chars=20)
        self.assertTrue(len(result) >= 2)
        joined = "\n\n".join(result)
        self.assertIn("Paragraph one.", joined)
        self.assertIn("Paragraph two.", joined)

    def test_split_at_line_boundary(self):
        text = "Line one.\nLine two.\nLine three."
        result = _split_text(text, max_chars=15)
        self.assertTrue(len(result) >= 2)
        for chunk in result:
            self.assertLessEqual(len(chunk), 15)

    def test_split_at_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence."
        result = _split_text(text, max_chars=25)
        self.assertTrue(len(result) >= 2)

    def test_hard_cut(self):
        text = "A" * 50
        result = _split_text(text, max_chars=10)
        self.assertEqual(len(result), 5)
        for chunk in result:
            self.assertEqual(len(chunk), 10)


# ═══════════════════════════════════════════════════════════════════════════════
# fsync_write_line
# ═══════════════════════════════════════════════════════════════════════════════


class TestFsyncWriteLine(unittest.TestCase):
    def test_writes_line_with_newline(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            path = f.name
            fsync_write_line(f, "hello")
        try:
            with open(path) as f:
                content = f.read()
            self.assertEqual(content, "hello\n")
        finally:
            os.unlink(path)

    def test_file_content_flushed(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            path = f.name
            fsync_write_line(f, "data")
            size = os.path.getsize(path)
            self.assertGreater(size, 0)
        os.unlink(path)

    def test_multiple_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            path = f.name
            fsync_write_line(f, "line1")
            fsync_write_line(f, "line2")
            fsync_write_line(f, "line3")
        try:
            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(lines, ["line1\n", "line2\n", "line3\n"])
        finally:
            os.unlink(path)

    def test_shared_file_handle_writes_are_serialized(self):
        expected = {f"line-{idx}" for idx in range(25)}
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            path = f.name
            with ThreadPoolExecutor(max_workers=5) as executor:
                list(executor.map(lambda value: fsync_write_line(f, value), expected))
        try:
            with open(path) as f:
                lines = {line.strip() for line in f if line.strip()}
            self.assertEqual(lines, expected)
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════════
# atomic_json_dump
# ═══════════════════════════════════════════════════════════════════════════════


class TestAtomicJsonDump(unittest.TestCase):
    def test_basic_dump_and_read_back(self):
        data = {"key": "value", "num": 42}
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.json")
            atomic_json_dump(data, path)
            with open(path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded, data)

    def test_nested_structures_preserved(self):
        data = {
            "list": [1, 2, {"nested": True}],
            "dict": {"a": {"b": [3, 4]}},
        }
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "nested.json")
            atomic_json_dump(data, path)
            with open(path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded, data)

    def test_custom_json_kw(self):
        data = {"hello": "世界"}
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "custom.json")
            atomic_json_dump(data, path, indent=2, ensure_ascii=False)
            with open(path) as f:
                raw = f.read()
            self.assertIn("世界", raw)
            self.assertIn("\n", raw)
            loaded = json.loads(raw)
            self.assertEqual(loaded, data)

    def test_atomicity_no_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "atom.json")
            original = {"original": True}
            atomic_json_dump(original, path)

            class Unserializable:
                pass

            with self.assertRaises(TypeError):
                atomic_json_dump({"bad": Unserializable()}, path)

            with open(path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded, original)


# ═══════════════════════════════════════════════════════════════════════════════
# _TokenBucketLimiter
# ═══════════════════════════════════════════════════════════════════════════════


class TestTokenBucketLimiter(unittest.TestCase):
    def test_rate_limiting(self):
        qps = 10
        limiter = _TokenBucketLimiter(qps)
        start = time.monotonic()
        limiter.acquire()
        limiter.acquire()
        elapsed = time.monotonic() - start
        min_expected = 1.0 / qps
        self.assertGreaterEqual(elapsed, min_expected * 0.9)


# ═══════════════════════════════════════════════════════════════════════════════
# extract_label_json
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractLabelJson(unittest.TestCase):
    def test_valid_json_with_label(self):
        text = '{"label": "correct"}'
        result = _extract_label_json(text)
        self.assertIsNotNone(result)
        self.assertIn('"label"', result)
        self.assertIn("correct", result)

    def test_json_embedded_in_text(self):
        text = 'The answer is {"label": "incorrect"} because it is wrong.'
        result = _extract_label_json(text)
        self.assertIsNotNone(result)
        self.assertIn("incorrect", result)

    def test_no_json(self):
        text = "Just plain text with no JSON at all."
        result = _extract_label_json(text)
        self.assertIsNone(result)

    def test_malformed_json(self):
        text = '{"label": "correct"'
        result = _extract_label_json(text)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

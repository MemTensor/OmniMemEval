import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.pipeline_status import STATUS_SUCCESS_EMPTY, classify_search_status
from utils.search_helpers import generic_text_search, unpack_search_result


class DummySearchClient:
    def __init__(self, results):
        self.results = results

    def search(self, query, user_id, top_k):
        return self.results


class TestSearchHelpers(unittest.TestCase):
    def test_empty_search_keeps_prompt_context_but_raw_context_is_empty(self):
        result = generic_text_search(DummySearchClient([]), "q", "u", 20)
        context, duration_ms, reflect_answer, raw_context = unpack_search_result(result)

        self.assertGreaterEqual(duration_ms, 0.0)
        self.assertIsNone(reflect_answer)
        self.assertTrue(context)
        self.assertEqual(raw_context, "")
        self.assertEqual(
            classify_search_status(
                context,
                reflect_answer,
                raw_context=raw_context,
            ),
            STATUS_SUCCESS_EMPTY,
        )

    def test_legacy_tuple_unpack_defaults_raw_context_to_context(self):
        context, duration_ms, reflect_answer, raw_context = unpack_search_result(
            ("context", 1.5)
        )
        self.assertEqual(context, "context")
        self.assertEqual(duration_ms, 1.5)
        self.assertIsNone(reflect_answer)
        self.assertEqual(raw_context, "context")


if __name__ == "__main__":
    unittest.main()

"""Tests for LoCoMo pipeline bookkeeping helpers."""

import os
import sys
import unittest


SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
LOCOMO_DIR = os.path.join(SCRIPTS_DIR, "locomo")
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, LOCOMO_DIR)

from locomo_common import (  # noqa: E402
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    STATUS_SUCCESS_EMPTY,
    classify_search_status,
    expected_answer_pairs,
    expected_questions,
    search_allowed_statuses,
    status_counts,
    status_records_with_skipped,
    validate_query_coverage,
)


class TestLoCoMoCommon(unittest.TestCase):
    def test_expected_questions_excludes_adversarial_category(self):
        qa_set = [
            {"question": "q1", "category": 1},
            {"question": "q2", "category": 5},
            {"question": "q3", "category": 4},
        ]
        self.assertEqual(expected_questions(qa_set), ["q1", "q3"])

    def test_classify_search_status_distinguishes_empty_success(self):
        self.assertEqual(classify_search_status("context"), STATUS_SUCCESS)
        self.assertEqual(classify_search_status("", "direct answer"), STATUS_SUCCESS)
        self.assertEqual(classify_search_status(""), STATUS_SUCCESS_EMPTY)
        self.assertEqual(
            classify_search_status(
                "Memories for user Alice:\n\nMemories for user Bob:\n\n",
                raw_context="",
            ),
            STATUS_SUCCESS_EMPTY,
        )
        self.assertEqual(
            classify_search_status(
                "Memories for user Alice:\n\nmemory\n\nMemories for user Bob:\n\n",
                raw_context="memory",
            ),
            STATUS_SUCCESS,
        )

    def test_validate_query_coverage_rejects_failed_status(self):
        records = [
            {"query": "q1", "status": STATUS_SUCCESS},
            {"query": "q2", "status": STATUS_FAILED},
        ]
        ok, issues = validate_query_coverage(
            records,
            ["q1", "q2"],
            allowed_statuses=search_allowed_statuses(
                allow_empty_search=True,
                allow_skipped=False,
            ),
        )
        self.assertFalse(ok)
        self.assertIn("disallowed statuses", "; ".join(issues))

    def test_validate_query_coverage_can_allow_skipped(self):
        records = [
            {"query": "q1", "status": STATUS_SUCCESS_EMPTY},
            {"query": "q2", "status": STATUS_SKIPPED},
        ]
        ok, issues = validate_query_coverage(
            records,
            ["q1", "q2"],
            allowed_statuses=search_allowed_statuses(
                allow_empty_search=True,
                allow_skipped=True,
            ),
        )
        self.assertTrue(ok, issues)

    def test_validate_query_coverage_allows_expected_duplicate_questions(self):
        records = [
            {"query": "q1", "status": STATUS_SUCCESS},
            {"query": "q1", "status": STATUS_SUCCESS},
            {"query": "q2", "status": STATUS_SUCCESS},
        ]
        ok, issues = validate_query_coverage(
            records,
            ["q1", "q1", "q2"],
            allowed_statuses={STATUS_SUCCESS},
        )
        self.assertTrue(ok, issues)

    def test_validate_query_coverage_counts_missing_duplicate_questions(self):
        records = [
            {"query": "q1", "status": STATUS_SUCCESS},
            {"query": "q2", "status": STATUS_SUCCESS},
        ]
        ok, issues = validate_query_coverage(
            records,
            ["q1", "q1", "q2"],
            allowed_statuses={STATUS_SUCCESS},
        )
        self.assertFalse(ok)
        self.assertIn("missing queries: 1", issues)

    def test_status_records_with_skipped_does_not_double_count_stored_skips(self):
        grouped_records = {
            "locomo_exp_user_0": [
                {"question": "q1", "status": STATUS_SUCCESS},
                {"question": "q2", "status": STATUS_SKIPPED},
            ]
        }
        skipped_records = [
            {"group_id": "locomo_exp_user_0", "query": "q2", "status": STATUS_SKIPPED},
            {"group_id": "locomo_exp_user_0", "query": "q3", "status": STATUS_SKIPPED},
        ]

        records = status_records_with_skipped(grouped_records, skipped_records)

        self.assertEqual(
            status_counts(records),
            {STATUS_SUCCESS: 1, STATUS_SKIPPED: 2},
        )


class TestLoCoMoResponses(unittest.TestCase):
    def test_expected_answer_questions_handles_statuses(self):
        qa_set = [
            {"question": "q1", "answer": "a1", "category": 1},
            {"question": "q2", "answer": "a2", "category": 2},
            {"question": "q3", "answer": "a3", "category": 3},
            {"question": "q4", "answer": "a4", "category": 4},
        ]
        search_results = [
            {"query": "q1", "context": "ctx", "status": STATUS_SUCCESS},
            {"query": "q2", "context": "", "status": STATUS_SUCCESS_EMPTY},
            {"query": "q3", "context": "", "status": STATUS_SKIPPED},
            {"query": "q4", "context": "", "status": STATUS_FAILED},
        ]

        pairs, skipped, failures, expected = expected_answer_pairs(qa_set, search_results)

        self.assertEqual([qa["question"] for qa, _ in pairs], ["q1", "q2"])
        self.assertEqual([record["query"] for record in skipped], ["q3"])
        self.assertEqual([record["query"] for record in failures], ["q4"])
        self.assertEqual(expected, ["q1", "q2", "q3", "q4"])

    def test_expected_answer_questions_reports_missing_search_result(self):
        qa_set = [{"question": "q1", "answer": "a1", "category": 1}]
        pairs, skipped, failures, _expected = expected_answer_pairs(qa_set, [])

        self.assertEqual(pairs, [])
        self.assertEqual(skipped, [])
        self.assertEqual(failures[0]["query"], "q1")

    def test_expected_answer_questions_pairs_duplicate_questions_in_order(self):
        qa_set = [
            {"question": "q1", "answer": "a1", "category": 1},
            {"question": "q1", "answer": "a2", "category": 1},
        ]
        search_results = [
            {"query": "q1", "context": "ctx1", "status": STATUS_SUCCESS},
            {"query": "q1", "context": "ctx2", "status": STATUS_SUCCESS},
        ]

        pairs, skipped, failures, expected = expected_answer_pairs(qa_set, search_results)

        self.assertEqual(expected, ["q1", "q1"])
        self.assertEqual([record["context"] for _, record in pairs], ["ctx1", "ctx2"])
        self.assertEqual(skipped, [])
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()

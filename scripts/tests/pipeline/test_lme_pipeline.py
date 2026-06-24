import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from longmemeval.lme_common import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    STATUS_SUCCESS_EMPTY,
    build_search_result,
    classify_search_status,
    error_payload,
    grade_complete,
    response_complete,
    search_allowed_statuses,
    skipped_response_record,
    status_counts,
    user_id_for,
    validate_single_search_result,
)


def _row():
    return {
        "question": "What train do I take?",
        "question_type": "temporal",
        "question_date": "2024-01-02",
        "answer": "The green line.",
        "haystack_session_ids": ["s1"],
        "answer_session_ids": ["s1"],
        "haystack_sessions": [
            [
                {
                    "role": "assistant",
                    "content": "You take the green line.",
                    "has_answer": True,
                }
            ]
        ],
    }


class TestLmePipelineStatus(unittest.TestCase):
    def test_search_status_distinguishes_success_empty_from_failure(self):
        self.assertEqual(classify_search_status("", None), STATUS_SUCCESS_EMPTY)
        self.assertEqual(classify_search_status("", "direct answer"), STATUS_SUCCESS)
        self.assertEqual(classify_search_status("context", None), STATUS_SUCCESS)
        self.assertEqual(
            classify_search_status("Conversation memories:\n\n", raw_context=""),
            STATUS_SUCCESS_EMPTY,
        )
        self.assertEqual(
            classify_search_status("Conversation memories:\n\nmemory", raw_context="memory"),
            STATUS_SUCCESS,
        )

    def test_search_result_validation_requires_allowed_status(self):
        user_id = user_id_for("v1", 3)
        result = build_search_result(
            _row(),
            user_id=user_id,
            context="",
            status=STATUS_SUCCESS_EMPTY,
        )

        ok, issues = validate_single_search_result(
            result,
            user_id=user_id,
            question=_row()["question"],
            allowed_statuses=search_allowed_statuses(
                allow_empty_search=True,
                allow_skipped=False,
            ),
        )
        self.assertTrue(ok, issues)

        ok, issues = validate_single_search_result(
            result,
            user_id=user_id,
            question=_row()["question"],
            allowed_statuses=search_allowed_statuses(
                allow_empty_search=False,
                allow_skipped=False,
            ),
        )
        self.assertFalse(ok)
        self.assertIn("disallowed status", issues[0])

    def test_skipped_response_is_complete_without_model_answer(self):
        user_id = user_id_for("v1", 0)
        search_result = build_search_result(
            _row(),
            user_id=user_id,
            status=STATUS_SKIPPED,
            error=error_payload("search", "rate limit"),
        )
        search_entry = search_result[user_id][0]

        response = skipped_response_record(
            user_id=user_id,
            search_entry=search_entry,
            reason="search was explicitly skipped",
            error=search_entry["error"],
        )

        ok, issues = response_complete(response, search_entry)
        self.assertTrue(ok, issues)
        self.assertEqual(response["status"], STATUS_SKIPPED)

    def test_grade_resume_checks_expected_judge_runs(self):
        response = {
            "question": _row()["question"],
            "status": STATUS_SUCCESS,
        }
        complete_grade = {
            "question": _row()["question"],
            "llm_judgments": {"judgment_1": True, "judgment_2": False},
        }
        ok, issues = grade_complete(complete_grade, response, num_runs=2)
        self.assertTrue(ok, issues)

        incomplete_grade = {
            "question": _row()["question"],
            "llm_judgments": {"judgment_1": True},
        }
        ok, issues = grade_complete(incomplete_grade, response, num_runs=2)
        self.assertFalse(ok)
        self.assertIn("missing judgment runs", issues[0])

        skipped_grade = {
            "question": response["question"],
            "status": STATUS_SKIPPED,
            "skip_reason": "eval_failed",
        }
        ok, issues = grade_complete(
            skipped_grade,
            response,
            num_runs=2,
            allow_skipped_grade=True,
        )
        self.assertTrue(ok, issues)

        ok, issues = grade_complete(skipped_grade, response, num_runs=2)
        self.assertFalse(ok)
        self.assertIn("existing grade is skipped", issues[0])

    def test_status_counts_defaults_legacy_records_to_success(self):
        records = [
            {"status": STATUS_FAILED},
            {"status": STATUS_SKIPPED},
            {"question": "legacy success record"},
        ]

        self.assertEqual(
            status_counts(records),
            {
                STATUS_FAILED: 1,
                STATUS_SKIPPED: 1,
                STATUS_SUCCESS: 1,
            },
        )


if __name__ == "__main__":
    unittest.main()

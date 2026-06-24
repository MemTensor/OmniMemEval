"""Contract tests for metric inputs and report rendering."""

import json
import os
import sys
import tempfile
import unittest


SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, os.path.join(SCRIPTS_DIR, "locomo"))

import locomo_metric  # noqa: E402
from locomo.locomo_report import LoCoMoReport  # noqa: E402
from longmemeval import lme_metric  # noqa: E402
from longmemeval.lme_report import LMEReport  # noqa: E402
from utils.duration_stats import add_duration_values, update_unit_duration_list  # noqa: E402
from utils.report_base import avg_prompt_tokens, render_token_usage  # noqa: E402


def _read_json(path):
    with open(path) as f:
        return json.load(f)


class TestMetricStageContracts(unittest.TestCase):
    def test_locomo_metric_counts_only_success_records(self):
        results = locomo_metric.calculate_scores({
            "locomo_exp_user_0": [
                {
                    "question": "q1",
                    "category": "1",
                    "llm_judgments": {"judgment_1": True},
                    "nlp_metrics": {"lexical": {"f1": 1.0}, "context_tokens": 5},
                    "search_duration_ms": 10,
                    "status": "success",
                },
                {
                    "question": "q2",
                    "category": "1",
                    "llm_judgments": {"judgment_1": False},
                    "status": "skipped",
                },
            ]
        })

        self.assertEqual(results["category_scores"]["1"]["total"], 1)
        self.assertEqual(results["metrics"]["llm_judge_score"], 1.0)

    def test_lme_metric_counts_only_success_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grade_path = os.path.join(tmpdir, "lib_lme_grades.json")
            excel_path = os.path.join(tmpdir, "lib_lme_results.xlsx")
            lme_metric.calculate_scores(
                {
                    "u1": {
                        "category": "temporal",
                        "llm_judgments": {"judgment_1": True},
                        "nlp_metrics": {"lexical": {"f1": 1.0}, "context_tokens": 8},
                        "search_duration_ms": 12,
                        "status": "success",
                    },
                    "u2": {
                        "category": "temporal",
                        "llm_judgments": {"judgment_1": False},
                        "status": "skipped",
                    },
                },
                grade_path,
                excel_path,
            )
            results = _read_json(grade_path)

        self.assertEqual(results["category_scores"]["temporal"]["total"], 1)
        self.assertEqual(results["metrics"]["llm_judge_score"], 1.0)

    def test_add_duration_values_prefer_per_unit_stats(self):
        stats = {
            "add_call_durations_by_unit": {"u1": [10, 20], "u2": [30]},
            "add_call_durations_ms": [999],
            "user_durations_ms": {"u1": 1000},
        }

        self.assertEqual(add_duration_values(stats), [10.0, 20.0, 30.0])

    def test_update_unit_duration_list_replaces_rerun_unit_values(self):
        stats = {}

        update_unit_duration_list(
            stats,
            "u1",
            [10, 20],
            map_key="add_call_durations_by_unit",
            flat_key="add_call_durations_ms",
        )
        update_unit_duration_list(
            stats,
            "u1",
            [30],
            map_key="add_call_durations_by_unit",
            flat_key="add_call_durations_ms",
        )

        self.assertEqual(stats["add_call_durations_by_unit"], {"u1": [30.0]})
        self.assertEqual(stats["add_call_durations_ms"], [30.0])


class TestReportStageContracts(unittest.TestCase):
    def test_answer_prompt_tokens_are_rendered_as_context_tokens(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "token_usage_answer.json"), "w") as f:
                json.dump(
                    {
                        "modules": {
                            "ANSWER": {
                                "model": "answer-model",
                                "call_count": 2,
                                "prompt_tokens": 100,
                                "estimated_prompt_tokens": 25,
                                "estimated_prompt_call_count": 1,
                                "completion_tokens": 10,
                                "total_tokens": 110,
                            }
                        }
                    },
                    f,
                )
            lines = []
            render_token_usage(lines, tmpdir)

        rendered = "\n".join(lines)
        self.assertIn("## Context Tokens", rendered)
        self.assertIn("same benchmark, answer model, and prompt template", rendered)
        self.assertIn("Context Tokens (avg) | 50.0", rendered)
        self.assertIn("Context Token Estimate Fallback Calls", rendered)
        self.assertNotIn("Context Tokens (total)", rendered)
        self.assertNotIn("Estimated Context Tokens", rendered)
        self.assertNotIn("Avg Completion / Call", rendered)

    def test_avg_prompt_tokens_reads_answer_prompt_proxy(self):
        self.assertEqual(
            avg_prompt_tokens({"ANSWER": {"call_count": 4, "prompt_tokens": 100}}),
            25,
        )

    def _render_report(self, report, grades_name, grades):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, grades_name)
            with open(path, "w") as f:
                json.dump(grades, f)
            rendered = report.generate_report(tmpdir, "lib", "v1")
        self.assertIn("## Evaluation Results", rendered)
        self.assertNotIn("No evaluation results found", rendered)
        return rendered

    def test_locomo_report_renders_minimal_metric_output(self):
        rendered = self._render_report(
            LoCoMoReport(),
            "lib_locomo_grades.json",
            {
                "metrics": {
                    "llm_judge_score": 1.0,
                    "llm_judge_std": 0.0,
                    "duration": {"search_duration_ms": 10.0},
                },
                "category_scores": {
                    "1": {"category_name": "multi hop", "llm_judge_score": 1.0, "total": 1}
                },
                "pipeline_status": {"answer": {"status_counts": {"success": 1}}},
            },
        )
        self.assertIn("LLM-as-Judge", rendered)

    def test_lme_report_renders_minimal_metric_output(self):
        rendered = self._render_report(
            LMEReport(),
            "lib_lme_grades.json",
            {
                "metrics": {
                    "llm_judge_score": 1.0,
                    "llm_judge_std": 0.0,
                    "context_tokens": 8,
                    "duration": {"search_duration_ms": 10.0},
                },
                "category_scores": {
                    "temporal": {"category_name": "temporal", "llm_judge_score": 1.0, "total": 1}
                },
            },
        )
        self.assertNotIn("| Context Tokens (avg) | 8 |", rendered)


if __name__ == "__main__":
    unittest.main()

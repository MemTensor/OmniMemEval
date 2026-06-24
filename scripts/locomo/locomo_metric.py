import argparse
import sys
import os
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from client_factory import SUPPORTED_LIBS, DEFAULT_LIB
from utils.checkpoint import atomic_json_dump
from utils.duration_stats import add_duration_values
from locomo_common import record_status



# Category mapping verified against original evaluation code and GitHub Issue #6
# Note: JSON category IDs do NOT follow the paper's prose numbering order
category_mapping = {
    "1": "multi hop",
    "2": "temporal reasoning",
    "3": "open domain",
    "4": "single hop",
}


def convert_numpy_types(obj):
    if isinstance(obj, np.number):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj


def load_pipeline_status(results_dir, lib):
    status_files = {
        "search": f"{lib}_locomo_search_status.json",
        "answer": f"{lib}_locomo_response_status.json",
        "eval": f"{lib}_locomo_eval_status.json",
    }
    status = {}
    for stage, filename in status_files.items():
        path = os.path.join(results_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as exc:
            status[stage] = {"load_error": str(exc)}
            continue
        failed_count = len(data.get("failed_groups", []) or data.get("failed_users", []))
        status[stage] = {
            "status_counts": data.get("status_counts", {}),
            "failed_units": failed_count,
            "failed_groups": failed_count,
            "skipped_records": len(data.get("skipped_records", [])),
        }
    return status


def calculate_scores(data):
    category_scores = {}
    category_question_count = {}

    overall_metrics = {
        "lexical": {
            m: []
            for m in [
                "f1",
                "rouge1_f",
                "rouge2_f",
                "rougeL_f",
                "bleu1",
                "bleu2",
                "bleu3",
                "bleu4",
                "meteor",
            ]
        },
        "semantic": {m: [] for m in ["bert_f1", "similarity"]},
        "context_tokens": [],
        "duration": {
            m: [] for m in ["search_duration_ms"]
        },
    }

    category_metrics = {}
    user_metrics = {}

    total_questions = 0

    all_judgment_keys = set()
    judgment_run_scores = {}

    for _user, questions in data.items():
        for question in questions:
            if "llm_judgments" in question:
                all_judgment_keys.update(question["llm_judgments"].keys())

    for key in all_judgment_keys:
        judgment_run_scores[key] = []

    for user, questions in data.items():
        user_total = 0

        # Initialize user_metrics with each judgment run
        user_metrics[user] = {
            "total": 0,
            "llm_judge_score": 0,
            "llm_judge_std": 0,
            "judgment_run_scores": {key: [] for key in all_judgment_keys},
            "lexical": {m: [] for m in overall_metrics["lexical"]},
            "semantic": {m: [] for m in overall_metrics["semantic"]},
            "context_tokens": [],
            "duration": {m: [] for m in overall_metrics["duration"]},
        }

        for question in questions:
            if record_status(question) != "success":
                continue
            total_questions += 1
            user_total += 1

            if "llm_judgments" in question:
                for judgment_key, judgment_value in question["llm_judgments"].items():
                    score = 1 if judgment_value else 0
                    judgment_run_scores[judgment_key].append(score)
                    user_metrics[user]["judgment_run_scores"][judgment_key].append(score)

            category = question.get("category", "unknown")
            if category not in category_scores:
                category_scores[category] = {
                    "total": 0,
                    "category_name": category_mapping.get(str(category), "Unknown"),
                    "judgment_run_scores": {key: [] for key in all_judgment_keys},
                }
                category_metrics[category] = {
                    "lexical": {m: [] for m in overall_metrics["lexical"]},
                    "semantic": {m: [] for m in overall_metrics["semantic"]},
                    "context_tokens": [],
                    "duration": {m: [] for m in overall_metrics["duration"]},
                }
                category_question_count[category] = 0

            category_scores[category]["total"] += 1
            category_question_count[category] += 1

            if "llm_judgments" in question:
                for judgment_key, judgment_value in question["llm_judgments"].items():
                    score = 1 if judgment_value else 0
                    category_scores[category]["judgment_run_scores"][judgment_key].append(score)

            nlp = question.get("nlp_metrics", {})
            for metric in overall_metrics["lexical"]:
                v = nlp.get("lexical", {}).get(metric)
                if v is not None:
                    overall_metrics["lexical"][metric].append(v)
                    category_metrics[category]["lexical"][metric].append(v)
                    user_metrics[user]["lexical"][metric].append(v)

            for metric in overall_metrics["semantic"]:
                v = nlp.get("semantic", {}).get(metric)
                if v is not None:
                    overall_metrics["semantic"][metric].append(v)
                    category_metrics[category]["semantic"][metric].append(v)
                    user_metrics[user]["semantic"][metric].append(v)

            ct = nlp.get("context_tokens")
            if ct is not None:
                overall_metrics["context_tokens"].append(ct)
                category_metrics[category]["context_tokens"].append(ct)
                user_metrics[user]["context_tokens"].append(ct)

            for metric in overall_metrics["duration"]:
                v = question.get(metric)
                if v is not None:
                    overall_metrics["duration"][metric].append(v)
                    category_metrics[category]["duration"][metric].append(v)
                    user_metrics[user]["duration"][metric].append(v)

        user_metrics[user]["total"] = user_total

        judgment_avgs = []
        for _judgment_key, scores in user_metrics[user]["judgment_run_scores"].items():
            if scores:
                avg = np.mean(scores)
                judgment_avgs.append(avg)

        user_metrics[user]["llm_judge_score"] = np.mean(judgment_avgs) if judgment_avgs else 0.0
        user_metrics[user]["llm_judge_std"] = (
            np.std(judgment_avgs) if len(judgment_avgs) > 1 else 0.0
        )

        for group in ["lexical", "semantic"]:
            for metric in user_metrics[user][group]:
                values = user_metrics[user][group][metric]
                user_metrics[user][group][metric] = float(np.mean(values)) if values else None

        user_metrics[user]["context_tokens"] = (
            np.mean(user_metrics[user]["context_tokens"])
            if user_metrics[user]["context_tokens"]
            else 0.0
        )

        duration_metrics = list(user_metrics[user]["duration"].keys())
        for metric in duration_metrics:
            values = user_metrics[user]["duration"][metric]
            if values:
                user_metrics[user]["duration"][metric] = np.mean(values)
                user_metrics[user]["duration"][f"{metric}_p50"] = np.percentile(values, 50)
                user_metrics[user]["duration"][f"{metric}_p95"] = np.percentile(values, 95)
            else:
                user_metrics[user]["duration"][metric] = 0.0
                user_metrics[user]["duration"][f"{metric}_p50"] = 0.0
                user_metrics[user]["duration"][f"{metric}_p95"] = 0.0

    judgment_run_averages = []
    for _judgment_key, scores in judgment_run_scores.items():
        if scores:
            judgment_run_averages.append(np.mean(scores))

    llm_judge_score = np.mean(judgment_run_averages) if judgment_run_averages else 0.0
    llm_judge_std = np.std(judgment_run_averages) if len(judgment_run_averages) > 1 else 0.0

    category_overall_scores = {}
    for category, score_data in category_scores.items():
        category_judgment_avgs = []
        for _judgment_key, scores in score_data["judgment_run_scores"].items():
            if scores:
                category_judgment_avgs.append(np.mean(scores))

        category_overall_scores[category] = {
            "category_name": score_data["category_name"],
            "llm_judge_score": np.mean(category_judgment_avgs) if category_judgment_avgs else 0.0,
            "llm_judge_std": np.std(category_judgment_avgs)
            if len(category_judgment_avgs) > 1
            else 0.0,
            "total": score_data["total"],
            "lexical": {},
            "semantic": {},
            "duration": {},
            "context_tokens": 0.0,
        }

        for group in ["lexical", "semantic"]:
            for metric in category_metrics[category][group]:
                values = category_metrics[category][group][metric]
                category_overall_scores[category][group][metric] = (
                    float(np.mean(values)) if values else None
                )

        category_overall_scores[category]["context_tokens"] = (
            np.mean(category_metrics[category]["context_tokens"])
            if category_metrics[category]["context_tokens"]
            else 0.0
        )

        # Calculate mean and percentiles for category duration metrics
        duration_metrics = list(
            category_metrics[category]["duration"].keys()
        )  # Create a list of keys first
        for metric in duration_metrics:
            values = category_metrics[category]["duration"][metric]
            if values:
                category_overall_scores[category]["duration"][metric] = np.mean(values)
                # Add P50 (median) and P95 percentiles
                category_overall_scores[category]["duration"][f"{metric}_p50"] = np.percentile(
                    values, 50
                )
                category_overall_scores[category]["duration"][f"{metric}_p95"] = np.percentile(
                    values, 95
                )
            else:
                category_overall_scores[category]["duration"][metric] = 0.0
                category_overall_scores[category]["duration"][f"{metric}_p50"] = 0.0
                category_overall_scores[category]["duration"][f"{metric}_p95"] = 0.0

    # calculate overall scores
    overall_metric_averages = {
        "llm_judge_score": llm_judge_score,
        "llm_judge_std": llm_judge_std,
        "lexical": {},
        "semantic": {},
        "context_tokens": 0.0,
        "duration": {},
    }

    for group in ["lexical", "semantic"]:
        for metric in overall_metrics[group]:
            values = overall_metrics[group][metric]
            overall_metric_averages[group][metric] = float(np.mean(values)) if values else None

    overall_metric_averages["context_tokens"] = (
        np.mean(overall_metrics["context_tokens"]) if overall_metrics["context_tokens"] else 0.0
    )

    duration_metrics = list(overall_metrics["duration"].keys())
    for metric in duration_metrics:
        values = overall_metrics["duration"][metric]
        if values:
            overall_metric_averages["duration"][metric] = np.mean(values)
            overall_metric_averages["duration"][f"{metric}_p50"] = np.percentile(values, 50)
            overall_metric_averages["duration"][f"{metric}_p95"] = np.percentile(values, 95)
        else:
            overall_metric_averages["duration"][metric] = 0.0
            overall_metric_averages["duration"][f"{metric}_p50"] = 0.0
            overall_metric_averages["duration"][f"{metric}_p95"] = 0.0

    return {
        "metrics": overall_metric_averages,
        "category_scores": category_overall_scores,
        "user_scores": user_metrics,
    }


def save_to_excel(results, output_path):
    # Create a combined data structure for metrics and category scores
    combined_data = []

    # Process overall metrics - flatten nested structures
    overall_row = {"category": "overall"}
    overall_row["llm_judge_score"] = results["metrics"]["llm_judge_score"]
    overall_row["llm_judge_std"] = results["metrics"]["llm_judge_std"]

    for metric, value in results["metrics"]["lexical"].items():
        overall_row[metric] = value if value is not None else "-"

    for metric, value in results["metrics"]["semantic"].items():
        overall_row[metric] = value if value is not None else "-"

    # Add context tokens
    overall_row["context_tokens"] = results["metrics"]["context_tokens"]

    # Add all duration metrics, including percentiles
    for metric, value in results["metrics"]["duration"].items():
        overall_row[metric] = value

    combined_data.append(overall_row)

    # Process category scores - flatten nested structures
    for _, scores in results["category_scores"].items():
        category_row = {"category": scores["category_name"]}
        category_row["llm_judge_score"] = scores["llm_judge_score"]
        category_row["llm_judge_std"] = scores["llm_judge_std"]

        for metric, value in scores["lexical"].items():
            category_row[metric] = value if value is not None else "-"

        for metric, value in scores["semantic"].items():
            category_row[metric] = value if value is not None else "-"

        # Add context tokens
        category_row["context_tokens"] = scores["context_tokens"]

        # Add all duration metrics, including percentiles
        for metric, value in scores["duration"].items():
            category_row[metric] = value

        combined_data.append(category_row)

    # Create DataFrame and save to Excel
    combined_df = pd.DataFrame(combined_data)

    # Create a pandas Excel writer
    with pd.ExcelWriter(output_path) as writer:
        combined_df.to_excel(writer, sheet_name="Metrics", index=False)

    print(f"Excel file saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lib",
        type=str,
        choices=SUPPORTED_LIBS,
        default=DEFAULT_LIB,
    )
    parser.add_argument(
        "--version",
        type=str,
        default="default",
        help="Version identifier for loading results (e.g., 1010)",
    )

    args = parser.parse_args()
    lib = args.lib
    version = args.version

    judged_path = f"results/locomo/{lib}-{version}/{lib}_locomo_judged.json"
    grade_path = f"results/locomo/{lib}-{version}/{lib}_locomo_grades.json"
    results_dir = f"results/locomo/{lib}-{version}"

    try:
        with open(judged_path) as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"❌ Input file not found: {judged_path}")
        raise SystemExit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {judged_path}: {e}")
        raise SystemExit(1)

    # Calculate scores
    results = calculate_scores(data)

    # Load ingestion stats for add_duration_ms
    ingestion_stats_path = f"{results_dir}/{lib}_locomo_ingestion_stats.json"
    if os.path.exists(ingestion_stats_path):
        with open(ingestion_stats_path) as sf:
            ingestion_stats = json.load(sf)
        add_values = add_duration_values(ingestion_stats)
        if add_values:
            results["metrics"]["duration"]["add_duration_ms"] = float(np.mean(add_values))
            results["metrics"]["duration"]["add_duration_ms_p50"] = float(np.percentile(add_values, 50))
            results["metrics"]["duration"]["add_duration_ms_p95"] = float(np.percentile(add_values, 95))

    results["pipeline_status"] = load_pipeline_status(results_dir, lib)

    # Output the result to a file
    results = convert_numpy_types(results)
    atomic_json_dump(results, grade_path, indent=4)

    # Save results to Excel
    excel_path = f"{results_dir}/{lib}_locomo_results.xlsx"
    save_to_excel(results, excel_path)

    # Print the LLM-as-a-Judge score to match the formatting in locomo_eval.py
    print("\n=== Metric Calculation Complete ===")
    total = sum(results["category_scores"][cat]["total"] for cat in results["category_scores"])
    print(
        f"LLM-as-a-Judge score: {results['metrics']['llm_judge_score']:.4f} ± {results['metrics']['llm_judge_std']:.4f}"
    )
    print(f"Total questions evaluated: {total}")

    # Print duration percentiles for overall metrics
    print("\n=== Duration Metrics ===")
    for metric in ["search_duration_ms", "add_duration_ms"]:
        dur = results["metrics"].get("duration", {})
        if metric in dur:
            print(f"{metric} (avg): {dur[metric]:.2f} ms")
            print(f"{metric} (P50): {dur.get(f'{metric}_p50', 0):.2f} ms")
            print(f"{metric} (P95): {dur.get(f'{metric}_p95', 0):.2f} ms")

    print(f"\nResults have been written to {grade_path}")
    print(f"Excel report has been saved to {excel_path}")


if __name__ == "__main__":
    main()

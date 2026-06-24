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
from longmemeval.lme_common import record_status


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
        "search": f"{lib}_lme_search_status.json",
        "answer": f"{lib}_lme_response_status.json",
        "eval": f"{lib}_lme_eval_status.json",
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
        status[stage] = {
            "status_counts": data.get("status_counts", {}),
            "failed_units": len(data.get("failed_users", [])),
            "failed_users": len(data.get("failed_users", [])),
            "skipped_records": len(data.get("skipped_records", [])),
        }
    return status


def save_to_excel(results, output_path):
    combined_data = []
    overall_row = {"category": "overall"}
    overall_row["llm_judge_score"] = results["metrics"]["llm_judge_score"]
    overall_row["llm_judge_std"] = results["metrics"]["llm_judge_std"]
    for metric, value in results["metrics"]["lexical"].items():
        overall_row[metric] = value if value is not None else "-"
    for metric, value in results["metrics"]["semantic"].items():
        overall_row[metric] = value if value is not None else "-"
    overall_row["context_tokens"] = results["metrics"]["context_tokens"]
    for metric, value in results["metrics"]["duration"].items():
        overall_row[metric] = value
    combined_data.append(overall_row)
    for _, scores in results["category_scores"].items():
        category_row = {"category": scores["category_name"]}
        category_row["llm_judge_score"] = scores["llm_judge_score"]
        category_row["llm_judge_std"] = scores["llm_judge_std"]
        for metric, value in scores["lexical"].items():
            category_row[metric] = value if value is not None else "-"
        for metric, value in scores["semantic"].items():
            category_row[metric] = value if value is not None else "-"
        category_row["context_tokens"] = scores["context_tokens"]
        for metric, value in scores["duration"].items():
            category_row[metric] = value
        combined_data.append(category_row)
    pd.DataFrame(combined_data).to_excel(output_path, sheet_name="Metrics", index=False)
    print(f"Excel file saved to: {output_path}")


def calculate_scores(data, grade_path, output_path):
    category_scores, category_question_count = {}, {}
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
    category_metrics, user_metrics = {}, {}
    all_judgment_keys = set()
    judgment_run_scores = {}

    for q in data.values():
        if "llm_judgments" in q:
            all_judgment_keys.update(q["llm_judgments"].keys())
    for k in all_judgment_keys:
        judgment_run_scores[k] = []

    for _, (user, q) in enumerate(data.items()):
        if record_status(q) != "success":
            continue
        user_metrics[user] = {
            "total": 0,
            "llm_judge_score": 0,
            "llm_judge_std": 0,
            "judgment_run_scores": {k: [] for k in all_judgment_keys},
            "lexical": {m: [] for m in overall_metrics["lexical"]},
            "semantic": {m: [] for m in overall_metrics["semantic"]},
            "context_tokens": [],
            "duration": {m: [] for m in overall_metrics["duration"]},
        }
        if "llm_judgments" in q:
            for k, v in q["llm_judgments"].items():
                score = 1 if v else 0
                judgment_run_scores[k].append(score)
                user_metrics[user]["judgment_run_scores"][k].append(score)
        cat = q["category"]
        if cat not in category_scores:
            category_scores[cat] = {
                "total": 0,
                "category_name": cat,
                "judgment_run_scores": {k: [] for k in all_judgment_keys},
            }
            category_metrics[cat] = {
                "lexical": {m: [] for m in overall_metrics["lexical"]},
                "semantic": {m: [] for m in overall_metrics["semantic"]},
                "context_tokens": [],
                "duration": {m: [] for m in overall_metrics["duration"]},
            }
            category_question_count[cat] = 0
        category_scores[cat]["total"] += 1
        category_question_count[cat] += 1
        if "llm_judgments" in q:
            for k, v in q["llm_judgments"].items():
                score = 1 if v else 0
                category_scores[cat]["judgment_run_scores"][k].append(score)
        nlp = q.get("nlp_metrics", {})
        for m in overall_metrics["lexical"]:
            v = nlp.get("lexical", {}).get(m)
            if v is not None:
                overall_metrics["lexical"][m].append(v)
                category_metrics[cat]["lexical"][m].append(v)
                user_metrics[user]["lexical"][m].append(v)
        for m in overall_metrics["semantic"]:
            v = nlp.get("semantic", {}).get(m)
            if v is not None:
                overall_metrics["semantic"][m].append(v)
                category_metrics[cat]["semantic"][m].append(v)
                user_metrics[user]["semantic"][m].append(v)
        ct = nlp.get("context_tokens")
        if ct is not None:
            overall_metrics["context_tokens"].append(ct)
            category_metrics[cat]["context_tokens"].append(ct)
            user_metrics[user]["context_tokens"].append(ct)
        for m in overall_metrics["duration"]:
            v = q.get(m)
            if v is not None:
                overall_metrics["duration"][m].append(v)
                category_metrics[cat]["duration"][m].append(v)
                user_metrics[user]["duration"][m].append(v)
        user_metrics[user]["total"] = 1
        judgment_avgs = [
            np.mean(scores)
            for scores in user_metrics[user]["judgment_run_scores"].values()
            if scores
        ]
        user_metrics[user]["llm_judge_score"] = np.mean(judgment_avgs) if judgment_avgs else 0.0
        user_metrics[user]["llm_judge_std"] = (
            np.std(judgment_avgs) if len(judgment_avgs) > 1 else 0.0
        )
        for group in ["lexical", "semantic"]:
            for m in user_metrics[user][group]:
                vals = user_metrics[user][group][m]
                user_metrics[user][group][m] = float(np.mean(vals)) if vals else None
        user_metrics[user]["context_tokens"] = (
            np.mean(user_metrics[user]["context_tokens"])
            if user_metrics[user]["context_tokens"]
            else 0.0
        )
        for m in list(user_metrics[user]["duration"].keys()):
            vals = user_metrics[user]["duration"][m]
            if vals:
                user_metrics[user]["duration"][m] = np.mean(vals)
                user_metrics[user]["duration"][f"{m}_p50"] = np.percentile(vals, 50)
                user_metrics[user]["duration"][f"{m}_p95"] = np.percentile(vals, 95)
            else:
                user_metrics[user]["duration"][m] = 0.0
                user_metrics[user]["duration"][f"{m}_p50"] = 0.0
                user_metrics[user]["duration"][f"{m}_p95"] = 0.0

    judgment_run_averages = [np.mean(scores) for scores in judgment_run_scores.values() if scores]
    llm_judge_score = np.mean(judgment_run_averages) if judgment_run_averages else 0.0
    llm_judge_std = np.std(judgment_run_averages) if len(judgment_run_averages) > 1 else 0.0

    category_overall_scores = {}
    for cat, score_data in category_scores.items():
        cat_judgment_avgs = [
            np.mean(scores) for scores in score_data["judgment_run_scores"].values() if scores
        ]
        category_overall_scores[cat] = {
            "category_name": score_data["category_name"],
            "llm_judge_score": np.mean(cat_judgment_avgs) if cat_judgment_avgs else 0.0,
            "llm_judge_std": np.std(cat_judgment_avgs) if len(cat_judgment_avgs) > 1 else 0.0,
            "total": score_data["total"],
            "lexical": {},
            "semantic": {},
            "duration": {},
            "context_tokens": 0.0,
        }
        for group in ["lexical", "semantic"]:
            for m in category_metrics[cat][group]:
                vals = category_metrics[cat][group][m]
                category_overall_scores[cat][group][m] = float(np.mean(vals)) if vals else None
        category_overall_scores[cat]["context_tokens"] = (
            np.mean(category_metrics[cat]["context_tokens"])
            if category_metrics[cat]["context_tokens"]
            else 0.0
        )
        for m in list(category_metrics[cat]["duration"].keys()):
            vals = category_metrics[cat]["duration"][m]
            if vals:
                category_overall_scores[cat]["duration"][m] = np.mean(vals)
                category_overall_scores[cat]["duration"][f"{m}_p50"] = np.percentile(vals, 50)
                category_overall_scores[cat]["duration"][f"{m}_p95"] = np.percentile(vals, 95)
            else:
                category_overall_scores[cat]["duration"][m] = 0.0
                category_overall_scores[cat]["duration"][f"{m}_p50"] = 0.0
                category_overall_scores[cat]["duration"][f"{m}_p95"] = 0.0

    overall_metric_averages = {
        "llm_judge_score": llm_judge_score,
        "llm_judge_std": llm_judge_std,
        "lexical": {},
        "semantic": {},
        "context_tokens": 0.0,
        "duration": {},
    }
    for group in ["lexical", "semantic"]:
        for m in overall_metrics[group]:
            vals = overall_metrics[group][m]
            overall_metric_averages[group][m] = float(np.mean(vals)) if vals else None
    overall_metric_averages["context_tokens"] = (
        np.mean(overall_metrics["context_tokens"]) if overall_metrics["context_tokens"] else 0.0
    )
    for m in list(overall_metrics["duration"].keys()):
        vals = overall_metrics["duration"][m]
        if vals:
            overall_metric_averages["duration"][m] = np.mean(vals)
            overall_metric_averages["duration"][f"{m}_p50"] = np.percentile(vals, 50)
            overall_metric_averages["duration"][f"{m}_p95"] = np.percentile(vals, 95)
        else:
            overall_metric_averages["duration"][m] = 0.0
            overall_metric_averages["duration"][f"{m}_p50"] = 0.0
            overall_metric_averages["duration"][f"{m}_p95"] = 0.0

    results = {
        "metrics": overall_metric_averages,
        "category_scores": category_overall_scores,
        "user_scores": user_metrics,
    }

    # Load ingestion stats for add_duration_ms
    ingestion_stats_path = os.path.join(os.path.dirname(grade_path),
        f"{os.path.basename(grade_path).split('_lme_')[0]}_lme_ingestion_stats.json")
    if os.path.exists(ingestion_stats_path):
        with open(ingestion_stats_path) as sf:
            ingestion_stats = json.load(sf)
        add_values = add_duration_values(ingestion_stats)
        if add_values:
            results["metrics"]["duration"]["add_duration_ms"] = float(np.mean(add_values))
            results["metrics"]["duration"]["add_duration_ms_p50"] = float(
                np.percentile(add_values, 50)
            )
            results["metrics"]["duration"]["add_duration_ms_p95"] = float(
                np.percentile(add_values, 95)
            )

    results_dir = os.path.dirname(grade_path)
    lib_name = os.path.basename(grade_path).split("_lme_")[0]
    results["pipeline_status"] = load_pipeline_status(results_dir, lib_name)
    results = convert_numpy_types(results)
    atomic_json_dump(results, grade_path, indent=4)
    save_to_excel(results, output_path)

    print("\n=== Metric Calculation Complete ===")
    total = sum(results["category_scores"][cat]["total"] for cat in results["category_scores"])
    judge_score = results["metrics"]["llm_judge_score"]
    judge_std = results["metrics"]["llm_judge_std"]
    print(f"LLM-as-a-Judge score: {judge_score:.4f} ± {judge_std:.4f}")
    print(f"Total questions evaluated: {total}")
    print("\n=== Duration Metrics ===")
    for m in ["search_duration_ms", "add_duration_ms"]:
        dur = results["metrics"].get("duration", {})
        if m in dur:
            print(f"{m} (avg): {dur[m]:.2f} ms")
            print(f"{m} (P50): {dur.get(f'{m}_p50', 0):.2f} ms")
            print(f"{m} (P95): {dur.get(f'{m}_p95', 0):.2f} ms")
    print(f"\nResults written to {grade_path}")
    print(f"Excel report saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LongMemeval Analysis Eval Metric Script")
    parser.add_argument(
        "--lib",
        type=str,
        choices=SUPPORTED_LIBS,
        default=DEFAULT_LIB,
    )
    parser.add_argument(
        "--version", type=str, default="default", help="Version of the evaluation framework."
    )
    args = parser.parse_args()
    lib, version = args.lib, args.version
    judged_path = f"results/lme/{lib}-{version}/{lib}_lme_judged.json"
    grade_path = f"results/lme/{lib}-{version}/{lib}_lme_grades.json"
    output_path = f"results/lme/{lib}-{version}/{lib}_lme_results.xlsx"
    try:
        with open(judged_path) as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"❌ Input file not found: {judged_path}")
        raise SystemExit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {judged_path}: {e}")
        raise SystemExit(1)
    calculate_scores(data, grade_path, output_path)

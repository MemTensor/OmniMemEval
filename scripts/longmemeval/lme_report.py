"""LongMemEval experiment report — delegates shared logic to report_base."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.report_base import BenchmarkReport, render_latency, report_main


class LMEReport(BenchmarkReport):
    benchmark_name = "LongMemEval"
    results_prefix = "lme"
    grades_suffix = "lme_grades"
    default_script = "run_lme_eval.sh"
    config_params = (
        "WORKERS", "LLM_WORKERS", "TOPK", "NUM_RUNS", "SAVE_MODEL_INPUT",
        "STREAMING", "START_IDX", "END_IDX", "ALLOW_EMPTY_SEARCH",
        "SKIP_FAILED_SEARCH", "SKIP_FAILED_ANSWER", "SKIP_FAILED_JUDGE",
        "SKIP_FAILED_STREAMING",
    )

    def render_scores(self, lines, grades):
        metrics = grades.get("metrics", {})
        category_scores = grades.get("category_scores", {})

        lines.append("## Evaluation Results")
        lines.append("")

        lines.append("### Overall Scores")
        lines.append("")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")

        judge_score = metrics.get("llm_judge_score", 0)
        judge_std = metrics.get("llm_judge_std", 0)
        lines.append(f"| LLM-as-Judge | {judge_score:.4f} ± {judge_std:.4f} |")

        lexical = metrics.get("lexical", {})
        for key, label in [
            ("f1", "F1"), ("rouge1_f", "ROUGE-1"), ("rouge2_f", "ROUGE-2"),
            ("rougeL_f", "ROUGE-L"), ("bleu1", "BLEU-1"), ("bleu4", "BLEU-4"),
            ("meteor", "METEOR"),
        ]:
            if key in lexical:
                v = lexical[key]
                lines.append(f"| {label} | {v:.4f} |" if v is not None else f"| {label} | - |")

        semantic = metrics.get("semantic", {})
        for key, label in [("bert_f1", "BERTScore F1"), ("similarity", "Semantic Similarity")]:
            if key in semantic:
                v = semantic[key]
                lines.append(f"| {label} | {v:.4f} |" if v is not None else f"| {label} | - |")

        lines.append("")

        if category_scores:
            lines.append("### Category Breakdown")
            lines.append("")
            lines.append("| Category | LLM-Judge | Questions |")
            lines.append("|----------|-----------|-----------|")
            for cat_id in sorted(category_scores.keys(), key=lambda x: str(x)):
                cat = category_scores[cat_id]
                name = cat.get("category_name", f"Category {cat_id}")
                lines.append(
                    f"| {name} | {cat.get('llm_judge_score', 0):.4f} | "
                    f"{cat.get('total', 0)} |"
                )
            lines.append("")

        pipeline_status = grades.get("pipeline_status", {})
        if pipeline_status:
            lines.append("### Pipeline Status")
            lines.append("")
            lines.append("| Stage | Failed Users | Skipped Records | Status Counts |")
            lines.append("|-------|--------------|-----------------|---------------|")
            for stage in ("search", "answer", "eval"):
                data = pipeline_status.get(stage)
                if not data:
                    continue
                counts = data.get("status_counts") or {}
                count_text = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "-"
                lines.append(
                    f"| {stage} | {data.get('failed_users', 0)} | "
                    f"{data.get('skipped_records', 0)} | {count_text} |"
                )
            lines.append("")

        render_latency(lines, metrics.get("duration", {}))

    def extract_dingtalk_data(self, grades):
        metrics = grades.get("metrics", {})
        cat_raw = grades.get("category_scores", {})
        cats = []
        for cat_id in sorted(cat_raw.keys(), key=lambda x: str(x)):
            cat = cat_raw[cat_id]
            cats.append({
                "name": cat.get("category_name", f"Category {cat_id}"),
                "score": cat.get("llm_judge_score", 0),
                "count": cat.get("total", 0),
            })
        return metrics.get("llm_judge_score", 0), metrics.get("llm_judge_std", 0), cats


if __name__ == "__main__":
    report_main(LMEReport())

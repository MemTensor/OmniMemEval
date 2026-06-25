"""Shared helpers and skeleton for benchmark report generation.

Each benchmark report script defines a small config dict and two callables
(``render_scores`` / ``extract_dingtalk_data``), then delegates to the
functions here for everything else — header, config table, token usage,
latency, CLI entry-point, and DingTalk notification.

Quick-start for adding a new benchmark report::

    from utils.report_base import BenchmarkReport, report_main

    class MyBenchReport(BenchmarkReport):
        benchmark_name = "MyBench"
        results_prefix = "mybench"          # results/{prefix}/{lib}-{version}
        grades_suffix  = "mybench_grades"   # {lib}_{suffix}.json
        default_script = "run_mybench_eval.sh"
        config_params  = ("WORKERS", "TOPK")

        def render_scores(self, lines, grades):
            ...  # append benchmark-specific markdown to *lines*

        def extract_dingtalk_data(self, grades):
            return overall_score, overall_std, category_scores_list

    if __name__ == "__main__":
        report_main(MyBenchReport())
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Any


# ── Shared helpers ────────────────────────────────────────────────────────────


def fmt_num(n: float | int) -> str:
    if isinstance(n, float):
        return f"{n:,.1f}"
    return f"{n:,}"


def load_json(path: str) -> dict | list | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠ Failed to load {path}: {e}")
        return None


def load_experiment_config(results_dir: str) -> dict[str, str]:
    """Parse ``experiment_config.sh`` into a dict."""
    path = os.path.join(results_dir, "experiment_config.sh")
    if not os.path.exists(path):
        return {}
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            m = re.match(r'^([A-Za-z_][A-Za-z_0-9]*)=(.*)$', line)
            if m:
                key, val = m.group(1), m.group(2)
                cfg[key] = val.strip('"').strip("'")
    return cfg


def load_env_snapshot(results_dir: str) -> list[tuple[str, str]]:
    """Read ``snapshot_eval.env``, return list of (key, value) excluding secrets."""
    path = os.path.join(results_dir, "snapshot_eval.env")
    if not os.path.exists(path):
        return []
    excluded_keys = {"ANSWER_BASE_URL", "EVAL_BASE_URL"}
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r'^([A-Z_][A-Z_0-9]*)=(.+)$', line)
            if m:
                key, val = m.group(1), m.group(2)
                if key in excluded_keys:
                    continue
                if any(s in key for s in ("_KEY", "_SECRET", "_TOKEN", "_PASSWORD")):
                    continue
                val = val.split("#")[0].strip().strip('"').strip("'")
                entries.append((key, val))
    return entries


def load_token_usage(results_dir: str) -> dict[str, Any]:
    """Load all ``token_usage_*.json`` files, return merged module dict."""
    modules = {}
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("token_usage_") and fname.endswith(".json"):
            data = load_json(os.path.join(results_dir, fname))
            if data:
                for mod, stats in data.get("modules", {}).items():
                    modules[mod] = stats
    return modules


def avg_prompt_tokens(token_modules: dict[str, Any]) -> float | None:
    """Return ANSWER average prompt tokens per API call, or ``None``."""
    answer = token_modules.get("ANSWER")
    if not answer:
        return None
    call_count = answer.get("call_count") or 0
    if call_count <= 0:
        return None
    return answer.get("prompt_tokens", 0) / call_count


# ── Reusable report sections ─────────────────────────────────────────────────


def render_header(lines: list[str], lib: str, version: str, now: str) -> None:
    lines.append(f"# OmniMemEval Experiment Report — {lib}-{version}")
    lines.append("")
    lines.append(f"> Generated: {now}")
    lines.append("")


def render_config(lines: list[str], cfg: dict[str, str], benchmark_name: str, lib: str, version: str, config_params: tuple[str, ...],
                  default_script: str, results_dir: str) -> None:
    lines.append("## Experiment Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Benchmark | {benchmark_name} |")
    lines.append(f"| Memory Framework | {lib} |")
    lines.append(f"| Version | {version} |")
    if cfg.get("RUN_TIMESTAMP"):
        lines.append(f"| Run Time | {cfg['RUN_TIMESTAMP']} |")
    if cfg.get("git_commit"):
        branch = cfg.get("git_branch", "—")
        dirty = cfg.get("git_dirty", "0")
        dirty_mark = " (dirty)" if dirty != "0" else ""
        lines.append(f"| Git | `{cfg['git_commit']}` ({branch}){dirty_mark} |")
    for k in config_params:
        if k in cfg:
            lines.append(f"| {k} | {cfg[k]} |")
    lines.append(f"| Results Directory | `{results_dir}` |")
    lines.append("")

    script_name = cfg.get("SCRIPT_NAME", default_script)
    lines.append("**Reproduce:**")
    lines.append("")
    lines.append("```bash")
    lines.append(f"./scripts/{script_name} --replay {results_dir}")
    lines.append("```")
    lines.append("")


def render_env_snapshot(lines: list[str], results_dir: str) -> None:
    env_entries = load_env_snapshot(results_dir)
    if env_entries:
        lines.append("### Environment Variables (snapshot_eval.env)")
        lines.append("")
        lines.append("| Variable | Value |")
        lines.append("|----------|-------|")
        for k, v in env_entries:
            lines.append(f"| `{k}` | `{v}` |")
        lines.append("")


def render_token_usage(lines: list[str], results_dir: str) -> None:
    token_modules = load_token_usage(results_dir)
    answer_stats = token_modules.get("ANSWER")
    lines.append("## Context Tokens")
    lines.append("")
    lines.append(
        "> Context Tokens are answer-stage prompt/input tokens. They are used "
        "as an approximate proxy for retrieved search-result length and should "
        "be compared only under the same benchmark, answer model, and prompt "
        "template."
    )
    lines.append("")
    if answer_stats:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        model = answer_stats.get("model") or "—"
        lines.append(f"| Answer Model | {model} |")
        cc = answer_stats.get("call_count", 0)
        pt = answer_stats.get("prompt_tokens", 0)
        estimated_calls = answer_stats.get("estimated_prompt_call_count", 0)
        lines.append(f"| Answer API Calls | {fmt_num(cc)} |")
        n = cc or 1
        lines.append(f"| Context Tokens (avg) | {pt/n:.1f} |")
        if estimated_calls:
            lines.append(f"| Context Token Estimate Fallback Calls | {fmt_num(estimated_calls)} |")
        lines.append("")
    else:
        lines.append("> No ANSWER context-token data found.")
        lines.append("")


def render_latency(lines: list[str], duration: dict[str, Any] | None) -> None:
    """Render Add / Search latency table from a ``duration`` dict.

    Supports flat duration dictionaries such as
    ``{add_duration_ms, add_duration_ms_p50, ...}``.
    """
    if not duration:
        return
    lines.append("### Latency Metrics")
    lines.append("")
    lines.append("| Metric | Avg (ms) | P50 (ms) | P95 (ms) |")
    lines.append("|--------|----------|----------|----------|")

    flat_keys = [("add_duration_ms", "Add Latency"),
                 ("search_duration_ms", "Search Latency")]
    nested_keys = [("add_duration", "Add Latency"),
                   ("search_duration", "Search Latency")]

    if any(k in duration for k, _ in flat_keys):
        for key, label in flat_keys:
            if key in duration:
                avg = duration[key]
                p50 = duration.get(f"{key}_p50", 0)
                p95 = duration.get(f"{key}_p95", 0)
                lines.append(f"| {label} | {avg:.1f} | {p50:.1f} | {p95:.1f} |")
            else:
                lines.append(f"| {label} | - | - | - |")
    else:
        for key, label in nested_keys:
            d = duration.get(key)
            if d:
                lines.append(f"| {label} | {d.get('mean', 0):.1f} | {d.get('p50', 0):.1f} | {d.get('p95', 0):.1f} |")
            else:
                lines.append(f"| {label} | - | - | - |")
    lines.append("")


# ── BenchmarkReport base class ───────────────────────────────────────────────


class BenchmarkReport:
    """Override the class attributes and two methods to create a new report."""

    benchmark_name: str = ""
    results_prefix: str = ""
    grades_suffix: str = ""
    default_script: str = ""
    config_params: tuple[str, ...] = ("WORKERS", "TOPK")
    dingtalk_metric_name: str = "LLM-as-Judge"

    def results_dir(self, lib: str, version: str) -> str:
        return f"results/{self.results_prefix}/{lib}-{version}"

    def grades_path(self, results_dir: str, lib: str) -> str:
        return os.path.join(results_dir, f"{lib}_{self.grades_suffix}.json")

    # ── Methods to override ───────────────────────────────────────────────

    def render_scores(self, lines: list[str], grades: dict) -> None:
        """Append benchmark-specific evaluation sections to *lines*.

        Called only when grades file exists and loads successfully.
        """
        raise NotImplementedError

    def extract_dingtalk_data(self, grades: dict) -> tuple[float, float, list[dict[str, Any]]]:
        """Return ``(overall_score, overall_std, category_scores)`` for DingTalk.

        *category_scores* is a list of ``{"name": str, "score": float, "count": int}``.
        """
        raise NotImplementedError

    # ── Concrete methods (shared logic) ───────────────────────────────────

    def generate_report(self, results_dir: str, lib: str, version: str) -> str:
        lines = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cfg = load_experiment_config(results_dir)

        render_header(lines, lib, version, now)
        render_config(lines, cfg, self.benchmark_name, lib, version,
                      self.config_params, self.default_script, results_dir)
        render_env_snapshot(lines, results_dir)
        render_token_usage(lines, results_dir)

        grades = load_json(self.grades_path(results_dir, lib))
        if not grades:
            lines.append("## Evaluation Results")
            lines.append("")
            lines.append(f"> No evaluation results found (`{os.path.basename(self.grades_path(results_dir, lib))}`)")
            lines.append("")
            return "\n".join(lines)

        self.render_scores(lines, grades)
        return "\n".join(lines)

    def notify_dingtalk(self, results_dir: str, lib: str, version: str) -> None:
        from utils.env import load_env
        load_env()
        from utils.dingtalk import send_eval_result

        cfg = load_experiment_config(results_dir)
        token_modules = load_token_usage(results_dir)
        grades = load_json(self.grades_path(results_dir, lib))
        if not grades:
            print("  [DingTalk] Skipped: no evaluation result data")
            return

        overall_score, overall_std, category_scores = self.extract_dingtalk_data(grades)

        send_eval_result(
            benchmark=self.benchmark_name,
            framework=lib,
            version=version,
            run_time=cfg.get("RUN_TIMESTAMP", "—"),
            overall_score=overall_score,
            overall_std=overall_std,
            category_scores=category_scores,
            context_tokens=avg_prompt_tokens(token_modules),
            metric_name=self.dingtalk_metric_name,
        )


# ── CLI entry-point ──────────────────────────────────────────────────────────


def report_main(bench: BenchmarkReport):
    """Generic CLI entry-point shared by all benchmark report scripts."""
    parser = argparse.ArgumentParser(
        description=f"Generate unified {bench.benchmark_name} experiment report"
    )
    parser.add_argument("--lib", type=str, required=True)
    parser.add_argument("--version", type=str, default="default")
    parser.add_argument(
        "--notify", action="store_true",
        help="Send evaluation summary to DingTalk after report generation",
    )
    args = parser.parse_args()

    results_dir = bench.results_dir(args.lib, args.version)
    if not os.path.isdir(results_dir):
        print(f"Error: results directory not found: {results_dir}")
        sys.exit(1)

    try:
        report = bench.generate_report(results_dir, args.lib, args.version)
    except Exception as e:
        print(f"❌ Failed to generate report: {e}")
        raise SystemExit(1)

    report_path = os.path.join(results_dir, "exp_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nExperiment report → {report_path}")
    print("=" * 66)
    print(report)

    if args.notify:
        bench.notify_dingtalk(results_dir, args.lib, args.version)

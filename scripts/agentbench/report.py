from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentbench.config import write_json


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _candidate_run_dirs(results_root: Path, agent: str, domain: str, version: str) -> list[Path]:
    return [
        results_root / f"{agent}-{version}-{domain}",
        results_root / f"{agent}-{domain}-{version}",
    ]


def _summary_path(results_root: Path, agent: str, domain: str, version: str) -> Path:
    checked = []
    for run_dir in _candidate_run_dirs(results_root, agent, domain, version):
        summary = run_dir / "test" / "summary.json"
        checked.append(str(summary))
        if summary.exists():
            return summary
    raise FileNotFoundError(
        "Missing AgentBench summary for "
        f"agent={agent} domain={domain} version={version}. Checked: {', '.join(checked)}"
    )


def _mean(values: list[float], *, digits: int = 4) -> float:
    if not values:
        return 0.0
    return round(statistics.mean(values), digits)


def _metric_mean(value: Any, default: float = 0.0) -> float:
    if isinstance(value, dict):
        value = value.get("mean", default)
    return float(value if value is not None else default)


def _avg_tokens(summary: dict[str, Any]) -> int:
    value = summary.get("avg_tokens", 0)
    if isinstance(value, dict):
        value = value.get("total", 0)
    return int(value or 0)


def _avg_chars(summary: dict[str, Any]) -> float:
    value = summary.get("response_chars")
    if isinstance(value, dict):
        value = value.get("avg")
    if value is None:
        value = summary.get("avg_chars", 0.0)
    return float(value or 0.0)


def _domain_report(summary_path: Path, domain: str) -> dict[str, Any]:
    summary = _read_json(summary_path)
    run_dir = summary_path.parents[1]
    pass_at_1 = _metric_mean(summary.get("pass@1", 0.0))
    infra_pass_at_1 = _metric_mean(
        (summary.get("infra_excluded") or {}).get("pass@1"),
        pass_at_1,
    )
    avg_turns = float(summary.get("averages", {}).get("avg_turns", summary.get("avg_turns", 0.0)) or 0.0)
    run = {
        "run": 1,
        "job": run_dir.name,
        "run_dir": str(run_dir),
        "summary": str(summary_path),
        "tasks": int(summary.get("tasks", 0) or 0),
        "total_trials": int(summary.get("total_trials", 0) or 0),
        "pass_at_1": pass_at_1,
        "infra_excluded_pass_at_1": infra_pass_at_1,
        "avg_pass_rate": float(summary.get("avg_pass_rate", pass_at_1) or 0.0),
        "avg_reward": float(summary.get("averages", {}).get("avg_reward", 0.0) or 0.0),
        "avg_turns": avg_turns,
        "avg_elapsed_sec": float(summary.get("avg_elapsed_sec", 0.0) or 0.0),
        "avg_chars": _avg_chars(summary),
        "avg_tokens": _avg_tokens(summary),
        "failure_breakdown": summary.get("failure_counts", summary.get("failure_breakdown", {})),
        "domain_metrics": summary.get("domain_metrics", {}),
    }
    return {
        "runs": [run],
        "average_pass_at_1": round(pass_at_1, 4),
        "average_infra_excluded_pass_at_1": round(infra_pass_at_1, 4),
        "average_pass_rate": round(run["avg_pass_rate"], 4),
        "average_reward": round(run["avg_reward"], 4),
        "average_turns": round(avg_turns, 2),
        "average_elapsed_sec": round(run["avg_elapsed_sec"], 1),
        "average_chars": round(run["avg_chars"], 1),
        "average_tokens": run["avg_tokens"],
        "tasks": run["tasks"],
        "total_trials": run["total_trials"],
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# OmniMemEval AgentBench Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Agent: `{report['agent']}`",
        f"- Version: `{report['version']}`",
        f"- Runs per domain: `{report['runs']}`",
        f"- Parallel: `{report['parallel']}`",
        "",
        "| Domain | Pass@1 | Avg Pass Rate | Avg Reward | Avg Turns | Avg Elapsed Sec | Avg Chars | Avg Tokens | Tasks | Run Dir |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for domain, item in report["domains"].items():
        run = item["runs"][0]
        lines.append(
            f"| `{domain}` | {item['average_pass_at_1'] * 100:.1f}% | "
            f"{item['average_pass_rate'] * 100:.1f}% | {item['average_reward']:.4f} | "
            f"{item['average_turns']:.2f} | {item['average_elapsed_sec']:.1f} | "
            f"{item['average_chars']:.1f} | {item['average_tokens']} | {item['tasks']} | "
            f"`{run['run_dir']}` |"
        )

    overall = report.get("overall") or {}
    lines.extend([
        "",
        "## Overall",
        "",
        f"- Average pass@1: {overall.get('average_pass_at_1', 0.0) * 100:.1f}%",
        f"- Average chars: {overall.get('average_chars', 0.0):.1f}",
        f"- Total tasks: {overall.get('tasks', 0)}",
        f"- Total trials: {overall.get('total_trials', 0)}",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_all_domains_report(
    *,
    results_root: Path | str,
    report_dir: Path | str,
    agent: str,
    version: str,
    domains: list[str],
    trials: int,
    parallel: int,
    report_name: str | None = None,
    plugin: str | None = None,
) -> dict[str, Any]:
    results_root = Path(results_root)
    report_dir = Path(report_dir)
    report_name = report_name or version
    domain_reports = {}
    for domain in domains:
        summary = _summary_path(results_root, agent, domain, version)
        domain_reports[domain] = _domain_report(summary, domain)

    pass_values = [item["average_pass_at_1"] for item in domain_reports.values()]
    task_values = [item["tasks"] for item in domain_reports.values()]
    trial_values = [item["total_trials"] for item in domain_reports.values()]
    token_values = [item["average_tokens"] for item in domain_reports.values()]
    char_values = [item["average_chars"] for item in domain_reports.values()]
    elapsed_values = [item["average_elapsed_sec"] for item in domain_reports.values()]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "plugin": plugin,
        "version": version,
        "job_prefix": report_name,
        "runs": trials,
        "parallel": parallel,
        "results_root": str(results_root),
        "domains": domain_reports,
        "overall": {
            "average_pass_at_1": _mean(pass_values),
            "tasks": sum(task_values),
            "total_trials": sum(trial_values),
            "average_chars": _mean(char_values, digits=1),
            "average_tokens": round(statistics.mean(token_values)) if token_values else 0,
            "average_elapsed_sec": _mean(elapsed_values, digits=1),
        },
    }
    write_json(report_dir / f"{report_name}_report.json", report)
    _write_markdown(report_dir / f"{report_name}_report.md", report)
    return report

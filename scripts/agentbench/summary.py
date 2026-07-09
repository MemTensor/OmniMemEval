from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from agentbench.config import write_json


INFRA_ERROR_PATTERNS = (
    "Assistant reply unavailable due to model error",
    "HTTP 503",
    "503",
    "upstream connect error",
    "LLM request timed out",
    "timed out",
    "GatewayClientRequestError",
    "GatewayTransportError",
    "unauthorized: gateway token missing",
)


def _rate(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) <= 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var / len(values))


def classify_failure(result: dict, reward: float) -> str:
    if reward > 0:
        return "resolved"
    agent_result = result.get("agent_result", {})
    verifier_result = result.get("verifier_result", {})
    error_text = " ".join(
        str(value)
        for value in (
            agent_result.get("error"),
            agent_result.get("stderr"),
            verifier_result.get("error"),
        )
        if value
    )
    if any(pattern in error_text for pattern in INFRA_ERROR_PATTERNS):
        return "infra_error"
    if verifier_result.get("error") == "No patch generated":
        return "no_patch"
    if verifier_result.get("error") == "Patch apply failed":
        return "patch_apply_failed"
    if verifier_result.get("resolved") is False:
        return "resolved_false"
    if result.get("exception_info"):
        return "exception"
    if agent_result.get("completion_status") == "timeout":
        return "timeout"
    if agent_result.get("completion_status") == "error":
        return "agent_error"
    if verifier_result.get("error"):
        return "verifier_error"
    return "failed"


def _load_result(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _text_from_openclaw_message(msg: dict[str, Any]) -> str:
    text_parts = []
    for item in msg.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
            text_parts.append(item["text"])
    return "\n".join(text_parts)


def _is_internal_context_compaction(text: str) -> bool:
    prefix = text.lstrip()[:512].upper()
    return "CONTEXT COMPACTION" in prefix and "REFERENCE ONLY" in prefix


def _final_answer_from_session(trial_dir: Path) -> str:
    session_file = trial_dir / "session.jsonl"
    if not session_file.exists():
        return ""

    final_text = ""
    try:
        with session_file.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if rec.get("type") == "message":
                    msg = rec.get("message", {})
                    if msg.get("role") != "assistant":
                        continue
                    text = _text_from_openclaw_message(msg)
                    if text and not _is_internal_context_compaction(text):
                        final_text = text
                    continue

                if rec.get("role") != "assistant":
                    continue
                if rec.get("tool_calls"):
                    continue
                content = rec.get("content")
                if isinstance(content, str) and content and not _is_internal_context_compaction(content):
                    final_text = content
    except OSError:
        return ""

    return final_text


def _all_assistant_text_from_session(
    trial_dir: Path,
    *,
    include_reasoning_content: bool = False,
) -> str:
    session_file = trial_dir / "session.jsonl"
    if not session_file.exists():
        return ""

    text_parts = []
    try:
        with session_file.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if rec.get("type") == "message":
                    msg = rec.get("message", {})
                    if msg.get("role") != "assistant":
                        continue
                    text = _text_from_openclaw_message(msg)
                    if text and not _is_internal_context_compaction(text):
                        text_parts.append(text)
                    continue

                if rec.get("role") != "assistant":
                    continue
                reasoning = rec.get("reasoning_content")
                if (
                    include_reasoning_content
                    and isinstance(reasoning, str)
                    and reasoning
                    and not _is_internal_context_compaction(reasoning)
                ):
                    text_parts.append(reasoning)
                content = rec.get("content")
                if isinstance(content, str) and content and not _is_internal_context_compaction(content):
                    text_parts.append(content)
    except OSError:
        return ""

    return "\n".join(text_parts)


def response_text_for_char_stats(result: dict[str, Any], trial_dir: Path) -> str:
    """Return assistant-generated text using the EvoAgentBench char-stat policy."""
    response = result.get("agent_result", {}).get("response", "") or ""
    agent = result.get("agent")
    if agent in {"openclaw", "hermes", "hermes-agent"}:
        session_text = _all_assistant_text_from_session(
            trial_dir,
            include_reasoning_content=agent in {"hermes", "hermes-agent"},
        )
        if session_text:
            return session_text
        return response

    session_text = _final_answer_from_session(trial_dir)
    if session_text:
        return session_text
    return response


def _char_summary(values: list[int], passed: list[bool]) -> dict[str, Any]:
    if not values:
        return {
            "avg": 0.0,
            "median": 0.0,
            "pass_avg": 0.0,
            "fail_avg": 0.0,
            "empty": 0,
        }

    pass_values = [value for value, ok in zip(values, passed) if ok]
    fail_values = [value for value, ok in zip(values, passed) if not ok]

    def avg(items: list[int]) -> float:
        return round(sum(items) / len(items), 1) if items else 0.0

    return {
        "avg": avg(values),
        "median": round(float(statistics.median(values)), 1),
        "pass_avg": avg(pass_values),
        "fail_avg": avg(fail_values),
        "empty": sum(1 for value in values if value == 0),
    }


def build_summary(phase_dir: Path, *, trials: int = 1, pass_at: int | None = None, domain=None) -> dict:
    pass_at = pass_at or trials
    if pass_at < 1:
        raise ValueError("pass_at must be >= 1")
    if pass_at > trials:
        raise ValueError(f"pass_at ({pass_at}) cannot exceed trials ({trials})")

    task_results: dict[str, list[dict]] = {}
    domain_metric_inputs = []
    trial_buckets: dict[int, list[dict]] = {}

    for result_file in phase_dir.glob("*/result.json"):
        if "_retry" in result_file.parent.name:
            continue
        result = _load_result(result_file)
        if not result:
            continue
        task_name = result.get("task_name") or result_file.parent.name.split("__trial_")[0]
        verifier_result = result.get("verifier_result", {})
        reward = verifier_result.get("reward", 0.0)
        token_usage = result.get("token_usage", {})
        agent_result = result.get("agent_result", {})
        response_chars = len(response_text_for_char_stats(result, result_file.parent))
        item = {
            "trial": int(result.get("trial") or 1),
            "reward": reward,
            "elapsed": agent_result.get("elapsed_sec", 0.0),
            "response_chars": response_chars,
            "turns": token_usage.get("turns", 0),
            "input_tokens": token_usage.get("input", 0),
            "output_tokens": token_usage.get("output", 0),
            "total_tokens": token_usage.get("total", 0),
            "failure_class": classify_failure(result, reward),
        }
        task_results.setdefault(task_name, []).append(item)
        trial_buckets.setdefault(item["trial"], []).append(item)
        domain_metric_inputs.append(verifier_result)

    threshold = domain.pass_threshold() if domain is not None else 0.0
    per_task = {}
    for task_name, results in sorted(task_results.items()):
        results = sorted(results, key=lambda item: item["trial"])
        passed = sum(1 for item in results if item["reward"] > threshold)
        first_trial_passed = bool(results and results[0]["reward"] > threshold)
        pass_at_results = results[:pass_at]
        pass_at_passed = any(item["reward"] > threshold for item in pass_at_results)
        per_task[task_name] = {
            "trials": len(results),
            "passed": passed,
            "pass@1": 1.0 if first_trial_passed else 0.0,
            f"pass@{pass_at}": 1.0 if pass_at_passed else 0.0,
            "avg_pass_rate": round(passed / len(results), 4) if results else 0.0,
            "avg_reward": round(sum(item["reward"] for item in results) / len(results), 4),
            "avg_elapsed_sec": round(sum(item["elapsed"] for item in results) / len(results), 1),
            "avg_chars": round(sum(item["response_chars"] for item in results) / len(results), 1),
            "avg_tokens": round(sum(item["total_tokens"] for item in results) / len(results)),
            "trial_results": [
                {
                    "trial": item["trial"],
                    "reward": item["reward"],
                    "passed": item["reward"] > threshold,
                    "elapsed_sec": item["elapsed"],
                    "chars": item["response_chars"],
                    "tokens": item["total_tokens"],
                    "failure_class": item["failure_class"],
                }
                for item in results
            ],
            "failure_classes": {
                name: sum(1 for item in results if item["failure_class"] == name)
                for name in sorted({item["failure_class"] for item in results})
            },
        }

    pass_values = [item["pass@1"] for item in per_task.values()]
    pass_mean, pass_se = _rate(pass_values)
    pass_n_values = [item.get(f"pass@{pass_at}", 0.0) for item in per_task.values()]
    pass_n_mean, pass_n_se = _rate(pass_n_values)
    avg_pass_values = [item["avg_pass_rate"] for item in per_task.values()]
    avg_pass_mean, avg_pass_se = _rate(avg_pass_values)
    infra_task_values = [
        item["pass@1"]
        for item in per_task.values()
        if item["failure_classes"].get("infra_error", 0) < item["trials"]
    ]
    infra_excluded_mean, infra_excluded_se = _rate(infra_task_values)
    all_results = [item for results in task_results.values() for item in results]
    total_trials = len(all_results)
    failure_counts = {}
    for item in all_results:
        failure_counts[item["failure_class"]] = failure_counts.get(item["failure_class"], 0) + 1
    response_chars = [item["response_chars"] for item in all_results]
    response_passed = [item["reward"] > threshold for item in all_results]

    trial_results = {}
    for trial_num, results in sorted(trial_buckets.items()):
        total = len(results)
        passed = sum(1 for item in results if item["reward"] > threshold)
        trial_results[str(trial_num)] = {
            "tasks": total,
            "passed": passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "avg_reward": round(sum(item["reward"] for item in results) / total, 4) if total else 0.0,
            "avg_elapsed_sec": round(sum(item["elapsed"] for item in results) / total, 1) if total else 0.0,
            "avg_chars": round(sum(item["response_chars"] for item in results) / total, 1) if total else 0.0,
            "avg_tokens": round(sum(item["total_tokens"] for item in results) / total) if total else 0,
        }

    summary = {
        "phase_dir": str(phase_dir),
        "tasks": len(task_results),
        "trials_per_task": trials,
        "pass_at": pass_at,
        "total_trials": total_trials,
        "pass@1": round(pass_mean, 4),
        "pass@1_stderr": round(pass_se, 4),
        f"pass@{pass_at}": round(pass_n_mean, 4),
        f"pass@{pass_at}_stderr": round(pass_n_se, 4),
        "avg_pass_rate": round(avg_pass_mean, 4),
        "avg_pass_rate_stderr": round(avg_pass_se, 4),
        "trial_results": trial_results,
        "averages": {
            "avg_reward": round(
                sum(item["reward"] for item in all_results) / total_trials, 4
            ) if total_trials else 0.0,
            "avg_turns": round(
                sum(item["turns"] for item in all_results) / total_trials, 1
            ) if total_trials else 0.0,
            "avg_elapsed_sec": round(
                sum(item["elapsed"] for item in all_results) / total_trials, 1
            ) if total_trials else 0.0,
            "avg_chars": round(
                sum(item["response_chars"] for item in all_results) / total_trials, 1
            ) if total_trials else 0.0,
            "avg_tokens": round(
                sum(item["total_tokens"] for item in all_results) / total_trials
            ) if total_trials else 0,
        },
        "avg_elapsed_sec": round(
            sum(item["elapsed"] for item in all_results) / total_trials, 1
        ) if total_trials else 0.0,
        "avg_turns": round(
            sum(item["turns"] for item in all_results) / total_trials, 1
        ) if total_trials else 0.0,
        "avg_chars": round(
            sum(item["response_chars"] for item in all_results) / total_trials, 1
        ) if total_trials else 0.0,
        "avg_tokens": round(
            sum(item["total_tokens"] for item in all_results) / total_trials
        ) if total_trials else 0,
        "response_chars": _char_summary(response_chars, response_passed),
        "failure_counts": failure_counts,
        "infra_excluded": {
            "tasks": len(infra_task_values),
            "excluded_tasks": len(per_task) - len(infra_task_values),
            "pass@1": {
                "mean": round(infra_excluded_mean, 4),
                "stderr": round(infra_excluded_se, 4),
            },
        },
        "per_task": per_task,
        "domain_metrics": domain.aggregate_metrics(domain_metric_inputs) if domain else {},
    }
    write_json(phase_dir / "summary.json", summary)
    write_phase_report(phase_dir / "report.md", summary)
    return summary


def write_phase_report(path: Path, summary: dict) -> None:
    """Write a compact markdown report for one AgentBench phase."""

    pass_at = int(summary.get("pass_at") or summary.get("trials_per_task") or 1)
    pass_at_key = f"pass@{pass_at}"
    lines = [
        "# AgentBench Phase Report",
        "",
        f"- Phase directory: `{summary.get('phase_dir', '')}`",
        f"- Tasks: {summary.get('tasks', 0)}",
        f"- Total trials: {summary.get('total_trials', 0)}",
        f"- Pass@1: {summary.get('pass@1', 0.0):.4f}",
        f"- Pass@1 stderr: {summary.get('pass@1_stderr', 0.0):.4f}",
        f"- Average pass rate: {summary.get('avg_pass_rate', 0.0):.4f}",
        f"- Average elapsed seconds: {summary.get('avg_elapsed_sec', 0.0)}",
        f"- Average chars: {summary.get('avg_chars', 0)}",
        f"- Average tokens: {summary.get('avg_tokens', 0)}",
        "",
        "## Failure Counts",
        "",
    ]
    if pass_at != 1:
        lines.insert(6, f"- Pass@{pass_at}: {summary.get(pass_at_key, 0.0):.4f}")
    failure_counts = summary.get("failure_counts") or {}
    if failure_counts:
        lines.extend(f"- `{name}`: {count}" for name, count in sorted(failure_counts.items()))
    else:
        lines.append("- None")

    domain_metrics = summary.get("domain_metrics") or {}
    if domain_metrics:
        lines.extend(["", "## Domain Metrics", ""])
        for key, value in domain_metrics.items():
            lines.append(f"- `{key}`: {value}")

    trial_results = summary.get("trial_results") or {}
    if trial_results:
        lines.extend([
            "",
            "## Per Trial",
            "",
            "| Trial | Tasks | Passed | Pass Rate | Avg Reward | Avg Elapsed Sec | Avg Chars | Avg Tokens |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for trial, item in sorted(trial_results.items(), key=lambda pair: int(pair[0])):
            lines.append(
                f"| {trial} | {item.get('tasks', 0)} | {item.get('passed', 0)} | "
                f"{item.get('pass_rate', 0.0):.4f} | {item.get('avg_reward', 0.0):.4f} | "
                f"{item.get('avg_elapsed_sec', 0.0)} | {item.get('avg_chars', 0)} | "
                f"{item.get('avg_tokens', 0)} |"
            )

    per_task = summary.get("per_task") or {}
    if per_task:
        lines.extend([
            "",
            "## Per Task",
            "",
            "| Task | Trials | Passed | Pass@1 | Avg Pass Rate | Avg Reward | Avg Elapsed Sec | Avg Chars |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for task, item in sorted(per_task.items()):
            lines.append(
                f"| `{task}` | {item.get('trials', 0)} | {item.get('passed', 0)} | "
                f"{item.get('pass@1', 0.0):.4f} | {item.get('avg_pass_rate', 0.0):.4f} | "
                f"{item.get('avg_reward', 0.0):.4f} | "
                f"{item.get('avg_elapsed_sec', 0.0)} | {item.get('avg_chars', 0)} |"
            )

    path.write_text("\n".join(lines) + "\n")

#!/usr/bin/env python3
"""Run LongMemEval in independent, checkpointed index shards."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


RUNNER = Path("/app/scripts/run_omnimemeval_lme.py")
OUTPUT_DIR = Path(os.getenv("OMNIMEMEVAL_LME_OUTPUT_DIR", "/tmp/omnimemeval/results"))
STAGES = ("clear", "ingest", "search", "answer", "judge")
_children: list[subprocess.Popen] = []


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def partition_cases(start: int, count: int, workers: int) -> list[tuple[int, int]]:
    """Return balanced non-overlapping (start, count) ranges."""
    if start < 0 or count < 1 or workers < 1:
        raise ValueError("start must be >= 0; count and workers must be >= 1")
    workers = min(workers, count)
    quotient, remainder = divmod(count, workers)
    ranges = []
    offset = start
    for worker in range(workers):
        size = quotient + (1 if worker < remainder else 0)
        ranges.append((offset, size))
        offset += size
    return ranges


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _refresh_state(state_path: Path, state: dict, shard_dirs: list[Path]) -> None:
    merged = {}
    for shard_dir in shard_dirs:
        shard = _read_json(shard_dir / "run-state.json", {})
        merged.update(shard.get("cases") or {})
    state["cases"].update(merged)
    _atomic_json(state_path, state)


def _merge_dict_files(shard_dirs: list[Path], filename: str) -> dict:
    result = {}
    for shard_dir in shard_dirs:
        payload = _read_json(shard_dir / filename, {})
        if isinstance(payload, dict):
            result.update(payload)
    return result


def _handle_signal(signum, _frame) -> None:
    for child in _children:
        if child.poll() is None:
            child.terminate()
    for child in _children:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
    raise SystemExit(128 + signum)


def main() -> int:
    start = int(os.getenv("OMNIMEMEVAL_LME_START_INDEX", "0"))
    count = int(os.getenv("OMNIMEMEVAL_LME_CASE_LIMIT", "500"))
    workers = int(os.getenv("OMNIMEMEVAL_LME_WORKERS", "10"))
    if not 0 <= start < 500 or not 1 <= count <= 500 - start:
        raise ValueError("invalid LongMemEval case range")
    if not 1 <= workers <= 10:
        raise ValueError("OMNIMEMEVAL_LME_WORKERS must be between 1 and 10")

    version = os.getenv("OMNIMEMEVAL_LME_VERSION", "hojo_lme_full_e2e")
    ranges = partition_cases(start, count, workers)
    root_config = {
        "version": version,
        "start_index": start,
        "case_limit": count,
        "workers": len(ranges),
        "ranges": [{"start_index": item_start, "case_limit": item_count} for item_start, item_count in ranges],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state_path = OUTPUT_DIR / "run-state.json"
    previous = _read_json(state_path, {})
    if previous and previous.get("config") != root_config:
        raise ValueError("existing evaluation checkpoint config differs; use a new version/output directory")
    state = previous or {
        "schema_version": "omnimemeval-lme-e2e-v1",
        "config": root_config,
        "status": "running",
        "cases": {str(index): {"status": "pending"} for index in range(start, start + count)},
    }
    state["status"] = "running"
    state.setdefault("cases", {})
    _atomic_json(state_path, state)

    shard_dirs = []
    children = []
    for shard_number, (shard_start, shard_count) in enumerate(ranges, start=1):
        shard_dir = OUTPUT_DIR / "shards" / f"worker-{shard_number:02d}"
        shard_dirs.append(shard_dir)
        env = os.environ.copy()
        env["OMNIMEMEVAL_LME_OUTPUT_DIR"] = str(shard_dir)
        env["OMNIMEMEVAL_LME_START_INDEX"] = str(shard_start)
        env["OMNIMEMEVAL_LME_CASE_LIMIT"] = str(shard_count)
        log_path = shard_dir / "worker.log"
        shard_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8")
        child = subprocess.Popen(
            [sys.executable, str(RUNNER)],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children.append((child, log_file, shard_number, shard_start, shard_count))
        _children.append(child)
        print(
            f"[LME] worker {shard_number}/{len(ranges)} started: "
            f"cases {shard_start}-{shard_start + shard_count - 1}, pid={child.pid}",
            flush=True,
        )

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    started = time.monotonic()
    while any(child.poll() is None for child, *_ in children):
        _refresh_state(state_path, state, shard_dirs)
        time.sleep(5)
    _refresh_state(state_path, state, shard_dirs)

    nonzero_workers = []
    for child, log_file, shard_number, shard_start, shard_count in children:
        log_file.close()
        if child.returncode != 0:
            nonzero_workers.append({
                "worker": shard_number,
                "start_index": shard_start,
                "case_limit": shard_count,
                "returncode": child.returncode,
            })
        for index in range(shard_start, shard_start + shard_count):
            if state["cases"].get(str(index), {}).get("status") in (None, "pending", "running"):
                state["cases"][str(index)] = {
                    "status": "failed",
                    "phase": "runner",
                    "error": f"worker {shard_number} exited without final case status",
                }

    judged = _merge_dict_files(shard_dirs, "hojo_lme_judged.json")
    searches = _merge_dict_files(shard_dirs, "hojo_lme_search_results.json")
    answers = _merge_dict_files(shard_dirs, "hojo_lme_responses.json")
    judged_indexes = {record.get("index") for record in judged.values()}
    for index in range(start, start + count):
        case = state["cases"].get(str(index), {})
        if index not in judged_indexes:
            user_id = f"lme_exper_user_{version}_{index}"
            failure_stage = case.get("failed_stage", case.get("phase", "runner"))
            failure_error = case.get("error", "worker exited without a judged result")
            state["cases"][str(index)] = {
                **case,
                "status": "failed",
                "failed_stage": failure_stage,
                "error": failure_error,
            }
            judged[user_id] = {
                "index": index,
                "user_id": user_id,
                "status": "failed",
                "failed_stage": failure_stage,
                "error_type": case.get("error_type", "WorkerError"),
                "error": failure_error,
            }
    completed = [item for item in judged.values() if item.get("status") == "success"]
    correct = sum(bool(item.get("correct")) for item in completed)
    failed = count - len(completed)
    failed_by_stage = {
        stage: sum(1 for item in judged.values() if item.get("status") != "success" and item.get("failed_stage") == stage)
        for stage in STAGES
    }
    failed_by_stage["runner"] = sum(
        1 for item in judged.values() if item.get("status") != "success" and item.get("failed_stage") == "runner"
    )
    categories: dict[str, list[bool]] = {}
    for item in completed:
        categories.setdefault(str(item.get("category") or "unknown"), []).append(bool(item.get("correct")))
    category_scores = {
        name: {"category_name": name, "llm_judge_score": sum(scores) / len(scores), "total": len(scores)}
        for name, scores in categories.items()
    }
    score = correct / len(completed) if completed else None
    shard_reports = [_read_json(shard_dir / "exp_report.json", {}) for shard_dir in shard_dirs]
    source_report = next((item for item in shard_reports if item), {})
    report = {
        "benchmark": "LongMemEval-S",
        "version": version,
        "metric": "OmniMemEval LLM Judge accuracy on valid cases only",
        "score": score,
        "valid_cases": len(completed),
        "total_cases": count,
        "start_index": start,
        "correct": correct,
        "incorrect": len(completed) - correct,
        "failed_cases": failed,
        "failed_by_stage": failed_by_stage,
        "worker_count": len(ranges),
        "workers_with_nonzero_exit": nonzero_workers,
        "dataset_sha256": source_report.get("dataset_sha256"),
        "answer_model": source_report.get("answer_model"),
        "eval_model": source_report.get("eval_model"),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "metrics": {"llm_judge_score": score, "llm_judge_std": 0.0, "duration": {}, "valid_cases": len(completed), "total_cases": count},
        "category_scores": category_scores,
        "pipeline_status": {
            "clear": {"failed_users": failed_by_stage["clear"], "skipped_records": 0},
            "ingest": {"failed_users": failed_by_stage["ingest"], "skipped_records": 0},
            "search": {"failed_users": failed_by_stage["search"], "skipped_records": 0},
            "answer": {"failed_users": failed_by_stage["answer"], "skipped_records": 0},
            "eval": {"failed_users": failed_by_stage["judge"], "skipped_records": 0},
            "runner": {"failed_users": failed_by_stage["runner"], "skipped_records": 0},
        },
    }
    for filename, payload in (
        ("hojo_lme_judged.json", judged),
        ("hojo_lme_search_results.json", searches),
        ("hojo_lme_responses.json", answers),
        ("exp_report.json", report),
    ):
        _atomic_json(OUTPUT_DIR / filename, payload)
    (OUTPUT_DIR / "exp_report.md").write_text(
        "\n".join([
            "# OmniMemEval LongMemEval-S End-to-End Report", "",
            f"- Version: `{version}`",
            f"- LLM-as-a-Judge accuracy (valid cases only): **{score:.4f}**" if score is not None else "- LLM-as-a-Judge accuracy (valid cases only): N/A",
            f"- Valid cases: **{len(completed)}/{count}**",
            f"- Correct / incorrect: **{correct} / {len(completed) - correct}**",
            f"- Failed cases: **{failed}**",
            f"- Workers: **{len(ranges)}**",
            f"- Elapsed: **{report['elapsed_seconds']:.1f}s**", "",
            "Failed cases are excluded from the accuracy denominator.",
        ]) + "\n",
        encoding="utf-8",
    )
    state["status"] = "completed" if len(completed) == count else "failed"
    state["summary"] = report
    _atomic_json(state_path, state)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if len(completed) == count else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[LME] parallel runner fatal: {type(exc).__name__}: {str(exc)}", file=sys.stderr, flush=True)
        raise SystemExit(1)

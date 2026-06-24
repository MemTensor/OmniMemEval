"""Shared LongMemEval pipeline bookkeeping helpers.

This module keeps status labels, result-shape construction, and checkpoint
validation in one place. It does not change retrieval, answer generation,
grading prompts, or metric formulas.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any


STATUS_SUCCESS = "success"
STATUS_SUCCESS_EMPTY = "success_empty"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def user_id_for(version: str, conv_idx: int) -> str:
    return f"lme_exper_user_{version}_{conv_idx}"


def record_status(record: Mapping[str, Any] | None) -> str:
    if not record:
        return STATUS_FAILED
    return str(record.get("status") or STATUS_SUCCESS)


def error_payload(stage: str, exc: BaseException | str) -> dict[str, str]:
    if isinstance(exc, BaseException):
        err_type = type(exc).__name__
        message = str(exc)
    else:
        err_type = "PipelineError"
        message = str(exc)
    return {
        "stage": stage,
        "type": err_type,
        "message": message,
        "timestamp": utc_now_iso(),
    }


def classify_search_status(
    context: str,
    reflect_answer: str | None = None,
    *,
    raw_context: str | None = None,
) -> str:
    if reflect_answer is not None and str(reflect_answer).strip():
        return STATUS_SUCCESS
    search_payload = context if raw_context is None else raw_context
    if str(search_payload or "").strip():
        return STATUS_SUCCESS
    return STATUS_SUCCESS_EMPTY


def search_allowed_statuses(
    *,
    allow_empty_search: bool,
    allow_skipped: bool,
) -> set[str]:
    statuses = {STATUS_SUCCESS}
    if allow_empty_search:
        statuses.add(STATUS_SUCCESS_EMPTY)
    if allow_skipped:
        statuses.add(STATUS_SKIPPED)
    return statuses


def status_counts(records: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        status = record_status(record)
        counts[status] = counts.get(status, 0) + 1
    return counts


def answer_evidences_for_row(row: Mapping[str, Any]) -> list[str]:
    sessions = row["haystack_sessions"]
    answer_session_ids = set(row["answer_session_ids"])
    haystack_session_ids = row["haystack_session_ids"]
    id_to_session = dict(zip(haystack_session_ids, sessions, strict=False))
    answer_sessions = [
        id_to_session[sid] for sid in answer_session_ids if sid in id_to_session
    ]

    evidences: list[str] = []
    for session in answer_sessions:
        for turn in session:
            if turn.get("has_answer"):
                evidences.append(f"{turn.get('role')} : {turn.get('content')}")
    return evidences


def build_search_entry(
    row: Mapping[str, Any],
    *,
    context: str,
    duration_ms: float,
    status: str,
    reflect_answer: str | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = {
        "question": row["question"],
        "category": row["question_type"],
        "date": row["question_date"],
        "golden_answer": row["answer"],
        "answer_evidences": answer_evidences_for_row(row),
        "search_context": context,
        "search_duration_ms": duration_ms,
        "status": status,
    }
    if reflect_answer is not None:
        entry["reflect_answer"] = reflect_answer
    if error is not None:
        entry["error"] = error
    return entry


def build_search_result(
    row: Mapping[str, Any],
    *,
    user_id: str,
    context: str = "",
    duration_ms: float = 0.0,
    status: str,
    reflect_answer: str | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    return {
        user_id: [
            build_search_entry(
                row,
                context=context,
                duration_ms=duration_ms,
                status=status,
                reflect_answer=reflect_answer,
                error=error,
            )
        ]
    }


def get_single_search_entry(
    search_results: Mapping[str, Any],
    user_id: str,
) -> dict[str, Any] | None:
    entries = search_results.get(user_id)
    if not isinstance(entries, list) or len(entries) != 1:
        return None
    entry = entries[0]
    if not isinstance(entry, dict):
        return None
    return entry


def validate_single_search_result(
    search_results: Mapping[str, Any],
    *,
    user_id: str,
    question: str,
    allowed_statuses: set[str],
    require_status: bool = True,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    entry = get_single_search_entry(search_results, user_id)
    if entry is None:
        return False, ["expected exactly one search entry"]
    if entry.get("question") != question:
        issues.append("question mismatch")
    if require_status and "status" not in entry:
        issues.append("missing status")
    if record_status(entry) not in allowed_statuses:
        issues.append(f"disallowed status: {record_status(entry)}")
    return not issues, issues


def response_complete(
    response: Mapping[str, Any] | None,
    search_entry: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    if not isinstance(response, Mapping):
        return False, ["missing response"]
    issues: list[str] = []
    if response.get("question") != search_entry.get("question"):
        issues.append("question mismatch")
    if record_status(response) == STATUS_SKIPPED:
        return not issues, issues
    for key in ("answer", "golden_answer", "response_duration_ms", "search_duration_ms"):
        if key not in response:
            issues.append(f"missing {key}")
    return not issues, issues


def grade_complete(
    grade: Mapping[str, Any] | None,
    response: Mapping[str, Any],
    num_runs: int,
    *,
    allow_skipped_grade: bool = False,
) -> tuple[bool, list[str]]:
    if not isinstance(grade, Mapping):
        return False, ["missing grade"]
    issues: list[str] = []
    if grade.get("question") != response.get("question"):
        issues.append("question mismatch")
    if record_status(response) == STATUS_SKIPPED:
        return not issues, issues
    if record_status(grade) == STATUS_SKIPPED:
        if allow_skipped_grade:
            return not issues, issues
        issues.append("existing grade is skipped")
        return False, issues

    judgments = grade.get("llm_judgments")
    if not isinstance(judgments, Mapping):
        issues.append("missing judgments")
    else:
        missing = [
            f"judgment_{idx}"
            for idx in range(1, num_runs + 1)
            if f"judgment_{idx}" not in judgments
        ]
        if missing:
            issues.append(f"missing judgment runs: {len(missing)}")
    return not issues, issues


def skipped_response_record(
    *,
    user_id: str,
    search_entry: Mapping[str, Any],
    reason: str,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "category": search_entry.get("category"),
        "question": search_entry.get("question"),
        "answer": "",
        "question_date": search_entry.get("date"),
        "golden_answer": search_entry.get("golden_answer"),
        "response_duration_ms": 0.0,
        "search_duration_ms": search_entry.get("search_duration_ms", 0.0),
        "answer_evidences": search_entry.get("answer_evidences", []),
        "status": STATUS_SKIPPED,
        "skip_reason": reason,
        "error": error,
    }

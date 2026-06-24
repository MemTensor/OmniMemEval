"""Shared pipeline status bookkeeping helpers.

This module intentionally contains only generic status labels, error payloads,
and small counting helpers. Benchmark-specific record shapes and validation stay
in each benchmark directory so retrieval, prompting, and scoring logic remain
unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

STATUS_SUCCESS = "success"
STATUS_SUCCESS_EMPTY = "success_empty"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def status_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        status = record_status(record)
        counts[status] = counts.get(status, 0) + 1
    return counts

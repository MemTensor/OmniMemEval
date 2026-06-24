"""Shared LoCoMo pipeline helpers.

The helpers in this module only describe pipeline bookkeeping: QA filtering,
status labels, query coverage validation, and structured error records. They
do not alter retrieval, answer generation, or grading logic.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from typing import Any


STATUS_SUCCESS = "success"
STATUS_SUCCESS_EMPTY = "success_empty"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def group_id_for(group_idx: int) -> str:
    return f"locomo_exp_user_{group_idx}"


def is_eval_qa(qa: dict[str, Any]) -> bool:
    return qa.get("category") != 5


def filter_eval_qas(qa_set: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [qa for qa in qa_set if is_eval_qa(qa)]


def qa_question(qa: dict[str, Any]) -> str:
    question = qa.get("question")
    return "" if question is None else str(question)


def expected_questions(qa_set: list[dict[str, Any]]) -> list[str]:
    return [qa_question(qa) for qa in filter_eval_qas(qa_set)]


def record_status(record: dict[str, Any]) -> str:
    return str(record.get("status") or STATUS_SUCCESS)


def index_records_by_query(
    records: list[dict[str, Any]],
    *,
    query_key: str = "query",
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for record in records:
        query = record.get(query_key)
        if query is None:
            continue
        query = str(query)
        if query in indexed:
            duplicates.append(query)
            continue
        indexed[query] = record
    return indexed, duplicates


def validate_query_coverage(
    records: list[dict[str, Any]],
    questions: list[str],
    *,
    query_key: str = "query",
    allowed_statuses: set[str] | None = None,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    expected_counts = Counter(str(q) for q in questions)
    record_counts = Counter(
        str(record.get(query_key))
        for record in records
        if record.get(query_key) is not None
    )

    missing_count = sum(
        max(expected_count - record_counts.get(query, 0), 0)
        for query, expected_count in expected_counts.items()
    )
    if missing_count:
        issues.append(f"missing queries: {missing_count}")

    extra_count = sum(
        max(record_count - expected_counts.get(query, 0), 0)
        for query, record_count in record_counts.items()
    )
    if extra_count:
        issues.append(f"unexpected queries: {extra_count}")

    if allowed_statuses is not None:
        bad = [
            record
            for record in records
            if record_status(record) not in allowed_statuses
        ]
        if bad:
            issues.append(f"disallowed statuses: {len(bad)}")

    return not issues, issues


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


def status_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        status = record_status(record)
        counts[status] = counts.get(status, 0) + 1
    return counts


def _record_question(record: dict[str, Any]) -> str:
    question = record.get("question", record.get("query", ""))
    return "" if question is None else str(question)


def status_records_with_skipped(
    grouped_records: dict[str, list[dict[str, Any]]],
    skipped_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge stored records with skipped-record metadata without double counting.

    LoCoMo keeps successful/skipped answer or grade records grouped by user and
    also writes a flat ``skipped_records`` list for diagnostics.  Some skipped
    records are present in both places; search-stage skips only exist in the
    diagnostics list.  This helper keeps the status summary complete without
    inflating the skipped count.
    """
    records: list[dict[str, Any]] = []
    represented: set[tuple[str, str, str]] = set()
    for group_id, group_records in grouped_records.items():
        for record in group_records:
            if not isinstance(record, dict):
                continue
            records.append(record)
            represented.add(
                (str(group_id), _record_question(record), record_status(record))
            )

    for record in skipped_records:
        if not isinstance(record, dict):
            continue
        key = (
            str(record.get("group_id", "")),
            _record_question(record),
            record_status(record),
        )
        if key not in represented:
            records.append(record)
    return records


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


def expected_answer_pairs(
    qa_set: list[dict[str, Any]],
    search_results: list[dict[str, Any]] | None,
) -> tuple[
    list[tuple[dict[str, Any], dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    """Return answerable QA/search pairs plus skipped and failed records."""
    eval_qas = filter_eval_qas(qa_set)
    search_results = search_results or []
    search_by_query = defaultdict(deque)
    for record in search_results:
        query = record.get("query")
        if query is not None:
            search_by_query[str(query)].append(record)

    pairs = []
    skipped = []
    failures = []
    expected = [qa_question(qa) for qa in eval_qas]

    for qa in eval_qas:
        question = qa_question(qa)
        matching_results = search_by_query.get(question)
        if not matching_results:
            failures.append({
                "query": question,
                "status": STATUS_FAILED,
                "error": error_payload("answer", "missing search result"),
            })
            continue
        search_result = matching_results.popleft()

        status = record_status(search_result)
        if status == STATUS_SKIPPED:
            skipped.append({
                "query": question,
                "status": STATUS_SKIPPED,
                "reason": "search was explicitly skipped",
            })
            continue
        if status == STATUS_FAILED:
            failures.append({
                "query": question,
                "status": STATUS_FAILED,
                "error": search_result.get("error")
                or error_payload("answer", "search result is failed"),
            })
            continue

        pairs.append((qa, search_result))

    extra_results = sum(len(records) for records in search_by_query.values())
    if extra_results:
        failures.append({
            "query": "",
            "status": STATUS_FAILED,
            "error": error_payload("answer", f"unexpected search results: {extra_results}"),
        })

    return pairs, skipped, failures, expected

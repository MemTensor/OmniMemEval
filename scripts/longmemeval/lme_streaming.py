#!/usr/bin/env python3
"""LongMemEval streaming add-search-delete pipeline.

Each LongMemEval conversation is treated as one independent evaluation unit:

1. add all haystack sessions for one conversation;
2. search that conversation's question;
3. save the normal LME search-result JSON shape;
4. delete the conversation/user from the memory service.

The output is intentionally compatible with lme_responses.py, lme_eval.py,
lme_metric.py, and lme_report.py.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client_factory import DEFAULT_LIB, SUPPORTED_LIBS, create_client
from longmemeval.lme_common import (
    STATUS_SKIPPED,
    STATUS_SUCCESS_EMPTY,
    build_search_result,
    classify_search_status,
    error_payload,
    status_counts,
    user_id_for,
)
from longmemeval.lme_data import load_lme_dataframe, sanitize_lme_message_content
from longmemeval.lme_ingestion import ingest_session
from utils.checkpoint import atomic_json_dump, fsync_write_line
from utils.duration_stats import update_unit_duration_list
from utils.env import load_env
from utils.ingest_helpers import AddCallTimer
from utils.response_options import parse_bool
from utils.search_helpers import dispatch_search, unpack_search_result
from utils.streaming import (
    configure_single_user_streaming,
    load_marker_set as load_added_chunks,
    log_event as _streaming_log_event,
    LongCallLogger,
    mark_marker as mark_added_chunk,
    prepare_user_after_delete,
    resolve_max_batch_chars,
    timed_delete_user_data,
)
from utils.time import parse_lme_time, to_iso

DEFAULT_GRAPHITI_CONVERSATION_RETRIES = 8
DEFAULT_CONVERSATION_RETRY_DELAY = 30.0
DEFAULT_CONVERSATION_RETRY_BACKOFF = 1.5
DEFAULT_CONVERSATION_RETRY_MAX_DELAY = 300.0


@dataclass(frozen=True)
class GraphitiContentChunk:
    start_session_idx: int
    end_session_idx: int
    content: str


def _results_dir(frame: str, version: str) -> Path:
    return Path("results") / "lme" / f"{frame}-{version}"


def _tmp_path(frame: str, version: str, conv_idx: int) -> Path:
    return (
        _results_dir(frame, version)
        / "tmp"
        / f"{frame}_lme_search_results_{conv_idx}.json"
    )


def _combined_path(frame: str, version: str) -> Path:
    return _results_dir(frame, version) / f"{frame}_lme_search_results.json"


def _completed_path(frame: str, version: str) -> Path:
    return _results_dir(frame, version) / "streaming_completed.txt"


def _added_chunks_path(frame: str, version: str, conv_idx: int) -> Path:
    return (
        _results_dir(frame, version)
        / "tmp"
        / f"{frame}_lme_added_chunks_{conv_idx}.txt"
    )


def _events_path(frame: str, version: str) -> Path:
    return _results_dir(frame, version) / "streaming_events.jsonl"


def _stats_path(frame: str, version: str) -> Path:
    return _results_dir(frame, version) / f"{frame}_lme_streaming_stats.json"


def per_session_checkpoint_id(conv_idx: int, session_idx: int) -> str:
    return f"{conv_idx}_{session_idx}"


def load_completed(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed: set[int] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                completed.add(int(line))
    return completed


def mark_completed(path: Path, completed: set[int], conv_idx: int) -> None:
    if conv_idx in completed:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fsync_write_line(f, str(conv_idx))
    completed.add(conv_idx)


def log_event(path: Path, event: str, conv_idx: int, **fields) -> None:
    _streaming_log_event(path, event, conv_idx=conv_idx, **fields)


def retry_delay_seconds(
    attempt: int,
    *,
    base_delay: float,
    backoff: float,
    max_delay: float,
) -> float:
    delay = base_delay * (backoff ** max(attempt - 1, 0))
    return max(0.0, min(delay, max_delay))


def write_combined_results(frame: str, version: str, completed: set[int]) -> None:
    tmp_dir = _results_dir(frame, version) / "tmp"
    combined: dict[str, list] = defaultdict(list)
    pattern = str(tmp_dir / f"{frame}_lme_search_results_*.json")

    def _idx(path: str) -> int:
        match = re.search(r"_lme_search_results_(\d+)\.json$", path)
        return int(match.group(1)) if match else 10**9

    for path in sorted(glob.glob(pattern), key=_idx):
        conv_idx = _idx(path)
        if conv_idx not in completed:
            continue
        with open(path) as f:
            data = json.load(f)
        for user_id, entries in data.items():
            combined[user_id].extend(entries)

    out = _combined_path(frame, version)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(dict(combined), out, indent=4)


def _completed_search_records(frame: str, version: str, completed: set[int]) -> list[dict]:
    records: list[dict] = []
    tmp_dir = _results_dir(frame, version) / "tmp"
    pattern = str(tmp_dir / f"{frame}_lme_search_results_*.json")

    def _idx(path: str) -> int:
        match = re.search(r"_lme_search_results_(\d+)\.json$", path)
        return int(match.group(1)) if match else 10**9

    for path in sorted(glob.glob(pattern), key=_idx):
        if _idx(path) not in completed:
            continue
        with open(path) as f:
            data = json.load(f)
        for entries in data.values():
            if isinstance(entries, list):
                records.extend(entry for entry in entries if isinstance(entry, dict))
    return records


def write_search_status(
    frame: str,
    version: str,
    completed: set[int],
    *,
    allow_empty_search: bool,
    skip_failed_streaming: bool,
    failed_users: list[dict],
    skipped_records: list[dict],
) -> None:
    records = _completed_search_records(frame, version, completed)
    atomic_json_dump(
        {
            "stage": "search",
            "mode": "streaming",
            "allow_empty_search": allow_empty_search,
            "skip_failed_streaming": skip_failed_streaming,
            "status_counts": status_counts(records),
            "failed_users": failed_users,
            "skipped_records": skipped_records,
        },
        _results_dir(frame, version) / f"{frame}_lme_search_status.json",
        indent=2,
    )


def clean_turn_content(turn) -> str:
    content = sanitize_lme_message_content(turn.get("content", ""))
    return str(content)


def build_batched_cognee_messages(sessions, dates) -> list[dict[str, str]]:
    """Flatten one LME conversation into timestamped messages."""
    messages: list[dict[str, str]] = []
    for session_idx, session in enumerate(sessions):
        date_value = parse_lme_time(dates[session_idx])
        timestamp = to_iso(date_value)
        messages.append(
            {
                "role": "system",
                "content": (
                    f"SESSION {session_idx} START. "
                    f"This session happened at {timestamp} UTC."
                ),
                "chat_time": timestamp,
            }
        )
        for turn_idx, turn in enumerate(session):
            role = turn.get("role", "unknown")
            content = clean_turn_content(turn)
            if not content.strip():
                continue
            messages.append(
                {
                    "role": role,
                    "content": (
                        f"SESSION {session_idx} TURN {turn_idx} "
                        f"{role}: {content}"
                    ),
                    "chat_time": timestamp,
                }
            )
    return messages


def build_batched_graphiti_content(
    sessions,
    dates,
    *,
    start_session_idx: int = 0,
) -> str:
    """Flatten LME sessions into one Graphiti episode body."""
    lines: list[str] = [
        "LONGMEMEVAL CONVERSATION",
        (
            "Each SESSION block is an independent historical session. "
            "Use the session timestamp when resolving temporal references."
        ),
    ]
    for local_idx, session in enumerate(sessions):
        session_idx = start_session_idx + local_idx
        date_value = parse_lme_time(dates[local_idx])
        timestamp = to_iso(date_value)
        lines.extend(
            [
                "",
                f"SESSION {session_idx} START",
                f"SESSION {session_idx} TIMESTAMP: {timestamp} UTC",
            ]
        )
        for turn_idx, turn in enumerate(session):
            role = turn.get("role", "unknown")
            content = clean_turn_content(turn)
            if not content.strip():
                continue
            lines.append(f"SESSION {session_idx} TURN {turn_idx} {role}: {content}")
        lines.append(f"SESSION {session_idx} END")
    return "\n".join(lines)


def _split_oversized_graphiti_content(
    content: str,
    max_chars: int,
    label: str,
) -> list[str]:
    if max_chars <= 0 or len(content) <= max_chars:
        return [content]

    prefix_template = (
        "LONGMEMEVAL CONVERSATION CHUNK\n"
        f"{label}; raw text part XX/YY. Preserve original SESSION/TURN labels.\n"
    )
    body_limit = max(1, max_chars - len(prefix_template) - 20)
    pieces: list[str] = []
    start = 0
    while start < len(content):
        end = min(start + body_limit, len(content))
        if end < len(content):
            newline = content.rfind("\n", start + 1, end)
            if newline > start + body_limit // 2:
                end = newline + 1
        pieces.append(content[start:end])
        start = end

    total = len(pieces)
    return [
        (
            "LONGMEMEVAL CONVERSATION CHUNK\n"
            f"{label}; raw text part {idx}/{total}. "
            "Preserve original SESSION/TURN labels.\n"
            f"{piece}"
        )
        for idx, piece in enumerate(pieces, start=1)
    ]


def build_graphiti_content_chunks(
    sessions,
    dates,
    *,
    max_chars: int = 0,
) -> list[GraphitiContentChunk]:
    """Pack a conversation into bounded Graphiti raw episode chunks."""
    if max_chars <= 0:
        return [
            GraphitiContentChunk(
                0,
                len(sessions) - 1,
                build_batched_graphiti_content(sessions, dates),
            )
        ]

    chunks: list[GraphitiContentChunk] = []
    current_sessions = []
    current_dates = []
    current_start_idx: int | None = None

    def flush_current() -> None:
        nonlocal current_sessions, current_dates, current_start_idx
        if current_start_idx is None:
            return
        content = build_batched_graphiti_content(
            current_sessions,
            current_dates,
            start_session_idx=current_start_idx,
        )
        chunks.append(
            GraphitiContentChunk(
                current_start_idx,
                current_start_idx + len(current_sessions) - 1,
                content,
            )
        )
        current_sessions = []
        current_dates = []
        current_start_idx = None

    for session_idx, session in enumerate(sessions):
        date = dates[session_idx]
        single_content = build_batched_graphiti_content(
            [session],
            [date],
            start_session_idx=session_idx,
        )
        if len(single_content) > max_chars:
            flush_current()
            parts = _split_oversized_graphiti_content(
                single_content,
                max_chars,
                f"SESSION {session_idx}",
            )
            for part in parts:
                chunks.append(GraphitiContentChunk(session_idx, session_idx, part))
            continue

        if current_start_idx is None:
            current_start_idx = session_idx
            current_sessions = [session]
            current_dates = [date]
            continue

        candidate_sessions = current_sessions + [session]
        candidate_dates = current_dates + [date]
        candidate = build_batched_graphiti_content(
            candidate_sessions,
            candidate_dates,
            start_session_idx=current_start_idx,
        )
        if len(candidate) > max_chars:
            flush_current()
            current_start_idx = session_idx
            current_sessions = [session]
            current_dates = [date]
        else:
            current_sessions = candidate_sessions
            current_dates = candidate_dates

    flush_current()
    return chunks


def graphiti_lme_chunk_id(batch_idx: int, chunk: GraphitiContentChunk) -> str:
    digest = hashlib.sha1(chunk.content.encode("utf-8")).hexdigest()[:16]
    return (
        f"batch_{batch_idx:03d}:sessions_"
        f"{chunk.start_session_idx}-{chunk.end_session_idx}:sha1_{digest}"
    )


def add_conversation_batched_cognee(
    lme_df,
    conv_idx: int,
    version: str,
    client,
) -> list[float]:
    row = lme_df.iloc[conv_idx]
    sessions = row["haystack_sessions"]
    dates = row["haystack_dates"]
    user_id = user_id_for(version, conv_idx)
    session_id = f"{user_id}_lme_exper_all_sessions"

    messages = build_batched_cognee_messages(sessions, dates)
    char_count = sum(len(str(msg.get("content", ""))) for msg in messages)
    timer = AddCallTimer(client)
    label = (
        f"cognee LME add conversation={conv_idx} sessions={len(sessions)} "
        f"messages={len(messages)} chars={char_count}"
    )
    with LongCallLogger(label):
        client.add(messages, user_id, session_key=session_id)
    print(
        f"[cognee] Conversation {conv_idx}: batched {len(sessions)} sessions, "
        f"{len(messages)} messages into one remember call"
    )
    return timer.durations_ms


def add_conversation_batched_graphiti(
    lme_df,
    conv_idx: int,
    version: str,
    client,
    *,
    max_chars: int = 0,
    added_chunks_path: Path | None = None,
    added_chunks: set[str] | None = None,
) -> list[float]:
    row = lme_df.iloc[conv_idx]
    sessions = row["haystack_sessions"]
    dates = row["haystack_dates"]
    user_id = user_id_for(version, conv_idx)
    chunks = build_graphiti_content_chunks(sessions, dates, max_chars=max_chars)
    timer = AddCallTimer(client)
    skipped_chunks = 0

    for batch_idx, chunk in enumerate(chunks):
        chunk_id = graphiti_lme_chunk_id(batch_idx, chunk)
        if added_chunks is not None and chunk_id in added_chunks:
            skipped_chunks += 1
            continue

        session_id = (
            f"{user_id}_lme_exper_batch_{batch_idx:03d}_"
            f"sessions_{chunk.start_session_idx}-{chunk.end_session_idx}"
        )
        timestamp = (
            to_iso(parse_lme_time(dates[chunk.start_session_idx]))
            if len(dates)
            else None
        )
        label = (
            f"graphiti LME add conversation={conv_idx} "
            f"batch={batch_idx + 1}/{len(chunks)} "
            f"sessions={chunk.start_session_idx}-{chunk.end_session_idx} "
            f"chars={len(chunk.content)}"
        )
        with LongCallLogger(label):
            client.add(
                [],
                user_id,
                session_key=session_id,
                raw_content=chunk.content,
                timestamp=timestamp,
                role="longmemeval_conversation_chunk",
                source_description=(
                    "LongMemEval conversation chunk with timestamped sessions "
                    f"{chunk.start_session_idx}-{chunk.end_session_idx}"
                ),
            )
        if added_chunks_path is not None and added_chunks is not None:
            mark_added_chunk(added_chunks_path, added_chunks, chunk_id)

    max_len = max((len(chunk.content) for chunk in chunks), default=0)
    print(
        f"[graphiti] Conversation {conv_idx}: chunked batched {len(sessions)} "
        f"sessions into {len(chunks)} raw episodes "
        f"({skipped_chunks} resumed, max_chars={max_chars}, largest={max_len})"
    )
    return timer.durations_ms


def add_conversation(
    lme_df,
    conv_idx: int,
    frame: str,
    version: str,
    client,
    ingest_mode: str,
    max_batch_chars: int = 0,
    added_parts_path: Path | None = None,
    added_parts: set[str] | None = None,
):
    if ingest_mode == "batched":
        if frame == "cognee":
            return add_conversation_batched_cognee(lme_df, conv_idx, version, client)
        if frame == "graphiti":
            return add_conversation_batched_graphiti(
                lme_df,
                conv_idx,
                version,
                client,
                max_chars=max_batch_chars,
                added_chunks_path=added_parts_path,
                added_chunks=added_parts,
            )
        raise RuntimeError(
            "batched add is currently supported only for cognee and graphiti"
        )

    row = lme_df.iloc[conv_idx]
    sessions = row["haystack_sessions"]
    dates = row["haystack_dates"]
    user_id = user_id_for(version, conv_idx)

    timer = AddCallTimer(client)
    skipped_sessions = 0
    total_sessions = len(sessions)
    for session_idx, session in enumerate(sessions):
        checkpoint_id = per_session_checkpoint_id(conv_idx, session_idx)
        if added_parts is not None and checkpoint_id in added_parts:
            skipped_sessions += 1
            print(
                f"[{frame}] LME conversation {conv_idx}: "
                f"session {session_idx + 1}/{total_sessions} already added, skipping",
                flush=True,
            )
            continue
        session_id = f"{user_id}_lme_exper_session_{session_idx}"
        date_value = parse_lme_time(dates[session_idx])
        ingest_session(session, date_value, user_id, session_id, frame, client)
        if added_parts_path is not None and added_parts is not None:
            mark_added_chunk(added_parts_path, added_parts, checkpoint_id)
    if skipped_sessions:
        print(
            f"[{frame}] LME conversation {conv_idx}: "
            f"{skipped_sessions}/{total_sessions} sessions resumed from checkpoint",
            flush=True,
        )
    return timer.durations_ms


def search_conversation(
    lme_df,
    conv_idx: int,
    frame: str,
    version: str,
    top_k: int,
    *,
    allow_empty_search: bool = True,
):
    row = lme_df.iloc[conv_idx]
    question = row["question"]
    question_type = row["question_type"]
    question_date = row["question_date"]
    user_id = user_id_for(version, conv_idx)

    client = create_client(frame)
    extra_kw = {"question_date": question_date}

    print("-" * 80)
    print(f"Searching conversation {conv_idx}")
    print(f"Question: {question}")
    print(f"Date: {question_date}")
    print(f"Type: {question_type}")
    print("-" * 80)

    result = dispatch_search(frame, client, question, user_id, top_k, **extra_kw)
    context, duration_ms, reflect_answer, raw_context = unpack_search_result(result)
    context = context or ""
    status = classify_search_status(
        context,
        reflect_answer,
        raw_context=raw_context,
    )
    if status == STATUS_SUCCESS_EMPTY and not allow_empty_search:
        raise RuntimeError(f"search returned no raw memories for conversation {conv_idx}")

    search_results = build_search_result(
        row,
        user_id=user_id,
        context=context,
        duration_ms=duration_ms,
        status=status,
        reflect_answer=reflect_answer,
    )
    path = _tmp_path(frame, version, conv_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(search_results, path, indent=4)
    print(f"Saved search result: {path}")
    return search_results


def save_stats(frame: str, version: str, stats: dict) -> None:
    path = _stats_path(frame, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(stats, path, indent=2)


def process_fresh_conversation(
    lme_df,
    conv_idx: int,
    frame: str,
    version: str,
    top_k: int,
    ingest_mode: str,
    max_batch_chars: int,
    wait_after_ingest: float,
    allow_empty_search: bool,
    completed: set[int],
    stats: dict,
    force_fresh: bool = False,
) -> None:
    user_id = user_id_for(version, conv_idx)
    start = time.time()
    print("\n" + "=" * 80)
    print(
        f"STREAM conversation {conv_idx}: delete, add({ingest_mode}), "
        "search, delete"
    )
    print("=" * 80)

    client = create_client(frame)
    added_parts_path = _added_chunks_path(frame, version, conv_idx)
    added_parts: set[str] | None = None
    can_resume_add_parts = ingest_mode == "per-session" or (
        frame == "graphiti" and ingest_mode == "batched"
    )
    if can_resume_add_parts:
        added_parts = set() if force_fresh else load_added_chunks(added_parts_path)
    should_start_fresh = force_fresh or not can_resume_add_parts or not added_parts

    initial_delete_ok = False
    initial_delete_error: str | None = None
    initial_delete_ms = 0.0
    if should_start_fresh:
        initial_delete_ok, initial_delete_error, initial_delete_ms = timed_delete_user_data(
            frame,
            client,
            user_id,
            phase="initial",
        )
        prepare_user_after_delete(frame, client, user_id)
        if can_resume_add_parts:
            if added_parts_path.exists():
                added_parts_path.unlink()
            added_parts = set()
            tmp_path = _tmp_path(frame, version, conv_idx)
            if tmp_path.exists():
                tmp_path.unlink()
    else:
        part_label = (
            "chunks" if frame == "graphiti" and ingest_mode == "batched" else "sessions"
        )
        print(
            f"Resuming LME conversation {conv_idx}: "
            f"{len(added_parts)} {part_label} recorded"
        )

    add_call_ms = add_conversation(
        lme_df,
        conv_idx,
        frame,
        version,
        client,
        ingest_mode,
        max_batch_chars,
        added_parts_path=added_parts_path if can_resume_add_parts else None,
        added_parts=added_parts if can_resume_add_parts else None,
    )
    if wait_after_ingest > 0:
        print(f"Waiting {wait_after_ingest}s after ingest")
        time.sleep(wait_after_ingest)

    search_conversation(
        lme_df,
        conv_idx,
        frame,
        version,
        top_k,
        allow_empty_search=allow_empty_search,
    )
    final_delete_ok, final_delete_error, final_delete_ms = timed_delete_user_data(
        frame,
        client,
        user_id,
        phase="final",
    )
    final_delete_status = "ok" if final_delete_ok else "error_skipped"
    if final_delete_ok and added_parts_path.exists():
        added_parts_path.unlink()
    if not final_delete_ok:
        log_event(
            _events_path(frame, version),
            "final_delete_error_skipped",
            conv_idx,
            user_id=user_id,
            error=final_delete_error,
        )

    mark_completed(_completed_path(frame, version), completed, conv_idx)
    stats.setdefault("modes", {})[str(conv_idx)] = f"fresh_streaming_{ingest_mode}"
    stats.setdefault("ingest_modes", {})[str(conv_idx)] = ingest_mode
    if frame == "graphiti" and ingest_mode == "batched":
        stats.setdefault("max_batch_chars", max_batch_chars)
    stats.setdefault("final_delete_statuses", {})[str(conv_idx)] = final_delete_status
    initial_delete_status = (
        "ok"
        if initial_delete_ok
        else ("not_run_add_part_resume" if not should_start_fresh else "error_skipped")
    )
    stats.setdefault("initial_delete_statuses", {})[str(conv_idx)] = initial_delete_status
    stats.setdefault("initial_delete_errors", {})[str(conv_idx)] = initial_delete_error
    stats.setdefault("final_delete_errors", {})[str(conv_idx)] = final_delete_error
    stats.setdefault("initial_delete_durations_ms", {})[str(conv_idx)] = round(
        initial_delete_ms,
        2,
    )
    stats.setdefault("final_delete_durations_ms", {})[str(conv_idx)] = round(
        final_delete_ms,
        2,
    )
    stats.setdefault("delete_call_counts", {})[str(conv_idx)] = (
        2 if should_start_fresh else 1
    )
    stats.setdefault("user_durations_ms", {})[str(conv_idx)] = round(
        (time.time() - start) * 1000,
        1,
    )
    update_unit_duration_list(
        stats,
        conv_idx,
        add_call_ms,
        map_key="add_call_durations_by_unit",
        flat_key="add_call_durations_ms",
    )
    log_event(
        _events_path(frame, version),
        "completed_fresh",
        conv_idx,
        add_calls=len(add_call_ms),
        ingest_mode=ingest_mode,
        max_batch_chars=(
            max_batch_chars
            if frame == "graphiti" and ingest_mode == "batched"
            else None
        ),
        final_delete_status=final_delete_status,
        initial_delete_ms=round(initial_delete_ms, 2),
        final_delete_ms=round(final_delete_ms, 2),
    )
    write_combined_results(frame, version, completed)
    save_stats(frame, version, stats)


def mark_streaming_failure_skipped(
    lme_df,
    conv_idx: int,
    frame: str,
    version: str,
    completed: set[int],
    exc: BaseException,
) -> dict:
    user_id = user_id_for(version, conv_idx)
    tmp_path = _tmp_path(frame, version, conv_idx)
    if not tmp_path.exists():
        row = lme_df.iloc[conv_idx]
        atomic_json_dump(
            build_search_result(
                row,
                user_id=user_id,
                status=STATUS_SKIPPED,
                error=error_payload("streaming", exc),
            ),
            tmp_path,
            indent=4,
        )
    mark_completed(_completed_path(frame, version), completed, conv_idx)
    return {
        "conv_idx": conv_idx,
        "user_id": user_id,
        "error": error_payload("streaming", exc),
    }


def load_stats(frame: str, version: str) -> dict:
    path = _stats_path(frame, version)
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {
        "mode": "lme_streaming",
        "user_durations_ms": {},
        "add_call_durations_by_unit": {},
        "add_call_durations_ms": [],
        "modes": {},
        "ingest_modes": {},
        "final_delete_statuses": {},
    }


def main() -> int:
    parser = argparse.ArgumentParser("LongMemEval streaming add-search-delete")
    parser.add_argument("--lib", choices=SUPPORTED_LIBS, default=DEFAULT_LIB)
    parser.add_argument("--env", help="Dotenv file to load")
    parser.add_argument("--version", default="default")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--allow-empty-search",
        "--allow_empty_search",
        type=parse_bool,
        default=True,
        help="Allow successful searches with no raw memories. Default: 1.",
    )
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int)
    parser.add_argument("--wait-after-ingest", type=float, default=0.0)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore streaming_completed.txt and process the selected range anyway.",
    )
    parser.add_argument(
        "--restart-unit",
        action="store_true",
        help="Delete each selected user and discard chunk/search checkpoints before reprocessing.",
    )
    parser.add_argument(
        "--skip-failed-streaming",
        action="store_true",
        help="Mark failed streaming units as skipped and continue instead of failing at the end.",
    )
    parser.add_argument(
        "--conversation-max-retries",
        type=int,
        help=(
            "Retry a failed conversation this many times before recording failure. "
            "For graphiti batched streaming, defaults to GRAPHITI_LME_CONV_MAX_RETRIES "
            f"or {DEFAULT_GRAPHITI_CONVERSATION_RETRIES}; otherwise defaults to "
            "LME_STREAMING_CONV_MAX_RETRIES or 0."
        ),
    )
    parser.add_argument(
        "--conversation-retry-delay",
        type=float,
        help=(
            "Base seconds to wait before retrying a failed conversation. "
            "Defaults to LME_STREAMING_RETRY_DELAY or 30."
        ),
    )
    parser.add_argument(
        "--conversation-retry-backoff",
        type=float,
        help=(
            "Backoff multiplier for conversation retries. "
            "Defaults to LME_STREAMING_RETRY_BACKOFF or 1.5."
        ),
    )
    parser.add_argument(
        "--conversation-retry-max-delay",
        type=float,
        help=(
            "Maximum seconds to wait between conversation retries. "
            "Defaults to LME_STREAMING_RETRY_MAX_DELAY or 300."
        ),
    )
    args = parser.parse_args()

    if args.env:
        env_path = Path(args.env)
        if not env_path.is_file():
            env_path = Path.cwd() / args.env
        if not env_path.is_file():
            raise SystemExit(f"Env file not found: {args.env}")
        os.environ["MEMEVAL_ENV_FILE"] = str(env_path.resolve())

    load_env()
    configure_single_user_streaming(args.lib)
    max_batch_chars = resolve_max_batch_chars(args.lib)

    ingest_mode = "batched" if args.lib in {"cognee", "graphiti"} else "per-session"
    if args.conversation_max_retries is None:
        if args.lib == "graphiti" and ingest_mode == "batched":
            args.conversation_max_retries = int(
                os.getenv(
                    "GRAPHITI_LME_CONV_MAX_RETRIES",
                    str(DEFAULT_GRAPHITI_CONVERSATION_RETRIES),
                )
            )
        else:
            args.conversation_max_retries = int(
                os.getenv("LME_STREAMING_CONV_MAX_RETRIES", "0")
            )
    if args.conversation_retry_delay is None:
        args.conversation_retry_delay = float(
            os.getenv(
                "LME_STREAMING_RETRY_DELAY",
                str(DEFAULT_CONVERSATION_RETRY_DELAY),
            )
        )
    if args.conversation_retry_backoff is None:
        args.conversation_retry_backoff = float(
            os.getenv(
                "LME_STREAMING_RETRY_BACKOFF",
                str(DEFAULT_CONVERSATION_RETRY_BACKOFF),
            )
        )
    if args.conversation_retry_max_delay is None:
        args.conversation_retry_max_delay = float(
            os.getenv(
                "LME_STREAMING_RETRY_MAX_DELAY",
                str(DEFAULT_CONVERSATION_RETRY_MAX_DELAY),
            )
        )
    if args.conversation_max_retries < 0:
        raise SystemExit("--conversation-max-retries must be >= 0")
    if args.conversation_retry_delay < 0:
        raise SystemExit("--conversation-retry-delay must be >= 0")
    if args.conversation_retry_backoff < 0:
        raise SystemExit("--conversation-retry-backoff must be >= 0")
    if args.conversation_retry_max_delay < 0:
        raise SystemExit("--conversation-retry-max-delay must be >= 0")

    if ingest_mode == "batched" and args.lib not in {"cognee", "graphiti"}:
        raise SystemExit(
            "batched add is currently supported only for --lib cognee or --lib graphiti"
        )

    results_dir = _results_dir(args.lib, args.version)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "tmp").mkdir(parents=True, exist_ok=True)

    lme_df = load_lme_dataframe()
    total = len(lme_df)
    end_idx = args.end_idx if args.end_idx is not None else total - 1
    if args.start_idx < 0 or end_idx >= total or end_idx < args.start_idx:
        raise SystemExit(
            f"Invalid range start={args.start_idx}, end={end_idx}, total={total}"
        )

    completed_path = _completed_path(args.lib, args.version)
    existing_completed = load_completed(completed_path)
    completed = set() if args.no_resume else set(existing_completed)

    stats = load_stats(args.lib, args.version)

    print("\n" + "=" * 80)
    print("LONGMEMEVAL STREAMING")
    print("=" * 80)
    print(f"lib={args.lib}")
    print(f"version={args.version}")
    print(f"range={args.start_idx}-{end_idx}")
    print(f"top_k={args.top_k}")
    print(f"allow_empty_search={args.allow_empty_search}")
    print(f"ingest_mode={ingest_mode}")
    if args.lib == "graphiti" and ingest_mode == "batched":
        print(f"max_batch_chars={max_batch_chars}")
    print(f"wait_after_ingest={args.wait_after_ingest}")
    print(f"conversation_max_retries={args.conversation_max_retries}")
    print(
        "conversation_retry="
        f"delay={args.conversation_retry_delay}, "
        f"backoff={args.conversation_retry_backoff}, "
        f"max_delay={args.conversation_retry_max_delay}"
    )
    print(f"already_completed={len(completed)}")
    print("=" * 80)

    failed_users: list[dict] = []
    skipped_records: list[dict] = []
    for conv_idx in range(args.start_idx, end_idx + 1):
        if conv_idx in completed and not args.no_resume:
            print(f"Skipping conversation {conv_idx}: already completed")
            continue
        failed_attempts = 0
        while True:
            try:
                process_fresh_conversation(
                    lme_df,
                    conv_idx,
                    args.lib,
                    args.version,
                    args.top_k,
                    ingest_mode,
                    max_batch_chars,
                    args.wait_after_ingest,
                    args.allow_empty_search,
                    completed,
                    stats,
                    force_fresh=(
                        args.restart_unit
                        or (args.no_resume and conv_idx in existing_completed)
                    ),
                )
                if failed_attempts:
                    log_event(
                        _events_path(args.lib, args.version),
                        "retry_success",
                        conv_idx,
                        attempts=failed_attempts,
                    )
                save_stats(args.lib, args.version, stats)
                break
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                raise
            except Exception as exc:
                failed_attempts += 1
                log_event(
                    _events_path(args.lib, args.version),
                    "failed",
                    conv_idx,
                    error=str(exc),
                    attempt=failed_attempts,
                    max_retries=args.conversation_max_retries,
                )
                if failed_attempts <= args.conversation_max_retries:
                    wait = retry_delay_seconds(
                        failed_attempts,
                        base_delay=args.conversation_retry_delay,
                        backoff=args.conversation_retry_backoff,
                        max_delay=args.conversation_retry_max_delay,
                    )
                    log_event(
                        _events_path(args.lib, args.version),
                        "retry_scheduled",
                        conv_idx,
                        error=str(exc),
                        attempt=failed_attempts,
                        max_retries=args.conversation_max_retries,
                        wait_seconds=round(wait, 2),
                    )
                    print(
                        f"WARNING conversation {conv_idx} failed "
                        f"({type(exc).__name__}: {exc}); retrying in {wait:.1f}s "
                        f"(attempt {failed_attempts}/{args.conversation_max_retries})",
                        flush=True,
                    )
                    save_stats(args.lib, args.version, stats)
                    if wait > 0:
                        time.sleep(wait)
                    continue

                failure = {
                    "conv_idx": conv_idx,
                    "user_id": user_id_for(args.version, conv_idx),
                    "error": error_payload("streaming", exc),
                }
                print(f"ERROR conversation {conv_idx}: {type(exc).__name__}: {exc}")
                if args.skip_failed_streaming:
                    skipped_records.append(
                        mark_streaming_failure_skipped(
                            lme_df,
                            conv_idx,
                            args.lib,
                            args.version,
                            completed,
                            exc,
                        )
                    )
                    write_combined_results(args.lib, args.version, completed)
                    save_stats(args.lib, args.version, stats)
                    break
                failed_users.append(failure)
                save_stats(args.lib, args.version, stats)
                break

    write_combined_results(args.lib, args.version, completed)
    save_stats(args.lib, args.version, stats)
    write_search_status(
        args.lib,
        args.version,
        completed,
        allow_empty_search=args.allow_empty_search,
        skip_failed_streaming=args.skip_failed_streaming,
        failed_users=failed_users,
        skipped_records=skipped_records,
    )
    if failed_users:
        print(f"\nStreaming failed for {len(failed_users)} conversation(s)")
        return 1
    print("\nStreaming complete")
    print(f"Combined search results: {_combined_path(args.lib, args.version)}")
    print(f"Completed conversations: {len(completed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

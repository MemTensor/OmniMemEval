#!/usr/bin/env python3
"""BEAM streaming add-search-delete pipeline.

Each BEAM conversation is treated as an independent evaluation unit:

1. add the conversation history;
2. search all probing questions for that conversation;
3. save BEAM-compatible search results;
4. delete the conversation/user from the memory service.

For Graphiti, large sessions are converted into bounded raw episodes with
chunk-level checkpoints. For the regular per-session path, each completed
session add is checkpointed. This keeps BEAM-10m recoverable without changing
the downstream BEAM response/eval/metric scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from beam.beam_ingestion import (  # noqa: E402
    flatten_session,
    ingest_session,
    load_beam_data,
    parse_time_anchor,
    session_time_anchor,
)
from beam.beam_search import parse_probing_questions  # noqa: E402
from beam.beam_common import (  # noqa: E402
    STATUS_SKIPPED,
    STATUS_SUCCESS_EMPTY,
    build_question_meta,
    build_search_entry,
    classify_search_status,
    error_payload,
    status_counts,
)
from client_factory import DEFAULT_LIB, SUPPORTED_LIBS, create_client  # noqa: E402
from utils.checkpoint import atomic_json_dump  # noqa: E402
from utils.duration_stats import update_unit_duration_list  # noqa: E402
from utils.env import load_env  # noqa: E402
from utils.ingest_helpers import AddCallTimer  # noqa: E402
from utils.response_options import parse_bool  # noqa: E402
from utils.search_helpers import dispatch_search, unpack_search_result  # noqa: E402
from utils.streaming import (  # noqa: E402
    configure_single_user_streaming,
    delete_user_data,
    load_marker_set as load_added_chunks,
    load_marker_set as load_completed,
    log_event,
    LongCallLogger,
    mark_marker as mark_added_chunk,
    mark_marker as mark_completed,
    prepare_user_after_delete,
    resolve_max_batch_chars,
)

@dataclass(frozen=True)
class BeamGraphitiChunk:
    conv_id: str
    session_idx: int
    part_idx: int
    total_parts: int
    content: str
    timestamp: str
    uuid: str

    @property
    def chunk_id(self) -> str:
        return f"{self.conv_id}_{self.session_idx}_{self.part_idx}"


def _results_dir(frame: str, version: str) -> Path:
    return Path("results") / "beam" / f"{frame}-{version}"


def _tmp_path(frame: str, version: str, conv_id: str) -> Path:
    return _results_dir(frame, version) / "tmp" / f"{frame}_beam_search_results_{conv_id}.json"


def _combined_path(frame: str, version: str) -> Path:
    return _results_dir(frame, version) / f"{frame}_beam_search_results.json"


def _completed_path(frame: str, version: str) -> Path:
    return _results_dir(frame, version) / "streaming_completed.txt"


def _events_path(frame: str, version: str) -> Path:
    return _results_dir(frame, version) / "streaming_events.jsonl"


def _stats_path(frame: str, version: str) -> Path:
    return _results_dir(frame, version) / f"{frame}_beam_ingestion_stats.json"


def _added_chunks_path(frame: str, version: str, conv_id: str) -> Path:
    return _results_dir(frame, version) / "tmp" / f"{frame}_beam_added_chunks_{conv_id}.txt"


def user_id_for(version: str, conv_id: str) -> str:
    return f"beam_exp_user_{version}_{conv_id}"


def per_session_checkpoint_id(conv_id: str, session_idx: int) -> str:
    return f"{conv_id}_session_{session_idx:03d}"


def _stable_uuid(*parts: object) -> str:
    key = ":".join(str(part) for part in parts)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"omnimemeval:{key}"))


def _split_with_header(body: str, max_chars: int, header_fn) -> list[str]:
    if max_chars <= 0:
        return [header_fn(1, 1) + body]

    probe = header_fn(9999, 9999)
    body_limit = max(1, max_chars - len(probe) - 20)
    pieces: list[str] = []
    start = 0
    while start < len(body):
        end = min(start + body_limit, len(body))
        if end < len(body):
            newline = body.rfind("\n", start + 1, end)
            if newline > start + body_limit // 2:
                end = newline + 1
        pieces.append(body[start:end])
        start = end

    total = len(pieces)
    return [header_fn(idx, total) + piece for idx, piece in enumerate(pieces, start=1)]


def build_graphiti_session_chunks(
    conv,
    session_idx: int,
    messages: list[dict],
    *,
    version: str,
    max_chars: int,
) -> list[BeamGraphitiChunk]:
    conv_id = str(conv["conversation_id"])
    scale = conv.get("_scale", "unknown")
    session_date = parse_time_anchor(session_time_anchor(messages))
    timestamp = session_date.isoformat()

    body_lines = [
        f"BEAM SCALE: {scale}",
        f"CONVERSATION ID: {conv_id}",
        f"SESSION INDEX: {session_idx}",
        f"SESSION TIMESTAMP: {timestamp}",
        "Use SESSION/TURN labels to preserve temporal order.",
        "",
        f"SESSION {session_idx} START",
    ]
    for turn_idx, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        body_lines.append(f"SESSION {session_idx} TURN {turn_idx} {role}: {content}")
    body_lines.append(f"SESSION {session_idx} END")
    body = "\n".join(body_lines)

    def header(part_idx: int, total_parts: int) -> str:
        return (
            "BEAM CONVERSATION CHUNK\n"
            f"conversation={conv_id}; session={session_idx}; "
            f"part={part_idx}/{total_parts}; scale={scale}.\n"
            "Preserve original SESSION/TURN labels and part order.\n"
        )

    contents = _split_with_header(body, max_chars, header)
    total = len(contents)
    return [
        BeamGraphitiChunk(
            conv_id=conv_id,
            session_idx=session_idx,
            part_idx=idx,
            total_parts=total,
            content=content,
            timestamp=timestamp,
            uuid=_stable_uuid("beam", version, conv_id, session_idx, idx),
        )
        for idx, content in enumerate(contents, start=1)
    ]


def add_conversation_graphiti_chunked(
    conv,
    version: str,
    client,
    *,
    max_chars: int,
    added_chunks_path: Path,
    added_chunks: set[str],
) -> tuple[list[float], dict]:
    conv_id = str(conv["conversation_id"])
    user_id = user_id_for(version, conv_id)
    timer = AddCallTimer(client)

    total_messages = 0
    total_chars = 0
    chunk_count = 0
    skipped_chunks = 0

    for session_idx, raw_session in enumerate(conv["chat"]):
        messages = flatten_session(raw_session)
        if not messages:
            continue
        total_messages += len(messages)
        total_chars += sum(len(msg.get("content", "")) for msg in messages)
        chunks = build_graphiti_session_chunks(
            conv,
            session_idx,
            messages,
            version=version,
            max_chars=max_chars,
        )
        for chunk in chunks:
            chunk_count += 1
            if chunk.chunk_id in added_chunks:
                skipped_chunks += 1
                continue
            session_key = (
                f"{user_id}_beam_session_{session_idx:03d}_"
                f"part_{chunk.part_idx:04d}_of_{chunk.total_parts:04d}"
            )
            label = (
                f"graphiti BEAM add conversation={conv_id} "
                f"session={session_idx + 1}/{len(conv['chat'])} "
                f"part={chunk.part_idx}/{chunk.total_parts} "
                f"chars={len(chunk.content)}"
            )
            with LongCallLogger(label):
                client.add(
                    [],
                    user_id,
                    session_key=session_key,
                    raw_content=chunk.content,
                    timestamp=chunk.timestamp,
                    role="beam_conversation_chunk",
                    uuid=chunk.uuid,
                    source_description=(
                        "BEAM conversation chunk "
                        f"conversation={conv_id} session={session_idx} "
                        f"part={chunk.part_idx}/{chunk.total_parts}"
                    ),
                )
            mark_added_chunk(added_chunks_path, added_chunks, chunk.chunk_id)

    meta = {
        "messages": total_messages,
        "chars": total_chars,
        "chunks": chunk_count,
        "skipped_chunks": skipped_chunks,
        "max_chars": max_chars,
    }
    print(
        f"[graphiti] BEAM conversation {conv_id}: "
        f"{total_messages} messages, {total_chars} chars, "
        f"{chunk_count} chunks ({skipped_chunks} resumed)"
    )
    return timer.durations_ms, meta


def add_conversation_per_session(
    conv,
    frame: str,
    version: str,
    client,
    *,
    added_sessions_path: Path,
    added_sessions: set[str],
) -> tuple[list[float], dict]:
    conv_id = str(conv["conversation_id"])
    user_id = user_id_for(version, conv_id)
    timer = AddCallTimer(client)
    total_messages = 0
    total_chars = 0
    session_count = 0
    skipped_sessions = 0
    total_sessions = len(conv["chat"])

    for session_idx, raw_session in enumerate(conv["chat"]):
        messages = flatten_session(raw_session)
        if not messages:
            print(
                f"[{frame}] BEAM conversation {conv_id}: "
                f"session {session_idx + 1}/{total_sessions} empty, skipping",
                flush=True,
            )
            continue
        session_count += 1
        checkpoint_id = per_session_checkpoint_id(conv_id, session_idx)
        session_chars = sum(len(msg.get("content", "")) for msg in messages)
        total_messages += len(messages)
        total_chars += session_chars
        session_date = parse_time_anchor(session_time_anchor(messages))
        session_id = f"{user_id}_session_{session_idx}"

        if checkpoint_id in added_sessions:
            skipped_sessions += 1
            print(
                f"[{frame}] BEAM conversation {conv_id}: "
                f"session {session_idx + 1}/{total_sessions} already added "
                f"({len(messages)} messages, {session_chars} chars), skipping",
                flush=True,
            )
            continue

        started = time.time()
        print(
            f"[{frame}] BEAM conversation {conv_id}: "
            f"adding session {session_idx + 1}/{total_sessions} "
            f"({len(messages)} messages, {session_chars} chars)",
            flush=True,
        )
        ingest_session(client, messages, frame, user_id, session_id, session_date)
        mark_added_chunk(added_sessions_path, added_sessions, checkpoint_id)
        elapsed_ms = round((time.time() - started) * 1000, 1)
        print(
            f"[{frame}] BEAM conversation {conv_id}: "
            f"session {session_idx + 1}/{total_sessions} added in {elapsed_ms} ms",
            flush=True,
        )

    return timer.durations_ms, {
        "messages": total_messages,
        "chars": total_chars,
        "chunks": session_count,
        "skipped_chunks": skipped_sessions,
        "max_chars": None,
    }


def load_existing_search_results(frame: str, version: str, conv_id: str):
    path = _tmp_path(frame, version, conv_id)
    if not path.exists():
        return defaultdict(list), set()
    try:
        with path.open() as f:
            data = json.load(f)
        done = {
            entry.get("question", "")
            for entries in data.values()
            for entry in entries
            if entry.get("question")
        }
        return defaultdict(list, data), done
    except Exception as exc:
        print(f"WARNING: ignoring invalid existing search tmp for {conv_id}: {exc}")
        return defaultdict(list), set()


def search_conversation(
    conv,
    frame: str,
    version: str,
    top_k: int,
    *,
    allow_empty_search: bool,
) -> tuple[dict, list[float]]:
    conv_id = str(conv["conversation_id"])
    user_id = user_id_for(version, conv_id)
    search_results, existing_questions = load_existing_search_results(frame, version, conv_id)
    path = _tmp_path(frame, version, conv_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    question_meta = build_question_meta(
        conv,
        version=version,
        parse_probing_questions=parse_probing_questions,
    )

    client = create_client(frame)
    durations: list[float] = []
    for meta in question_meta:
        question_text = meta["question"].get("question", "")
        if question_text in existing_questions:
            continue

        print(f"  Searching BEAM conv {conv_id}: {question_text[:100]}...")
        result = dispatch_search(frame, client, question_text, user_id, top_k)
        context, duration_ms, reflect_answer, raw_context = unpack_search_result(result)
        durations.append(duration_ms)
        context = context or ""
        status = classify_search_status(
            context,
            reflect_answer,
            raw_context=raw_context,
        )
        if status == STATUS_SUCCESS_EMPTY and not allow_empty_search:
            raise RuntimeError(f"search returned no raw memories for BEAM conversation {conv_id}")

        entry = build_search_entry(
            meta,
            context=context,
            duration_ms=duration_ms,
            status=status,
            reflect_answer=reflect_answer,
        )
        search_results[user_id].append(entry)
        existing_questions.add(question_text)

        atomic_json_dump(dict(search_results), path, indent=4)

    return dict(search_results), durations


def write_combined_results(frame: str, version: str, completed: set[str]) -> None:
    combined: dict[str, list] = defaultdict(list)
    for conv_id in sorted(completed, key=str):
        path = _tmp_path(frame, version, conv_id)
        if not path.exists():
            continue
        with path.open() as f:
            data = json.load(f)
        for user_id, entries in data.items():
            combined[user_id].extend(entries)

    out = _combined_path(frame, version)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(dict(combined), out, indent=4)


def write_search_status(
    frame: str,
    version: str,
    completed: set[str],
    *,
    allow_empty_search: bool,
    skip_failed_streaming: bool,
    failed_users: list[dict],
    skipped_records: list[dict],
) -> None:
    records = []
    for conv_id in sorted(completed, key=str):
        path = _tmp_path(frame, version, conv_id)
        if not path.exists():
            continue
        with path.open() as f:
            data = json.load(f)
        for entries in data.values():
            if isinstance(entries, list):
                records.extend(entry for entry in entries if isinstance(entry, dict))

    atomic_json_dump(
        {
            "stage": "search",
            "mode": "beam_streaming",
            "allow_empty_search": allow_empty_search,
            "skip_failed_streaming": skip_failed_streaming,
            "status_counts": status_counts(records),
            "failed_users": failed_users,
            "skipped_records": skipped_records,
        },
        _results_dir(frame, version) / f"{frame}_beam_search_status.json",
        indent=2,
    )


def load_stats(frame: str, version: str) -> dict:
    path = _stats_path(frame, version)
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {
        "mode": "beam_streaming",
        "unit_durations_ms": {},
        "user_durations_ms": {},
        "add_call_durations_by_unit": {},
        "add_call_durations_ms": [],
        "search_call_durations_ms": [],
        "final_delete_statuses": {},
        "ingest_modes": {},
        "chunk_counts": {},
        "message_counts": {},
        "char_counts": {},
    }


def save_stats(frame: str, version: str, stats: dict) -> None:
    path = _stats_path(frame, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(stats, path, indent=2)


def process_conversation(
    conv,
    *,
    frame: str,
    version: str,
    top_k: int,
    allow_empty_search: bool,
    ingest_mode: str,
    max_batch_chars: int,
    wait_after_ingest: float,
    completed: set[str],
    stats: dict,
    restart_unit: bool,
) -> None:
    conv_id = str(conv["conversation_id"])
    user_id = user_id_for(version, conv_id)
    started = time.time()
    print("\n" + "=" * 80)
    print(f"BEAM STREAM conversation {conv_id}: add({ingest_mode}), search, delete")
    print("=" * 80)

    client = create_client(frame)
    added_path = _added_chunks_path(frame, version, conv_id)
    added_chunks = load_added_chunks(added_path)
    supports_add_resume = ingest_mode in ("chunked", "per-session")
    should_start_fresh = restart_unit or not (supports_add_resume and added_chunks)

    if should_start_fresh:
        delete_user_data(frame, client, user_id)
        prepare_user_after_delete(frame, client, user_id)
        if added_path.exists():
            added_path.unlink()
        added_chunks = set()
        tmp_path = _tmp_path(frame, version, conv_id)
        if tmp_path.exists():
            tmp_path.unlink()
    else:
        unit_label = "chunks" if ingest_mode == "chunked" else "sessions"
        print(
            f"Resuming BEAM conversation {conv_id}: "
            f"{len(added_chunks)} {unit_label} recorded",
            flush=True,
        )

    if ingest_mode == "chunked":
        if frame != "graphiti":
            raise RuntimeError("chunked add requires a raw-chunk capable client")
        add_durations, meta = add_conversation_graphiti_chunked(
            conv,
            version,
            client,
            max_chars=max_batch_chars,
            added_chunks_path=added_path,
            added_chunks=added_chunks,
        )
    else:
        add_durations, meta = add_conversation_per_session(
            conv,
            frame,
            version,
            client,
            added_sessions_path=added_path,
            added_sessions=added_chunks,
        )

    if wait_after_ingest > 0:
        print(f"Waiting {wait_after_ingest}s after ingest")
        time.sleep(wait_after_ingest)

    _, search_durations = search_conversation(
        conv,
        frame,
        version,
        top_k,
        allow_empty_search=allow_empty_search,
    )

    final_delete_ok, final_delete_error = delete_user_data(
        frame,
        client,
        user_id,
    )
    final_delete_status = "ok" if final_delete_ok else "error_skipped"
    if final_delete_ok and added_path.exists():
        added_path.unlink()
    if not final_delete_ok:
        log_event(
            _events_path(frame, version),
            "final_delete_error_skipped",
            conv_id=conv_id,
            user_id=user_id,
            error=final_delete_error,
        )

    mark_completed(_completed_path(frame, version), completed, conv_id)
    elapsed_ms = round((time.time() - started) * 1000, 1)

    stats.setdefault("unit_durations_ms", {})[conv_id] = elapsed_ms
    stats.setdefault("user_durations_ms", {})[conv_id] = elapsed_ms
    update_unit_duration_list(
        stats,
        conv_id,
        add_durations,
        map_key="add_call_durations_by_unit",
        flat_key="add_call_durations_ms",
    )
    stats.setdefault("search_call_durations_ms", []).extend(round(v, 2) for v in search_durations)
    stats.setdefault("final_delete_statuses", {})[conv_id] = final_delete_status
    stats.setdefault("ingest_modes", {})[conv_id] = ingest_mode
    stats.setdefault("chunk_counts", {})[conv_id] = meta["chunks"]
    stats.setdefault("message_counts", {})[conv_id] = meta["messages"]
    stats.setdefault("char_counts", {})[conv_id] = meta["chars"]
    if ingest_mode == "chunked":
        stats["max_batch_chars"] = max_batch_chars

    log_event(
        _events_path(frame, version),
        "completed",
        conv_id=conv_id,
        ingest_mode=ingest_mode,
        chunks=meta["chunks"],
        skipped_chunks=meta["skipped_chunks"],
        messages=meta["messages"],
        chars=meta["chars"],
        final_delete_status=final_delete_status,
    )
    write_combined_results(frame, version, completed)
    save_stats(frame, version, stats)


def mark_streaming_failure_skipped(
    conv,
    frame: str,
    version: str,
    completed: set[str],
    exc: BaseException,
) -> dict:
    conv_id = str(conv["conversation_id"])
    user_id = user_id_for(version, conv_id)
    tmp_path = _tmp_path(frame, version, conv_id)
    if not tmp_path.exists():
        entries = [
            build_search_entry(
                meta,
                context="",
                duration_ms=0.0,
                status=STATUS_SKIPPED,
                error=error_payload("streaming", exc),
            )
            for meta in build_question_meta(
                conv,
                version=version,
                parse_probing_questions=parse_probing_questions,
            )
        ]
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_dump({user_id: entries}, tmp_path, indent=4)
    mark_completed(_completed_path(frame, version), completed, conv_id)
    return {
        "conv_id": conv_id,
        "user_id": user_id,
        "error": error_payload("streaming", exc),
    }


def main() -> int:
    parser = argparse.ArgumentParser("BEAM streaming add-search-delete")
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
    parser.add_argument(
        "--scale",
        choices=["100k", "500k", "1m", "10m", "all"],
        default="10m",
    )
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int)
    parser.add_argument("--wait-after-ingest", type=float, default=0.0)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore streaming_completed.txt and process selected conversations anyway.",
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
    args = parser.parse_args()

    if args.env:
        env_path = Path(args.env)
        if not env_path.is_file():
            env_path = Path.cwd() / args.env
        if not env_path.is_file():
            raise SystemExit(f"Env file not found: {args.env}")
        os.environ["OMNIMEMEVAL_ENV_FILE"] = str(env_path.resolve())

    load_env()
    configure_single_user_streaming(args.lib)

    max_batch_chars = resolve_max_batch_chars(args.lib)

    ingest_mode = "chunked" if args.lib == "graphiti" else "per-session"

    if ingest_mode == "chunked" and args.lib != "graphiti":
        raise SystemExit("chunked add is currently supported only for raw-chunk capable clients")

    results_dir = _results_dir(args.lib, args.version)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "tmp").mkdir(parents=True, exist_ok=True)

    conversations = load_beam_data(args.scale)
    total = len(conversations)
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
    print("BEAM STREAMING")
    print("=" * 80)
    print(f"lib={args.lib}")
    print(f"version={args.version}")
    print(f"scale={args.scale}")
    print(f"range={args.start_idx}-{end_idx}")
    print(f"top_k={args.top_k}")
    print(f"allow_empty_search={args.allow_empty_search}")
    print(f"ingest_mode={ingest_mode}")
    if ingest_mode == "chunked":
        print(f"max_batch_chars={max_batch_chars}")
    print(f"wait_after_ingest={args.wait_after_ingest}")
    print(f"already_completed={len(completed)}")
    print("=" * 80)

    failed_users: list[dict] = []
    skipped_records: list[dict] = []
    for row_idx in range(args.start_idx, end_idx + 1):
        conv = conversations[row_idx]
        conv_id = str(conv["conversation_id"])
        if conv_id in completed and not args.no_resume:
            print(f"Skipping BEAM conversation {conv_id}: already completed")
            continue
        try:
            process_conversation(
                conv,
                frame=args.lib,
                version=args.version,
                top_k=args.top_k,
                allow_empty_search=args.allow_empty_search,
                ingest_mode=ingest_mode,
                max_batch_chars=max_batch_chars,
                wait_after_ingest=args.wait_after_ingest,
                completed=completed,
                stats=stats,
                restart_unit=args.restart_unit or (args.no_resume and conv_id in existing_completed),
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            raise
        except Exception as exc:
            log_event(
                _events_path(args.lib, args.version),
                "failed",
                conv_id=conv_id,
                error=str(exc),
            )
            print(f"ERROR BEAM conversation {conv_id}: {type(exc).__name__}: {exc}")
            failure = {
                "conv_id": conv_id,
                "user_id": user_id_for(args.version, conv_id),
                "error": error_payload("streaming", exc),
            }
            if args.skip_failed_streaming:
                skipped_records.append(
                    mark_streaming_failure_skipped(
                        conv,
                        args.lib,
                        args.version,
                        completed,
                        exc,
                    )
                )
                write_combined_results(args.lib, args.version, completed)
                save_stats(args.lib, args.version, stats)
                continue
            failed_users.append(failure)
            continue

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
        print(f"\nBEAM streaming failed for {len(failed_users)} conversation(s)")
        return 1
    print("\nBEAM streaming complete")
    print(f"Combined search results: {_combined_path(args.lib, args.version)}")
    print(f"Completed conversations: {len(completed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

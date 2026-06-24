import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from utils.env import load_env
from utils.progress import track
from utils.time import parse_lme_time, to_iso
from utils.checkpoint import fsync_write_line
from utils.ingest_helpers import inject_time, session_id_kwargs, AddCallTimer
from utils.streaming import LongCallLogger
from client_factory import SUPPORTED_LIBS, DEFAULT_LIB
from longmemeval.lme_data import load_lme_dataframe, sanitize_lme_message_content

_LME_RETAIN_MISSION = (
    "Extract ALL factual claims from this conversation, paying special attention to: "
    "1) TEMPORAL facts — dates, times, durations, sequences of events, and when things "
    "happened relative to each other; "
    "2) Personal details — names, preferences, habits, relationships, occupations; "
    "3) Plans and intentions — future plans, goals, commitments with their timeframes; "
    "4) Changes over time — if something changed (moved cities, switched jobs, etc.), "
    "extract BOTH the before and after states with their timestamps. "
    "Preserve exact dates and times whenever mentioned. "
    "Extract negative statements ('I have never...', 'I don't...') as separate facts."
)


def _clean_session_messages(session, session_id):
    """Return LME messages after dropping empty text turns."""
    messages = []
    dropped = 0
    sanitized = 0
    for msg in session:
        raw_content = msg.get("content", "")
        content = sanitize_lme_message_content(raw_content)
        if content != raw_content:
            sanitized += 1
        if content is None or str(content).strip() == "":
            dropped += 1
            continue
        messages.append({"role": msg["role"], "content": content})
    if sanitized:
        print(f"  ⚠ Session {session_id}: sanitized {sanitized} special-token message(s)")
    if dropped:
        print(f"  ⚠ Session {session_id}: dropped {dropped} empty message(s)")
    return messages


def ingest_session(session, date, user_id, session_id, frame, client):
    messages = _clean_session_messages(session, session_id)
    if not messages:
        print(f"[{frame}] Session {session_id}: skipped empty session at {to_iso(date)}")
        return
    char_count = sum(len(str(msg.get("content", ""))) for msg in messages)

    time_kw = inject_time(messages, date, frame)
    sess_kw = session_id_kwargs(frame, session_id)

    label = (
        f"{frame} LME add user={user_id} session={session_id} "
        f"messages={len(messages)} chars={char_count}"
    )
    with LongCallLogger(label):
        if frame == "letta":
            client.add(messages, user_id, **sess_kw, **time_kw)
        elif frame == "hindsight":
            date_display = date.strftime("%Y-%m-%d %H:%M:%S")
            context_str = (
                f"Session {session_id} - you are the assistant in this "
                f"conversation - happened on {date_display} UTC."
            )
            client.add(
                [], user_id,
                raw_content=json.dumps(messages),
                context=context_str,
                retain_mission=_LME_RETAIN_MISSION,
                **sess_kw, **time_kw,
            )
        elif frame == "supermemory":
            client.add(messages, user_id, session_id=session_id, **time_kw)
        else:
            client.add(messages, user_id, **sess_kw, **time_kw)

    print(f"[{frame}] Session {session_id}: Ingested {len(messages)} messages at {to_iso(date)}")


def _everos_lme_flush_once():
    return os.getenv("EVEROS_LME_FLUSH_ONCE", "false").lower() in ("true", "1", "yes")


def ingest_conv(lme_df, version, conv_idx, frame, success_records, f, clear=False):
    import time as _time
    conv_start = _time.time()

    conversation = lme_df.iloc[conv_idx]
    sessions = conversation["haystack_sessions"]
    dates = conversation["haystack_dates"]

    user_id = f"lme_exper_user_{version}_{conv_idx}"

    print("\n" + "=" * 80)
    print(f"🔄 [INGESTING CONVERSATION {conv_idx}".center(80))
    print("=" * 80)

    from client_factory import create_client

    client = create_client(frame)
    timer = AddCallTimer(client)
    everos_flush_once = frame == "everos" and _everos_lme_flush_once()
    everos_conv_session_id = f"{user_id}_lme_exper_all_sessions"
    has_completed_sessions = any(
        f"{conv_idx}_{session_idx}" in success_records
        for session_idx in range(len(sessions))
    )
    if has_completed_sessions:
        print(f"  Resuming conversation {conv_idx}: keeping existing user memory")
    else:
        try:
            if frame == "zep":
                client.delete_user(user_id)
                client.add_user(user_id)
            elif "mem0" in frame:
                client.delete_all(user_id=user_id)
            elif frame == "everos" and clear:
                if hasattr(client, "delete"):
                    client.delete(user_id)
            elif frame == "supermemory" and clear:
                client.delete(user_id)
            elif clear and hasattr(client, "delete"):
                client.delete(user_id)
            elif clear and hasattr(client, "delete_user"):
                client.delete_user(user_id)
        except Exception as exc:
            print(f"  ⚠ Cleanup failed for {user_id}, continuing ingestion: {exc}")

    failed_sessions = []
    pending_success_records = []
    for idx, session in enumerate(sessions):
        record_key = f"{conv_idx}_{idx}"
        if record_key not in success_records:
            session_id = user_id + "_lme_exper_session_" + str(idx)
            add_session_id = everos_conv_session_id if everos_flush_once else session_id
            date_string = parse_lme_time(dates[idx])

            try:
                if everos_flush_once:
                    messages = _clean_session_messages(session, session_id)
                    if not messages:
                        print(f"[{frame}] Session {session_id}: skipped empty session at {to_iso(date_string)}")
                        fsync_write_line(f, record_key)
                        continue
                    time_kw = inject_time(messages, date_string, frame)
                    char_count = sum(len(str(msg.get("content", ""))) for msg in messages)
                    label = (
                        f"{frame} LME add user={user_id} session={session_id} "
                        f"conv_id={add_session_id} messages={len(messages)} "
                        f"chars={char_count} flush=false"
                    )
                    with LongCallLogger(label):
                        client.add(
                            messages,
                            user_id,
                            conv_id=add_session_id,
                            flush=False,
                            **time_kw,
                        )
                    print(f"[{frame}] Session {session_id}: Ingested {len(messages)} messages at {to_iso(date_string)}")
                else:
                    ingest_session(session, date_string, user_id, session_id, frame, client)
                    fsync_write_line(f, record_key)
                pending_success_records.append(record_key)
            except Exception as e:
                import traceback
                print(f"❌ Error ingesting session {record_key}: {type(e).__name__}: {e}")
                traceback.print_exc()
                failed_sessions.append(record_key)
        else:
            print(f"✅ Session {record_key} already ingested")

    if failed_sessions:
        raise RuntimeError(
            f"Conversation {conv_idx}: {len(failed_sessions)}/{len(sessions)} sessions failed: {failed_sessions}"
        )

    if everos_flush_once and pending_success_records:
        with LongCallLogger(
            f"{frame} LME flush user={user_id} session={everos_conv_session_id} "
            f"sessions={len(pending_success_records)}"
        ):
            client.flush(user_id, session_id=everos_conv_session_id)
        for record_key in pending_success_records:
            fsync_write_line(f, record_key)

    print("=" * 80)
    return round((_time.time() - conv_start) * 1000, 1), timer.durations_ms


def main(frame, version, num_workers=2, clear=False):
    load_env()

    print("\n" + "=" * 80)
    print(f"🚀 LONGMEMEVAL INGESTION - {frame.upper()} v{version}".center(80))
    print("=" * 80)
    if clear:
        print("🧹 --clear enabled: will delete existing memories before ingestion")

    lme_df = load_lme_dataframe()

    num_multi_sessions = len(lme_df)
    print(f"👥 Number of users: {num_multi_sessions}")
    print("-" * 80)

    start_time = datetime.now()
    os.makedirs(f"results/lme/{frame}-{version}/", exist_ok=True)
    success_records = set()
    record_file = f"results/lme/{frame}-{version}/success_records.txt"
    if clear and os.path.exists(record_file):
        os.remove(record_file)
        print("🧹 Cleared progress records")
    if os.path.exists(record_file):
        with open(record_file) as f:
            for i in f.readlines():
                success_records.add(i.strip())

    user_durations = {}
    all_add_call_durations = []
    failed_conversations = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor, open(record_file, "a+") as f:
        future_to_idx = {}
        for session_idx in range(num_multi_sessions):
            future = executor.submit(
                ingest_conv, lme_df, version, session_idx, frame, success_records, f, clear
            )
            future_to_idx[future] = session_idx

        for future in track(
            as_completed(future_to_idx), total=len(future_to_idx), description="Ingesting conversations",
        ):
            idx = future_to_idx[future]
            try:
                dur_ms, add_call_ms = future.result()
                user_durations[str(idx)] = dur_ms
                all_add_call_durations.extend(add_call_ms)
            except Exception as e:
                import traceback
                print(f"❌ Error processing conversation {idx}: {type(e).__name__}: {e}")
                traceback.print_exc()
                failed_conversations.append(idx)

    stats_path = os.path.join(f"results/lme/{frame}-{version}", f"{frame}_lme_ingestion_stats.json")
    with open(stats_path, "w") as sf:
        json.dump({"user_durations_ms": user_durations, "add_call_durations_ms": [round(d, 2) for d in all_add_call_durations]}, sf, indent=2)
    print(f"Ingestion stats saved to {stats_path}")

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    elapsed_time_str = str(elapsed_time).split(".")[0]

    if failed_conversations:
        print("\n" + "=" * 80)
        print(f"❌ INGESTION FAILED: {len(failed_conversations)}/{num_multi_sessions} conversations had errors".center(80))
        print("=" * 80)
        print(f"⏱️  Total time: {elapsed_time_str}")
        print("💡 Fix errors and re-run — successfully ingested sessions are saved in success_records.txt")
        print("=" * 80 + "\n")
        raise SystemExit(1)

    print("\n" + "=" * 80)
    print("✅ INGESTION COMPLETE".center(80))
    print("=" * 80)
    print(f"⏱️  Total time taken to ingest {num_multi_sessions} multi-sessions: {elapsed_time_str}")
    print(f"🔄 Framework: {frame} | Version: {version} | Workers: {num_workers}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LongMemeval Ingestion Script")
    parser.add_argument(
        "--lib",
        type=str,
        choices=SUPPORTED_LIBS,
        default=DEFAULT_LIB,
    )
    parser.add_argument(
        "--version", type=str, default="default", help="Version of the evaluation framework."
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Number of parallel ingestion workers."
    )
    parser.add_argument(
        "--clear", action="store_true", help="Clear existing memories before ingestion"
    )

    args = parser.parse_args()
    main(frame=args.lib, version=args.version, num_workers=args.workers, clear=args.clear)

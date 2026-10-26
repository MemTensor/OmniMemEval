import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)

sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, SCRIPT_DIR)

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from utils.env import load_env
from utils.progress import track
from utils.checkpoint import atomic_json_dump, fsync_write_line
from utils.ingest_helpers import inject_time, session_id_kwargs, AddCallTimer
from utils.streaming import LongCallLogger
from utils.time import parse_beam_time
from client_factory import SUPPORTED_LIBS, DEFAULT_LIB

SCALE_FILE_MAP = {
    "100k": "beam_100k.json",
    "500k": "beam_500k.json",
    "1m": "beam_1m.json",
    "10m": "beam_10m_10m.json",
}


def load_beam_data(scale="all"):
    data_dir = os.path.join(PROJECT_DIR, "data", "beam")
    all_conversations = []

    if scale == "all":
        scales = list(SCALE_FILE_MAP.keys())
    else:
        scales = [scale]

    for s in scales:
        filepath = os.path.join(data_dir, SCALE_FILE_MAP[s])
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                conv = json.loads(line)
                conv["_scale"] = s
                all_conversations.append(conv)

    return all_conversations


def flatten_session(session):
    """Normalize a BEAM session to a flat list of messages.

    100k/500k/1m sessions are already ``list[message]``.
    10m sessions are ``dict{plan-N: [batch{turns: [[msg, ...], ...]}]}``;
    we flatten them into a single ordered message list.
    """
    if isinstance(session, list):
        return session

    messages = []
    for plan_key in sorted(session.keys()):
        batches = session[plan_key]
        if not batches:
            continue
        for batch in batches:
            for turn in batch.get("turns", []):
                if isinstance(turn, list):
                    messages.extend(turn)
                elif isinstance(turn, dict):
                    messages.append(turn)
    return messages


def parse_time_anchor(time_anchor_str):
    return parse_beam_time(time_anchor_str)


def session_time_anchor(messages, default="January-01-2024"):
    """Return the first non-empty ``time_anchor`` in a session, else ``default``.

    BEAM-10M conversation 7 session 0 opens with null anchors; its first
    anchor (index 244) is March-05-2020. Every other 10M session starts with
    an anchor, so they keep the date of their first message.
    """
    for msg in messages:
        if isinstance(msg, dict) and msg.get("time_anchor"):
            return msg["time_anchor"]
    return default


_BEAM_RETAIN_MISSION = (
    "Extract ALL factual claims the user makes about themselves, their project, "
    "and their experience — including NEGATIVE statements (e.g. 'I have never done X', "
    "'I don't know Y', 'I haven't used Z'). Negative self-assessments and denials "
    "are as important as positive ones. Also preserve contradictions: if the user "
    "says opposite things at different points, extract BOTH statements as separate facts. "
    "Preserve specific numbers, dates, versions, and quantities exactly as stated."
)


def ingest_session(client, messages, frame, user_id, session_id, session_date):
    chat_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
    char_count = sum(len(str(msg.get("content", ""))) for msg in chat_messages)

    time_kw = inject_time(chat_messages, session_date, frame)
    sess_kw = {"session_id": session_id} if frame == "lcmx" else session_id_kwargs(frame, session_id)

    label = (
        f"{frame} BEAM add user={user_id} session={session_id} "
        f"messages={len(chat_messages)} chars={char_count}"
    )
    with LongCallLogger(label):
        if frame == "letta":
            client.add(chat_messages, user_id, **sess_kw, **time_kw)
        elif frame == "hindsight":
            date_display = session_date.strftime("%Y-%m-%d %H:%M:%S") if session_date else ""
            context_str = (
                f"Session {session_id} - you are the assistant in this "
                f"conversation - happened on {date_display} UTC."
            )
            client.add(
                [], user_id,
                raw_content=json.dumps(messages),
                context=context_str,
                retain_mission=_BEAM_RETAIN_MISSION,
                **sess_kw, **time_kw,
            )
        elif frame == "supermemory":
            client.add(chat_messages, user_id, session_id=session_id, **time_kw)
        else:
            client.add(chat_messages, user_id, **sess_kw, **time_kw)

    return len(chat_messages)


def ingest_conversation(conv, version, frame, success_records, f, clear=False):
    import time as _time
    conv_start = _time.time()

    conv_id = conv["conversation_id"]
    user_id = f"beam_exp_user_{version}_{conv_id}"
    sessions = conv["chat"]

    print(f"\n{'=' * 80}")
    print(f"🔄 [INGESTING CONVERSATION {conv_id}]".center(80))
    print(f"{'=' * 80}")

    from client_factory import create_client

    client = create_client(frame)
    timer = AddCallTimer(client)
    has_completed_sessions = any(
        f"{conv_id}_{sess_idx}" in success_records
        for sess_idx in range(len(sessions))
    )
    if has_completed_sessions:
        print(f"  Resuming conversation {conv_id}: keeping existing user memory")
    else:
        try:
            if frame == "zep":
                client.delete_user(user_id)
                client.add_user(user_id)
            elif "mem0" in frame:
                client.delete_all(user_id=user_id)
            elif clear and hasattr(client, "delete"):
                client.delete(user_id)
            elif clear and hasattr(client, "delete_user"):
                client.delete_user(user_id)
        except Exception as exc:
            print(f"  ⚠ Cleanup failed for {user_id}, continuing ingestion: {exc}")

    failed_sessions = []
    for sess_idx, raw_session in enumerate(sessions):
        record_key = f"{conv_id}_{sess_idx}"
        if record_key in success_records:
            print(f"✅ Session {record_key} already ingested")
            continue

        session = flatten_session(raw_session)
        if not session:
            continue

        session_date = parse_time_anchor(session_time_anchor(session))
        session_id = f"{user_id}_session_{sess_idx}"

        try:
            msg_count = ingest_session(
                client, session, frame, user_id, session_id, session_date
            )
            fsync_write_line(f, record_key)
            print(
                f"[{frame}] ✅ Session {sess_idx}: Ingested {msg_count} messages "
                f"at {session_date.isoformat()}"
            )
        except Exception as e:
            import traceback
            print(f"❌ Error ingesting session {record_key}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed_sessions.append(record_key)

    if failed_sessions:
        print(
            f"  ⚠ Conversation {conv_id}: {len(failed_sessions)}/{len(sessions)} "
            f"sessions failed: {failed_sessions} — continuing with remaining data"
        )

    print(f"{'=' * 80}")
    return (
        conv_id,
        round((_time.time() - conv_start) * 1000, 1),
        timer.durations_ms,
        failed_sessions,
    )


def main(frame, version, num_workers=2, clear=False, scale="all"):
    load_env()

    if frame == "everos":
        os.environ["EVEROS_USE_GROUP"] = "false"

    print(f"\n{'=' * 80}")
    print(f"🚀 BEAM INGESTION - {frame.upper()} v{version} (scale={scale})".center(80))
    print(f"{'=' * 80}")
    if clear:
        print("🧹 --clear enabled: will delete existing memories before ingestion")

    conversations = load_beam_data(scale)
    num_conversations = len(conversations)
    print(f"📚 Loaded {num_conversations} BEAM conversations (scale={scale})")
    print(f"{'-' * 80}")

    start_time = datetime.now()
    results_dir = f"results/beam/{frame}-{version}"
    os.makedirs(results_dir, exist_ok=True)

    success_records = set()
    record_file = os.path.join(results_dir, "success_records.txt")
    if clear and os.path.exists(record_file):
        os.remove(record_file)
        print("🧹 Cleared progress records")
    if os.path.exists(record_file):
        with open(record_file) as f:
            for line in f.readlines():
                success_records.add(line.strip())

    user_durations = {}
    all_add_call_durations = []
    failed_conversations = []
    with (
        ThreadPoolExecutor(max_workers=num_workers) as executor,
        open(record_file, "a+") as f,
    ):
        futures = []
        for conv in conversations:
            future = executor.submit(
                ingest_conversation, conv, version, frame, success_records, f, clear
            )
            futures.append(future)

        for future in track(
            as_completed(futures),
            total=len(futures),
            description="Ingesting conversations",
        ):
            try:
                cid, dur_ms, add_call_ms, failed_sessions = future.result()
                user_durations[str(cid)] = dur_ms
                all_add_call_durations.extend(add_call_ms)
                if failed_sessions:
                    failed_conversations.append({
                        "conv_id": str(cid),
                        "failed_sessions": failed_sessions,
                    })
            except Exception as e:
                import traceback
                print(f"❌ Error processing conversation: {type(e).__name__}: {e}")
                traceback.print_exc()
                failed_conversations.append(str(e))

    stats_path = os.path.join(results_dir, f"{frame}_beam_ingestion_stats.json")
    atomic_json_dump(
        {
            "user_durations_ms": user_durations,
            "add_call_durations_ms": [round(d, 2) for d in all_add_call_durations],
            "failed_conversations": failed_conversations,
        },
        stats_path,
        indent=2,
    )
    print(f"Ingestion stats saved to {stats_path}")

    end_time = datetime.now()
    elapsed = str(end_time - start_time).split(".")[0]

    if failed_conversations:
        print(f"\n{'=' * 80}")
        print(f"❌ INGESTION FAILED: {len(failed_conversations)}/{num_conversations} conversations had errors".center(80))
        print(f"{'=' * 80}")
        print(f"⏱️  Total time: {elapsed}")
        print("💡 Fix errors and re-run — successfully ingested sessions are saved in success_records.txt")
        print(f"{'=' * 80}\n")
        raise SystemExit(1)

    print(f"\n{'=' * 80}")
    print("✅ INGESTION COMPLETE".center(80))
    print(f"{'=' * 80}")
    print(f"⏱️  Total time: {elapsed}")
    print(f"🔄 Framework: {frame} | Version: {version} | Workers: {num_workers}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEAM Ingestion Script")
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
    parser.add_argument(
        "--scale",
        type=str,
        default="all",
        choices=["100k", "500k", "1m", "10m", "all"],
        help="BEAM evaluation scale (100k/500k/1m/10m/all).",
    )

    args = parser.parse_args()
    main(
        frame=args.lib,
        version=args.version,
        num_workers=args.workers,
        clear=args.clear,
        scale=args.scale,
    )

import argparse
import concurrent.futures
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, SCRIPT_DIR)

import pandas as pd

from utils.env import load_env
from utils.progress import track
from utils.time import parse_locomo_time, to_iso, to_unix
from utils.checkpoint import fsync_write_line
from utils.ingest_helpers import inject_time, AddCallTimer
from client_factory import SUPPORTED_LIBS, DEFAULT_LIB


def ingest_session(client, session, frame, metadata):
    dt = parse_locomo_time(metadata["session_date"])
    iso_date = to_iso(dt)
    conv_idx = metadata["conv_idx"]
    conv_id = f"locomo_exp_user_{conv_idx}"
    session_key = metadata["session_key"]
    print(f"Processing conv {conv_id}, session {session_key}")
    start_time = time.time()

    speaker_a_user_id = metadata["speaker_a_user_id"]
    speaker_b_user_id = metadata["speaker_b_user_id"]

    speaker_a_messages = []
    speaker_b_messages = []
    for chat in session:
        data = chat.get("speaker") + ": " + chat.get("text")
        if chat.get("speaker") == metadata["speaker_a"]:
            speaker_a_messages.append({
                "role": "user", "name": metadata["speaker_a"], "content": data,
            })
            speaker_b_messages.append({
                "role": "assistant", "name": metadata["speaker_a"], "content": data,
            })
        elif chat.get("speaker") == metadata["speaker_b"]:
            speaker_a_messages.append({
                "role": "assistant", "name": metadata["speaker_b"], "content": data,
            })
            speaker_b_messages.append({
                "role": "user", "name": metadata["speaker_b"], "content": data,
            })

    # ── Zep: graph or thread mode ────────────────────────────────────────
    if frame == "zep":
        group_id = metadata.get("group_id")
        if group_id:
            for chat in session:
                speaker = chat.get("speaker", "")
                blip_caption = chat.get("blip_captions")
                img_desc = (
                    f" (description of attached image: {blip_caption})"
                    if blip_caption else ""
                )
                data = speaker + ": " + chat.get("text") + img_desc
                client.sdk_graph_add(
                    graph_id=group_id, data=data,
                    data_type="message", created_at=iso_date,
                )
        else:
            ts = to_unix(dt)
            client.add(speaker_a_messages, speaker_a_user_id, ts,
                        thread_id=f"{speaker_a_user_id}_{session_key}")
            client.add(speaker_b_messages, speaker_b_user_id, ts,
                        thread_id=f"{speaker_b_user_id}_{session_key}")

    # ── EverOS: group or personal mode ───────────────────────────────────
    elif frame == "everos":
        use_group = os.environ.get("EVEROS_USE_GROUP", "true").lower() in ("true", "1", "yes")
        if use_group:
            all_messages = [{
                "role": "user",
                "name": chat.get("speaker"),
                "content": chat.get("speaker") + ": " + chat.get("text"),
                "chat_time": iso_date,
            } for chat in session]
            client.add_group(all_messages, speaker_a_user_id)
        else:
            inject_time(speaker_a_messages, dt, frame)
            inject_time(speaker_b_messages, dt, frame)
            client.add(speaker_a_messages, speaker_a_user_id,
                        f"{conv_id}_{session_key}")
            client.add(speaker_b_messages, speaker_b_user_id,
                        f"{conv_id}_{session_key}")

    # ── Supermemory: single combined session ─────────────────────────────
    elif frame == "supermemory":
        all_messages = [{
            "role": "user" if chat.get("speaker") == metadata["speaker_a"] else "assistant",
            "content": chat.get("text"),
            "speaker": chat.get("speaker"),
        } for chat in session]
        time_kw = inject_time(all_messages, dt, frame)
        client.add(all_messages, speaker_a_user_id,
                    session_id=f"{conv_id}_{session_key}", **time_kw)

    # ── Backboard: single combined session ───────────────────────────────
    elif frame == "backboard":
        all_messages = [{
            "role": "user",
            "name": chat.get("speaker"),
            "content": chat.get("text"),
        } for chat in session]
        inject_time(all_messages, dt, frame)
        client.add(all_messages, speaker_a_user_id, session_key=session_key)

    # ── Graphiti: two user graphs, one full-session episode per user ─────
    elif frame == "graphiti":
        lines = [
            (
                f"{chat.get('speaker')}: {chat.get('text')}"
                + (
                    f" (description of attached image: {chat.get('blip_captions')})"
                    if chat.get("blip_captions")
                    else ""
                )
            )
            for chat in session
        ]
        transcript = "\n".join(lines)
        source_description = (
            f"LoCoMo conversation between {metadata['speaker_a']} and "
            f"{metadata['speaker_b']} ({session_key} of {conv_id})"
        )
        client.add(
            [],
            speaker_a_user_id,
            session_key=session_key,
            timestamp=iso_date,
            raw_content=transcript,
            role=f"{metadata['speaker_a']}-{metadata['speaker_b']}",
            source_description=source_description,
        )
        client.add(
            [],
            speaker_b_user_id,
            session_key=session_key,
            timestamp=iso_date,
            raw_content=transcript,
            role=f"{metadata['speaker_a']}-{metadata['speaker_b']}",
            source_description=source_description,
        )

    # ── Hindsight: raw JSON content ──────────────────────────────────────
    elif frame == "hindsight":
        time_kw = inject_time(speaker_a_messages, dt, frame)
        context_str = (
            f"Conversation between {metadata['speaker_a']} and "
            f"{metadata['speaker_b']} ({session_key} of {conv_id})"
        )
        client.add(
            speaker_a_messages, speaker_a_user_id,
            session_key=session_key,
            raw_content=json.dumps(session),
            context=context_str,
            **time_kw,
        )

    # ── Letta: single-agent per conversation ─────────────────────────────
    elif frame == "letta":
        letta_msgs = [
            {**m, "content": m["content"][len(m["name"]) + 2:]}
            if m.get("name") and m["content"].startswith(f"{m['name']}: ")
            else m
            for m in speaker_a_messages
        ]
        time_kw = inject_time(letta_msgs, dt, frame)
        client.add(letta_msgs, speaker_a_user_id,
                    session_key=session_key, **time_kw)

    # ── Cognee: single-dataset per conversation ──────────────────────────
    elif frame == "cognee":
        inject_time(speaker_a_messages, dt, frame)
        client.add(speaker_a_messages, speaker_a_user_id,
                    session_key=session_key)

    # ── Default: dual-speaker with inject_time ───────────────────────────
    else:
        inject_time(speaker_a_messages, dt, frame)
        inject_time(speaker_b_messages, dt, frame)
        sess_id = f"{conv_id}_{session_key}"

        if frame == "memos":
            client.add(speaker_a_messages, speaker_a_user_id, sess_id)
            client.add(speaker_b_messages, speaker_b_user_id, sess_id)
        elif frame == "memorylake":
            client.add(speaker_a_messages, speaker_a_user_id, session_key=session_key)
            client.add(speaker_b_messages, speaker_b_user_id, session_key=session_key)
        else:
            client.add(speaker_a_messages, speaker_a_user_id)
            client.add(speaker_b_messages, speaker_b_user_id)

    end_time = time.time()
    return round(end_time - start_time, 2)


def process_user(conv_idx, frame, locomo_df, version, success_records, f, clear=False):
    conversation = locomo_df["conversation"].iloc[conv_idx]
    max_session_count = 35
    start_time = time.time()
    total_session_time = 0
    valid_sessions = 0
    speaker_a_user_id = f"locomo_exp_user_{conv_idx}_speaker_a_{version}"
    speaker_b_user_id = f"locomo_exp_user_{conv_idx}_speaker_b_{version}"

    from client_factory import create_client

    client = create_client(frame)
    timer = AddCallTimer(client)
    group_id = None
    zep_use_group = False
    try:
        if frame == "zep":
            zep_use_group = os.environ.get("ZEP_USE_GROUP", "true").lower() in ("true", "1", "yes")
            if zep_use_group:
                group_id = f"locomo_exp_group_{conv_idx}_{version}"
                client.sdk_graph_create(group_id)
            else:
                client.delete_user(speaker_a_user_id)
                client.delete_user(speaker_b_user_id)
                client.add_user(speaker_a_user_id)
                client.add_user(speaker_b_user_id)
        elif "mem0" in frame:
            from utils.prompts import MEMOS_CUSTOM_INSTRUCTIONS
            client.set_custom_instructions(MEMOS_CUSTOM_INSTRUCTIONS)
            if clear:
                client.delete_all(user_id=speaker_a_user_id)
                client.delete_all(user_id=speaker_b_user_id)
        elif frame == "everos" and clear:
            use_group = os.environ.get("EVEROS_USE_GROUP", "true").lower() in ("true", "1", "yes")
            if use_group:
                client.delete_group(speaker_a_user_id)
            else:
                client.delete(speaker_a_user_id)
                client.delete(speaker_b_user_id)
        elif frame == "supermemory" and clear:
            client.delete(speaker_a_user_id)
        elif clear and hasattr(client, "delete"):
            client.delete(speaker_a_user_id)
            client.delete(speaker_b_user_id)
        elif clear and hasattr(client, "delete_user"):
            client.delete_user(speaker_a_user_id)
            client.delete_user(speaker_b_user_id)
    except Exception as exc:
        print(f"  ⚠ Cleanup failed for user {conv_idx}, continuing ingestion: {exc}")
    sessions_to_process = []
    for session_idx in range(max_session_count):
        session_key = f"session_{session_idx}"
        session = conversation.get(session_key)
        if session is None:
            continue

        metadata = {
            "session_date": conversation.get(f"session_{session_idx}_date_time") + " UTC",
            "speaker_a": conversation.get("speaker_a"),
            "speaker_b": conversation.get("speaker_b"),
            "speaker_a_user_id": speaker_a_user_id,
            "speaker_b_user_id": speaker_b_user_id,
            "conv_idx": conv_idx,
            "session_key": session_key,
            "group_id": group_id,
        }
        sessions_to_process.append((session, metadata))
        valid_sessions += 1

    print(f"Processing {valid_sessions} sessions for user {conv_idx}")

    failed_sessions = []
    for session_idx, (session, metadata) in enumerate(sessions_to_process):
        record_key = f"{conv_idx}_{session_idx}"
        if record_key not in success_records:
            try:
                session_time = ingest_session(client, session, frame, metadata)
                total_session_time += session_time
                print(f"User {conv_idx}, {metadata['session_key']} processed in {session_time} seconds")
                fsync_write_line(f, record_key)
            except Exception as e:
                import traceback
                print(f"❌ Error ingesting session {record_key}: {type(e).__name__}: {e}")
                traceback.print_exc()
                failed_sessions.append(record_key)
        else:
            print(f"Session {record_key} already ingested")

    if failed_sessions:
        raise RuntimeError(
            f"User {conv_idx}: {len(failed_sessions)}/{len(sessions_to_process)} sessions failed: {failed_sessions}"
        )

    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    print(f"User {conv_idx} processed successfully in {elapsed_time} seconds")

    return elapsed_time, timer.durations_ms


def main(frame, version="default", num_workers=2, clear=False):
    load_env()
    locomo_df = pd.read_json("data/locomo/locomo10.json")
    num_users = len(locomo_df)
    start_time = time.time()
    total_time = 0
    print(
        f"Starting processing for {num_users} users in serial mode, each user using {num_workers} workers for sessions..."
    )
    if clear:
        print("🧹 --clear enabled: will delete existing memories before ingestion")
    os.makedirs(f"results/locomo/{frame}-{version}/", exist_ok=True)
    success_records = set()
    record_file = f"results/locomo/{frame}-{version}/success_records.txt"
    if clear and os.path.exists(record_file):
        os.remove(record_file)
        print("🧹 Cleared progress records")
    if os.path.exists(record_file):
        with open(record_file) as f:
            for i in f.readlines():
                success_records.add(i.strip())

    user_durations = {}
    all_add_call_durations = []
    failed_users = []
    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor,
        open(record_file, "a+") as f,
    ):
        future_to_uid = {
            executor.submit(process_user, uid, frame, locomo_df, version, success_records, f, clear): uid
            for uid in range(num_users)
        }
        for future in track(
            concurrent.futures.as_completed(future_to_uid),
            total=num_users,
            description="Ingesting users",
        ):
            uid = future_to_uid[future]
            try:
                session_time, add_call_ms = future.result()
                total_time += session_time
                user_durations[str(uid)] = round(session_time * 1000, 1)
                all_add_call_durations.extend(add_call_ms)
            except Exception as e:
                import traceback
                print(f"❌ Error processing user {uid}: {type(e).__name__}: {e}")
                traceback.print_exc()
                failed_users.append(uid)

    stats_path = os.path.join(f"results/locomo/{frame}-{version}", f"{frame}_locomo_ingestion_stats.json")
    with open(stats_path, "w") as sf:
        json.dump({
            "user_durations_ms": user_durations,
            "add_call_durations_ms": [round(d, 2) for d in all_add_call_durations],
        }, sf, indent=2)
    print(f"Ingestion stats saved to {stats_path}")

    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    elapsed_time_str = f"{minutes} minutes and {seconds} seconds"

    if failed_users:
        print("\n" + "=" * 80)
        print(f"❌ INGESTION FAILED: {len(failed_users)}/{num_users} users had errors".center(80))
        print("=" * 80)
        print(f"⏱️  Total time: {elapsed_time_str}")
        print("💡 Fix errors and re-run — successfully ingested sessions are saved in success_records.txt")
        print("=" * 80 + "\n")
        raise SystemExit(1)

    print("\n" + "=" * 80)
    print("✅ INGESTION COMPLETE".center(80))
    print("=" * 80)
    print(f"⏱️  Total time: {elapsed_time_str}")
    print(f"🔄 Framework: {frame} | Version: {version} | Workers: {num_workers}")
    print(f"📊 Processed: {num_users} users")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lib",
        type=str,
        choices=SUPPORTED_LIBS,
        default=DEFAULT_LIB,
    )
    parser.add_argument(
        "--version",
        type=str,
        default="default",
        help="Version identifier for saving results (e.g., 1010)",
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Number of parallel ingestion workers."
    )
    parser.add_argument(
        "--clear", action="store_true", help="Clear existing memories before ingestion"
    )
    args = parser.parse_args()
    lib = args.lib
    version = args.version
    workers = args.workers

    main(lib, version, workers, clear=args.clear)

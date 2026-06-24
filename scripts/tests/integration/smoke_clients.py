"""Smoke test clients against real memory backends using LoCoMo sample data.

Tests add/search/delete for each client, reporting PASS/FAIL/SKIP.
Unlike ``test_*.py`` unit tests in this directory, this script performs real
API calls and requires an environment file for each selected memory product.

Usage:
    python scripts/tests/integration/smoke_clients.py --lib memos --env .env.memos
    python scripts/tests/integration/smoke_clients.py --lib memos --lib mem0
    cd scripts && python -m tests.integration.smoke_clients --lib memos --env ../.env.memos
"""

import argparse
import json
import os
import sys
import time
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(SCRIPT_DIR)
SCRIPTS_DIR = os.path.dirname(TESTS_DIR)
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)

sys.path.insert(0, SCRIPTS_DIR)

from dotenv import load_dotenv

from client_factory import SUPPORTED_LIBS, create_client
from utils.time import parse_locomo_time, to_iso
from utils.ingest_helpers import inject_time, session_id_kwargs

DATA_PATH = os.path.join(PROJECT_DIR, "data", "locomo", "locomo10.json")
DEFAULT_SESSION_LIMIT = 2
MAX_SESSION_SCAN = 35
SEARCH_TOPK = 5


def _zep_use_group():
    return os.environ.get("ZEP_USE_GROUP", "true").lower() in ("true", "1", "yes")


def _env_truthy(name, default="false"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


def check_time_in_result(result_str, ingested_dates):
    """Check if search results contain time info from ingested sessions.

    Tries multiple date format fragments (ISO prefix, month-day, year).
    Returns (found: bool, detail: str).
    """
    if not result_str or not ingested_dates:
        return False, "no data to check"

    for dt in ingested_dates:
        candidates = [
            dt.strftime("%Y-%m-%dT"),
            dt.strftime("%Y-%m-%d"),
            dt.strftime("%B %-d"),
            dt.strftime("%-d %B"),
            to_iso(dt),
        ]
        for c in candidates:
            if c in result_str:
                return True, f"matched '{c}'"
    return False, f"none of {[d.strftime('%Y-%m-%d') for d in ingested_dates]} found"


def has_search_result(result):
    """Return whether a backend search response contains usable content."""
    if isinstance(result, dict):
        return any(bool(value) for value in result.values())
    if isinstance(result, str):
        text = result.strip()
        if not text:
            return False
        return text not in ("Conversation memories:", "User Profile:", "Agent Memory:")
    if isinstance(result, list):
        return len(result) > 0
    return bool(result)


TEST_VERSION = "smoke_test"
CONV_IDX = 0


def load_first_user():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"LoCoMo sample data not found: {DATA_PATH}. "
            "Run data/locomo/prepare_locomo.py first."
        )
    with open(DATA_PATH) as f:
        data = json.load(f)
    item = data[CONV_IDX]
    conv = item["conversation"]
    qa = item.get("qa", [])
    return conv, qa


def build_messages(conv, session_key, lib_name):
    """Build speaker_a message list from a session, with time injected per lib."""
    session = conv[session_key]
    raw_date = conv[f"{session_key}_date_time"]
    dt = parse_locomo_time(raw_date)
    speaker_a = conv["speaker_a"]

    messages = []
    for chat in session:
        content = f"{chat['speaker']}: {chat['text']}"
        role = "user" if chat["speaker"] == speaker_a else "assistant"
        messages.append({"role": role, "name": chat["speaker"], "content": content})

    time_kw = inject_time(messages, dt, lib_name)
    return messages, time_kw, dt


def get_delete_fn(client):
    """Return the appropriate delete function for a client."""
    if hasattr(client, "delete"):
        return client.delete
    if hasattr(client, "delete_user"):
        return client.delete_user
    if hasattr(client, "delete_all"):
        return lambda uid: client.delete_all(user_id=uid)
    return None


def run_smoke_test(
    lib_name,
    conv,
    qa_list,
    env_path=None,
    session_limit=DEFAULT_SESSION_LIMIT,
    print_result=False,
):
    """Run add/search/delete smoke test for one lib. Returns (status, detail)."""
    user_id = f"locomo_exp_user_{CONV_IDX}_speaker_a_{TEST_VERSION}"
    conv_id = f"locomo_exp_user_{CONV_IDX}"

    print(f"\n{'=' * 60}")
    print(f"  {lib_name.upper()}")
    print(f"{'=' * 60}")

    # ── Load env for this lib ────────────────────────────────────────
    env_file = env_path or os.path.join(PROJECT_DIR, f".env.{lib_name}")
    if not os.path.exists(env_file):
        print(f"  SKIP: env file not found: {env_file}")
        return "SKIP", "no env file"
    load_dotenv(env_file, override=True)

    # ── Create client ────────────────────────────────────────────────
    try:
        client = create_client(lib_name)
    except Exception as e:
        print(f"  SKIP: cannot create client — {e}")
        return "SKIP", str(e)

    # ── Zep graph mode detection ────────────────────────────────────
    zep_graph = (lib_name == "zep" and _zep_use_group()
                 and hasattr(client, "sdk_graph_create"))
    zep_graph_id = f"smoke_test_graph_{CONV_IDX}" if zep_graph else None

    # ── 1. Delete (clean slate) ──────────────────────────────────────
    if zep_graph:
        try:
            client.sdk_graph_delete(zep_graph_id)
            print(f"  [1/4] delete (graph): OK")
        except Exception as e:
            print(f"  [1/4] delete (graph): WARN — {e}")
    else:
        delete_fn = get_delete_fn(client)
        if delete_fn:
            try:
                delete_fn(user_id)
                print(f"  [1/4] delete: OK")
            except Exception as e:
                print(f"  [1/4] delete: WARN — {e}")
        else:
            print(f"  [1/4] delete: N/A (no delete method)")

    # ── 2. Add (ingest 2 sessions) ───────────────────────────────────
    sessions_ingested = 0
    ingested_dates = []
    try:
        if zep_graph:
            client.sdk_graph_create(zep_graph_id)
        elif lib_name == "zep" and hasattr(client, "add_user"):
            client.add_user(user_id)

        for idx in range(MAX_SESSION_SCAN):
            key = f"session_{idx}"
            if key not in conv:
                continue
            messages, time_kw, dt = build_messages(conv, key, lib_name)
            ingested_dates.append(dt)
            sess_kw = session_id_kwargs(lib_name, f"{conv_id}_{key}")

            if zep_graph:
                session = conv[key]
                iso_date = to_iso(dt)
                for chat in session:
                    data = f"{chat['speaker']}: {chat['text']}"
                    client.sdk_graph_add(
                        graph_id=zep_graph_id, data=data,
                        data_type="message", created_at=iso_date,
                    )
            elif lib_name == "letta":
                letta_msgs = [
                    {**m, "content": m["content"][len(m["name"]) + 2:]}
                    if m.get("name") and m["content"].startswith(f"{m['name']}: ")
                    else m
                    for m in messages
                ]
                client.add(letta_msgs, user_id, **sess_kw, **time_kw)
            elif lib_name == "hindsight":
                session_data = conv[key]
                client.add(
                    messages, user_id,
                    raw_content=json.dumps(session_data),
                    context=f"Smoke test session {key}",
                    **sess_kw, **time_kw,
                )
            elif lib_name == "supermemory":
                client.add(messages, user_id,
                           session_id=f"{conv_id}_{key}", **time_kw)
            else:
                client.add(messages, user_id, **sess_kw, **time_kw)

            sessions_ingested += 1
            print(f"  [2/4] add: ingested {key} ({len(messages)} msgs)")
            if sessions_ingested >= session_limit:
                break

        if sessions_ingested == 0:
            print(f"  [2/4] add: FAIL — no sessions found")
            return "FAIL", "no sessions ingested"
        print(f"  [2/4] add: OK ({sessions_ingested} sessions)")
    except Exception as e:
        print(f"  [2/4] add: FAIL — {e}")
        traceback.print_exc()
        return "FAIL", f"add error: {e}"

    # ── Wait for async indexing ──────────────────────────────────────
    if lib_name == "supermemory" and hasattr(client, "await_indexing"):
        print(f"  [--] waiting for supermemory indexing...")
        client.await_indexing(timeout=300)
    elif lib_name == "hindsight" and hasattr(client, "await_extraction"):
        print(f"  [--] waiting for hindsight extraction...")
        client.await_extraction(user_id, max_wait_s=120, poll_interval=5)
    elif lib_name == "graphiti":
        if _env_truthy("GRAPHITI_SYNC_ADD", "0"):
            print(f"  [--] graphiti sync mode — no extra wait needed")
        else:
            wait_s = int(os.environ.get("GRAPHITI_WAIT_AFTER_INGEST", "60"))
            print(f"  [--] waiting for graphiti async processing ({wait_s}s)...")
            time.sleep(wait_s)
    elif lib_name == "viking":
        print(f"  [--] waiting for viking LLM extraction (30s)...")
        time.sleep(30)
    elif zep_graph:
        print(f"  [--] waiting for zep graph processing (15s)...")
        time.sleep(15)
    else:
        time.sleep(3)

    # ── 3. Search ────────────────────────────────────────────────────
    try:
        query = qa_list[0]["question"] if qa_list else "What happened recently?"
        print(f"  [3/4] search: query={query[:60]}...")
        t0 = time.time()
        if zep_graph:
            result = client.sdk_graph_search(
                graph_id=zep_graph_id, query=query, top_k=SEARCH_TOPK,
            )
        else:
            result = client.search(query=query, user_id=user_id, top_k=SEARCH_TOPK)
        elapsed = round(time.time() - t0, 2)

        if has_search_result(result):
            preview = str(result)[:150].replace("\n", " ")
            print(f"  [3/4] search: OK in {elapsed}s — {preview}...")
            if print_result:
                print("  [3/4] search result:")
                print(str(result))
        else:
            print(f"  [3/4] search: WARN — empty result (may need more indexing time)")
    except Exception as e:
        print(f"  [3/4] search: FAIL — {e}")
        traceback.print_exc()
        return "FAIL", f"search error: {e}"

    # ── 3.5. Verify time info in search results ─────────────────────
    time_ok, time_detail = check_time_in_result(str(result), ingested_dates)
    if time_ok:
        print(f"  [T/4] time verify: OK — {time_detail}")
    else:
        print(f"  [T/4] time verify: WARN — {time_detail}")

    # ── 4. Delete (cleanup) ──────────────────────────────────────────
    if zep_graph:
        try:
            client.sdk_graph_delete(zep_graph_id)
            print(f"  [4/4] delete cleanup (graph): OK")
        except Exception as e:
            print(f"  [4/4] delete cleanup (graph): WARN — {e}")
    else:
        delete_fn = get_delete_fn(client)
        if delete_fn:
            try:
                delete_fn(user_id)
                print(f"  [4/4] delete cleanup: OK")
            except Exception as e:
                print(f"  [4/4] delete cleanup: WARN — {e}")
        else:
            print(f"  [4/4] delete cleanup: N/A")

    time_tag = "time:OK" if time_ok else "time:WARN"
    return "PASS", f"{sessions_ingested} sessions, search in {elapsed}s, {time_tag}"


def main():
    parser = argparse.ArgumentParser(description="Smoke test all memory clients")
    parser.add_argument(
        "--lib", type=str, action="append", default=None,
        help="Specific lib(s) to test (can be repeated). Default: all.",
    )
    parser.add_argument(
        "--env", type=str, default=None,
        help="Env file to load for every selected lib. Relative paths are resolved from project root.",
    )
    parser.add_argument(
        "--sessions", type=int, default=DEFAULT_SESSION_LIMIT,
        help=f"Number of LoCoMo sessions to ingest per smoke test. Default: {DEFAULT_SESSION_LIMIT}.",
    )
    parser.add_argument(
        "--print-result", action="store_true",
        help="Print the full search result instead of only a short preview.",
    )
    args = parser.parse_args()
    if args.sessions < 1:
        parser.error("--sessions must be >= 1")

    libs = args.lib if args.lib else SUPPORTED_LIBS
    env_path = args.env
    if env_path and not os.path.isabs(env_path):
        env_path = os.path.join(PROJECT_DIR, env_path)

    conv, qa_list = load_first_user()

    results = {}
    for lib_name in libs:
        if lib_name not in SUPPORTED_LIBS:
            print(f"\nUnknown lib: {lib_name}, skipping")
            continue
        status, detail = run_smoke_test(
            lib_name,
            conv,
            qa_list,
            env_path=env_path,
            session_limit=args.sessions,
            print_result=args.print_result,
        )
        results[lib_name] = (status, detail)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  SMOKE TEST SUMMARY")
    print(f"{'=' * 60}")
    for lib_name, (status, detail) in results.items():
        icon = {"PASS": "OK", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
        print(f"  [{icon:4s}] {lib_name:14s} — {detail}")

    passed = sum(1 for s, _ in results.values() if s == "PASS")
    failed = sum(1 for s, _ in results.values() if s == "FAIL")
    skipped = sum(1 for s, _ in results.values() if s == "SKIP")
    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'=' * 60}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

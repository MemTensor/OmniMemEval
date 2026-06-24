import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from utils.env import load_env
from utils.checkpoint import atomic_json_dump
from utils.progress import track
from utils.search_helpers import dispatch_search, unpack_search_result
from utils.response_options import parse_bool
from client_factory import SUPPORTED_LIBS, DEFAULT_LIB
from longmemeval.lme_data import load_lme_dataframe
from longmemeval.lme_common import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    build_search_result,
    classify_search_status,
    error_payload,
    get_single_search_entry,
    search_allowed_statuses,
    status_counts,
    user_id_for,
    validate_single_search_result,
)

def process_user(
    lme_df,
    conv_idx,
    frame,
    version,
    top_k=20,
    *,
    allow_empty_search=True,
    skip_failed_search=False,
):
    row = lme_df.iloc[conv_idx]
    question = row["question"]
    question_type = row["question_type"]
    question_date = row["question_date"]
    user_id = user_id_for(version, conv_idx)

    print("\n" + "-" * 80)
    print(f"🔎 [{conv_idx + 1}/{len(lme_df)}] Processing conversation {conv_idx}")
    print(f"❓ Question: {question}")
    print(f"📅 Date: {question_date}")
    print(f"🏷️  Type: {question_type}")
    print("-" * 80)

    allowed_statuses = search_allowed_statuses(
        allow_empty_search=allow_empty_search,
        allow_skipped=skip_failed_search,
    )

    existing_results, exists = load_existing_results(
        frame,
        version,
        conv_idx,
        user_id=user_id,
        question=question,
        allowed_statuses=allowed_statuses,
    )
    if exists:
        print(f"♻️  Using existing results for conversation {conv_idx}")
        return existing_results, []

    from client_factory import create_client

    client = create_client(frame)
    extra_kw = {"question_date": question_date}
    try:
        result = dispatch_search(frame, client, question, user_id, top_k, **extra_kw)

        context, duration_ms, reflect_answer, raw_context = unpack_search_result(result)
    except Exception as e:
        print(f"  ❌ Search failed for conversation {conv_idx}: {e}")
        status = STATUS_SKIPPED if skip_failed_search else STATUS_FAILED
        search_results = build_search_result(
            row,
            user_id=user_id,
            status=status,
            error=error_payload("search", e),
        )
    else:
        context = context or ""
        status = classify_search_status(
            context,
            reflect_answer,
            raw_context=raw_context,
        )
        search_results = build_search_result(
            row,
            user_id=user_id,
            context=context,
            duration_ms=duration_ms,
            status=status,
            reflect_answer=reflect_answer,
        )

    entry = get_single_search_entry(search_results, user_id)
    blocking_records = []
    if entry is None:
        blocking_records.append({
            "status": STATUS_FAILED,
            "error": error_payload("search", "missing search entry"),
        })
    elif entry.get("status") not in allowed_statuses:
        blocking_records.append(entry)

    os.makedirs(f"results/lme/{frame}-{version}/tmp", exist_ok=True)
    tmp_path = f"results/lme/{frame}-{version}/tmp/{frame}_lme_search_results_{conv_idx}.json"
    atomic_json_dump(search_results, tmp_path, indent=4)
    print(f"💾 Search results for conversation {conv_idx} saved...")
    print("-" * 80)

    return search_results, blocking_records


def load_existing_results(
    frame,
    version,
    group_idx,
    *,
    user_id,
    question,
    allowed_statuses,
):
    result_path = f"results/lme/{frame}-{version}/tmp/{frame}_lme_search_results_{group_idx}.json"
    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                data = json.load(f)
            ok, issues = validate_single_search_result(
                data,
                user_id=user_id,
                question=question,
                allowed_statuses=allowed_statuses,
                require_status=True,
            )
            if ok:
                return data, True
            print(
                f"Existing results for group {group_idx} are incomplete; "
                f"will retry ({'; '.join(issues)})"
            )
        except Exception as e:
            print(f"❌ Error loading existing results for group {group_idx}: {e}")
    return {}, False


def main(
    frame,
    version,
    top_k=20,
    num_workers=2,
    *,
    allow_empty_search=True,
    skip_failed_search=False,
):
    load_env()

    print("\n" + "=" * 80)
    print(f"🔍 LONGMEMEVAL SEARCH - {frame.upper()} v{version}".center(80))
    print("=" * 80)

    lme_df = load_lme_dataframe()
    num_multi_sessions = len(lme_df)
    print(f"👥 Number of users: {num_multi_sessions}")
    print(f"⚙️  Search parameters: top_k={top_k}, workers={num_workers}")
    print("-" * 80)

    all_search_results = {}
    start_time = datetime.now()
    failed_users = []
    all_status_records = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(
                process_user,
                lme_df,
                idx,
                frame,
                version,
                top_k,
                allow_empty_search=allow_empty_search,
                skip_failed_search=skip_failed_search,
            ): idx
            for idx in range(num_multi_sessions)
        }

        for future in track(
            as_completed(future_to_idx), total=num_multi_sessions, description="Searching users",
        ):
            _idx = future_to_idx[future]
            try:
                search_results, blocking_records = future.result()
                for user_id, results in search_results.items():
                    all_status_records.extend(results)
                    if not blocking_records:
                        all_search_results[user_id] = results
                if blocking_records:
                    failed_users.append({
                        "conv_idx": _idx,
                        "user_id": user_id_for(version, _idx),
                        "failures": blocking_records,
                    })
            except Exception as e:
                print(f"❌ Error searching user {_idx}: {e}")
                failed_users.append({
                    "conv_idx": _idx,
                    "user_id": user_id_for(version, _idx),
                    "error": error_payload("search", e),
                })

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    elapsed_time_str = str(elapsed_time).split(".")[0]

    results_dir = f"results/lme/{frame}-{version}"
    atomic_json_dump(
        dict(all_search_results),
        f"{results_dir}/{frame}_lme_search_results.json",
        indent=4,
    )
    atomic_json_dump(
        {
            "stage": "search",
            "allow_empty_search": allow_empty_search,
            "skip_failed_search": skip_failed_search,
            "status_counts": status_counts(all_status_records),
            "failed_users": failed_users,
        },
        f"{results_dir}/{frame}_lme_search_status.json",
        indent=2,
    )
    print(f"📁 Results saved to: results/lme/{frame}-{version}/{frame}_lme_search_results.json")

    if failed_users:
        print("\n" + "=" * 80)
        failure_message = (
            f"❌ SEARCH FAILED: {len(failed_users)}/{num_multi_sessions} "
            "users had errors"
        )
        print(failure_message.center(80))
        print("=" * 80)
        print(f"⏱️  Total time: {elapsed_time_str}")
        print(f"🔄 Framework: {frame} | Version: {version} | Workers: {num_workers}")
        print("=" * 80 + "\n")
        raise SystemExit(1)

    print("\n" + "=" * 80)
    print("✅ SEARCH COMPLETE".center(80))
    print("=" * 80)
    print(f"⏱️  Total time taken to search {num_multi_sessions} users: {elapsed_time_str}")
    print(f"🔄 Framework: {frame} | Version: {version} | Workers: {num_workers}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LongMemeval Search Script")
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
        "--top-k", type=int, default=20, help="Number of top results to retrieve from the search."
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Number of parallel search workers."
    )
    parser.add_argument(
        "--allow-empty-search",
        "--allow_empty_search",
        type=parse_bool,
        default=True,
        help="Allow successful searches with no raw memories. Default: 1.",
    )
    parser.add_argument(
        "--skip-failed-search",
        "--skip_failed_search",
        type=parse_bool,
        default=False,
        help=(
            "Explicitly mark failed search calls as skipped instead of failing "
            "the step. Default: 0."
        ),
    )

    args = parser.parse_args()

    main(
        frame=args.lib,
        version=args.version,
        top_k=args.top_k,
        num_workers=args.workers,
        allow_empty_search=args.allow_empty_search,
        skip_failed_search=args.skip_failed_search,
    )

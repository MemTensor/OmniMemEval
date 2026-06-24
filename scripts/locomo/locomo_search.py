import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, SCRIPT_DIR)

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time

import pandas as pd

from utils.env import load_env
from utils.checkpoint import atomic_json_dump
from utils.progress import track
from utils.response_options import parse_bool
from utils.search_helpers import unpack_search_result
from client_factory import SUPPORTED_LIBS, DEFAULT_LIB
from locomo_common import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    classify_search_status,
    error_payload,
    expected_questions,
    filter_eval_qas,
    group_id_for,
    record_status,
    search_allowed_statuses,
    status_counts,
    validate_query_coverage,
)


def _memory_text(results):
    if results is None:
        return ""
    if isinstance(results, str):
        return results
    return "\n".join(results)


def _combined_raw_context(*parts):
    return "\n".join(part for part in parts if part)


def zep_group_search(client, query, group_id, top_k):
    """Search Zep standalone graph via SDK (official LoCoMo configuration)."""
    start = time()
    context = client.sdk_graph_search(graph_id=group_id, query=query, top_k=top_k)
    duration_ms = (time() - start) * 1000
    return context, duration_ms, None, context or ""


def generic_text_search(
    client, query, speaker_a_user_id, speaker_b_user_id, top_k, speaker_a, speaker_b
):
    """Generic search for clients that return plain text from search()."""
    from utils.prompts import DUAL_SPEAKER_TEMPLATE

    start = time()
    results_a = client.search(query, speaker_a_user_id, top_k)
    results_b = client.search(query, speaker_b_user_id, top_k)

    mem_a = _memory_text(results_a)
    mem_b = _memory_text(results_b)

    context = DUAL_SPEAKER_TEMPLATE.format(
        speaker_1=speaker_a,
        speaker_1_memories=mem_a,
        speaker_2=speaker_b,
        speaker_2_memories=mem_b,
    )
    duration_ms = (time() - start) * 1000
    return context, duration_ms, None, _combined_raw_context(mem_a, mem_b)


def shared_conv_search(
    client, query, speaker_a_user_id, speaker_b_user_id, top_k, speaker_a, speaker_b
):
    """Single search for products that share one store per conversation
    (e.g. Letta agent, Cognee dataset, Hindsight bank, Backboard assistant).

    Both speakers get the same search results since they share the
    underlying data store.
    """
    from utils.prompts import DUAL_SPEAKER_TEMPLATE

    start = time()
    memories = client.search(query, speaker_a_user_id, top_k)
    mem_text = _memory_text(memories)

    context = DUAL_SPEAKER_TEMPLATE.format(
        speaker_1=speaker_a,
        speaker_1_memories=mem_text,
        speaker_2=speaker_b,
        speaker_2_memories=mem_text,
    )
    duration_ms = (time() - start) * 1000
    return context, duration_ms, None, mem_text


def cognee_search(
    client, query, speaker_a_user_id, speaker_b_user_id, top_k, speaker_a, speaker_b
):
    """Cognee uses one dataset per conversation.

    Supports two modes depending on the configured SearchType:
    - completion types (GRAPH_COMPLETION, etc.): Cognee LLM generates the
      answer directly — returned as ``reflect_answer`` so the answer stage
      skips the external ANSWER_MODEL (same pattern as Hindsight reflect).
    - retrieval-only types (CHUNKS, SUMMARIES, …): raw context is returned
      for external answer generation.
    """
    from utils.prompts import DUAL_SPEAKER_TEMPLATE

    start = time()
    memories = client.search(query, speaker_a_user_id, top_k)
    mem_text = _memory_text(memories)

    context = DUAL_SPEAKER_TEMPLATE.format(
        speaker_1=speaker_a,
        speaker_1_memories=mem_text,
        speaker_2=speaker_b,
        speaker_2_memories=mem_text,
    )
    duration_ms = (time() - start) * 1000

    if client.is_completion_search and not client._only_context:
        return context, duration_ms, mem_text, mem_text
    return context, duration_ms, None, mem_text


def letta_search(
    client, query, speaker_a_user_id, speaker_b_user_id, top_k, speaker_a, speaker_b
):
    """Letta uses one agent per conversation.

    Supports two eval modes controlled by ``LETTA_EVAL_MODE`` env var:
    - ``direct``: agent answers using its own tools → returned as reflect_answer
    - ``rag``   : retrieve passages, feed to external ANSWER LLM
    """
    from utils.prompts import DUAL_SPEAKER_TEMPLATE

    start = time()
    memories = client.search(query, speaker_a_user_id, top_k)

    if isinstance(memories, dict) and "answer" in memories:
        answer = memories["answer"]
        raw_context = memories.get("context", "")
        context = DUAL_SPEAKER_TEMPLATE.format(
            speaker_1=speaker_a,
            speaker_1_memories=raw_context,
            speaker_2=speaker_b,
            speaker_2_memories=raw_context,
        )
        duration_ms = (time() - start) * 1000
        return context, duration_ms, answer, raw_context

    mem_text = _memory_text(memories)
    context = DUAL_SPEAKER_TEMPLATE.format(
        speaker_1=speaker_a,
        speaker_1_memories=mem_text,
        speaker_2=speaker_b,
        speaker_2_memories=mem_text,
    )
    duration_ms = (time() - start) * 1000
    return context, duration_ms, None, mem_text


def hindsight_search(
    client, query, speaker_a_user_id, speaker_b_user_id, top_k, speaker_a, speaker_b
):
    """Hindsight uses one bank per conversation.

    Supports two modes controlled by ``HINDSIGHT_MODE`` env var:
    - ``recall`` (default): retrieve memories, return as context for external LLM
    - ``reflect``: Hindsight generates the answer directly via Reflect API
    """
    mode = os.getenv("HINDSIGHT_MODE", "recall").lower()
    if mode == "reflect":
        from utils.prompts import DUAL_SPEAKER_TEMPLATE

        start = time()
        answer, based_on = client.reflect(query, speaker_a_user_id)
        if isinstance(based_on, dict):
            memories = based_on.get("memories", [])
            sources = "\n".join(
                m.get("text", "") if isinstance(m, dict) else str(m)
                for m in memories
            )
        else:
            sources = "\n".join(str(s) for s in based_on) if based_on else ""
        context = DUAL_SPEAKER_TEMPLATE.format(
            speaker_1=speaker_a,
            speaker_1_memories=sources,
            speaker_2=speaker_b,
            speaker_2_memories=sources,
        )
        duration_ms = (time() - start) * 1000
        return context, duration_ms, answer, sources
    else:
        return shared_conv_search(
            client, query, speaker_a_user_id, speaker_b_user_id,
            top_k, speaker_a, speaker_b,
        )


def backboard_search(
    client, query, speaker_a_user_id, speaker_b_user_id, top_k, speaker_a, speaker_b
):
    """Backboard search with two eval modes:

    - ``rag``     (default): retrieve memories → feed to external ANSWER LLM
    - ``reflect``: Backboard's built-in LLM answers using ``memory="Auto"``
                   + ``send_to_llm=true``  (returned as ``reflect_answer``)
    """
    eval_mode = os.getenv("BACKBOARD_EVAL_MODE", "rag").strip().lower()
    if eval_mode == "reflect":
        from utils.prompts import DUAL_SPEAKER_TEMPLATE

        start = time()
        answer, mem_text = client.reflect(query, speaker_a_user_id, top_k)
        context = DUAL_SPEAKER_TEMPLATE.format(
            speaker_1=speaker_a,
            speaker_1_memories=mem_text,
            speaker_2=speaker_b,
            speaker_2_memories=mem_text,
        )
        duration_ms = (time() - start) * 1000
        return context, duration_ms, answer, mem_text
    else:
        return shared_conv_search(
            client, query, speaker_a_user_id, speaker_b_user_id,
            top_k, speaker_a, speaker_b,
        )


def search_query(client, query, metadata, frame, top_k=20):
    _conv_id = metadata.get("conv_id")
    speaker_a = metadata.get("speaker_a")
    speaker_b = metadata.get("speaker_b")
    speaker_a_user_id = metadata.get("speaker_a_user_id")
    speaker_b_user_id = metadata.get("speaker_b_user_id")

    group_id = metadata.get("group_id")
    if frame == "zep" and group_id:
        return zep_group_search(client, query, group_id, top_k)

    _search_dispatch = {
        "zep": generic_text_search,
        "mem0": generic_text_search,
        "memos": generic_text_search,
        "everos": shared_conv_search,
        "supermemory": shared_conv_search,
        "letta": letta_search,
        "cognee": cognee_search,
        "hindsight": hindsight_search,
        "graphiti": generic_text_search,
        "viking": generic_text_search,
        "memori": generic_text_search,
        "memmachine": generic_text_search,
        "memorylake": generic_text_search,
        "backboard": backboard_search,
        "mem9": generic_text_search,
    }
    search_fn = _search_dispatch.get(frame)
    if search_fn is None:
        raise ValueError(f"No search function for lib: {frame!r}")
    result = search_fn(
        client, query, speaker_a_user_id, speaker_b_user_id, top_k, speaker_a, speaker_b
    )
    return result


def load_existing_results(
    frame,
    version,
    group_idx,
    conv_id,
    questions,
    *,
    allow_empty_search=True,
    allow_skipped=False,
):
    result_path = (
        f"results/locomo/{frame}-{version}/tmp/{frame}_locomo_search_results_{group_idx}.json"
    )
    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                data = json.load(f)
            records = data.get(conv_id, [])
            ok, issues = validate_query_coverage(
                records,
                questions,
                allowed_statuses=search_allowed_statuses(
                    allow_empty_search=allow_empty_search,
                    allow_skipped=allow_skipped,
                ),
            )
            if ok:
                return data, True
            print(
                f"Existing results for group {group_idx} are incomplete; "
                f"will retry ({'; '.join(issues)})"
            )
        except Exception as e:
            print(f"Error loading existing results for group {group_idx}: {e}")
    return {}, False


def process_user(
    conv_idx,
    locomo_df,
    frame,
    version,
    top_k=20,
    num_workers=2,
    *,
    allow_empty_search=True,
    skip_failed_search=False,
):
    search_results = defaultdict(list)
    qa_set = locomo_df["qa"].iloc[conv_idx]
    eval_qas = filter_eval_qas(qa_set)
    questions = expected_questions(qa_set)
    conversation = locomo_df["conversation"].iloc[conv_idx]
    speaker_a = conversation.get("speaker_a")
    speaker_b = conversation.get("speaker_b")
    speaker_a_user_id = f"locomo_exp_user_{conv_idx}_speaker_a_{version}"
    speaker_b_user_id = f"locomo_exp_user_{conv_idx}_speaker_b_{version}"
    conv_id = group_id_for(conv_idx)

    group_id = None
    if frame == "zep":
        zep_use_group = os.environ.get("ZEP_USE_GROUP", "true").lower() in ("true", "1", "yes")
        if zep_use_group:
            group_id = f"locomo_exp_group_{conv_idx}_{version}"

    existing_results, loaded = load_existing_results(
        frame,
        version,
        conv_idx,
        conv_id,
        questions,
        allow_empty_search=allow_empty_search,
        allow_skipped=skip_failed_search,
    )
    if loaded:
        print(f"Loaded existing results for group {conv_idx}")
        return existing_results, []

    from client_factory import create_client

    client = create_client(frame)

    metadata = {
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "speaker_a_user_id": speaker_a_user_id,
        "speaker_b_user_id": speaker_b_user_id,
        "conv_idx": conv_idx,
        "conv_id": conv_id,
        "group_id": group_id,
    }

    def process_qa(qa):
        query = str(qa.get("question") or "")
        try:
            result = search_query(client, query, metadata, frame, top_k=top_k)

            context, duration_ms, reflect_answer, raw_context = unpack_search_result(result)
        except Exception as e:
            print(f"  ❌ Search failed for user {conv_idx}, query: {query[:60]}: {e}")
            status = STATUS_SKIPPED if skip_failed_search else STATUS_FAILED
            return {
                "query": query,
                "context": "",
                "duration_ms": 0.0,
                "status": status,
                "error": error_payload("search", e),
            }

        if not context:
            print(f"No context found for query: {query}")
            context = ""
        status = classify_search_status(
            context,
            reflect_answer,
            raw_context=raw_context,
        )
        entry = {
            "query": query,
            "context": context,
            "duration_ms": duration_ms,
            "status": status,
        }
        if reflect_answer is not None:
            entry["reflect_answer"] = reflect_answer
        return entry

    results_by_index = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_qa, qa): idx
            for idx, qa in enumerate(eval_qas)
        }

        for future in track(
            as_completed(futures), total=len(futures), description=f"Processing user {conv_idx}"
        ):
            idx = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                query = str(eval_qas[idx].get("question") or "")
                result = {
                    "query": query,
                    "context": "",
                    "duration_ms": 0.0,
                    "status": STATUS_SKIPPED if skip_failed_search else STATUS_FAILED,
                    "error": error_payload("search", exc),
                }
            results_by_index[idx] = result

    records = [results_by_index[idx] for idx in sorted(results_by_index)]
    search_results[conv_id].extend(records)

    allowed_statuses = search_allowed_statuses(
        allow_empty_search=allow_empty_search,
        allow_skipped=skip_failed_search,
    )
    ok, issues = validate_query_coverage(
        records,
        questions,
        allowed_statuses=allowed_statuses,
    )
    blocking_records = [
        record for record in records
        if record_status(record) not in allowed_statuses
    ]
    if not ok:
        print(f"  ❌ Search results for user {conv_idx} are incomplete: {'; '.join(issues)}")
        blocking_records.append({
            "query": "",
            "context": "",
            "duration_ms": 0.0,
            "status": STATUS_FAILED,
            "error": error_payload("search", "; ".join(issues)),
        })

    os.makedirs(f"results/locomo/{frame}-{version}/tmp/", exist_ok=True)
    tmp_path = f"results/locomo/{frame}-{version}/tmp/{frame}_locomo_search_results_{conv_idx}.json"
    atomic_json_dump(dict(search_results), tmp_path, indent=2)
    print(f"Save search results {conv_idx}")

    return search_results, blocking_records


def main(
    frame,
    version="default",
    num_workers=2,
    top_k=20,
    *,
    allow_empty_search=True,
    skip_failed_search=False,
):
    load_env()
    locomo_df = pd.read_json("data/locomo/locomo10.json")
    num_users = len(locomo_df)
    os.makedirs(f"results/locomo/{frame}-{version}/", exist_ok=True)
    all_search_results = defaultdict(list)

    failed_users = []
    all_status_records = []
    for idx in range(num_users):
        print(f"\n[{idx + 1}/{num_users}] Processing user {idx}")
        try:
            user_results, blocking_records = process_user(
                idx,
                locomo_df,
                frame,
                version,
                top_k,
                num_workers,
                allow_empty_search=allow_empty_search,
                skip_failed_search=skip_failed_search,
            )
            for conv_id, results in user_results.items():
                all_status_records.extend(results)
                if not blocking_records:
                    all_search_results[conv_id].extend(results)
            if blocking_records:
                failed_users.append({
                    "group_idx": idx,
                    "group_id": group_id_for(idx),
                    "failures": blocking_records,
                })
        except Exception as e:
            print(f"❌ Error searching user {idx}: {e}")
            failed_users.append({
                "group_idx": idx,
                "group_id": group_id_for(idx),
                "error": error_payload("search", e),
            })

    results_dir = f"results/locomo/{frame}-{version}"
    atomic_json_dump(
        dict(all_search_results),
        f"{results_dir}/{frame}_locomo_search_results.json",
        indent=2,
    )
    atomic_json_dump(
        {
            "stage": "search",
            "allow_empty_search": allow_empty_search,
            "skip_failed_search": skip_failed_search,
            "status_counts": status_counts(all_status_records),
            "failed_users": failed_users,
        },
        f"{results_dir}/{frame}_locomo_search_status.json",
        indent=2,
    )

    if failed_users:
        print(f"\n❌ SEARCH FAILED: {len(failed_users)}/{num_users} users had errors")
        raise SystemExit(1)

    print("✅ Save all search results")


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
        "--workers", type=int, default=2, help="Number of parallel search workers."
    )
    parser.add_argument(
        "--top-k", type=int, default=20, help="Number of results to retrieve in search queries"
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
        help="Explicitly mark failed search calls as skipped instead of failing the step. Default: 0.",
    )
    args = parser.parse_args()
    lib = args.lib
    version = args.version
    workers = args.workers
    top_k = args.top_k

    main(
        lib,
        version,
        workers,
        top_k,
        allow_empty_search=args.allow_empty_search,
        skip_failed_search=args.skip_failed_search,
    )

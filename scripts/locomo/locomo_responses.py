import argparse
import asyncio
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, SCRIPT_DIR)

from time import time

import pandas as pd

from utils.env import load_env
from utils.checkpoint import atomic_json_dump
from utils.prompts import LOCOMO_ANSWER_PROMPT
from utils.progress import create_progress
from utils.response_options import add_save_model_input_arg, parse_bool
from client_factory import SUPPORTED_LIBS, DEFAULT_LIB
from locomo_common import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    error_payload,
    expected_answer_pairs,
    group_id_for,
    qa_question,
    record_status,
    status_counts,
    status_records_with_skipped,
)


_SPEAKER_MARKER = "Memories for user "


def _dedup_shared_context(context: str) -> str:
    """Deduplicate identical speaker sections in shared-conversation contexts.

    When both speakers share the same memory store (e.g. Backboard, Hindsight,
    Cognee), ``shared_conv_search`` duplicates the same text under both speaker
    headers.  This wastes tokens and adds noise.  If the two sections are
    identical we merge them into a single block.
    """
    if not context or _SPEAKER_MARKER not in context:
        return context or ""

    parts = context.split(_SPEAKER_MARKER)
    if len(parts) < 3:
        return context

    sections = []
    for part in parts[1:]:
        colon_idx = part.find(":")
        if colon_idx == -1:
            return context
        name = part[:colon_idx].strip()
        content = part[colon_idx + 1:].strip()
        sections.append((name, content))

    if len(sections) == 2 and sections[0][1] == sections[1][1]:
        return (
            f"Shared memories from conversation between "
            f"{sections[0][0]} and {sections[1][0]}:\n\n{sections[0][1]}"
        )
    return context


async def locomo_response(frame, llm_client, model_name, context: str, question: str):
    """Generate answer and return (answer, messages_sent_to_model)."""
    prompt = LOCOMO_ANSWER_PROMPT.format(
        context=context,
        question=question,
    )
    messages = [{"role": "user", "content": prompt}]

    response = await llm_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
    )
    result = response.choices[0].message.content or ""
    return result, messages


async def process_qa(frame, qa, search_result, oai_client, model_name, semaphore, save_model_input):
    async with semaphore:
        start = time()
        query = qa.get("question")
        gold_answer = qa.get("answer")
        qa_category = qa.get("category")

        raw_context = search_result.get("context") or ""
        context = _dedup_shared_context(raw_context)
        reflect_answer = search_result.get("reflect_answer")

        if reflect_answer:
            answer = reflect_answer
            model_input = None
        else:
            answer, model_input = await locomo_response(frame, oai_client, model_name, context, query)

        response_duration_ms = (time() - start) * 1000

        response_record = {
            "question": query,
            "answer": answer,
            "category": qa_category,
            "golden_answer": gold_answer,
            "response_duration_ms": response_duration_ms,
            "search_duration_ms": search_result.get("duration_ms", 0),
            "status": STATUS_SUCCESS,
        }
        if save_model_input:
            response_record["model_input"] = model_input
        return response_record


def _response_record_complete(record, expected_question):
    if not isinstance(record, dict):
        return False, ["missing response"]
    issues = []
    if record.get("question") != expected_question:
        issues.append("question mismatch")
    if record_status(record) == STATUS_SKIPPED:
        return not issues, issues
    for key in ("answer", "golden_answer", "response_duration_ms", "search_duration_ms"):
        if key not in record:
            issues.append(f"missing {key}")
    return not issues, issues


async def main(
    frame,
    version="default",
    llm_workers=10,
    save_model_input=False,
    *,
    skip_failed_answer=False,
):
    search_path = f"results/locomo/{frame}-{version}/{frame}_locomo_search_results.json"
    response_path = f"results/locomo/{frame}-{version}/{frame}_locomo_responses.json"

    load_env()
    from utils.llm_client import create_async_openai_client

    oai_client, answer_model = create_async_openai_client("ANSWER")
    print(f"[ANSWER] model={answer_model}")

    locomo_df = pd.read_json("data/locomo/locomo10.json")
    with open(search_path) as file:
        locomo_search_results = json.load(file)

    num_users = len(locomo_df)
    semaphore = asyncio.Semaphore(llm_workers)

    all_responses = {}
    failed_groups = []
    skipped_records = []
    if os.path.exists(response_path):
        try:
            with open(response_path) as f:
                all_responses = json.load(f)
            print(f"♻️  Loaded {len(all_responses)} existing groups for checkpoint/resume")
        except Exception:
            all_responses = {}

    total_questions = 0
    for group_idx in range(num_users):
        qa_set = locomo_df["qa"].iloc[group_idx]

        group_id = group_id_for(group_idx)
        search_results = locomo_search_results.get(group_id)
        matched_pairs, search_skipped, search_failures, _expected = expected_answer_pairs(
            qa_set,
            search_results,
        )
        expected_answer_questions = [qa_question(qa) for qa, _ in matched_pairs]
        skipped_records.extend(
            {"group_id": group_id, **record} for record in search_skipped
        )

        existing_records = all_responses.get(group_id, [])
        existing_by_question = {
            str(record.get("question")): record
            for record in existing_records
            if isinstance(record, dict) and record.get("question") is not None
        }
        responses_by_question = {}
        for qa, _search_result in matched_pairs:
            question = qa_question(qa)
            existing_record = existing_by_question.get(question)
            ok, issues = _response_record_complete(existing_record, question)
            if ok:
                responses_by_question[question] = existing_record
            elif existing_record is not None:
                print(
                    f"♻️  Reprocessing {group_id}/{question[:60]}; "
                    f"existing response incomplete ({'; '.join(issues)})"
                )

        if search_failures:
            failed_groups.append({
                "group_id": group_id,
                "stage": "answer",
                "failures": search_failures,
            })
            print(f"❌ {group_id}: cannot answer {len(search_failures)} questions due to search issues")

        pending_pairs = [
            (qa, search_result)
            for qa, search_result in matched_pairs
            if qa_question(qa) not in responses_by_question
        ]

        if not pending_pairs:
            if responses_by_question:
                all_responses[group_id] = [
                    responses_by_question[question]
                    for question in expected_answer_questions
                    if question in responses_by_question
                ]
                atomic_json_dump(all_responses, response_path, indent=2)
                total_questions += len(all_responses[group_id])
            if not search_failures:
                print(f"♻️  Skipping {group_id} (already processed)")
            continue

        async def run_pending_pair(qa, search_result):
            try:
                result = await process_qa(
                    frame,
                    qa,
                    search_result,
                    oai_client,
                    answer_model,
                    semaphore,
                    save_model_input,
                )
            except Exception as exc:
                result = exc
            return qa, search_result, result

        tasks = [
            run_pending_pair(qa, search_result)
            for qa, search_result in pending_pairs
        ]

        pbar_desc = f"[{group_idx+1}/{num_users}] {group_id}"
        answer_failures = []
        with create_progress() as progress:
            task_id = progress.add_task(pbar_desc, total=len(tasks))
            for coro in asyncio.as_completed(tasks):
                qa, _search_result, result = await coro
                question = qa_question(qa)
                if isinstance(result, Exception):
                    failure = {
                        "query": question,
                        "status": STATUS_SKIPPED if skip_failed_answer else STATUS_FAILED,
                        "error": error_payload("answer", result),
                    }
                    if skip_failed_answer:
                        skipped_records.append({"group_id": group_id, **failure})
                        skipped_response = {
                            "question": question,
                            "answer": "",
                            "category": qa.get("category"),
                            "golden_answer": qa.get("answer"),
                            "response_duration_ms": 0.0,
                            "search_duration_ms": _search_result.get("duration_ms", 0),
                            "status": STATUS_SKIPPED,
                            "skip_reason": "answer_failed",
                            "error": failure["error"],
                        }
                        responses_by_question[question] = skipped_response
                    else:
                        answer_failures.append(failure)
                    print(f"❌ Error generating response for {group_id}: {result}")
                else:
                    responses_by_question[question] = result
                all_responses[group_id] = [
                    responses_by_question[question]
                    for question in expected_answer_questions
                    if question in responses_by_question
                ]
                atomic_json_dump(all_responses, response_path, indent=2)
                progress.advance(task_id)

        if answer_failures:
            failed_groups.append({
                "group_id": group_id,
                "stage": "answer",
                "failures": answer_failures,
            })
        total_questions += len(all_responses.get(group_id, []))

    total_questions = sum(len(v) for v in all_responses.values())
    print(f"Total: {total_questions} questions across {len(all_responses)} users")

    atomic_json_dump(all_responses, response_path, indent=2)
    print(f"✅ All responses saved to {response_path}")

    from utils.token_tracker import get_tracker

    results_dir = f"results/locomo/{frame}-{version}"
    get_tracker().save(f"{results_dir}/token_usage_answer.json")
    atomic_json_dump(
        {
            "stage": "answer",
            "skip_failed_answer": skip_failed_answer,
            "status_counts": status_counts(
                status_records_with_skipped(all_responses, skipped_records)
            ),
            "failed_groups": failed_groups,
            "skipped_records": skipped_records,
        },
        f"{results_dir}/{frame}_locomo_response_status.json",
        indent=2,
    )

    if failed_groups:
        print(f"\n❌ RESPONSE GENERATION FAILED: {len(failed_groups)}/{num_users} groups had errors")
        raise SystemExit(1)


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
        help="Version identifier for loading results (e.g., 1010)",
    )
    parser.add_argument(
        "--llm-workers", "--llm_workers", type=int, default=10, help="Max concurrent LLM API calls."
    )
    add_save_model_input_arg(parser)
    parser.add_argument(
        "--skip-failed-answer",
        "--skip_failed_answer",
        type=parse_bool,
        default=False,
        help="Explicitly skip failed answer-generation calls instead of failing the step. Default: 0.",
    )
    args = parser.parse_args()
    lib = args.lib
    version = args.version
    asyncio.run(
        main(
            lib,
            version,
            llm_workers=args.llm_workers,
            save_model_input=args.save_model_input,
            skip_failed_answer=args.skip_failed_answer,
        )
    )

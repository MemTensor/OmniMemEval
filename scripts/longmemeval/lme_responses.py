import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from time import time

from utils.env import load_env
from utils.checkpoint import atomic_json_dump
from utils.progress import create_progress
from utils.prompts import LME_ANSWER_PROMPT
from utils.response_options import add_save_model_input_arg, parse_bool
from client_factory import SUPPORTED_LIBS, DEFAULT_LIB
from longmemeval.lme_common import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    error_payload,
    get_single_search_entry,
    record_status,
    response_complete,
    skipped_response_record,
    status_counts,
)


async def lme_response(llm_client, model_name, context, question, question_date, frame=None):
    """Generate answer and return (answer, messages_sent_to_model)."""
    prompt = LME_ANSWER_PROMPT.format(
        question=question,
        question_date=question_date,
        context=context,
    )
    messages = [{"role": "user", "content": prompt}]
    response = await llm_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
    )
    result = response.choices[0].message.content or ""
    return result, messages


async def process_qa(
    user_id,
    search_result,
    llm_client,
    model_name,
    semaphore,
    frame=None,
    save_model_input=False,
):
    async with semaphore:
        start = time()
        question = search_result.get("question")
        question_date = search_result.get("date")
        context = search_result.get("search_context", "")
        reflect_answer = search_result.get("reflect_answer")

        if reflect_answer:
            answer = reflect_answer
            model_input = None
        else:
            answer, model_input = await lme_response(
                llm_client,
                model_name,
                context,
                question,
                question_date,
                frame=frame,
            )

        response_duration_ms = (time() - start) * 1000

        print("\n" + "-" * 80)
        print(f"🤖 Processed User: {user_id}")
        print(f"⏱️  Duration: {response_duration_ms:.2f} ms")
        print(f"❓ Question: {question}")
        print(f"💬 Answer: {answer[:150]}..." if len(answer) > 150 else f"💬 Answer: {answer}")
        print("-" * 80)

        response_record = {
            "user_id": user_id,
            "category": search_result.get("category"),
            "question": question,
            "answer": answer,
            "question_date": question_date,
            "golden_answer": search_result.get("golden_answer"),
            "response_duration_ms": response_duration_ms,
            "search_duration_ms": search_result.get("search_duration_ms"),
            "answer_evidences": search_result.get("answer_evidences", []),
            "status": STATUS_SUCCESS,
        }
        if save_model_input:
            response_record["model_input"] = model_input
        return response_record


async def _with_key(key, coro):
    try:
        return key, await coro, None
    except Exception as exc:
        return key, None, exc


async def main(
    frame,
    version,
    llm_workers=10,
    save_model_input=False,
    *,
    skip_failed_answer=False,
):
    print("\n" + "=" * 80)
    print(f"🚀 LONGMEMEVAL RESPONSE GENERATION - {frame.upper()} v{version}".center(80))
    print("=" * 80)

    load_env()
    from utils.llm_client import create_async_openai_client

    oai_client, answer_model = create_async_openai_client("ANSWER")
    print(f"🔌 [ANSWER] model={answer_model}")

    search_path = f"results/lme/{frame}-{version}/{frame}_lme_search_results.json"
    response_path = f"results/lme/{frame}-{version}/{frame}_lme_responses.json"

    print(f"📂 Loading search results from: {search_path}")
    with open(search_path) as file:
        lme_search_results = json.load(file)
    print(f"📊 Found {len(lme_search_results)} users to process")
    print(f"⚙️  Using {llm_workers} LLM worker threads")
    print("-" * 80)

    lme_responses = {}
    if os.path.exists(response_path):
        try:
            with open(response_path) as f:
                lme_responses = json.load(f)
            print(f"♻️  Loaded {len(lme_responses)} existing results for checkpoint/resume")
        except Exception:
            lme_responses = {}

    start_time = time()
    semaphore = asyncio.Semaphore(llm_workers)

    tasks = []
    failed_users = []
    skipped_records = []
    for user_id, search_results in lme_search_results.items():
        search_entry = get_single_search_entry(lme_search_results, user_id)
        if search_entry is None:
            failed_users.append({
                "user_id": user_id,
                "error": error_payload("answer", "missing or malformed search result"),
            })
            continue

        if user_id in lme_responses:
            ok, issues = response_complete(lme_responses.get(user_id), search_entry)
            if ok:
                print(f"♻️  Skipping {user_id} (already processed)")
                continue
            print(
                f"♻️  Reprocessing {user_id}; existing response incomplete "
                f"({'; '.join(issues)})"
            )
            lme_responses.pop(user_id, None)

        status = record_status(search_entry)
        if status == STATUS_SKIPPED:
            skipped = skipped_response_record(
                user_id=user_id,
                search_entry=search_entry,
                reason="search was explicitly skipped",
                error=search_entry.get("error"),
            )
            lme_responses[user_id] = skipped
            skipped_records.append(skipped)
            atomic_json_dump(lme_responses, response_path, indent=4)
            continue
        if status == STATUS_FAILED:
            failed_users.append({
                "user_id": user_id,
                "error": search_entry.get("error")
                or error_payload("answer", "search result is failed"),
            })
            continue

        tasks.append(
            _with_key(
                user_id,
                process_qa(
                    user_id,
                    search_entry,
                    oai_client,
                    answer_model,
                    semaphore,
                    frame=frame,
                    save_model_input=save_model_input,
                ),
            )
        )

    with create_progress() as progress:
        task_id = progress.add_task("Generating responses", total=len(tasks))
        for coro in asyncio.as_completed(tasks):
            user_id, result, exc = await coro
            if exc:
                print(f"❌ Error processing user {user_id}: {exc}")
                search_entry = get_single_search_entry(lme_search_results, user_id) or {}
                failure = {
                    "user_id": user_id,
                    "error": error_payload("answer", exc),
                }
                if skip_failed_answer:
                    skipped = skipped_response_record(
                        user_id=user_id,
                        search_entry=search_entry,
                        reason="answer_failed",
                        error=failure["error"],
                    )
                    lme_responses[user_id] = skipped
                    skipped_records.append(skipped)
                    atomic_json_dump(lme_responses, response_path, indent=4)
                else:
                    failed_users.append(failure)
            else:
                lme_responses[user_id] = result
                atomic_json_dump(lme_responses, response_path, indent=4)
            progress.advance(task_id)

    end_time = time()
    elapsed_time = end_time - start_time
    elapsed_sec = int(elapsed_time)

    atomic_json_dump(lme_responses, response_path, indent=4)
    print(f"📁 Responses saved to: {response_path}")

    from utils.token_tracker import get_tracker

    results_dir = f"results/lme/{frame}-{version}"
    get_tracker().save(f"{results_dir}/token_usage_answer.json")
    atomic_json_dump(
        {
            "stage": "answer",
            "skip_failed_answer": skip_failed_answer,
            "status_counts": status_counts(list(lme_responses.values())),
            "failed_users": failed_users,
            "skipped_records": skipped_records,
        },
        f"{results_dir}/{frame}_lme_response_status.json",
        indent=2,
    )

    if failed_users:
        print(
            f"\n❌ RESPONSE GENERATION FAILED: {len(failed_users)}/"
            f"{len(lme_search_results)} users had errors"
        )
        raise SystemExit(1)

    print("\n" + "=" * 80)
    print("✅ RESPONSE GENERATION COMPLETE".center(80))
    print("=" * 80)
    print(f"⏱️ Total time: {elapsed_sec // 60}m {elapsed_sec % 60}s")
    print(f"📊 Processed: {len(lme_responses)} users")
    print(f"🔄 Framework: {frame} | Version: {version}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LongMemeval Response Generation Script")
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
        "--llm-workers", "--llm_workers", type=int, default=10, help="Max concurrent LLM API calls."
    )
    add_save_model_input_arg(parser)
    parser.add_argument(
        "--skip-failed-answer",
        "--skip_failed_answer",
        type=parse_bool,
        default=False,
        help=(
            "Explicitly skip failed answer-generation calls instead of failing "
            "the step. Default: 0."
        ),
    )

    args = parser.parse_args()
    asyncio.run(
        main(
            frame=args.lib,
            version=args.version,
            llm_workers=args.llm_workers,
            save_model_input=args.save_model_input,
            skip_failed_answer=args.skip_failed_answer,
        )
    )

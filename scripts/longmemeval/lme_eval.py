import argparse
import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import transformers

from utils.env import load_env
from utils.checkpoint import atomic_json_dump
from utils.nlp_metrics import (
    init_nlp, calculate_nlp_metrics, extract_label_json, LLMGrade,
)
from utils.progress import create_progress
from utils.response_options import parse_bool
from utils.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_PROMPT
from client_factory import SUPPORTED_LIBS, DEFAULT_LIB
from longmemeval.lme_common import (
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    error_payload,
    grade_complete,
    record_status,
    status_counts,
)

logging.basicConfig(level=logging.CRITICAL)
transformers.logging.set_verbosity_error()


async def lme_grader(
    llm_client,
    eval_model_name,
    question,
    golden_answer,
    response,
    semaphore: asyncio.Semaphore,
):
    judge_prompt = JUDGE_PROMPT.format(
        question=question, golden_answer=golden_answer, response=response
    )

    async with semaphore:
        api_response = await llm_client.chat.completions.create(
            model=eval_model_name,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0,
        )

    message_content = api_response.choices[0].message.content or ""
    label_json = extract_label_json(text=message_content)
    if label_json is None:
        raise ValueError(
            f"could not extract judge label from response: {message_content[:200]}"
        )
    label = json.loads(label_json)["label"]
    parsed = LLMGrade(llm_judgment=label, llm_reasoning="")
    return parsed.llm_judgment.strip().lower() == "correct"


async def process_qa(
    user_id,
    response_data,
    llm_client,
    eval_model_name,
    num_runs: int,
    llm_semaphore: asyncio.Semaphore,
    nlp_options=None,
):
    question = response_data.get("question")
    golden_answer = response_data.get("golden_answer", "")
    context = response_data.get("search_context", "")
    response = response_data.get("answer", "")

    grading_tasks = [
        lme_grader(
            llm_client,
            eval_model_name,
            question,
            golden_answer,
            response,
            llm_semaphore,
        )
        for _ in range(num_runs)
    ]
    judgments = await asyncio.gather(*grading_tasks, return_exceptions=True)
    errors = [judgment for judgment in judgments if isinstance(judgment, Exception)]
    if errors:
        raise RuntimeError(f"judge failed: {errors[0]}") from errors[0]
    judgments_dict = {f"judgment_{i + 1}": j for i, j in enumerate(judgments)}

    nlp_metrics = calculate_nlp_metrics(
        gold_answer=golden_answer, response=response, context=context, options=nlp_options
    )

    print("\n" + "=" * 80)
    print(f"🔍 Processed User: {user_id}")
    print("-" * 80)
    print(f"❓ Question: \n   {question}")
    print("-" * 80)
    print(
        f"📖 Golden Answer: \n   {golden_answer[:150]}..."
        if len(str(golden_answer)) > 150
        else f"📖 Golden Answer: \n   {golden_answer}"
    )
    print("-" * 80)
    print(
        f"💬 LLM Response: \n   {response[:150]}..."
        if len(str(response)) > 150
        else f"💬 Answer: \n   {response}"
    )
    print("-" * 80)

    judgments_formatted = []
    for run, correct in judgments_dict.items():
        status = "✓ CORRECT" if correct else "✗ WRONG"
        judgments_formatted.append(f"{run}: {status}")

    print(f"⚖️  Judgments: \n   {', '.join(judgments_formatted)}")
    print("=" * 80)

    graded_response = {
        "user_id": user_id,
        "category": response_data.get("category"),
        "question": question,
        "question_date": response_data.get("question_date"),
        "golden_answer": response_data.get("golden_answer"),
        "answer": response,
        "llm_judgments": judgments_dict,
        "nlp_metrics": nlp_metrics,
        "response_duration_ms": response_data.get("response_duration_ms"),
        "search_duration_ms": response_data.get("search_duration_ms"),
        "total_duration_ms": response_data.get("response_duration_ms")
        + response_data.get("search_duration_ms", 0),
        "status": STATUS_SUCCESS,
    }
    return graded_response


async def _with_key(key, coro):
    try:
        return key, await coro, None
    except Exception as exc:
        return key, None, exc


def skipped_grade_record(user_id, response_data, *, reason, error=None):
    response_duration = response_data.get("response_duration_ms") or 0.0
    search_duration = response_data.get("search_duration_ms") or 0.0
    record = {
        "user_id": user_id,
        "category": response_data.get("category"),
        "question": response_data.get("question"),
        "question_date": response_data.get("question_date"),
        "golden_answer": response_data.get("golden_answer"),
        "answer": response_data.get("answer", ""),
        "response_duration_ms": response_duration,
        "search_duration_ms": search_duration,
        "total_duration_ms": response_duration + search_duration,
        "status": STATUS_SKIPPED,
        "skip_reason": reason,
    }
    if error is not None:
        record["error"] = error
    return record


def convert_numpy_types(obj):
    if isinstance(obj, np.number):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj


def evaluate_accuracy(results, num_runs):
    run_scores = []
    evaluated_count = 0

    for i in range(1, num_runs + 1):
        judgment_key = f"judgment_{i}"
        correct, total = 0, 0
        for _, response in results.items():
            if record_status(response) != STATUS_SUCCESS:
                continue
            judgments = response.get("llm_judgments", {})
            if judgment_key in judgments:
                total += 1
                if judgments[judgment_key]:
                    correct += 1
        if total > 0:
            run_scores.append(correct / total)
            evaluated_count += total
    evaluated_count = evaluated_count // num_runs
    return run_scores, evaluated_count


async def main(
    frame,
    version="default",
    nlp_options=None,
    num_runs=1,
    llm_workers=10,
    *,
    skip_failed_judge=False,
):
    init_nlp()
    print(f"Starting evaluation for {frame} version {version}...")

    load_env()
    from utils.llm_client import create_async_openai_client

    oai_client, eval_model = create_async_openai_client("EVAL")
    print(f"[EVAL] model={eval_model}")

    response_path = f"results/lme/{frame}-{version}/{frame}_lme_responses.json"
    search_path = f"results/lme/{frame}-{version}/{frame}_lme_search_results.json"
    judged_path = f"results/lme/{frame}-{version}/{frame}_lme_judged.json"

    with open(response_path) as file:
        lme_responses = json.load(file)

    if os.path.exists(search_path):
        with open(search_path) as f:
            lme_search_data = json.load(f)
        for uid, entries in lme_search_data.items():
            if uid in lme_responses:
                ctx = (
                    entries[0].get("search_context", "")
                    if isinstance(entries, list) and entries
                    else ""
                )
                lme_responses[uid].setdefault("search_context", ctx)
        print(f"📂 Loaded search contexts from: {search_path}")

    lme_eval_results = {}
    if os.path.exists(judged_path):
        try:
            with open(judged_path) as f:
                lme_eval_results = json.load(f)
            print(f"♻️  Loaded {len(lme_eval_results)} existing results for checkpoint/resume")
        except Exception:
            lme_eval_results = {}

    tasks = []
    skipped_records = []
    failed_users = []
    already_done = 0
    llm_semaphore = asyncio.Semaphore(llm_workers)

    for user_id, response_data in lme_responses.items():
        if record_status(response_data) == STATUS_SKIPPED:
            skipped = skipped_grade_record(
                user_id,
                response_data,
                reason=response_data.get("skip_reason", "response was skipped"),
                error=response_data.get("error"),
            )
            lme_eval_results[user_id] = skipped
            skipped_records.append({
                "user_id": user_id,
                "question": response_data.get("question"),
                "reason": response_data.get("skip_reason", "response was skipped"),
                "error": response_data.get("error"),
            })
            continue

        if user_id in lme_eval_results:
            ok, issues = grade_complete(
                lme_eval_results.get(user_id),
                response_data,
                num_runs,
                allow_skipped_grade=skip_failed_judge,
            )
            if ok:
                already_done += 1
                continue
            print(
                f"♻️  Reprocessing {user_id}; existing grade incomplete "
                f"({'; '.join(issues)})"
            )
            lme_eval_results.pop(user_id, None)

        tasks.append(
            _with_key(
                user_id,
                process_qa(
                    user_id,
                    response_data,
                    oai_client,
                    eval_model,
                    num_runs,
                    llm_semaphore,
                    nlp_options,
                ),
            )
        )

    if already_done:
        print(f"♻️  Skipping {already_done} already-evaluated users")

    with create_progress() as progress:
        task_id = progress.add_task("Evaluating users", total=len(tasks))
        for coro in asyncio.as_completed(tasks):
            user_id, result, exc = await coro
            if exc:
                response_data = lme_responses.get(user_id, {})
                failure = {
                    "user_id": user_id,
                    "error": error_payload("eval", exc),
                }
                if skip_failed_judge:
                    lme_eval_results[user_id] = skipped_grade_record(
                        user_id,
                        response_data,
                        reason="eval_failed",
                        error=failure["error"],
                    )
                    skipped_records.append(failure)
                    atomic_json_dump(
                        convert_numpy_types(lme_eval_results),
                        judged_path,
                        indent=4,
                    )
                else:
                    failed_users.append(failure)
                print(f"[ERROR] Processing user {user_id} failed: {exc}")
            else:
                user_id = result["user_id"]
                lme_eval_results[user_id] = result
                atomic_json_dump(
                    convert_numpy_types(lme_eval_results),
                    judged_path,
                    indent=4,
                )
            progress.advance(task_id)

    run_scores, evaluated_count = evaluate_accuracy(lme_eval_results, num_runs)

    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY".center(80))
    print("=" * 80)

    if evaluated_count > 0:
        print(f"📋 Evaluated: {evaluated_count} responses across {num_runs} runs")
        print(f"🎯 LLM-as-a-Judge Mean Accuracy: {np.mean(run_scores):.4f}")
        print(f"🔍 Standard Deviation: {np.std(run_scores):.4f}")

        run_scores_formatted = [f"{round(s, 4):.4f}" for s in run_scores]
        print(f"🔢 Individual run scores: [{', '.join(run_scores_formatted)}]")
    else:
        print("⚠️  No responses were evaluated. LLM-as-a-Judge score: N/A (0/0)")

    print("-" * 80)

    lme_eval_results = convert_numpy_types(lme_eval_results)
    atomic_json_dump(lme_eval_results, judged_path, indent=4)
    print(f"📁 Results saved to: {judged_path}")

    results_dir = f"results/lme/{frame}-{version}"
    atomic_json_dump(
        {
            "stage": "eval",
            "skip_failed_judge": skip_failed_judge,
            "status_counts": status_counts(list(lme_eval_results.values())),
            "failed_users": failed_users,
            "skipped_records": skipped_records,
        },
        f"{results_dir}/{frame}_lme_eval_status.json",
        indent=2,
    )

    if failed_users:
        print(f"\n❌ EVALUATION FAILED: {len(failed_users)} users had errors")
        raise SystemExit(1)

    print("✅ Evaluation completed successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM responses using LLM-as-a-Judge.")
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
        "--options",
        type=str,
        nargs="+",
        default=["lexical"],
        choices=["lexical", "semantic"],
        help="NLP options to use for evaluation.",
    )
    parser.add_argument(
        "--num_runs", type=int, default=1, help="Number of runs for LLM-as-a-Judge evaluation."
    )
    parser.add_argument(
        "--llm-workers", "--llm_workers", type=int, default=10, help="Max concurrent LLM API calls."
    )
    parser.add_argument(
        "--skip-failed-judge",
        "--skip_failed_judge",
        type=parse_bool,
        default=False,
        help="Explicitly skip failed judge calls instead of failing the step. Default: 0.",
    )

    args = parser.parse_args()
    asyncio.run(
        main(
            frame=args.lib,
            version=args.version,
            nlp_options=args.options,
            num_runs=args.num_runs,
            llm_workers=args.llm_workers,
            skip_failed_judge=args.skip_failed_judge,
        )
    )

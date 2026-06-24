import argparse
import asyncio
import json
import logging
import os
import sys
import time

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
from client_factory import SUPPORTED_LIBS, DEFAULT_LIB
from utils.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_PROMPT
from locomo_common import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    error_payload,
    record_status,
    status_counts,
    status_records_with_skipped,
    validate_query_coverage,
)


logging.basicConfig(level=logging.CRITICAL)
transformers.logging.set_verbosity_error()


async def locomo_grader(
    llm_client,
    eval_model_name,
    question: str,
    gold_answer: str,
    response: str,
    semaphore: asyncio.Semaphore,
) -> bool:
    accuracy_prompt = JUDGE_PROMPT.format(
        question=question, golden_answer=gold_answer, response=response
    )
    async with semaphore:
        api_response = await llm_client.chat.completions.create(
            model=eval_model_name,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": accuracy_prompt},
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


def convert_numpy_types(obj):
    if isinstance(obj, np.number):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj


def _existing_grades_complete(
    existing_grades,
    group_responses,
    num_runs,
    *,
    allow_skipped_grade=False,
):
    expected_questions = [
        response.get("question", "")
        for response in group_responses
        if record_status(response) != STATUS_SKIPPED
    ]
    ok, issues = validate_query_coverage(
        existing_grades,
        expected_questions,
        query_key="question",
    )
    if not ok:
        return False, issues

    required_keys = {f"judgment_{i}" for i in range(1, num_runs + 1)}
    for grade in existing_grades:
        if record_status(grade) == STATUS_SKIPPED:
            if allow_skipped_grade:
                continue
            return False, ["existing grade is skipped"]
        judgments = grade.get("llm_judgments", {})
        if not required_keys.issubset(judgments):
            return False, ["missing judgment runs"]
    return True, []


def skipped_grade_record(response, *, reason, error=None):
    response_duration = response.get("response_duration_ms") or 0.0
    search_duration = response.get("search_duration_ms") or 0.0
    record = {
        "question": response.get("question"),
        "answer": response.get("answer", ""),
        "golden_answer": response.get("golden_answer"),
        "category": response.get("category"),
        "response_duration_ms": response_duration,
        "search_duration_ms": search_duration,
        "total_duration_ms": response_duration + search_duration,
        "status": STATUS_SKIPPED,
        "skip_reason": reason,
    }
    if error is not None:
        record["error"] = error
    return record


async def process_group_responses(
    group_id,
    group_responses,
    oai_client,
    eval_model,
    nlp_options,
    num_runs: int,
    llm_semaphore: asyncio.Semaphore,
    *,
    skip_failed_judge=False,
):
    graded_responses = []
    blocking_errors = []
    skipped_errors = []

    # Process responses with asyncio for concurrent API calls
    for i, response in enumerate(group_responses, 1):
        if record_status(response) == STATUS_SKIPPED:
            skipped_errors.append({
                "question": response.get("question"),
                "status": STATUS_SKIPPED,
                "reason": response.get("skip_reason", "response was skipped"),
                "error": response.get("error"),
            })
            continue

        print(f"  [{i}/{len(group_responses)}] Grading {group_id}")
        question = response.get("question")
        answer = response.get("answer")
        ground_truth = response.get("golden_answer")
        category = response.get("category")

        context = response.get("search_context", "")
        response_duration_ms = response.get("response_duration_ms", 0.0)
        search_duration_ms = response.get("search_duration_ms", 0.0)

        if ground_truth is None:
            failure = {
                "question": question,
                "status": STATUS_SKIPPED if skip_failed_judge else STATUS_FAILED,
                "error": error_payload("eval", "missing golden_answer"),
            }
            if skip_failed_judge:
                graded_responses.append(
                    skipped_grade_record(
                        response,
                        reason="eval_failed",
                        error=failure["error"],
                    )
                )
                skipped_errors.append(failure)
            else:
                blocking_errors.append(failure)
            continue

        grading_tasks = [
            locomo_grader(
                oai_client,
                eval_model,
                question,
                ground_truth,
                answer,
                llm_semaphore,
            )
            for _ in range(num_runs)
        ]
        judgments = await asyncio.gather(*grading_tasks, return_exceptions=True)
        errors = [judgment for judgment in judgments if isinstance(judgment, Exception)]
        if errors:
            failure = {
                "question": question,
                "status": STATUS_SKIPPED if skip_failed_judge else STATUS_FAILED,
                "error": error_payload("eval", errors[0]),
            }
            if skip_failed_judge:
                graded_responses.append(
                    skipped_grade_record(
                        response,
                        reason="eval_failed",
                        error=failure["error"],
                    )
                )
                skipped_errors.append(failure)
            else:
                blocking_errors.append(failure)
            continue

        judgments_dict = {f"judgment_{i + 1}": j for i, j in enumerate(judgments)}

        nlp_metrics = calculate_nlp_metrics(ground_truth, answer, context, nlp_options)

        graded_response = {
            "question": question,
            "answer": answer,
            "golden_answer": ground_truth,
            "category": category,
            "llm_judgments": judgments_dict,
            "nlp_metrics": nlp_metrics,
            "response_duration_ms": response_duration_ms,
            "search_duration_ms": search_duration_ms,
            "total_duration_ms": response_duration_ms + search_duration_ms,
            "status": STATUS_SUCCESS,
        }
        graded_responses.append(graded_response)

    return group_id, graded_responses, blocking_errors, skipped_errors


async def process_single_group(
    group_id,
    group_responses,
    oai_client,
    eval_model,
    nlp_options,
    num_runs,
    llm_semaphore,
    *,
    skip_failed_judge=False,
):
    start_time = time.time()
    result = await process_group_responses(
        group_id,
        group_responses,
        oai_client,
        eval_model,
        nlp_options,
        num_runs,
        llm_semaphore,
        skip_failed_judge=skip_failed_judge,
    )
    elapsed_time = round(time.time() - start_time, 2)
    print(f"Group {group_id} processed in {elapsed_time} seconds")
    return result


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
    print(
        f"\n=== Starting LoCoMo evaluation for {frame} (version: {version}) with {num_runs} run(s) per question ==="
    )
    print(f"Using {llm_workers} max concurrent LLM API calls")

    results_dir = f"results/locomo/{frame}-{version}"
    response_path = f"{results_dir}/{frame}_locomo_responses.json"
    search_path = f"{results_dir}/{frame}_locomo_search_results.json"
    judged_path = f"{results_dir}/{frame}_locomo_judged.json"

    os.makedirs(results_dir, exist_ok=True)

    load_env()
    from utils.llm_client import create_async_openai_client

    oai_client, eval_model = create_async_openai_client("EVAL")
    print(f"[EVAL] model={eval_model}")

    with open(response_path) as file:
        locomo_responses = json.load(file)

    if os.path.exists(search_path):
        with open(search_path) as f:
            locomo_search_data = json.load(f)
        for uid, resp_list in locomo_responses.items():
            search_list = locomo_search_data.get(uid, [])
            ctx_by_query = {s.get("query", ""): s.get("context", "") for s in search_list}
            for resp in resp_list:
                resp.setdefault("search_context", ctx_by_query.get(resp.get("question", ""), ""))
        print(f"📂 Loaded search contexts from: {search_path}")

    all_grades = {}

    if os.path.exists(judged_path):
        try:
            with open(judged_path) as f:
                all_grades = json.load(f)
            print(f"♻️  Loaded {len(all_grades)} existing groups for checkpoint/resume")
        except Exception:
            all_grades = {}

    user_ids = sorted(locomo_responses.keys())
    num_users = len(user_ids)
    total_responses_count = sum(len(locomo_responses[uid]) for uid in user_ids)
    print(f"Found {total_responses_count} total responses across {num_users} users to evaluate")

    tasks = []
    active_users = 0
    llm_semaphore = asyncio.Semaphore(llm_workers)
    for group_id in user_ids:
        if group_id in all_grades:
            ok, issues = _existing_grades_complete(
                all_grades.get(group_id, []),
                locomo_responses.get(group_id, []),
                num_runs,
                allow_skipped_grade=skip_failed_judge,
            )
            if ok:
                print(f"♻️  Skipping {group_id} (already evaluated)")
                continue
            print(
                f"♻️  Reprocessing {group_id}; existing grades incomplete "
                f"({'; '.join(issues)})"
            )
            all_grades.pop(group_id, None)

        group_responses = locomo_responses.get(group_id, [])
        if not group_responses:
            print(f"No responses found for group {group_id}; marking empty group complete")
            all_grades[group_id] = []
            atomic_json_dump(convert_numpy_types(all_grades), judged_path, indent=2)
            continue

        active_users += 1
        tasks.append(
            process_single_group(
                group_id,
                group_responses,
                oai_client,
                eval_model,
                nlp_options,
                num_runs,
                llm_semaphore,
                skip_failed_judge=skip_failed_judge,
            )
        )

    print(f"Starting evaluation of {active_users} user groups with responses")

    failed_groups = []
    skipped_records = []

    with create_progress() as progress:
        task_id = progress.add_task("Evaluating groups", total=len(tasks))
        for coro in asyncio.as_completed(tasks):
            try:
                group_id, graded_responses, blocking_errors, group_skipped = await coro
                skipped_records.extend(
                    {"group_id": group_id, **record} for record in group_skipped
                )
                if blocking_errors:
                    failed_groups.append({
                        "group_id": group_id,
                        "stage": "eval",
                        "failures": blocking_errors,
                    })
                else:
                    all_grades[group_id] = graded_responses
                    atomic_json_dump(convert_numpy_types(all_grades), judged_path, indent=2)
            except Exception as exc:
                print(f"[ERROR] Processing group failed: {exc}")
                failed_groups.append({
                    "group_id": None,
                    "stage": "eval",
                    "error": error_payload("eval", exc),
                })
            progress.advance(task_id)

    print("\n=== Evaluation Complete: Calculating final scores ===")

    run_scores = []
    evaluated_count = 0
    if num_runs > 0:
        for i in range(1, num_runs + 1):
            judgment_key = f"judgment_{i}"
            current_run_correct_count = 0
            current_run_total_count = 0
            for group in all_grades.values():
                for response in group:
                    if record_status(response) != STATUS_SUCCESS:
                        continue
                    judgments = response.get("llm_judgments", {})
                    if judgment_key in judgments:
                        if judgments[judgment_key]:
                            current_run_correct_count += 1
                        current_run_total_count += 1

            if current_run_total_count > 0:
                run_accuracy = current_run_correct_count / current_run_total_count
                run_scores.append(run_accuracy)

        evaluated_count = current_run_total_count

    if evaluated_count > 0:
        mean_of_scores = np.mean(run_scores)
        std_of_scores = np.std(run_scores)
        print(f"LLM-as-a-Judge Mean Score: {mean_of_scores:.4f}")
        print(f"LLM-as-a-Judge Standard Deviation: {std_of_scores:.4f}")
        print(f"(Calculated from {num_runs} separate runs over {evaluated_count} questions)")
        print(f"Individual run scores: {[round(s, 4) for s in run_scores]}")
    else:
        print("No responses were evaluated")
        print("LLM-as-a-Judge score: N/A (0/0)")

    all_grades = convert_numpy_types(all_grades)
    atomic_json_dump(all_grades, judged_path, indent=2)
    print(f"Saved detailed evaluation results to {judged_path}")

    atomic_json_dump(
        {
            "stage": "eval",
            "skip_failed_judge": skip_failed_judge,
            "status_counts": status_counts(
                status_records_with_skipped(all_grades, skipped_records)
            ),
            "failed_groups": failed_groups,
            "skipped_records": skipped_records,
        },
        f"{results_dir}/{frame}_locomo_eval_status.json",
        indent=2,
    )

    if failed_groups:
        print(f"\n❌ EVALUATION FAILED: {len(failed_groups)} groups had errors")
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
        "--num_runs",
        type=int,
        default=1,
        help="Number of times to run the LLM grader for each question",
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

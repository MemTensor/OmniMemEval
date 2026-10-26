#!/usr/bin/env python3
"""Run the bundled end-to-end OmniMemEval LongMemEval-S benchmark.

The evaluator writes dialogue-message claims through MemoryOS' existing v2
ingest API, then uses v2 retrieval and separate answer/Judge models. Results
are checkpointed after each case so a restarted evaluator resumes safely.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from omnimemeval_badcase_ids import selected_ids


DATA_PATH = Path(os.getenv(
    "OMNIMEMEVAL_LME_DATA_PATH",
    "/app/data/omnimemeval/longmemeval_s_cleaned.json.gz",
))
OUTPUT_DIR = Path(os.getenv("OMNIMEMEVAL_LME_OUTPUT_DIR", "/tmp/omnimemeval/results"))
ANSWER_PROMPT = """\
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from a conversation. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories.
2. Pay special attention to the timestamps to determine the answer.
3. If the question asks about a specific event or fact, look for direct evidence in the memories.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question.
2. Examine the timestamps and content of these memories carefully.
3. Look for explicit mentions of dates, times, locations, or events that answer the question.
4. If the answer requires calculation (e.g., converting relative time references), show your work.
5. Formulate a precise, concise answer based solely on the evidence in the memories.
6. Double-check that your answer directly addresses the question asked.
7. Ensure your final answer is specific and avoids vague time references.

{context}

Current Date: {question_date}

Question: {question}

Answer:
"""
JUDGE_SYSTEM = "You are an expert grader that determines if answers to questions match a gold standard answer."
JUDGE_PROMPT = """\
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic. The generated answer might be much longer, but be generous: if it touches on the same topic as the gold answer, it is CORRECT.
For time-related questions, accept different formats when they refer to the same date or time period.

Question: {question}
Gold answer: {gold}
Generated answer: {answer}

First provide a short one-sentence explanation, then return CORRECT or WRONG as JSON with the key "label". Do not include both labels.
"""


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"required environment variable {name} is not set")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _load_dataset(path: Path) -> tuple[list[dict], str]:
    raw = gzip.decompress(path.read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    if not isinstance(data, list) or len(data) != 500:
        raise ValueError(f"expected 500 LongMemEval-S cases, found {len(data) if isinstance(data, list) else 'invalid data'}")
    return data, digest


def _parse_date(value: str) -> str | None:
    try:
        return datetime.strptime(value, "%Y/%m/%d (%a) %H:%M").isoformat(timespec="minutes") + "+00:00"
    except (TypeError, ValueError):
        return None


def _session_claim_batches(
    session: list[dict],
    *,
    run_version: str,
    case_index: int,
    session_id: str,
    session_index: int,
    occurred_at: str | None,
    include_assistant: bool = True,
    batch_size: int = 200,
) -> list[dict]:
    """Represent each dialogue message as a searchable v2 claim.

    This intentionally uses the existing ``/api/v2/memory/ingest`` contract;
    it does not invoke MemoryOS parse rules. Batch size follows the API's
    MemoryIngestRequest limit, and long single messages are split safely.
    """
    if not 1 <= batch_size <= 200:
        raise ValueError("claim batch size must be between 1 and 200")
    claims = []
    max_content_chars = 15970  # leave room for speaker prefix under text's 16k cap
    for message_index, message in enumerate(session):
        role = str(message.get("role") or "user").strip().lower() or "user"
        if role == "assistant" and not include_assistant:
            continue
        content = str(message.get("content") or "").replace("<|endoftext|>", "").strip()
        if not content:
            continue
        parts = [content[i:i + max_content_chars] for i in range(0, len(content), max_content_chars)]
        for part_index, part in enumerate(parts):
            identity = f"{run_version}:{case_index}:{session_index}:{message_index}:{part_index}"
            claim_key = "msg_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
            claims.append({
                "dimension": "conversation",
                "claim_type": "conversation_message",
                "claim_key": claim_key,
                "value": f"{role}: {part}",
                "text": f"{role}: {part}",
                "fact_status": "stated",
                "confidence": 1.0,
                "importance": 3,
            })

    batches = []
    for batch_index, offset in enumerate(range(0, len(claims), batch_size)):
        batch_claims = claims[offset:offset + batch_size]
        event_id = f"{run_version}:{case_index}:{session_index}:{batch_index}"
        batches.append({
            "claims": batch_claims,
            "source_event": {
                "event_id": event_id,
                "source_type": "dialogue",
                "source_id": session_id,
                "meeting_id": f"lme_{case_index}:{session_id}",
                "segment_id": str(batch_index),
                "occurred_at": occurred_at,
                "raw_text": "\n".join(claim["text"] for claim in batch_claims)[:200000] or None,
            },
        })
    return batches


def _load_state(path: Path, config: dict) -> dict:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("config") != config:
            raise ValueError("existing evaluation checkpoint config differs; use a new output directory/version")
        return state
    return {"schema_version": "omnimemeval-lme-e2e-v1", "config": config, "status": "running", "cases": {}}


def _http_client(base_url: str, api_key: str, tenant_id: str, org_id: str, timeout: float) -> httpx.Client:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Tenant-Id": tenant_id,
        "X-Org-Id": org_id,
        "Content-Type": "application/json",
    }
    return httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)


def _post(client: httpx.Client, path: str, payload: dict | None = None) -> dict:
    response = client.post(path, json=payload or {})
    response.raise_for_status()
    result = response.json()
    if result.get("code", 0) != 0:
        raise RuntimeError(f"{path} returned code={result.get('code')}: {result.get('message')}")
    return result.get("data") or {}


def _claim_context(items: list[dict]) -> str:
    memories = []
    for item in items:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        dates = []
        for provenance in item.get("provenance") or []:
            stamp = provenance.get("occurred_at")
            if stamp and stamp not in dates:
                dates.append(str(stamp))
        prefix = f"[occurred_at: {', '.join(dates)}] " if dates else ""
        memories.append(f"- {prefix}{text}")
    return "Conversation memories:\n\n" + "\n".join(memories)


def _assistant_recall_query(question: str) -> str | None:
    """Search prior assistant utterances only when the question asks for one.

    The benchmark's question type and gold answer must not influence retrieval.
    """
    if not re.search(
        r"\b(?:you|your|assistant)\s+(?:recommended|suggested|said|told|provided|"
        r"listed|explained|advised|gave|created|wrote|generated|showed)\b"
        r"|\b(?:recommendation|suggestion|answer|response)\s+you\s+(?:gave|provided|made)\b",
        question, flags=re.I,
    ):
        return None
    return f"{question} Prior assistant response, recommendation, or exact list of details."


def _merge_assistant_evidence(
    base_items: list[dict], assistant_items: list[dict], *, top_k: int, max_tokens: int,
) -> tuple[list[dict], dict]:
    """Reserve a small share of context for direct assistant statements.

    Preserve base retrieval for normal facts; prepend only a bounded number of
    previously missing assistant utterances. Never exceed the item or token budget.
    """
    selected: list[dict] = []
    seen: set[str] = set()
    used_tokens = 0

    def add(item: dict) -> bool:
        nonlocal used_tokens
        value = str(item.get("text") or "").strip()
        identity = str(item.get("claim_id") or value)
        cost = (len(value) + 1) // 2
        if not value or identity in seen or used_tokens + cost > max_tokens or len(selected) >= top_k:
            return False
        selected.append(item)
        seen.add(identity)
        used_tokens += cost
        return True

    assistant_added = 0
    base_ids = {str(item.get("claim_id") or item.get("text") or "") for item in base_items}
    for item in assistant_items:
        if assistant_added >= min(4, max(1, top_k // 4)):
            break
        identity = str(item.get("claim_id") or item.get("text") or "")
        if (identity not in base_ids
                and re.match(r"^assistant:\s", str(item.get("text") or ""), flags=re.I)
                and add(item)):
            assistant_added += 1
    if not assistant_added:
        return base_items, {"assistant_evidence_added": 0, "selected": len(base_items),
                            "estimated_tokens": sum((len(str(item.get("text") or "")) + 1) // 2
                                                    for item in base_items)}
    for item in base_items:
        add(item)
    return selected, {"assistant_evidence_added": assistant_added,
                      "selected": len(selected), "estimated_tokens": used_tokens}


def _label(text: str) -> bool:
    match = re.search(r'\{\s*"label"\s*:\s*["\']([^"\']*)["\']\s*\}', text)
    if match:
        return match.group(1).strip().lower() == "correct"
    normalized = text.strip().lower()
    if normalized in {"correct", "wrong"}:
        return normalized == "correct"
    raise ValueError("judge response did not contain a JSON label")


def main() -> int:
    base_url = _required("OMNIMEMEVAL_MEMORY_BASE_URL")
    api_key = _required("OMNIMEMEVAL_MEMORY_API_KEY")
    tenant_id = _required("OMNIMEMEVAL_TENANT_ID")
    org_id = _required("OMNIMEMEVAL_ORG_ID")
    answer_model = _required("OMNIMEMEVAL_ANSWER_MODEL")
    answer_key = _required("OMNIMEMEVAL_ANSWER_API_KEY")
    answer_url = _required("OMNIMEMEVAL_ANSWER_BASE_URL")
    eval_model = _required("OMNIMEMEVAL_EVAL_MODEL")
    eval_key = _required("OMNIMEMEVAL_EVAL_API_KEY")
    eval_url = _required("OMNIMEMEVAL_EVAL_BASE_URL")
    run_version = os.getenv("OMNIMEMEVAL_LME_VERSION", "hojo_lme_full_e2e")
    top_k = int(os.getenv("OMNIMEMEVAL_LME_TOP_K", "20"))
    candidate_limit = int(os.getenv("OMNIMEMEVAL_LME_CANDIDATE_LIMIT", "200"))
    min_score = float(os.getenv("OMNIMEMEVAL_LME_MIN_SCORE", "0.5"))
    max_tokens = int(os.getenv("OMNIMEMEVAL_LME_MAX_TOKENS", "1500"))
    timeout = float(os.getenv("OMNIMEMEVAL_LME_TIMEOUT_SECONDS", "300"))
    start_index = int(os.getenv("OMNIMEMEVAL_LME_START_INDEX", "0"))
    case_limit = int(os.getenv("OMNIMEMEVAL_LME_CASE_LIMIT", "500"))
    include_assistant = os.getenv("HOJO_INDEX_ASSISTANT_MESSAGES", "1").strip().lower() in {"1", "true", "yes", "on"}
    if not 1 <= top_k <= 50 or not top_k <= candidate_limit <= 500:
        raise ValueError("top-k must be 1..50 and candidate limit must be between top-k and 500")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("minimum retrieval score must be between 0 and 1")
    if not 0 <= start_index < 500 or not 1 <= case_limit <= 500 - start_index:
        raise ValueError("invalid LongMemEval case range")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset, data_hash = _load_dataset(DATA_PATH)
    case_indexes = selected_ids(
        "OMNIMEMEVAL_LME_CASE_INDEXES",
        {str(index) for index in range(start_index, start_index + case_limit)},
    )
    selected_indexes = (
        sorted(int(value) for value in case_indexes)
        if case_indexes is not None else list(range(start_index, start_index + case_limit))
    )
    config = {
        "dataset_sha256": data_hash,
        "version": run_version,
        "start_index": start_index,
        "case_limit": case_limit,
        "case_indexes": selected_indexes if case_indexes is not None else None,
        "ingest_mode": "direct_message_claims_v2",
        "index_assistant_messages": include_assistant,
        "answer_model": answer_model,
        "eval_model": eval_model,
        "top_k": top_k,
        "candidate_limit": candidate_limit,
        "min_score": min_score,
        "max_tokens": max_tokens,
        "assistant_recall_profile": "bounded_second_pass_v1",
    }
    state_path = OUTPUT_DIR / "run-state.json"
    judged_path = OUTPUT_DIR / "hojo_lme_judged.json"
    search_path = OUTPUT_DIR / "hojo_lme_search_results.json"
    answer_path = OUTPUT_DIR / "hojo_lme_responses.json"
    state = _load_state(state_path, config)
    judged = json.loads(judged_path.read_text(encoding="utf-8")) if judged_path.exists() else {}
    searches = json.loads(search_path.read_text(encoding="utf-8")) if search_path.exists() else {}
    answers = json.loads(answer_path.read_text(encoding="utf-8")) if answer_path.exists() else {}

    answer_client = OpenAI(api_key=answer_key, base_url=answer_url, timeout=timeout)
    judge_client = OpenAI(api_key=eval_key, base_url=eval_url, timeout=timeout)
    started = time.monotonic()
    with _http_client(base_url, api_key, tenant_id, org_id, timeout) as memory_client:
        health = memory_client.get("/health")
        health.raise_for_status()
        for index in selected_indexes:
            row = dataset[index]
            key = str(index)
            if state["cases"].get(key, {}).get("status") == "completed":
                continue
            user_id = f"lme_exper_user_{run_version}_{index}"
            namespace = {"type": "longmemeval", "id": f"{run_version}_{index}"}
            case_state = {"status": "running", "phase": "clear", "user_id": user_id, "started_at": datetime.utcnow().isoformat() + "Z"}
            state["cases"][key] = case_state
            _atomic_json(state_path, state)
            try:
                _post(memory_client, "/api/v2/memory/clear", {"user_id": user_id})
                session_count = 0
                for session_index, session in enumerate(row["haystack_sessions"]):
                    session_id = str(row["haystack_session_ids"][session_index])
                    occurred_at = _parse_date(row["haystack_dates"][session_index])
                    batches = _session_claim_batches(
                        session,
                        run_version=run_version,
                        case_index=index,
                        session_id=session_id,
                        session_index=session_index,
                        occurred_at=occurred_at,
                        include_assistant=include_assistant,
                    )
                    for batch in batches:
                        case_state["phase"] = "ingest"
                        _post(memory_client, "/api/v2/memory/ingest", {
                            "user_id": user_id,
                            "namespace": namespace,
                            "source_event": batch["source_event"],
                            "claims": batch["claims"],
                        })
                        session_count += 1

                search_started = time.monotonic()
                case_state["phase"] = "search"
                _atomic_json(state_path, state)
                retrieve_data = _post(memory_client, "/api/v2/memory/retrieve", {
                    "user_id": user_id,
                    "query": row["question"],
                    "required_dimensions": [],
                    "method": "SEMANTIC",
                    "usage_context": {
                        "module_id": "OmniMemEval",
                        "scenario": "longmemeval",
                        "scope": "self",
                        "namespace": namespace,
                        "caller_role": "user",
                    },
                    "top_k": top_k,
                    "candidate_limit": candidate_limit,
                    "max_tokens": max_tokens,
                    "min_score": min_score,
                })
                items = retrieve_data.get("items") or []
                assistant_recall = _assistant_recall_query(row["question"])
                assistant_coverage = None
                assistant_recall_error = None
                if assistant_recall and include_assistant:
                    try:
                        supplemental = _post(memory_client, "/api/v2/memory/retrieve", {
                            "user_id": user_id,
                            "query": assistant_recall,
                            "required_dimensions": [],
                            "method": "SEMANTIC",
                            "usage_context": {
                                "module_id": "OmniMemEval", "scenario": "longmemeval",
                                "scope": "self", "namespace": namespace, "caller_role": "user",
                            },
                            "top_k": top_k, "candidate_limit": candidate_limit,
                            "max_tokens": max_tokens, "min_score": min_score,
                        })
                        items, assistant_coverage = _merge_assistant_evidence(
                            items, supplemental.get("items") or [],
                            top_k=top_k, max_tokens=max_tokens,
                        )
                    except Exception as exc:
                        # A supplemental lookup must not invalidate the primary result.
                        assistant_recall_error = type(exc).__name__
                search_duration_ms = round((time.monotonic() - search_started) * 1000, 2)
                context = _claim_context(items)
                prompt = ANSWER_PROMPT.format(
                    context=context,
                    question_date=row.get("question_date", "unknown"),
                    question=row["question"],
                )
                answer_started = time.monotonic()
                case_state["phase"] = "answer"
                _atomic_json(state_path, state)
                answer_result = answer_client.chat.completions.create(
                    model=answer_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                answer = answer_result.choices[0].message.content or ""
                response_duration_ms = round((time.monotonic() - answer_started) * 1000, 2)
                judge_prompt = JUDGE_PROMPT.format(
                    question=row["question"], gold=row["answer"], answer=answer,
                )
                case_state["phase"] = "judge"
                _atomic_json(state_path, state)
                judge_result = judge_client.chat.completions.create(
                    model=eval_model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": judge_prompt},
                    ],
                    temperature=0,
                )
                judge_raw = judge_result.choices[0].message.content or ""
                correct = _label(judge_raw)
                record = {
                    "index": index,
                    "user_id": user_id,
                    "question_id": row.get("question_id"),
                    "question": row["question"],
                    "category": row["question_type"],
                    "question_date": row.get("question_date"),
                    "golden_answer": row["answer"],
                    "answer": answer,
                    "correct": correct,
                    "llm_judgments": {"judgment_1": correct},
                    "session_ingest_calls": session_count,
                    "retrieval_trace": retrieve_data.get("retrieval_trace", {}),
                    "assistant_evidence_coverage": assistant_coverage,
                    "assistant_recall_error": assistant_recall_error,
                    "search_duration_ms": search_duration_ms,
                    "response_duration_ms": response_duration_ms,
                    "total_duration_ms": response_duration_ms + search_duration_ms,
                    "status": "success",
                }
                answer_evidences = []
                answer_session_ids = set(row.get("answer_session_ids") or [])
                for session_id, session in zip(row.get("haystack_session_ids") or [], row["haystack_sessions"]):
                    if session_id not in answer_session_ids:
                        continue
                    for message in session:
                        if message.get("has_answer"):
                            answer_evidences.append(f"{message.get('role')} : {message.get('content')}")
                searches[user_id] = [{
                    "question": row["question"],
                    "category": row["question_type"],
                    "date": row.get("question_date"),
                    "golden_answer": row["answer"],
                    "answer_evidences": answer_evidences,
                    "search_context": context,
                    "search_duration_ms": search_duration_ms,
                    "retrieval_trace": record["retrieval_trace"],
                    "assistant_evidence_coverage": assistant_coverage,
                    "assistant_recall_error": assistant_recall_error,
                    "status": "success",
                }]
                answers[user_id] = {
                    "user_id": user_id,
                    "question": row["question"],
                    "category": row["question_type"],
                    "question_date": row.get("question_date"),
                    "golden_answer": row["answer"],
                    "answer": answer,
                    "response_duration_ms": response_duration_ms,
                    "search_duration_ms": search_duration_ms,
                    "answer_evidences": answer_evidences,
                    "status": "success",
                }
                judged[user_id] = {
                    **record,
                    "answer_evidences": answer_evidences,
                    "nlp_metrics": {},
                }
                case_state.update({"status": "completed", "correct": record["correct"], "finished_at": datetime.utcnow().isoformat() + "Z"})
                print(f"[LME] case {index + 1}/500 complete judge={record['correct']} sessions={session_count}", flush=True)
            except Exception as exc:
                case_state.update({"status": "failed", "failed_stage": case_state.get("phase"), "error_type": type(exc).__name__, "error": str(exc)[:1000], "finished_at": datetime.utcnow().isoformat() + "Z"})
                judged[user_id] = {"index": index, "user_id": user_id, "question": row.get("question"), "category": row.get("question_type"), "golden_answer": row.get("answer"), "status": "failed", "failed_stage": case_state.get("phase"), "error_type": type(exc).__name__, "error": str(exc)[:1000]}
                print(f"[LME] case {index + 1}/500 failed: {type(exc).__name__}: {str(exc)[:300]}", file=sys.stderr, flush=True)
            finally:
                try:
                    _post(memory_client, "/api/v2/memory/clear", {"user_id": user_id})
                except Exception as cleanup_exc:
                    case_state["cleanup_error"] = f"{type(cleanup_exc).__name__}: {str(cleanup_exc)[:300]}"
                _atomic_json(state_path, state)
                _atomic_json(judged_path, judged)
                _atomic_json(search_path, searches)
                _atomic_json(answer_path, answers)

    selected_user_ids = {f"lme_exper_user_{run_version}_{idx}" for idx in selected_indexes}
    judged = {uid: record for uid, record in judged.items() if uid in selected_user_ids}
    completed = [r for r in judged.values() if r.get("status") == "success"]
    correct = sum(1 for r in completed if r.get("correct"))
    failures = sum(1 for r in judged.values() if r.get("status") != "success")
    failed_by_stage = {
        stage: sum(1 for r in judged.values() if r.get("status") != "success" and r.get("failed_stage") == stage)
        for stage in ("clear", "ingest", "search", "answer", "judge")
    }
    categories: dict[str, list[bool]] = {}
    for record in completed:
        categories.setdefault(str(record.get("category") or "unknown"), []).append(bool(record.get("correct")))
    category_scores = {
        name: {"category_name": name, "llm_judge_score": sum(scores) / len(scores), "total": len(scores)}
        for name, scores in categories.items()
    }
    score = correct / len(completed) if completed else None
    report = {
        "benchmark": "LongMemEval-S",
        "version": run_version,
        "metric": "OmniMemEval LLM Judge accuracy on valid cases only",
        "score": score,
        "valid_cases": len(completed),
        "total_cases": len(selected_indexes),
        "scope": "badcase_subset" if case_indexes is not None else "range",
        "start_index": start_index,
        "correct": correct,
        "incorrect": len(completed) - correct,
        "failed_cases": failures,
        "failed_by_stage": failed_by_stage,
        "dataset_sha256": data_hash,
        "answer_model": answer_model,
        "eval_model": eval_model,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "metrics": {
            "llm_judge_score": score,
            "llm_judge_std": 0.0,
            "duration": {},
            "valid_cases": len(completed),
            "total_cases": len(selected_indexes),
        },
        "category_scores": category_scores,
        "pipeline_status": {
            "clear": {"failed_users": failed_by_stage["clear"], "skipped_records": 0},
            "ingest": {"failed_users": failed_by_stage["ingest"], "skipped_records": 0},
            "search": {"failed_users": failed_by_stage["search"], "skipped_records": 0},
            "answer": {"failed_users": failed_by_stage["answer"], "skipped_records": 0},
            "eval": {"failed_users": failed_by_stage["judge"], "skipped_records": 0},
        },
    }
    _atomic_json(OUTPUT_DIR / "exp_report.json", report)
    md = [
        "# OmniMemEval LongMemEval-S End-to-End Report", "",
        f"- Version: `{run_version}`",
        f"- LLM-as-a-Judge accuracy (valid cases only): **{score:.4f}**" if score is not None else "- LLM-as-a-Judge accuracy (valid cases only): N/A",
        f"- Valid cases: **{len(completed)}/{case_limit}**",
        f"- Correct / incorrect: **{correct} / {len(completed) - correct}**",
        f"- Failed cases: **{failures}**",
        f"- Elapsed: **{report['elapsed_seconds']:.1f}s**",
        "",
        "This is the end-to-end OmniMemEval QA + LLM Judge score. Failed cases are excluded from the accuracy denominator.",
    ]
    (OUTPUT_DIR / "exp_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    state["status"] = "completed" if len(completed) == case_limit else "failed"
    state["summary"] = report
    _atomic_json(state_path, state)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if len(completed) == case_limit else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[LME] fatal: {type(exc).__name__}: {str(exc)}", file=sys.stderr, flush=True)
        raise SystemExit(1)

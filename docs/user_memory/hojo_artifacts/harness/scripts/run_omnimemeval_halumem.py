#!/usr/bin/env python3
"""Run the full HaluMem-Medium QA benchmark from bundled data in one Pod."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from omnimemeval_badcase_ids import selected_ids


DATA_PATH = Path(__file__).resolve().parents[1] / "data/omnimemeval/HaluMem-Medium.jsonl"
DATA_SHA256 = "486fbc130a5c8781a2af27ffa508a1d7855245137aa449c193ac4d29c45634e7"
DATA_URL = "https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/resolve/main/HaluMem-Medium.jsonl"
OUTPUT_DIR = Path(os.getenv("OMNIMEMEVAL_HALUMEM_OUTPUT_DIR") or "/tmp/omnimemeval/results/halumem")

ANSWER_PROMPT = """\
You are an intelligent memory assistant tasked with answering questions based on retrieved conversation memories.

# CONTEXT:
You have access to memories from prior conversations with a user. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories.
2. Pay special attention to timestamps and the chronological order of events.
3. Answer ONLY based on information present in the provided memories.
4. If the memories contain conflicting information, use the most recent one unless the question specifically asks about an earlier time.
5. If the memories do not contain sufficient information to answer the question, say so clearly.
6. Do NOT fabricate or hallucinate any information not present in the memories.

# APPROACH (Think step by step):
1. Identify which memories are relevant to the question.
2. Check for any updates or changes over time in the relevant memories.
3. Formulate a precise, concise answer grounded solely in the evidence from memories.
4. Double-check that your answer directly addresses the question asked.

{context}

Question: {question}

Answer:
"""
JUDGE_SYSTEM = (
    "You are an expert grader that determines if answers to questions "
    "match a gold standard answer."
)
GROUNDED_INFERENCE_GUIDANCE = """\
# GROUNDED INFERENCE:
For a question about a role, condition, plan, or outcome, a direct description
of that fact is sufficient; do not demand an exact formal title unless the
question asks for one. Combine explicitly stated premises for a limited
inference, but never invent missing premises. For a dated question, use only
evidence from that date or earlier, and distinguish a later update from the
earlier state. If no supporting evidence exists, state that it is unknown.

"""
JUDGE_PROMPT = """\
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given:
    (1) a question about a user's personal information, preferences, or experiences,
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The question tests whether a memory system can accurately recall and reason about user information
from prior conversations. The gold answer is the factually correct answer based on the conversation history.

Grading guidelines:
- Be generous: as long as the generated answer conveys the same core meaning as the gold answer, mark it CORRECT.
- For factual questions (names, dates, numbers, locations), the generated answer must match the gold answer's key facts.
- For questions about updates or changes, the answer must reflect the LATEST state unless the question asks about a prior state.
- If the generated answer says "I don't know" or "no information available" but the gold answer exists, mark it WRONG.
- If the generated answer fabricates information not in the gold answer, mark it WRONG.

Question: {question}
Gold answer: {golden_answer}
Generated answer: {response}

Return exactly one JSON object, with no explanation or Markdown:
{{"label": "CORRECT"}} or {{"label": "WRONG"}}
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"required environment variable {name} is not set")
    return value


def _configured_workers() -> int:
    workers = int(os.getenv("OMNIMEMEVAL_HALUMEM_WORKERS", "6"))
    if not 1 <= workers <= 20:
        raise ValueError("OMNIMEMEVAL_HALUMEM_WORKERS must be between 1 and 20")
    return workers


def _run_users(user_indexes: list[int], workers: int, process_user: Any) -> None:
    with ThreadPoolExecutor(max_workers=min(workers, len(user_indexes) or 1)) as executor:
        futures = [executor.submit(process_user, user_idx) for user_idx in user_indexes]
        for future in as_completed(futures):
            future.result()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _load_dataset(path: Path) -> tuple[list[dict], str]:
    if not path.is_file():
        raise FileNotFoundError(f"bundled HaluMem-Medium dataset is missing: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(f"bundled HaluMem-Medium SHA-256 mismatch: {digest}")
    users = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(users) != 20:
        raise ValueError(f"expected 20 HaluMem-Medium users, found {len(users)}")
    count = sum(
        len(session.get("questions") or [])
        for user in users for session in user.get("sessions") or []
    )
    if count != 3467:
        raise ValueError(f"expected 3467 HaluMem-Medium questions, found {count}")
    return users, digest


def _questions(user: dict, version: str) -> list[dict]:
    user_uuid = str(user["uuid"])
    user_id = f"hm_exp_user_{version}_{user_uuid}"
    result = []
    for session_idx, session in enumerate(user.get("sessions") or []):
        for question_idx, question in enumerate(session.get("questions") or []):
            key = f"{user_id}__s{session_idx}__q{question_idx}"
            result.append({
                "key": key,
                "user_id": user_id,
                "user_uuid": user_uuid,
                "session_idx": session_idx,
                "question_idx": question_idx,
                "question": question,
            })
    return result


def _parse_time(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%b %d, %Y, %H:%M:%S").replace(
            tzinfo=timezone.utc
        ).isoformat()
    except ValueError:
        return None


def _answer_prompt(context: str, question: str, profile: str) -> str:
    prompt = ANSWER_PROMPT.format(context=context, question=question)
    if profile == "grounded_inference_v1":
        return prompt.replace("# APPROACH", GROUNDED_INFERENCE_GUIDANCE + "# APPROACH", 1)
    return prompt


def _select_question_coordinates(questions: list[dict]) -> tuple[list[dict], set[str] | None]:
    selected = selected_ids(
        "OMNIMEMEVAL_HALUMEM_QUESTION_COORDINATES",
        {meta["coordinate"] for meta in questions},
    )
    if selected is None:
        return questions, None
    return [meta for meta in questions if meta["coordinate"] in selected], selected


def _claim_batches(session: dict, *, version: str, user_idx: int, session_idx: int,
                   time_profile: str = "session") -> list[dict]:
    claims_with_times: list[tuple[dict, str | None]] = []
    dialogue = session.get("dialogue") or []
    session_time = _parse_time(session.get("end_time") or session.get("start_time"))
    for message_idx, message in enumerate(dialogue):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").strip().lower()
        content = str(message.get("content") or "").replace("<|endoftext|>", "").strip()
        if not content:
            continue
        message_time = (_parse_time(message.get("timestamp")) or session_time
                        if time_profile == "message_day_v1" else session_time)
        for chunk_idx, offset in enumerate(range(0, len(content), 15_900)):
            stamp = f"[message_time: {message_time}] " if time_profile == "message_day_v1" and message_time else ""
            text = f"{role}: {stamp}{content[offset:offset + 15_900]}"
            identity = f"{version}:{user_idx}:{session_idx}:{message_idx}:{chunk_idx}"
            claim = {
                "dimension": "conversation",
                "claim_type": "conversation_message",
                "claim_key": "msg_" + hashlib.sha256(identity.encode()).hexdigest(),
                "value": text,
                "text": text,
                "fact_status": "stated",
                "confidence": 1.0,
                "importance": 3,
            }
            claims_with_times.append((claim, message_time))
    session_id = f"hm_user_{user_idx}_session_{session_idx}"
    batches = []
    grouped: list[tuple[list[dict], str | None]] = []
    for claim, stamp in claims_with_times:
        day = stamp[:10] if stamp and time_profile == "message_day_v1" else None
        previous_day = (grouped[-1][1][:10] if grouped and grouped[-1][1] else None)
        if not grouped or len(grouped[-1][0]) >= 200 or (time_profile == "message_day_v1" and day != previous_day):
            grouped.append(([], stamp))
        grouped[-1][0].append(claim)
    for batch_idx, (chunk, first_stamp) in enumerate(grouped):
        batches.append({
            "claims": chunk,
            "source_event": {
                "event_id": f"{version}:{user_idx}:{session_idx}:{batch_idx}",
                "source_type": "dialogue",
                "source_id": session_id,
                "meeting_id": f"halumem_{user_idx}:{session_id}",
                "segment_id": str(batch_idx),
                "occurred_at": first_stamp,
                "raw_text": "\n".join(claim["text"] for claim in chunk)[:200000] or None,
            },
        })
    return batches


def _post(client: httpx.Client, route: str, payload: dict | None = None) -> dict:
    response = client.post(route, json=payload or {})
    response.raise_for_status()
    body = response.json()
    if body.get("code", 0) != 0:
        raise RuntimeError(f"{route} returned code={body.get('code')}: {body.get('message')}")
    return body.get("data") or {}


def _context(items: list[dict]) -> str:
    memories = []
    for item in items:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        dates = []
        for provenance in item.get("provenance") or []:
            stamp = provenance.get("occurred_at")
            if stamp and str(stamp) not in dates:
                dates.append(str(stamp))
        prefix = f"[occurred_at: {', '.join(dates)}] " if dates else ""
        memories.append(f"- {prefix}{text}")
    return "Conversation memories:\n\n" + "\n".join(memories)


def _label(text: str) -> bool:
    # Accept harmless explanatory fields or a fenced JSON object, but never
    # infer a grade from prose that may mention both possible labels.
    for match in re.finditer(r"\{[^{}]*\}", text, flags=re.S):
        try:
            obj = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        label = str(obj.get("label", "")).strip().lower() if isinstance(obj, dict) else ""
        if label in {"correct", "wrong"}:
            return label == "correct"
    normalized = text.strip().lower()
    if normalized in {"correct", "wrong"}:
        return normalized == "correct"
    raise ValueError("judge response did not contain a valid JSON CORRECT/WRONG label")


def _judge_label(client: OpenAI, model: str, question: str, gold: str, answer: str) -> bool:
    prompt = JUDGE_PROMPT.format(question=question, golden_answer=gold, response=answer)
    for attempt in range(2):
        messages = [{"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt}]
        if attempt:
            messages.append({"role": "user", "content":
                             'Your previous output could not be parsed. Return only {"label":"CORRECT"} '
                             'or {"label":"WRONG"}.'})
        graded = client.chat.completions.create(
            model=model, messages=messages, temperature=0,
        )
        try:
            return _label(graded.choices[0].message.content or "")
        except ValueError:
            if attempt:
                raise
    raise AssertionError("unreachable")


def _fail(meta: dict, stage: str, exc: BaseException) -> dict:
    q = meta["question"]
    return {
        "key": meta["key"], "user_id": meta["user_id"],
        "question": q.get("question"), "golden_answer": q.get("answer"),
        "category": q.get("question_type", "unknown"),
        "difficulty": q.get("difficulty", "unknown"),
        "session_idx": meta["session_idx"], "question_idx": meta["question_idx"],
        "status": "failed", "failed_stage": stage,
        "error_type": type(exc).__name__, "error": str(exc)[:800],
    }


def main() -> int:
    memory_base = _required("OMNIMEMEVAL_MEMORY_BASE_URL")
    memory_key = _required("OMNIMEMEVAL_MEMORY_API_KEY")
    tenant_id = _required("OMNIMEMEVAL_TENANT_ID")
    org_id = _required("OMNIMEMEVAL_ORG_ID")
    answer_model = _required("OMNIMEMEVAL_ANSWER_MODEL")
    answer_key = _required("OMNIMEMEVAL_ANSWER_API_KEY")
    answer_url = _required("OMNIMEMEVAL_ANSWER_BASE_URL")
    eval_model = _required("OMNIMEMEVAL_EVAL_MODEL")
    eval_key = _required("OMNIMEMEVAL_EVAL_API_KEY")
    eval_url = _required("OMNIMEMEVAL_EVAL_BASE_URL")

    version = os.getenv("OMNIMEMEVAL_HALUMEM_VERSION", "hojo_halumem_medium_full_e2e")
    start_index = int(os.getenv("OMNIMEMEVAL_HALUMEM_START_INDEX", "0"))
    question_limit = int(os.getenv("OMNIMEMEVAL_HALUMEM_QUESTION_LIMIT", "3467"))
    top_k = int(os.getenv("OMNIMEMEVAL_HALUMEM_TOP_K", "20"))
    candidate_limit = int(os.getenv("OMNIMEMEVAL_HALUMEM_CANDIDATE_LIMIT", "200"))
    min_score = float(os.getenv("OMNIMEMEVAL_HALUMEM_MIN_SCORE", "0.5"))
    max_tokens = int(os.getenv("OMNIMEMEVAL_HALUMEM_MAX_TOKENS", "1500"))
    timeout = float(os.getenv("OMNIMEMEVAL_HALUMEM_TIMEOUT_SECONDS", "300"))
    startup_wait = int(os.getenv("OMNIMEMEVAL_HALUMEM_STARTUP_WAIT_SECONDS", "900"))
    workers = _configured_workers()
    answer_profile = os.getenv("OMNIMEMEVAL_HALUMEM_ANSWER_PROFILE", "baseline").strip() or "baseline"
    time_profile = os.getenv("OMNIMEMEVAL_HALUMEM_TIME_PROFILE", "session").strip() or "session"
    if answer_profile not in {"baseline", "grounded_inference_v1"}:
        raise ValueError("invalid HaluMem answer profile")
    if time_profile not in {"session", "message_day_v1"}:
        raise ValueError("invalid HaluMem time profile")
    if not 1 <= top_k <= 50 or not top_k <= candidate_limit <= 500:
        raise ValueError("HaluMem top-k must be 1..50 and candidate limit between top-k and 500")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("HaluMem minimum score must be between 0 and 1")
    if start_index < 0 or question_limit < 1 or start_index + question_limit > 3467:
        raise ValueError("invalid HaluMem question range; expected start 0..3466 and limit 1..3467")
    if startup_wait < 0 or timeout <= 0:
        raise ValueError("invalid HaluMem timeout configuration")

    users, dataset_hash = _load_dataset(DATA_PATH)
    # Accept the historical pipeline value that already ends with the version.
    version_dir = OUTPUT_DIR if OUTPUT_DIR.name == version else OUTPUT_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)
    state_path = version_dir / "run-state.json"
    checkpoint_dir = version_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    all_questions = [
        {**meta, "coordinate": f"{user_idx}:{meta['session_idx']}:{meta['question_idx']}"}
        for user_idx, user in enumerate(users) for meta in _questions(user, version)
    ]
    selected = all_questions[start_index:start_index + question_limit]
    selected, badcase_coordinates = _select_question_coordinates(selected)
    selected_by_user: dict[int, list[dict]] = {}
    selected_keys = {meta["key"] for meta in selected}
    for user_idx, user in enumerate(users):
        metas = [meta for meta in _questions(user, version) if meta["key"] in selected_keys]
        if metas:
            selected_by_user[user_idx] = metas

    config = {
        "benchmark": "HaluMem-Medium", "version": version,
        "dataset_sha256": dataset_hash, "dataset_url": DATA_URL,
        "user_count": 20, "question_count": len(selected),
        "start_index": start_index, "question_limit": question_limit,
        "question_coordinates": sorted(badcase_coordinates) if badcase_coordinates is not None else None,
        "top_k": top_k, "candidate_limit": candidate_limit, "min_score": min_score,
        "max_tokens": max_tokens, "answer_model": answer_model, "eval_model": eval_model,
        "ingest_mode": "direct_message_claims_v2", "workers": workers,
        "answer_profile": answer_profile, "time_profile": time_profile,
    }
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config") != config:
            raise ValueError("existing HaluMem checkpoint config differs; use a new version")
    else:
        state = {"schema_version": "omnimemeval-halumem-e2e-v1", "config": config,
                 "status": "running", "users": {}}

    checkpoint_path = checkpoint_dir / "records.jsonl"
    records: dict[str, dict] = {}
    if checkpoint_path.exists():
        with checkpoint_path.open(encoding="utf-8") as source:
            for line in source:
                try:
                    item = json.loads(line)
                    if item.get("key") in selected_keys:
                        records[item["key"]] = item
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

    records_lock = threading.RLock()
    state_lock = threading.RLock()

    def save_record(item: dict) -> None:
        with records_lock:
            records[item["key"]] = item
            with checkpoint_path.open("a", encoding="utf-8") as target:
                target.write(json.dumps(item, ensure_ascii=False) + "\n")
                target.flush()
                os.fsync(target.fileno())

    def save_state() -> None:
        with state_lock:
            _atomic_json(state_path, state)

    def update_user_state(user_state: dict, **updates: Any) -> None:
        with state_lock:
            user_state.update(updates)
            _atomic_json(state_path, state)

    stage_failures = {name: 0 for name in ("clear", "ingest", "search", "answer", "judge")}
    started = time.monotonic()
    request_timeout = timeout
    answer_client = OpenAI(api_key=answer_key, base_url=answer_url, timeout=timeout)
    judge_client = OpenAI(api_key=eval_key, base_url=eval_url, timeout=timeout)
    memory_headers = {
        "Authorization": f"Bearer {memory_key}", "X-Tenant-Id": tenant_id,
        "X-Org-Id": org_id, "Content-Type": "application/json",
    }
    print(f"[HALUMEM] start version={version} users={len(selected_by_user)} "
          f"questions={len(selected)}/3467 workers={workers}", flush=True)

    with httpx.Client(base_url=memory_base.rstrip("/"), headers=memory_headers,
                      timeout=request_timeout) as memory:
        memory.get("/health").raise_for_status()
        def process_user(user_idx: int) -> None:
            user = users[user_idx]
            metas = selected_by_user.get(user_idx)
            if not metas:
                return
            user_id = metas[0]["user_id"]
            user_state = {"status": "running", "phase": "clear", "user_id": user_id,
                          "question_count": len(metas), "started_at": _now()}
            with state_lock:
                prior = state.setdefault("users", {}).get(str(user_idx), {})
                with records_lock:
                    already_complete = prior.get("status") == "completed" and all(
                        records.get(meta["key"], {}).get("judged", {}).get("status") == "success"
                        for meta in metas
                    )
                if already_complete:
                    return
                state["users"][str(user_idx)] = user_state
                state["status"] = "running"
                _atomic_json(state_path, state)
            try:
                _post(memory, "/api/v2/memory/clear", {"user_id": user_id})
                update_user_state(user_state, phase="ingest")
                ingest_calls = 0
                for session_idx, session in enumerate(user.get("sessions") or []):
                    for batch in _claim_batches(session, version=version,
                                                user_idx=user_idx, session_idx=session_idx,
                                                time_profile=time_profile):
                        _post(memory, "/api/v2/memory/ingest", {
                            "user_id": user_id,
                            "namespace": {"type": "halumem", "id": version},
                            **batch,
                        })
                        ingest_calls += 1
                update_user_state(user_state, ingest_calls=ingest_calls)
                for meta in metas:
                    q = meta["question"]
                    key = meta["key"]
                    with records_lock:
                        record = dict(records.get(key, {"key": key}))
                    if record.get("judged", {}).get("status") == "success":
                        continue
                    try:
                        if "search" not in record:
                            update_user_state(user_state, phase="search")
                            retrieve = _post(memory, "/api/v2/memory/retrieve", {
                                "user_id": user_id, "query": q["question"],
                                "required_dimensions": [], "method": "SEMANTIC",
                                "usage_context": {"module_id": "OmniMemEval", "scenario": "halumem",
                                    "scope": "self", "namespace": {"type": "halumem", "id": version},
                                    "caller_role": "user"},
                                "top_k": top_k, "candidate_limit": candidate_limit,
                                "max_tokens": max_tokens, "min_score": min_score,
                            })
                            context = _context(retrieve.get("items") or [])
                            record["search"] = {
                                "search_context": context,
                                "retrieval_trace": retrieve.get("retrieval_trace") or {},
                                "search_duration_ms": 0.0,
                            }
                        if "answer" not in record:
                            update_user_state(user_state, phase="answer")
                            prompt = _answer_prompt(
                                record["search"]["search_context"], q["question"], answer_profile)
                            response = answer_client.chat.completions.create(
                                model=answer_model, messages=[{"role": "user", "content": prompt}],
                                temperature=0,
                            )
                            record["answer"] = response.choices[0].message.content or ""
                        update_user_state(user_state, phase="judge")
                        correct = _judge_label(judge_client, eval_model, q["question"],
                                               q.get("answer", ""), record["answer"])
                        record["judged"] = {"status": "success", "correct": correct}
                        record.update({
                            "user_id": user_id, "question": q.get("question"),
                            "golden_answer": q.get("answer"),
                            "category": q.get("question_type", "unknown"),
                            "difficulty": q.get("difficulty", "unknown"),
                            "evidence": q.get("evidence", ""),
                            "session_idx": meta["session_idx"], "question_idx": meta["question_idx"],
                        })
                    except Exception as exc:
                        with state_lock:
                            stage = user_state.get("phase", "unknown")
                            stage_failures[stage] += 1
                        record.update({
                            "user_id": user_id, "question": q.get("question"),
                            "golden_answer": q.get("answer"),
                            "category": q.get("question_type", "unknown"),
                            "difficulty": q.get("difficulty", "unknown"),
                            "evidence": q.get("evidence", ""),
                            "session_idx": meta["session_idx"], "question_idx": meta["question_idx"],
                            "judged": {"status": "failed", "failed_stage": stage,
                                        "error_type": type(exc).__name__, "error": str(exc)[:800]},
                        })
                        print(f"[HALUMEM] failed {key} at {stage}: {type(exc).__name__}: {str(exc)[:180]}",
                              file=sys.stderr, flush=True)
                    save_record(record)
                    with records_lock:
                        processed_questions = sum(
                            1 for m in metas if records.get(m["key"], {}).get("judged", {}).get("status")
                            in {"success", "failed"}
                        )
                    update_user_state(user_state, processed_questions=processed_questions)
                with records_lock:
                    failed_count = sum(records.get(m["key"], {}).get("judged", {}).get("status") != "success"
                                       for m in metas)
                update_user_state(user_state, status="failed" if failed_count else "completed",
                                  failed_questions=failed_count, finished_at=_now())
                print(f"[HALUMEM] user {user_idx + 1}/20 {user_id}: "
                      f"{len(metas) - failed_count}/{len(metas)} valid", flush=True)
            except Exception as exc:
                with state_lock:
                    stage = user_state.get("phase", "ingest")
                    stage_failures[stage] += 1
                    user_state.update({"status": "failed", "failed_stage": stage,
                                       "error_type": type(exc).__name__, "error": str(exc)[:800],
                                       "finished_at": _now()})
                    _atomic_json(state_path, state)
                for meta in metas:
                    with records_lock:
                        prior_record = records.get(meta["key"], {})
                    if prior_record.get("judged", {}).get("status") == "success":
                        continue
                    record = dict(prior_record) if prior_record else {"key": meta["key"]}
                    record.update({
                        "user_id": user_id, "question": meta["question"].get("question"),
                        "golden_answer": meta["question"].get("answer"),
                        "category": meta["question"].get("question_type", "unknown"),
                        "difficulty": meta["question"].get("difficulty", "unknown"),
                        "evidence": meta["question"].get("evidence", ""),
                        "session_idx": meta["session_idx"], "question_idx": meta["question_idx"],
                        "judged": {"status": "failed", "failed_stage": stage,
                                    "error_type": type(exc).__name__, "error": str(exc)[:800]},
                    })
                    save_record(record)
                print(f"[HALUMEM] user {user_id} failed at {stage}: "
                      f"{type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr, flush=True)
            finally:
                try:
                    _post(memory, "/api/v2/memory/clear", {"user_id": user_id})
                except Exception as exc:
                    update_user_state(user_state,
                                      cleanup_error=f"{type(exc).__name__}: {str(exc)[:300]}")
                save_state()

        user_indexes = sorted(selected_by_user)
        _run_users(user_indexes, workers, process_user)

    ordered_records = [records[meta["key"]] for meta in selected if meta["key"] in records]
    searches: dict[str, list[dict]] = {}
    responses: dict[str, dict] = {}
    judged: dict[str, dict] = {}
    for record in ordered_records:
        user_id = record.get("user_id", "")
        judged_part = record.get("judged") or {}
        common = {k: record.get(k) for k in (
            "key", "user_id", "question", "golden_answer", "category", "difficulty",
            "evidence", "session_idx", "question_idx")}
        search = {**common, **(record.get("search") or {}),
                  "status": "success" if "search" in record else "failed"}
        if judged_part.get("failed_stage") == "search":
            search.update({"status": "failed", "error": judged_part})
        searches.setdefault(user_id, []).append(search)
        responses[record["key"]] = {
            **common, "answer": record.get("answer", ""),
            "search_context": (record.get("search") or {}).get("search_context", ""),
            "status": "success" if record.get("answer") is not None else "failed",
        }
        judged[record["key"]] = {
            **common, "answer": record.get("answer", ""),
            "status": judged_part.get("status", "failed"),
            "failed_stage": judged_part.get("failed_stage"),
            "error_type": judged_part.get("error_type"), "error": judged_part.get("error"),
            "llm_judgments": {"judgment_1": "CORRECT" if judged_part.get("correct") else "WRONG"}
            if judged_part.get("status") == "success" else {},
        }
    valid = [record for record in ordered_records if record.get("judged", {}).get("status") == "success"]
    failed = [record for record in ordered_records if record.get("judged", {}).get("status") != "success"]
    score = sum(bool(record["judged"].get("correct")) for record in valid) / len(valid) if valid else None
    categories: dict[str, list[dict]] = {}
    for record in valid:
        categories.setdefault(str(record.get("category") or "unknown"), []).append(record)
    category_scores = {
        name: {"qa_accuracy": sum(bool(row["judged"].get("correct")) for row in rows) / len(rows),
               "correct": sum(bool(row["judged"].get("correct")) for row in rows), "total": len(rows)}
        for name, rows in sorted(categories.items())
    }
    by_stage = {name: sum(r.get("judged", {}).get("failed_stage") == name for r in failed)
                for name in stage_failures}
    report = {
        "benchmark": "HaluMem-Medium", "version": version,
        "run_id": version, "created_at": _now(),
        "metric": "OmniMemEval QA Accuracy (LLM-as-Judge; successful questions only)",
        "workers": min(workers, len(selected_by_user) or 1),
        "score": score, "valid_questions": len(valid), "total_questions": len(selected),
        "dataset_questions": 3467, "failed_questions": len(failed), "failed_by_stage": by_stage,
        "valid_users": sum(user.get("status") == "completed" for user in state.get("users", {}).values()),
        "total_users": len(selected_by_user), "dataset_sha256": dataset_hash,
        "dataset_url": DATA_URL, "answer_model": answer_model, "eval_model": eval_model,
        "elapsed_seconds": round(time.monotonic() - started, 2), "updated_at": _now(),
        "category_scores": category_scores,
        "summary": {
            "primary_metric": "QA Accuracy", "primary_score": score,
            "total_cases": len(selected), "valid_cases": len(valid),
            "failed_cases": len(failed), "skipped_cases": 0,
            "judge_accuracy": score,
        },
        "categories": [
            {"name": name, "label": name, "primary_score": values["qa_accuracy"],
             "total_cases": values["total"], "valid_cases": values["total"]}
            for name, values in category_scores.items()
        ],
        "configuration": config,
    }
    _atomic_json(version_dir / "hojo_hm_search_results.json", searches)
    _atomic_json(version_dir / "hojo_hm_responses.json", responses)
    _atomic_json(version_dir / "hojo_hm_judged.json", judged)
    _atomic_json(version_dir / "exp_report.json", report)
    score_text = f"{score:.4f}" if score is not None else "N/A"
    (version_dir / "exp_report.md").write_text(
        "# OmniMemEval HaluMem-Medium End-to-End Report\n\n"
        f"- Version: `{version}`\n- QA Accuracy: **{score_text}**\n"
        f"- Workers: **{report['workers']}**\n"
        f"- Valid questions: **{len(valid)}/{len(selected)}**\n"
        f"- Failed questions: **{len(failed)}**\n"
        f"- Failures by stage: `{json.dumps(by_stage, sort_keys=True)}`\n"
        f"- Elapsed: **{report['elapsed_seconds']:.1f}s**\n",
        encoding="utf-8",
    )
    state["summary"] = report
    all_users_done = all(
        item.get("status") in {"completed", "failed"}
        for item in state.get("users", {}).values()
    ) and len(state.get("users", {})) == len(selected_by_user)
    state["status"] = "completed" if all_users_done and not failed else (
        "completed_with_failures" if all_users_done else "running")
    _atomic_json(state_path, state)
    print(f"[HALUMEM] {state['status']} score={score_text} "
          f"valid={len(valid)}/{len(selected)} failed={len(failed)}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[HALUMEM] fatal: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)

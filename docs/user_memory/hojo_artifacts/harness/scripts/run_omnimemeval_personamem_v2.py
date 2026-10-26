#!/usr/bin/env python3
"""Run PersonaMem v2's personalized multiple-choice benchmark from bundled data."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data/omnimemeval/personamem_v2"
ROOT = Path(os.getenv("OMNIMEMEVAL_PERSONAMEM_DATA_PATH") or DEFAULT_ROOT)
BENCHMARK_PATH = ROOT / "benchmark/text/benchmark.csv"
HISTORY_DIR = ROOT / "data/chat_history_32k"
DATASET_REVISION = "ed956dea41521fc4499acbc63f966e0fd3c053ba"
DATASET_SHA256 = "686f5fba06cebb5f60bed59386f8f6321cfe6f0cb763e1f319adebb2fdc20cbb"
DATASET_URL = "https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2"
OUTPUT_ROOT = Path(os.getenv("OMNIMEMEVAL_PERSONAMEM_OUTPUT_DIR") or "/tmp/omnimemeval/results/personamem_v2")

ANSWER_PROMPT = """You are answering a personalized multiple-choice question using only the user's conversation memories.
Use relevant stated preferences and personal facts. Do not infer a stereotype. If memories conflict, prefer the most recent explicit preference. Select exactly one option.

Retrieved memories:
{context}

Question: {question}

Options:
{options}

Return only the selected label, such as (a)."""

ANSWER_PROMPT_GUARDED = """You are answering a personalized multiple-choice question using only the user's conversation memories.
The options are possible answers, not evidence about the user. Assistant suggestions are not proof of a user preference.
If the user asked to forget or stop considering a personal detail, do not use that detail to justify an option, even if older memories mention it. Prefer the most recent explicit user preference or correction. Select exactly one option.

Retrieved memories:
{context}

Question: {question}

Options:
{options}

Return only the selected label, such as (a)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"required environment variable {name} is not set")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _load_dataset() -> tuple[list[dict[str, str]], dict[int, dict], str]:
    if not BENCHMARK_PATH.is_file():
        raise FileNotFoundError(f"bundled PersonaMem v2 benchmark is missing: {BENCHMARK_PATH}")
    required = {"persona_id", "chat_history_32k_link", "user_query", "correct_answer", "incorrect_answers", "pref_type"}
    with BENCHMARK_PATH.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing_columns = required.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"PersonaMem v2 benchmark misses columns: {sorted(missing_columns)}")
        rows = list(reader)
    if len(rows) != 5000:
        raise ValueError(f"expected 5000 PersonaMem v2 benchmark rows, found {len(rows)}")

    referenced = {row["chat_history_32k_link"].replace("\\", "/").rsplit("/", 1)[-1] for row in rows}
    histories: dict[int, dict] = {}
    digest = hashlib.sha256()
    digest.update(BENCHMARK_PATH.read_bytes())
    for filename in sorted(referenced):
        path = HISTORY_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"PersonaMem v2 history referenced by benchmark is missing: {path}")
        raw = path.read_bytes()
        data = json.loads(raw)
        if not isinstance(data.get("chat_history"), list):
            raise ValueError(f"invalid chat_history in {path}")
        match = re.search(r"_persona(\d+)\.json$", filename)
        if not match:
            raise ValueError(f"cannot determine persona id from history filename: {filename}")
        histories[int(match.group(1))] = data
        digest.update(filename.encode("utf-8") + b"\0" + raw)
    persona_ids = {int(row["persona_id"]) for row in rows}
    if persona_ids != set(histories):
        raise ValueError("PersonaMem v2 benchmark persona ids do not match bundled history files")
    actual_digest = digest.hexdigest()
    if actual_digest != DATASET_SHA256:
        raise ValueError(f"bundled PersonaMem v2 dataset SHA-256 mismatch: {actual_digest}")
    return rows, histories, actual_digest


def _query_text(value: str) -> str:
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
    return str(parsed.get("content", value)) if isinstance(parsed, dict) else value


def _choices(row: dict[str, str], row_index: int) -> tuple[str, str]:
    incorrect = json.loads(row["incorrect_answers"])
    if not isinstance(incorrect, list) or len(incorrect) != 3:
        raise ValueError(f"row {row_index} must contain exactly three incorrect answers")
    values = [("correct", row["correct_answer"])] + [("incorrect", str(x)) for x in incorrect]
    random.Random(row_index).shuffle(values)
    labels = "abcd"
    lines, gold = [], ""
    for label, (kind, answer) in zip(labels, values):
        lines.append(f"({label}) {answer}")
        if kind == "correct":
            gold = f"({label})"
    return "\n".join(lines), gold


def _is_correct(answer: str, gold: str) -> bool:
    text = answer.strip().lower()
    if "<final_answer>" in text:
        text = text.split("<final_answer>")[-1].split("</final_answer>")[0]
    labels = set(re.findall(r"\(([a-d])\)", text)) or set(re.findall(r"\b([a-d])\b", text))
    return labels == {gold.lower().strip("() ")}


def _claim_batches(history: dict, *, version: str, persona_id: int) -> list[dict]:
    claims = []
    for message_idx, message in enumerate(history.get("chat_history") or []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").strip().lower()
        # The system entry contains the hidden persona profile and must not be ingested.
        if role == "system":
            continue
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        for chunk_idx, offset in enumerate(range(0, len(content), 15_900)):
            text = f"{role}: {content[offset:offset + 15_900]}"
            identity = f"{version}:{persona_id}:{message_idx}:{chunk_idx}"
            claims.append({
                "dimension": "conversation", "claim_type": "conversation_message",
                "claim_key": "pm_" + hashlib.sha256(identity.encode()).hexdigest(),
                "value": text, "text": text, "fact_status": "stated",
                "confidence": 1.0, "importance": 3,
            })
    batches = []
    for batch_idx, offset in enumerate(range(0, len(claims), 200)):
        chunk = claims[offset:offset + 200]
        batches.append({
            "claims": chunk,
            "source_event": {
                "event_id": f"{version}:{persona_id}:{batch_idx}",
                "source_type": "dialogue", "source_id": f"personamem_persona_{persona_id}",
                "meeting_id": f"personamem_{version}:{persona_id}",
                "segment_id": str(batch_idx),
                "raw_text": "\n".join(c["text"] for c in chunk)[:200000] or None,
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
    lines = []
    for item in items:
        text = str(item.get("text") or "").strip()
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines) or "(No relevant memories retrieved.)"


def _option_values(options: str) -> list[str]:
    values = re.findall(r"^\([a-d]\)\s*(.+)$", options, flags=re.M)
    if len(values) != 4:
        raise ValueError("PersonaMem retrieval requires four labeled options")
    return values


def _option_queries(question: str, options: str) -> list[str]:
    """Probe every visible option symmetrically; never use the hidden gold label."""
    values = _option_values(options)
    return [f"{question[:300]}\nPossible answer: {value[:200]}" for value in values]


_REVOCATION_RE = re.compile(
    r"\bforget\b|\bno longer\b|\b(?:do not|don't) (?:use|consider|remember)\b|\bset aside\b",
    flags=re.I,
)
_REVOCATION_STOPWORDS = {
    "about", "after", "also", "answer", "because", "could", "details", "enjoy", "forget",
    "from", "given", "have", "ideas", "into", "like", "might", "more", "most", "other",
    "personal", "possible", "prefer", "question", "remember", "since", "some", "someone",
    "their", "there", "these", "those", "this", "that", "user", "what", "when", "where",
    "which", "while", "with", "would", "your",
}


def _distinctive_words(value: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z]{4,}", value.lower())
        if word not in _REVOCATION_STOPWORDS
    }


def _revocation_query(question: str, options: str) -> str:
    """Search for withdrawn personal facts without reading the hidden gold label."""
    option_details = " ".join(value[:180] for value in _option_values(options))
    return (
        "User requests to forget, stop considering, or correct prior personal preferences and roles. "
        f"Question: {question[:240]}\nPossible details: {option_details[:720]}"
    )


def _relevant_revocations(items: list[dict], question: str, options: str) -> list[dict]:
    """Keep explicit withdrawals related to the question or a visible option."""
    topic = _distinctive_words(question + " " + " ".join(
        value[:180] for value in _option_values(options)
    ))
    selected: list[dict] = []
    seen: set[str] = set()
    for item in items:
        value = str(item.get("text") or "").strip()
        lower = value.lower()
        if not _REVOCATION_RE.search(value):
            continue
        assistant_ack = lower.startswith("assistant:") and re.search(
                r"\b(?:i(?:'ll| will) forget|we can set aside|i won't factor)\b", lower
            )
        if not lower.startswith("user:") and not assistant_ack:
            continue
        if len(topic.intersection(_distinctive_words(value))) < (1 if assistant_ack else 2):
            continue
        key = str(item.get("claim_id") or value)
        if key in seen:
            continue
        selected.append(item)
        seen.add(key)
    return selected[:2]


def _merge_option_evidence(
    base_items: list[dict], option_items: list[list[dict]], *, max_items: int, max_chars: int,
    min_option_score: float,
) -> list[dict]:
    """Preserve baseline order and admit at most one strong user claim per option."""
    merged: list[dict] = []
    seen: set[str] = set()
    used_chars = 0

    def add(item: dict) -> bool:
        nonlocal used_chars
        if len(merged) >= max_items or used_chars >= max_chars:
            return False
        value = str(item.get("text") or "").strip()
        key = str(item.get("claim_id") or value)
        if not value or key in seen:
            return False
        value = value[:max_chars - used_chars]
        merged.append({**item, "text": value})
        seen.add(key)
        used_chars += len(value)
        return True

    for item in base_items:
        add(item)
    for group in option_items:
        for item in group:
            value = str(item.get("text") or "").strip()
            if not value.lower().startswith("user:"):
                continue
            try:
                score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            if score < min_option_score:
                continue
            if add(item):
                break
    return merged


def main() -> int:
    memory_base = _required("OMNIMEMEVAL_MEMORY_BASE_URL")
    memory_key = _required("OMNIMEMEVAL_MEMORY_API_KEY")
    tenant_id = _required("OMNIMEMEVAL_TENANT_ID")
    org_id = _required("OMNIMEMEVAL_ORG_ID")
    answer_model = _required("OMNIMEMEVAL_ANSWER_MODEL")
    answer_key = _required("OMNIMEMEVAL_ANSWER_API_KEY")
    answer_url = _required("OMNIMEMEVAL_ANSWER_BASE_URL")
    version = os.getenv("OMNIMEMEVAL_PERSONAMEM_VERSION", "hojo_personamem_v2_full_e2e").strip()
    start_index = _env_int("OMNIMEMEVAL_PERSONAMEM_START_INDEX", 0)
    question_limit = _env_int("OMNIMEMEVAL_PERSONAMEM_QUESTION_LIMIT", 5000)
    workers = _env_int("OMNIMEMEVAL_PERSONAMEM_WORKERS", 6)
    top_k = _env_int("OMNIMEMEVAL_PERSONAMEM_TOP_K", 20)
    candidate_limit = _env_int("OMNIMEMEVAL_PERSONAMEM_CANDIDATE_LIMIT", 200)
    min_score = _env_float("OMNIMEMEVAL_PERSONAMEM_MIN_SCORE", 0.5)
    max_tokens = _env_int("OMNIMEMEVAL_PERSONAMEM_MAX_TOKENS", 1500)
    timeout = _env_float("OMNIMEMEVAL_PERSONAMEM_TIMEOUT_SECONDS", 300.0)
    option_retrieval = os.getenv("OMNIMEMEVAL_PERSONAMEM_OPTION_RETRIEVAL", "false").lower() == "true"
    option_trigger_items = _env_int("OMNIMEMEVAL_PERSONAMEM_OPTION_TRIGGER_ITEMS", 3)
    option_top_k = _env_int("OMNIMEMEVAL_PERSONAMEM_OPTION_TOP_K", 4)
    context_max_chars = _env_int("OMNIMEMEVAL_PERSONAMEM_CONTEXT_MAX_CHARS", 12000)
    if not version or not 1 <= workers <= 20 or not 1 <= top_k <= 50 or not top_k <= candidate_limit <= 500:
        raise ValueError("invalid PersonaMem version/workers/top-k/candidate-limit configuration")
    if not 0 <= min_score <= 1 or timeout <= 0:
        raise ValueError("invalid PersonaMem minimum score or timeout")
    if not 1 <= option_trigger_items <= top_k or not 1 <= option_top_k <= top_k or context_max_chars < 1000:
        raise ValueError("invalid PersonaMem option retrieval configuration")

    rows, histories, dataset_hash = _load_dataset()
    if start_index < 0 or question_limit < 1 or start_index + question_limit > len(rows):
        raise ValueError("invalid PersonaMem question range; expected a range within 0..4999")
    selected = [(idx, row) for idx, row in enumerate(rows[start_index:start_index + question_limit], start_index)]
    grouped: dict[int, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for idx, row in selected:
        grouped[int(row["persona_id"])].append((idx, row))

    output_dir = OUTPUT_ROOT if OUTPUT_ROOT.name == version else OUTPUT_ROOT / version
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    state_path, checkpoint_path = output_dir / "run-state.json", checkpoint_dir / "records.jsonl"
    config = {
        "benchmark": "PersonaMem v2", "version": version, "dataset_revision": DATASET_REVISION,
        "dataset_sha256": dataset_hash, "dataset_url": DATASET_URL,
        "question_count": len(selected), "start_index": start_index,
        "question_limit": question_limit, "persona_count": len(grouped), "workers": workers,
        "top_k": top_k, "candidate_limit": candidate_limit, "min_score": min_score,
        "max_tokens": max_tokens, "answer_model": answer_model,
        "ingest_mode": "direct_message_claims_v1", "metric": "multiple_choice_accuracy",
        "option_retrieval": option_retrieval,
        "option_trigger_items": option_trigger_items if option_retrieval else None,
        "option_top_k": option_top_k if option_retrieval else None,
        "context_max_chars": context_max_chars if option_retrieval else None,
        "option_strategy": "guarded_revocation_v2" if option_retrieval else None,
    }
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config") != config:
            raise ValueError("existing PersonaMem checkpoint config differs; use a new version")
    else:
        state = {"schema_version": "omnimemeval-personamem-v2-e2e-v1", "config": config,
                 "status": "running", "personas": {}}

    records: dict[str, dict] = {}
    if checkpoint_path.exists():
        with checkpoint_path.open(encoding="utf-8") as source:
            for line in source:
                try:
                    item = json.loads(line)
                    if item.get("row_index") in {idx for idx, _ in selected}:
                        records[str(item["row_index"])] = item
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    state_lock = threading.RLock()
    records_lock = threading.RLock()

    def save_state() -> None:
        with state_lock:
            _atomic_json(state_path, state)

    def save_record(record: dict) -> None:
        with records_lock:
            records[str(record["row_index"])] = record
            with checkpoint_path.open("a", encoding="utf-8") as target:
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                target.flush()
                os.fsync(target.fileno())

    answer_client = OpenAI(api_key=answer_key, base_url=answer_url, timeout=timeout)
    headers = {"Authorization": f"Bearer {memory_key}", "X-Tenant-Id": tenant_id,
               "X-Org-Id": org_id, "Content-Type": "application/json"}
    memory = httpx.Client(base_url=memory_base.rstrip("/"), headers=headers, timeout=timeout)
    memory.get("/health").raise_for_status()
    started = time.monotonic()

    def process_persona(persona_id: int) -> None:
        questions = grouped[persona_id]
        user_id = f"pm_exp_user_{version}_{persona_id}"
        persona_state = {"status": "running", "phase": "clear", "user_id": user_id,
                         "question_count": len(questions), "started_at": _now()}
        with state_lock:
            prior = state.setdefault("personas", {}).get(str(persona_id), {})
            with records_lock:
                done = prior.get("status") == "completed" and all(
                    records.get(str(idx), {}).get("status") == "success" for idx, _ in questions
                )
            if done:
                return
            state["personas"][str(persona_id)] = persona_state
            state["status"] = "running"
            _atomic_json(state_path, state)
        try:
            _post(memory, "/api/v2/memory/clear", {"user_id": user_id})
            persona_state.update(phase="ingest")
            save_state()
            for batch in _claim_batches(histories[persona_id], version=version, persona_id=persona_id):
                _post(memory, "/api/v2/memory/ingest", {
                    "user_id": user_id,
                    "namespace": {"type": "personamem_v2", "id": version},
                    **batch,
                })
            for idx, row in questions:
                with records_lock:
                    if records.get(str(idx), {}).get("status") == "success":
                        continue
                record = {"row_index": idx, "persona_id": persona_id,
                          "user_id": user_id, "question": _query_text(row["user_query"]),
                          "category": row.get("pref_type") or "unknown",
                          "topic": row.get("topic_query") or "", "golden_answer": ""}
                try:
                    options, gold = _choices(row, idx)
                    record["golden_answer"] = gold
                    persona_state.update(phase="search", current_row=idx)
                    save_state()
                    request = {
                        "user_id": user_id, "query": record["question"],
                        "required_dimensions": [], "method": "SEMANTIC",
                        "usage_context": {"module_id": "OmniMemEval", "scenario": "personamem_v2",
                            "scope": "self", "namespace": {"type": "personamem_v2", "id": version},
                            "caller_role": "user"},
                        "top_k": top_k, "candidate_limit": candidate_limit,
                        "max_tokens": max_tokens, "min_score": min_score,
                    }
                    retrieved = _post(memory, "/api/v2/memory/retrieve", request)
                    base_items = retrieved.get("items") or []
                    selected_items = base_items
                    option_trace: dict[str, Any] = {"triggered": False}
                    context = _context(selected_items)
                    if option_retrieval:
                        option_trace = {"triggered": False, "base_items": len(base_items),
                                        "revocation_candidates": 0, "revocation_matches": 0,
                                        "supplemental_added": 0}
                        revocation_items: list[dict] = []
                        try:
                            revoked = _post(memory, "/api/v2/memory/retrieve", {
                                **request, "query": _revocation_query(record["question"], options),
                                "top_k": min(8, top_k), "min_score": min(min_score, 0.25),
                            })
                            revocation_items = revoked.get("items") or []
                            option_trace["revocation_candidates"] = len(revocation_items)
                        except Exception as exc:
                            # A supplementary probe must not fail a valid baseline answer.
                            option_trace["revocation_error"] = type(exc).__name__
                        revocations = _relevant_revocations(
                            [*base_items, *revocation_items], record["question"], options
                        )
                        option_trace["revocation_matches"] = len(revocations)
                        revoked_ids = {
                            str(item.get("claim_id") or item.get("text")) for item in revocations
                        }
                        remaining_base = [
                            item for item in base_items
                            if str(item.get("claim_id") or item.get("text")) not in revoked_ids
                        ]
                        option_items: list[list[dict]] = []
                        if len(base_items) < option_trigger_items:
                            option_trace["triggered"] = True
                            option_trace["counts"] = []
                            for option_query in _option_queries(record["question"], options):
                                try:
                                    extra = _post(memory, "/api/v2/memory/retrieve", {
                                        **request, "query": option_query, "top_k": option_top_k,
                                    })
                                    items = extra.get("items") or []
                                    option_items.append(items)
                                    option_trace["counts"].append(len(items))
                                except Exception as exc:
                                    option_items.append([])
                                    option_trace["counts"].append(0)
                                    option_trace.setdefault("errors", []).append(type(exc).__name__)
                        revoked_context = "\n".join(
                            f"- {str(item.get('text') or '').strip()[:600]}"
                            for item in revocations
                        )
                        heading = (
                            "Withdrawn or corrected personal details (do not use as preferences):\n"
                            f"{revoked_context}\n\nOther retrieved memories:\n"
                            if revoked_context else ""
                        )
                        selected_items = _merge_option_evidence(
                            remaining_base, option_items, max_items=top_k,
                            max_chars=max(0, context_max_chars - len(heading)),
                            min_option_score=min_score,
                        )
                        base_ids = {
                            str(item.get("claim_id") or item.get("text")) for item in remaining_base
                        }
                        option_trace["supplemental_added"] = sum(
                            str(item.get("claim_id") or item.get("text")) not in base_ids
                            for item in selected_items
                        )
                        option_trace["merged_items"] = len(selected_items)
                        context = heading + _context(selected_items)
                    persona_state.update(phase="answer")
                    save_state()
                    answer_prompt = ANSWER_PROMPT_GUARDED if option_retrieval else ANSWER_PROMPT
                    response = answer_client.chat.completions.create(
                        model=answer_model,
                        messages=[{"role": "user", "content": answer_prompt.format(
                            context=context, question=record["question"], options=options)}],
                        temperature=0,
                    )
                    answer = response.choices[0].message.content or ""
                    correct = _is_correct(answer, gold)
                    record.update({"status": "success", "answer": answer, "is_correct": correct,
                                   "options": options, "search_context": context,
                                   "retrieval_trace": retrieved.get("retrieval_trace") or {},
                                   "option_retrieval_trace": option_trace})
                except Exception as exc:
                    record.update({"status": "failed", "failed_stage": persona_state.get("phase"),
                                   "error_type": type(exc).__name__, "error": str(exc)[:800]})
                    print(f"[PERSONAMEM] failed row={idx} persona={persona_id} "
                          f"stage={record['failed_stage']} {type(exc).__name__}: {str(exc)[:200]}",
                          file=sys.stderr, flush=True)
                save_record(record)
                with records_lock:
                    processed = sum(records.get(str(i), {}).get("status") in {"success", "failed"}
                                    for i, _ in questions)
                persona_state.update(processed_questions=processed)
                save_state()
            with records_lock:
                failed = sum(records.get(str(i), {}).get("status") != "success" for i, _ in questions)
            persona_state.update(status="failed" if failed else "completed", failed_questions=failed,
                                 finished_at=_now())
            print(f"[PERSONAMEM] persona={persona_id} {len(questions)-failed}/{len(questions)} valid", flush=True)
        except Exception as exc:
            persona_state.update(status="failed", failed_stage=persona_state.get("phase"),
                                 error_type=type(exc).__name__, error=str(exc)[:800], finished_at=_now())
            print(f"[PERSONAMEM] persona={persona_id} failed stage={persona_state['failed_stage']} "
                  f"{type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr, flush=True)
            for idx, row in questions:
                with records_lock:
                    if records.get(str(idx), {}).get("status") == "success":
                        continue
                options, gold = _choices(row, idx)
                save_record({"row_index": idx, "persona_id": persona_id, "user_id": user_id,
                             "question": _query_text(row["user_query"]), "category": row.get("pref_type") or "unknown",
                             "golden_answer": gold, "options": options, "status": "failed",
                             "failed_stage": persona_state["failed_stage"],
                             "error_type": type(exc).__name__, "error": str(exc)[:800]})
        finally:
            try:
                _post(memory, "/api/v2/memory/clear", {"user_id": user_id})
            except Exception as exc:
                persona_state["cleanup_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            save_state()

    print(f"[PERSONAMEM] start version={version} questions={len(selected)}/5000 "
          f"personas={len(grouped)} workers={min(workers, len(grouped))}", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=min(workers, len(grouped) or 1)) as pool:
            futures = [pool.submit(process_persona, pid) for pid in sorted(grouped)]
            for future in as_completed(futures):
                future.result()
    finally:
        memory.close()

    ordered = [records[str(idx)] for idx, _ in selected if str(idx) in records]
    valid = [record for record in ordered if record.get("status") == "success"]
    failed = [record for record in ordered if record.get("status") != "success"]
    correct = sum(bool(record.get("is_correct")) for record in valid)
    score = correct / len(valid) if valid else None
    vector_observed = [
        record for record in valid
        if isinstance(record.get("retrieval_trace", {}).get("vector_hits"), int)
    ]
    vector_zero_queries = sum(
        record["retrieval_trace"]["vector_hits"] == 0 for record in vector_observed
    )
    option_triggered_queries = sum(
        bool(record.get("option_retrieval_trace", {}).get("triggered")) for record in valid
    )
    option_added_queries = sum(
        record.get("option_retrieval_trace", {}).get("supplemental_added", 0) > 0
        for record in valid
    )
    revocation_matched_queries = sum(
        record.get("option_retrieval_trace", {}).get("revocation_matches", 0) > 0
        for record in valid
    )
    categories: dict[str, list[dict]] = defaultdict(list)
    for record in valid:
        categories[record.get("category") or "unknown"].append(record)
    category_scores = {name: {"accuracy": sum(bool(x.get("is_correct")) for x in items) / len(items),
                              "correct": sum(bool(x.get("is_correct")) for x in items), "total": len(items)}
                       for name, items in sorted(categories.items())}
    def common(record: dict) -> dict:
        return {k: record.get(k) for k in (
            "row_index", "persona_id", "user_id", "question", "category", "topic", "golden_answer",
        )}

    searches = {str(r["row_index"]): {**common(r), "search_context": r.get("search_context", ""),
                "retrieval_trace": r.get("retrieval_trace", {}),
                "option_retrieval_trace": r.get("option_retrieval_trace", {}),
                "status": "success" if r.get("search_context") is not None else "failed"}
                for r in ordered}
    responses = {str(r["row_index"]): {**common(r), "answer": r.get("answer", ""),
                  "options": r.get("options", ""), "is_correct": r.get("is_correct"), "status": r.get("status")}
                 for r in ordered}
    judged = {str(r["row_index"]): {**common(r), "answer": r.get("answer", ""),
                "is_correct": r.get("is_correct"), "status": r.get("status"),
                "failed_stage": r.get("failed_stage"), "error_type": r.get("error_type"), "error": r.get("error")}
              for r in ordered}
    report = {"benchmark": "PersonaMem v2", "version": version, "run_id": version,
              "metric": "Multiple-choice accuracy on valid questions only", "score": score,
              "accuracy": score, "correct_questions": correct, "valid_questions": len(valid),
              "total_questions": len(selected), "dataset_questions": 5000,
              "failed_questions": len(failed), "dataset_sha256": dataset_hash,
              "vector_observed_queries": len(vector_observed),
              "vector_zero_queries": vector_zero_queries,
              "vector_zero_rate": vector_zero_queries / len(vector_observed) if vector_observed else None,
              "option_triggered_queries": option_triggered_queries,
              "option_added_queries": option_added_queries,
              "revocation_matched_queries": revocation_matched_queries,
              "retrieval_quality_warning": (
                  "More than half of baseline retrievals had zero vector hits; check Qdrant/index health before interpreting accuracy."
                  if len(vector_observed) >= 20 and vector_zero_queries > len(vector_observed) / 2
                  else None
              ),
              "dataset_revision": DATASET_REVISION, "dataset_url": DATASET_URL,
              "answer_model": answer_model, "workers": min(workers, len(grouped)),
              "elapsed_seconds": round(time.monotonic() - started, 2), "updated_at": _now(),
              "category_scores": category_scores, "configuration": config}
    _atomic_json(output_dir / "hojo_personamem_v2_search_results.json", searches)
    _atomic_json(output_dir / "hojo_personamem_v2_responses.json", responses)
    _atomic_json(output_dir / "hojo_personamem_v2_judged.json", judged)
    _atomic_json(output_dir / "exp_report.json", report)
    score_text = f"{score:.4f}" if score is not None else "N/A"
    (output_dir / "exp_report.md").write_text(
        "# OmniMemEval PersonaMem v2 End-to-End Report\n\n"
        f"- Version: `{version}`\n- Accuracy: **{score_text}**\n"
        f"- Valid questions: **{len(valid)}/{len(selected)}**\n"
        f"- Failed questions: **{len(failed)}**\n"
        f"- Workers: **{report['workers']}**\n",
        encoding="utf-8")
    state["summary"] = report
    state["status"] = "completed" if not failed else "completed_with_failures"
    _atomic_json(state_path, state)
    print(f"[PERSONAMEM] {state['status']} accuracy={score if score is not None else 'N/A'} "
          f"valid={len(valid)}/{len(selected)} failed={len(failed)}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[PERSONAMEM] fatal: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)

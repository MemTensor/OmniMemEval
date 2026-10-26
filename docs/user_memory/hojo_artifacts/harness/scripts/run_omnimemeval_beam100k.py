#!/usr/bin/env python3
"""Run the bundled, full 20-conversation BEAM 100K benchmark against MemoryOS."""

from __future__ import annotations

import ast
import gzip
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from omnimemeval_badcase_ids import selected_ids


DATA_PATH = Path(os.getenv(
    "OMNIMEMEVAL_BEAM100K_DATA_PATH",
    "/app/data/omnimemeval/beam_100k.json.gz",
))
OUTPUT_DIR = Path(os.getenv(
    "OMNIMEMEVAL_BEAM100K_OUTPUT_DIR",
    "/tmp/omnimemeval/results/beam100k",
))
# MemoryOS estimates tokens as ceil(characters / 2). Keep every claim below
# the default 1,500-token retrieval budget, including its provenance header.
CLAIM_CONTENT_CHARS = 2_200
CLAIM_MAX_CHARS = 2_400
INGEST_RAW_TEXT_MAX_CHARS = 200_000
BROAD_EVIDENCE_QUERY = re.compile(
    r"\b(?:order in which|chronological(?:ly)?|sequence|in order|summary|"
    r"summari[sz](?:e|ing)|progress(?:ed|ion)?|over time|throughout|"
    r"across (?:our|the) (?:conversations|discussions))\b",
    re.I,
)
ORDERING_QUERY = re.compile(r"\b(?:order in which|chronological(?:ly)?|sequence|in order)\b", re.I)
CLAIM_ORDER = re.compile(r"^\[session=(\d+) message=(\d+) source_message_id=\d+ part=(\d+)/\d+\]")
CLAIM_ROLE = re.compile(r"^\[session=\d+ message=\d+ source_message_id=[^ ]+ part=\d+/\d+\]\s+(user|assistant):", re.I)
ASSISTANT_EVIDENCE_QUERY = re.compile(
    r"\b(?:assistant|you (?:said|suggested|recommended|advised|told)|"
    r"(?:your|the) (?:advice|suggestions|recommendations|response|reply))\b", re.I,
)
FENCED_CODE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
EVIDENCE_STOPWORDS = frozenset({
    "about", "after", "across", "before", "different", "from", "have", "into",
    "list", "many", "only", "order", "over", "please", "question", "summarize",
    "their", "them", "these", "this", "throughout", "what", "when", "which",
    "with", "your", "user", "conversation", "conversations",
})
ANSWER_PROMPT = """
You are a knowledgeable and helpful AI assistant with access to memory context about a user.

# CONTEXT:
The following are relevant memories retrieved from prior conversations with the user.

{context}

# INSTRUCTIONS:
1. Carefully analyze the provided memory context.
2. Answer the question as thoroughly and accurately as possible based on the memories.
3. If the memories contain specific details (dates, names, numbers, preferences), include them in your answer.
4. If the memories are insufficient to fully answer the question, provide the best answer you can with available information and note any gaps.
5. Be detailed and comprehensive in your response.
{dimension_guidance}

# QUESTION:
{question}

# ANSWER:
"""
RUBRIC_PROMPT = """\
You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## QUESTION:
{question}

## RUBRIC CRITERION (what to check):
{rubric_item}

## RESPONSE TO EVALUATE:
{response}

## SCORING:
- **1.0** = Fully satisfied: The response clearly and completely addresses this rubric criterion.
- **0.5** = Partially satisfied: The response addresses this criterion but is incomplete, vague, or only partially correct.
- **0.0** = Not satisfied: The response does not address this criterion at all, or is incorrect.

## SEMANTIC TOLERANCE:
Judge by meaning, not exact wording. Accept paraphrases and synonyms that preserve intent. Numbers/currencies/dates may appear in equivalent forms (e.g. "$68,000", "68k"). Treat them as equal when numerically equivalent.

Evaluate the response against ONLY this specific rubric criterion.

Return your evaluation as JSON:
{{"score": <0.0 or 0.5 or 1.0>, "reason": "<brief explanation>"}}\
"""
ORDERING_PROMPT = """\
You are an expert evaluator. The question asked the user to list events in chronological order. The reference ordering of events is given below as a numbered list. Your task is to determine the order in which these same events appear in the response.

## QUESTION:
{question}

## REFERENCE ORDERING (correct chronological order):
{reference_ordering}

## RESPONSE TO EVALUATE:
{response}

## INSTRUCTIONS:
For each event in the reference list, determine its position (1-based) in the response. If an event is NOT mentioned in the response at all, assign position -1.

Return a JSON object mapping each event label to its detected position:
{{"positions": [{{"event": "<event label>", "position": <int or -1>}}, ...]}}\
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"required environment variable {name} is not set")
    return value


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _load_dataset(path: Path) -> tuple[list[dict], str]:
    raw = gzip.decompress(path.read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 20:
        raise ValueError(f"expected 20 BEAM 100K conversations, found {len(rows)}")
    return rows, digest


class _JoinedParts(io.RawIOBase):
    """Expose bounded Git blobs as one seek-free compressed byte stream."""

    def __init__(self, parts: list[Path]):
        self.parts = iter(parts)
        self.current = None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        while True:
            if self.current is None:
                path = next(self.parts, None)
                if path is None:
                    return 0
                self.current = path.open("rb")
            count = self.current.readinto(buffer)
            if count:
                return count
            self.current.close()
            self.current = None

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
            self.current = None
        super().close()


def _beam10m_parts(prefix: Path) -> list[Path]:
    parts = sorted(prefix.parent.glob(prefix.name + "[0-9][0-9]"))
    if not parts or [path.name[-2:] for path in parts] != [f"{i:02d}" for i in range(len(parts))]:
        raise FileNotFoundError(f"BEAM 10M data parts missing or noncontiguous: {prefix}NN")
    return parts


def _iter_beam10m(prefix: Path):
    with _JoinedParts(_beam10m_parts(prefix)) as source:
        with gzip.GzipFile(fileobj=io.BufferedReader(source), mode="rb") as stream:
            for line in stream:
                if line.strip():
                    yield line


def _scan_beam10m(prefix: Path) -> tuple[list[dict], str]:
    digest = hashlib.sha256()
    metadata = []
    for line in _iter_beam10m(prefix):
        digest.update(line)
        row = json.loads(line)
        metadata.append({"conversation_id": row["conversation_id"],
                         "probing_questions": row.get("probing_questions")})
    if len(metadata) != 10:
        raise ValueError(f"expected 10 BEAM 10M conversations, found {len(metadata)}")
    return metadata, digest.hexdigest()


def _questions(row: dict) -> list[dict]:
    raw = row.get("probing_questions") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = ast.literal_eval(raw)
    result = []
    for dimension, entries in raw.items():
        for index, item in enumerate(entries):
            if not isinstance(item, dict) or not item.get("question"):
                continue
            result.append({
                **item,
                "dimension": str(dimension),
                "question_idx": index,
                "key": f"{row['conversation_id']}:{dimension}:{index}",
            })
    return result


def _select_workload(rows: list[dict], start_index: int, conversation_limit: int,
                     dimensions: str, scale: str = "100K") -> tuple[list[dict], list[list[dict]], list[str]]:
    if not 0 <= start_index < len(rows) or not 1 <= conversation_limit <= len(rows) - start_index:
        raise ValueError(f"invalid BEAM {scale} conversation range")
    available = {question["dimension"] for row in rows for question in _questions(row)}
    selected_dimensions = ([name.strip() for name in dimensions.split(",") if name.strip()]
                           if dimensions.strip() else sorted(available))
    if not selected_dimensions or len(selected_dimensions) != len(set(selected_dimensions)):
        raise ValueError(f"BEAM {scale} dimensions must be nonempty and unique")
    unknown = set(selected_dimensions) - available
    if unknown:
        raise ValueError(f"unknown BEAM {scale} dimensions: {', '.join(sorted(unknown))}")
    selected_rows = rows[start_index:start_index + conversation_limit]
    questions = [[question for question in _questions(row)
                  if question["dimension"] in selected_dimensions]
                 for row in selected_rows]
    return selected_rows, questions, selected_dimensions


def _sessions(row: dict) -> list[list[dict]]:
    chat = row.get("chat") or []
    sessions = []
    for raw_session in chat:
        if isinstance(raw_session, list):
            messages = [item for item in raw_session if isinstance(item, dict) and item.get("content")]
            if messages:
                sessions.append(messages)
        elif isinstance(raw_session, dict):
            # BEAM 10M nests plans -> batches -> turns -> messages.
            for plan in sorted(raw_session, key=lambda key: int(key.split("-")[-1])):
                for batch in raw_session[plan] or []:
                    messages = [message for turn in batch.get("turns") or []
                                for message in (turn if isinstance(turn, list) else [turn])
                                if isinstance(message, dict) and message.get("content")]
                    if messages:
                        sessions.append(messages)
    return sessions


def _parse_time(value: str | None) -> str | None:
    if not value:
        return None
    for fmt in ("%B-%d-%Y", "%b-%d-%Y", "%Y-%m-%d", "%B %d, %Y"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    return None


def _split_message(content: str, max_chars: int = CLAIM_CONTENT_CHARS) -> list[str]:
    """Split at readable boundaries without dropping or reordering source text."""
    parts = []
    start = 0
    while start < len(content):
        end = min(start + max_chars, len(content))
        if end < len(content):
            minimum = start + max_chars // 2
            for separator in ("\n\n", "\n", " "):
                boundary = content.rfind(separator, minimum, end)
                if boundary >= minimum:
                    end = boundary + len(separator)
                    break
        parts.append(content[start:end])
        start = end
    return parts


def _claims_for_session(messages: list[dict], version: str, conv_index: int,
                        session_index: int) -> list[dict]:
    claims = []
    for msg_index, message in enumerate(messages):
        role = str(message.get("role") or "user").strip().lower()
        content = str(message.get("content") or "").replace("<|endoftext|>", "").strip()
        if not content:
            continue
        parts = _split_message(content)
        source_id = message.get("id")
        source_id = str(source_id) if isinstance(source_id, int) else "unknown"
        for part_index, part in enumerate(parts):
            # Session timestamps alone cannot order turns within a BEAM session.
            # Put the original message position in every independently retrieved claim.
            header = (f"[session={session_index + 1} message={msg_index + 1} "
                      f"source_message_id={source_id} part={part_index + 1}/{len(parts)}] ")
            text = f"{header}{role}: {part}"
            if len(text) > CLAIM_MAX_CHARS:
                raise ValueError("BEAM claim exceeds retrieval-safe character budget")
            identity = f"{version}:{conv_index}:{session_index}:{msg_index}:{part_index}"
            claims.append({
                "dimension": "conversation",
                "claim_type": "conversation_message",
                "claim_key": "msg_" + hashlib.sha256(identity.encode()).hexdigest(),
                "value": text,
                "text": text,
                "fact_status": "stated",
                "confidence": 1.0,
                "importance": 3,
            })
    return claims


def _claim_batches(claims: list[dict], *, max_claims: int = 200,
                   max_raw_chars: int = INGEST_RAW_TEXT_MAX_CHARS) -> list[list[dict]]:
    """Keep every ingest source raw_text complete and within schema limits."""
    if not 1 <= max_claims <= 200 or not 1 <= max_raw_chars <= INGEST_RAW_TEXT_MAX_CHARS:
        raise ValueError("invalid ingest batch limits")
    batches: list[list[dict]] = []
    batch: list[dict] = []
    chars = 0
    for claim in claims:
        claim_chars = len(claim["text"])
        if batch and (len(batch) == max_claims or chars + 1 + claim_chars > max_raw_chars):
            batches.append(batch)
            batch = []
            chars = 0
        batch.append(claim)
        chars += claim_chars + (1 if len(batch) > 1 else 0)
    if batch:
        batches.append(batch)
    return batches


def _needs_broad_evidence(query: str) -> bool:
    return bool(BROAD_EVIDENCE_QUERY.search(query))


def _answer_prompt(context: str, question: dict) -> str:
    dimension = question["dimension"]
    guidance = {
        "event_ordering": (
            "6. Before answering, identify distinct user-stated events in every represented "
            "session and put them in source message order, not retrieval relevance order. List only "
            "events the user actually raised. The assistant's explanations and suggestions "
            "are not events the user raised. Do not substitute other related topics just to "
            "fill the requested count; if a step lacks evidence, say so."
        ),
        "summarization": (
            "6. Before writing, identify the distinct user-stated problems, attempted fixes, "
            "decisions, people, dates, numbers, and outcomes in each represented session. "
            "Summarize those concrete developments in source order before adding general "
            "background. Do not replace a specific fact with a vague project timeline, and "
            "do not turn assistant suggestions or examples into things the user did."
        ),
        "multi_session_reasoning": (
            "6. Combine only user-stated facts from the relevant sessions. Compare earlier "
            "and later statements before drawing a conclusion; distinguish what changed "
            "from what remained true. Cite concrete names, amounts, dates, and constraints "
            "where the memories provide them. Do not treat assistant suggestions as user decisions."
        ),
        "abstention": (
            "6. Questions asking what the user specifically did, chose, received, or was "
            "taught require evidence of that fact. Assistant advice, examples, and possible "
            "plans do not establish that the user adopted them. If the specific fact is not "
            "recorded, say it is unknown; do not infer it from general knowledge."
        ),
    }.get(dimension, "")
    return ANSWER_PROMPT.format(
        context=context, question=question["question"], dimension_guidance=guidance,
    )


def _retrieval_limits(question: dict, candidate_limit: int, min_score: float) -> tuple[int, float]:
    if question["dimension"] in {"event_ordering", "summarization", "multi_session_reasoning"}:
        return max(candidate_limit, 400), min(min_score, 0.35)
    return candidate_limit, min_score


def _evidence_queries(question: dict) -> list[str]:
    query = question["question"]
    if question["dimension"] not in {"event_ordering", "summarization", "multi_session_reasoning"}:
        return [query]
    return [
        query,
        f"{query} Find the user's earliest relevant statements, specific problems, "
        "initial decisions, and first actions across conversations.",
        f"{query} Find the user's later changes, attempted fixes, concrete outcomes, "
        "people, dates, and numbers across conversations.",
    ]


def _source_session(item: dict) -> int | None:
    match = CLAIM_ORDER.match(str(item.get("text") or ""))
    return int(match.group(1)) if match else None


def _user_evidence_only(question: dict) -> bool:
    """Use the user's statements for questions about what the user did or raised."""
    return (question["dimension"] in {"event_ordering", "summarization", "multi_session_reasoning"}
            and not ASSISTANT_EVIDENCE_QUERY.search(question["question"]))


def _exclude_assistant_evidence(pools: list[list[dict]]) -> tuple[list[list[dict]], int]:
    filtered = []
    excluded = 0
    for pool in pools:
        kept = []
        for item in pool:
            role = CLAIM_ROLE.match(str(item.get("text") or ""))
            if role and role.group(1).lower() == "assistant":
                excluded += 1
            else:
                kept.append(item)
        filtered.append(kept)
    return filtered, excluded


def _compact_code_evidence(item: dict, query: str) -> dict:
    """Bound fenced source code without discarding its surrounding user prose.

    This is question-text-only compression. It never inspects the gold answer,
    rubric, or judge feedback. The original claim remains unchanged in MemoryOS.
    """
    source = str(item.get("text") or "")
    terms = {
        term for term in re.findall(r"[a-z][a-z0-9_]{3,}", query.lower())
        if term not in EVIDENCE_STOPWORDS
    }

    def replace(match: re.Match[str]) -> str:
        block = match.group(1)
        if len(block) <= 500:
            return match.group(0)
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        matched = [line for line in lines if any(term in line.lower() for term in terms)]
        chosen = list(dict.fromkeys(lines[:2] + matched[:10]))
        excerpt = "\n".join(chosen)[:700]
        return f"[source code excerpt; {len(block)} original characters]\n{excerpt}\n"

    compacted = FENCED_CODE.sub(replace, source)
    return {**item, "text": compacted} if compacted != source else item


def _select_question_keys(prefix: str, questions_by_conv: list[list[dict]]) -> tuple[list[list[dict]], set[str] | None]:
    selected = selected_ids(
        f"{prefix}_QUESTION_KEYS",
        {question["key"] for questions in questions_by_conv for question in questions},
    )
    if selected is None:
        return questions_by_conv, None
    return [
        [question for question in questions if question["key"] in selected]
        for questions in questions_by_conv
    ], selected


def _merge_evidence(passes: list[list[dict]], max_tokens: int, top_k: int) -> tuple[list[dict], dict]:
    """Keep complementary retrieved claims under one answer-context budget."""
    pools: list[list[dict]] = []
    seen: set[str] = set()
    for items in passes:
        pool = []
        for item in items:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            identity = str(item.get("claim_id") or text)
            if identity in seen:
                continue
            seen.add(identity)
            pool.append(item)
        pools.append(pool)

    # Interleave query variants so a broad first search cannot monopolize the
    # budget. First cover distinct source sessions, then fill remaining space.
    interleaved = [item for index in range(max(map(len, pools), default=0))
                   for pool in pools for item in pool[index:index + 1]]
    selected: list[dict] = []
    selected_ids: set[int] = set()
    covered_sessions: set[int] = set()
    tokens = 0

    def add(item: dict) -> bool:
        nonlocal tokens
        cost = (len(str(item["text"])) + 1) // 2
        if tokens + cost > max_tokens or len(selected) >= top_k:
            return False
        selected.append(item)
        selected_ids.add(id(item))
        tokens += cost
        session = _source_session(item)
        if session is not None:
            covered_sessions.add(session)
        return True

    for item in interleaved:
        session = _source_session(item)
        if session is not None and session not in covered_sessions:
            add(item)
    for item in interleaved:
        if id(item) not in selected_ids:
            add(item)
    return selected, {
        "unique_candidates": len(interleaved), "selected": len(selected),
        "source_sessions_covered": len(covered_sessions), "estimated_tokens": tokens,
    }


def _post(client: httpx.Client, route: str, payload: dict | None = None) -> dict:
    # Ingest has deterministic claim keys and event IDs, so repeating the same
    # request after a lost response confirms the existing claims rather than
    # creating new ones. Do not retry clear/retrieve or HTTP 4xx responses.
    attempts = 3 if route == "/api/v2/memory/ingest" else 1
    for attempt in range(attempts):
        try:
            response = client.post(route, json=payload or {})
            break
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError):
            if attempt + 1 == attempts:
                raise
            time.sleep(min(2 ** attempt, 2))
    response.raise_for_status()
    body = response.json()
    if body.get("code", 0) != 0:
        raise RuntimeError(f"{route} returned code={body.get('code')}: {body.get('message')}")
    return body.get("data") or {}


def _retrieve_evidence(client: httpx.Client, request: dict, question: dict,
                       max_tokens: int, top_k: int, context_profile: str = "raw",
                       evidence_profile: str = "session_coverage_v1") -> tuple[list[dict], list[dict], list[dict], dict | None]:
    queries = _evidence_queries(question)
    user_grounded = evidence_profile == "user_grounded_v1" and _user_evidence_only(question)
    # Retrieve a larger candidate pool before excluding assistant turns. The
    # answer context still obeys the original top_k and token budget below.
    pass_request = ({**request, "top_k": min(50, max(top_k * 2, 40)),
                     "max_tokens": min(16000, max(max_tokens * 2, 12000))}
                    if user_grounded else request)
    retrievals = []
    supplemental_errors = []
    for pass_index, query in enumerate(queries):
        try:
            retrievals.append(_post(client, "/api/v2/memory/retrieve", {**pass_request, "query": query}))
        except Exception as exc:
            if pass_index == 0:
                raise
            supplemental_errors.append({"pass": pass_index, "error_type": type(exc).__name__})
    if len(queries) == 1:
        return retrievals[0].get("items") or [], retrievals, supplemental_errors, None
    pools = [result.get("items") or [] for result in retrievals]
    excluded_assistant = 0
    if user_grounded:
        pools, excluded_assistant = _exclude_assistant_evidence(pools)
    chars_saved = 0
    if context_profile == "code_compact_v1":
        compacted_pools = [[_compact_code_evidence(item, question["question"]) for item in pool]
                           for pool in pools]
        chars_saved = sum(
            len(str(original.get("text") or "")) - len(str(compacted.get("text") or ""))
            for originals, compacteds in zip(pools, compacted_pools)
            for original, compacted in zip(originals, compacteds)
        )
        pools = compacted_pools
    evidence, coverage = _merge_evidence(
        pools, max_tokens, top_k,
    )
    coverage["context_chars_saved"] = chars_saved
    coverage["assistant_candidates_excluded"] = excluded_assistant
    coverage["user_evidence_only"] = user_grounded
    coverage["per_pass_top_k"] = pass_request["top_k"]
    coverage["per_pass_max_tokens"] = pass_request.get("max_tokens", max_tokens)
    return evidence, retrievals, supplemental_errors, coverage


def _context(items: list[dict], query: str = "", *, chronological: bool = False) -> str:
    if chronological or ORDERING_QUERY.search(query):
        def source_order(item: dict) -> tuple[int, int, int]:
            match = CLAIM_ORDER.match(str(item.get("text") or ""))
            return tuple(map(int, match.groups())) if match else (10**9, 10**9, 10**9)

        items = sorted(items, key=source_order)
    memories = []
    for item in items:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        stamps = []
        for provenance in item.get("provenance") or []:
            stamp = provenance.get("occurred_at")
            if stamp and stamp not in stamps:
                stamps.append(str(stamp))
        prefix = f"[occurred_at: {', '.join(stamps)}] " if stamps else ""
        memories.append(f"- {prefix}{text}")
    return "Conversation memories:\n\n" + "\n".join(memories)


def _extract_score(raw: str) -> tuple[float, str]:
    match = re.search(r'\{[^{}]*"score"\s*:\s*([0-9.]+)[^{}]*\}', raw)
    if not match:
        raise ValueError("judge response has no JSON score")
    item = json.loads(match.group(0))
    score = float(item["score"])
    if score >= 0.75:
        score = 1.0
    elif score >= 0.25:
        score = 0.5
    else:
        score = 0.0
    return score, str(item.get("reason") or "")


def _kendall_tau_b(ref: list[int], pred: list[int]) -> float:
    concordant = discordant = ties_ref = ties_pred = 0
    for i in range(len(ref)):
        for j in range(i + 1, len(ref)):
            rd, pd = ref[i] - ref[j], pred[i] - pred[j]
            if rd == 0:
                ties_ref += 1
                ties_pred += pd == 0
            elif pd == 0:
                ties_pred += 1
            elif (rd > 0) == (pd > 0):
                concordant += 1
            else:
                discordant += 1
    pairs = len(ref) * (len(ref) - 1) / 2
    denom = ((pairs - ties_ref) * (pairs - ties_pred)) ** 0.5
    return (concordant - discordant) / denom if denom else 0.0


def _judge(answer: str, question: dict, client: OpenAI, model: str) -> tuple[float, dict]:
    dimension = question["dimension"]
    rubric = question.get("rubric") or []
    if isinstance(rubric, str):
        rubric = [rubric]
    if not rubric:
        rubric = [f"The response should contain: {question.get('ideal_response', '')}"]
    if dimension == "event_ordering":
        prompt = ORDERING_PROMPT.format(
            question=question["question"],
            reference_ordering="\n".join(f"{i + 1}. {item}" for i, item in enumerate(rubric)),
            response=answer,
        )
        raw = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a precise and fair evaluation judge."},
                      {"role": "user", "content": prompt}],
            temperature=0,
        ).choices[0].message.content or ""
        match = re.search(r'\{\s*"positions"\s*:\s*\[.*?\]\s*\}', raw, re.DOTALL)
        if not match:
            raise ValueError("event-ordering judge response has no positions")
        positions = [int(item.get("position", -1)) for item in json.loads(match.group(0))["positions"]]
        positions = (positions + [-1] * len(rubric))[:len(rubric)]
        found = [(idx, position) for idx, position in enumerate(positions) if position > 0]
        coverage = len(found) / len(rubric)
        if len(found) < 2:
            score = coverage * 0.5
        else:
            tau = _kendall_tau_b([idx for idx, _ in found], [pos for _, pos in found])
            score = max(0.0, (tau + 1.0) / 2.0) * coverage
        return score, {"scoring_method": "kendall_tau_b", "positions": positions, "raw": raw}

    scores = []
    details = []
    for item in rubric:
        prompt = RUBRIC_PROMPT.format(
            question=question["question"], rubric_item=item, response=answer,
        )
        raw = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a precise and fair evaluation judge."},
                      {"role": "user", "content": prompt}],
            temperature=0,
        ).choices[0].message.content or ""
        score, reason = _extract_score(raw)
        scores.append(score)
        details.append({"rubric_item": item, "score": score, "reason": reason})
    return sum(scores) / len(scores), {"scoring_method": "per_rubric_item", "rubric_item_scores": details}


def _fail_record(question: dict, user_id: str, stage: str, exc: Exception) -> dict:
    return {
        "key": question["key"], "conv_id": question.get("conv_id"),
        "question": question["question"], "dimension": question["dimension"],
        "status": "failed", "failed_stage": stage,
        "error_type": type(exc).__name__, "error": str(exc)[:800], "user_id": user_id,
    }


def main(scale: str = "100K") -> int:
    if scale not in {"100K", "10M"}:
        raise ValueError(f"unsupported BEAM scale: {scale}")
    prefix = f"OMNIMEMEVAL_BEAM{scale}"
    benchmark = f"BEAM {scale}"
    total_conversations = 20 if scale == "100K" else 10
    total_questions = 400 if scale == "100K" else 200
    data_path = (DATA_PATH if scale == "100K" else Path(os.getenv(
        "OMNIMEMEVAL_BEAM10M_DATA_PREFIX", "/app/data/omnimemeval/beam_10m.json.gz.part-")))
    output_dir = (OUTPUT_DIR if scale == "100K" else Path(os.getenv(
        "OMNIMEMEVAL_BEAM10M_OUTPUT_DIR", "/tmp/omnimemeval/results/beam10m")))
    base_url = _required("OMNIMEMEVAL_MEMORY_BASE_URL")
    answer_key = _required("OMNIMEMEVAL_ANSWER_API_KEY")
    answer_url = _required("OMNIMEMEVAL_ANSWER_BASE_URL")
    eval_key = _required("OMNIMEMEVAL_EVAL_API_KEY")
    eval_url = _required("OMNIMEMEVAL_EVAL_BASE_URL")
    answer_model = _required("OMNIMEMEVAL_ANSWER_MODEL")
    eval_model = _required("OMNIMEMEVAL_EVAL_MODEL")
    version = os.getenv(f"{prefix}_VERSION", "").strip() or f"hojo_beam{scale.lower()}_full_e2e"
    top_k = _env_int(f"{prefix}_TOP_K", 20)
    candidate_limit = _env_int(f"{prefix}_CANDIDATE_LIMIT", 200)
    min_score = _env_float(f"{prefix}_MIN_SCORE", 0.5)
    max_tokens = _env_int(f"{prefix}_MAX_TOKENS", 1500)
    broad_max_tokens = _env_int(f"{prefix}_BROAD_MAX_TOKENS", 6000)
    start_index = _env_int(f"{prefix}_START_INDEX", 0)
    conversation_limit = _env_int(f"{prefix}_CONVERSATION_LIMIT", total_conversations)
    workers = _env_int(f"{prefix}_WORKERS", 6 if scale == "100K" else 1)
    dimensions = os.getenv(f"{prefix}_DIMENSIONS", "")
    context_profile = os.getenv(f"{prefix}_CONTEXT_PROFILE", "raw").strip() or "raw"
    evidence_profile = os.getenv(f"{prefix}_EVIDENCE_PROFILE", "session_coverage_v1").strip() or "session_coverage_v1"
    if context_profile not in {"raw", "code_compact_v1"}:
        raise ValueError(f"{prefix}_CONTEXT_PROFILE must be raw or code_compact_v1")
    if evidence_profile not in {"session_coverage_v1", "user_grounded_v1"}:
        raise ValueError(f"{prefix}_EVIDENCE_PROFILE must be session_coverage_v1 or user_grounded_v1")
    max_workers = 10 if scale == "100K" else 4
    if not 1 <= workers <= max_workers:
        raise ValueError(f"{prefix}_WORKERS must be between 1 and {max_workers}")
    if not 1 <= top_k <= 50 or not top_k <= candidate_limit <= 500:
        raise ValueError(f"invalid {benchmark} retrieval limits")
    if not 100 <= max_tokens <= 16000 or not 100 <= broad_max_tokens <= 16000:
        raise ValueError(f"invalid {benchmark} retrieval token budget")
    rows, digest = (_load_dataset(data_path) if scale == "100K" else _scan_beam10m(data_path))
    if sum(len(_questions(row)) for row in rows) != total_questions:
        raise ValueError(f"bundled {benchmark} dataset does not contain {total_questions} questions")
    selected_rows, questions_by_conv, selected_dimensions = _select_workload(
        rows, start_index, conversation_limit, dimensions, scale,
    )
    questions_by_conv, question_keys = _select_question_keys(prefix, questions_by_conv)
    question_count = sum(map(len, questions_by_conv))
    selected_conversation_count = sum(bool(questions) for questions in questions_by_conv)
    full_scope = (question_keys is None and start_index == 0
                  and conversation_limit == total_conversations and len(selected_dimensions) == 10)

    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "run-state.json"
    report_path = output_dir / "exp_report.json"
    judged_path = output_dir / "hojo_beam_judged.json"
    searches_path = output_dir / "hojo_beam_search_results.json"
    responses_path = output_dir / "hojo_beam_responses.json"
    config = {
        "benchmark": benchmark, "version": version, "dataset_sha256": digest,
        "conversation_count": conversation_limit, "question_count": question_count,
        "selected_conversation_count": selected_conversation_count,
        "question_keys": sorted(question_keys) if question_keys is not None else None,
        "start_index": start_index, "dimensions": selected_dimensions,
        "scope": "full" if full_scope else "subset", "top_k": top_k,
        "candidate_limit": candidate_limit, "min_score": min_score,
        "ordering_summary_candidate_limit": max(candidate_limit, 400),
        "ordering_summary_min_score": min(min_score, 0.35),
        "max_tokens": max_tokens, "broad_max_tokens": broad_max_tokens,
        "answer_model": answer_model, "eval_model": eval_model,
        "ingest_mode": "direct_message_claims_v3", "prompt_profile": "omnimemeval_beam_evidence_v3",
        "retrieval_profile": "multi_pass_session_coverage_v1",
        "context_profile": context_profile,
        "evidence_profile": evidence_profile,
        "workers": workers,
    }
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config") != config:
            raise ValueError("existing BEAM checkpoint config differs; use a new version/output directory")
    else:
        state = {"schema_version": f"omnimemeval-beam{scale.lower()}-e2e-v1", "config": config,
                 "status": "running", "cases": {}}
    judged = json.loads(judged_path.read_text(encoding="utf-8")) if judged_path.exists() else {}
    searches = json.loads(searches_path.read_text(encoding="utf-8")) if searches_path.exists() else {}
    answers = json.loads(responses_path.read_text(encoding="utf-8")) if responses_path.exists() else {}
    request_timeout = _env_float(f"{prefix}_TIMEOUT_SECONDS", 300.0)
    answer_client = OpenAI(api_key=answer_key, base_url=answer_url, timeout=request_timeout)
    judge_client = OpenAI(api_key=eval_key, base_url=eval_url, timeout=request_timeout)
    started = time.monotonic()
    stage_failures = {name: 0 for name in ("clear", "ingest", "search", "answer", "judge")}
    cleanup_failures = 0
    state_lock = threading.RLock()

    def update_case(case_index: int, **updates: Any) -> None:
        with state_lock:
            state.setdefault("cases", {}).setdefault(str(case_index), {}).update(updates)
            _atomic_json(state_path, state)

    def save_record(target: dict, path: Path, key: str, record: dict) -> None:
        with state_lock:
            target[key] = record
            _atomic_json(path, target)

    def is_judged_success(key: str) -> bool:
        with state_lock:
            return judged.get(key, {}).get("status") == "success"

    print(f"[BEAM{scale}] start version={version} conversations={conversation_limit} "
          f"questions={question_count} workers={min(workers, selected_conversation_count)} "
          f"scope={config['scope']}", flush=True)

    with httpx.Client(
        base_url=base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {_required('OMNIMEMEVAL_MEMORY_API_KEY')}",
                 "X-Tenant-Id": _required("OMNIMEMEVAL_TENANT_ID"),
                 "X-Org-Id": _required("OMNIMEMEVAL_ORG_ID"), "Content-Type": "application/json"},
        timeout=request_timeout,
    ) as memory:
        memory.get("/health").raise_for_status()

        def process_conversation(conv_index: int, row: dict, questions: list[dict]) -> None:
            nonlocal cleanup_failures
            if not questions:
                return
            conv_id = str(row["conversation_id"])
            user_id = f"beam_exp_user_hojo_{version}_{conv_id}"
            namespace = {"type": f"beam{scale.lower()}", "id": f"{version}_{conv_id}"}
            for q in questions:
                q["conv_id"] = conv_id
            with state_lock:
                prior = state.get("cases", {}).get(str(conv_index), {})
                already_complete = prior.get("status") == "completed" and all(
                    judged.get(q["key"], {}).get("status") == "success" for q in questions
                )
            if already_complete:
                return
            with state_lock:
                state.setdefault("cases", {})[str(conv_index)] = {
                    "status": "running", "phase": "clear", "conversation_id": conv_id,
                    "user_id": user_id, "started_at": _now(),
                }
                _atomic_json(state_path, state)
            stage = "clear"
            try:
                _post(memory, "/api/v2/memory/clear", {"user_id": user_id})
                stage = "ingest"
                update_case(conv_index, phase=stage)
                ingest_calls = 0
                for session_index, session in enumerate(_sessions(row)):
                    occurred_at = _parse_time(session[0].get("time_anchor"))
                    claims = _claims_for_session(session, version, conv_index, session_index)
                    session_id = f"{user_id}_session_{session_index}"
                    # 10M sessions produce much larger payloads than 100K;
                    # bounded batches avoid keeping an ingest request open for
                    # the full client timeout while preserving every claim.
                    batches = (_claim_batches(claims, max_claims=50, max_raw_chars=60_000)
                               if scale == "10M" else _claim_batches(claims))
                    for batch_idx, batch in enumerate(batches):
                        event_id = f"{version}:{conv_index}:{session_index}:{batch_idx}"
                        _post(memory, "/api/v2/memory/ingest", {
                            "user_id": user_id, "namespace": namespace,
                            "source_event": {"event_id": event_id, "source_type": "dialogue",
                                "source_id": session_id, "meeting_id": f"beam{scale.lower()}_{conv_index}:{session_id}",
                                "segment_id": str(batch_idx), "occurred_at": occurred_at,
                                "raw_text": "\n".join(c["text"] for c in batch)},
                            "claims": batch,
                        })
                        ingest_calls += 1
                update_case(conv_index, ingest_calls=ingest_calls)
                for question in questions:
                    key = question["key"]
                    if is_judged_success(key):
                        continue
                    stage = "search"
                    try:
                        update_case(conv_index, phase=stage)
                        question_max_tokens = (max(max_tokens, broad_max_tokens)
                                               if (question["dimension"] == "multi_session_reasoning"
                                                   or _needs_broad_evidence(question["question"]))
                                               else max_tokens)
                        question_candidate_limit, question_min_score = _retrieval_limits(
                            question, candidate_limit, min_score,
                        )
                        request = {
                            "user_id": user_id, "query": question["question"],
                            "required_dimensions": [], "method": "SEMANTIC",
                            "usage_context": {"module_id": "OmniMemEval", "scenario": f"beam{scale.lower()}",
                                "scope": "self", "namespace": namespace, "caller_role": "user"},
                            "top_k": top_k, "candidate_limit": question_candidate_limit,
                            "max_tokens": question_max_tokens, "min_score": question_min_score,
                        }
                        evidence, retrievals, supplemental_errors, coverage = _retrieve_evidence(
                            memory, request, question, question_max_tokens, top_k,
                            context_profile=context_profile,
                            evidence_profile=evidence_profile,
                        )
                        context = _context(evidence, question["question"], chronological=coverage is not None)
                        save_record(searches, searches_path, key, {"key": key, "conv_id": conv_id,
                            "question": question["question"], "dimension": question["dimension"],
                            "search_context": context,
                            "retrieval_trace": retrievals[0].get("retrieval_trace", {}),
                            "retrieval_passes": [result.get("retrieval_trace", {}) for result in retrievals],
                            "supplemental_retrieval_errors": supplemental_errors,
                            "evidence_coverage": coverage,
                            "retrieve_max_tokens": question_max_tokens,
                            "status": "success"})
                        stage = "answer"
                        update_case(conv_index, phase=stage)
                        prompt = _answer_prompt(context, question)
                        answer = answer_client.chat.completions.create(
                            model=answer_model, messages=[{"role": "user", "content": prompt}], temperature=0,
                        ).choices[0].message.content or ""
                        save_record(answers, responses_path, key, {
                            "key": key, "conv_id": conv_id, "question": question["question"],
                            "answer": answer, "ideal_response": question.get("ideal_response"), "status": "success",
                        })
                        stage = "judge"
                        update_case(conv_index, phase=stage)
                        score, detail = _judge(answer, question, judge_client, eval_model)
                        save_record(judged, judged_path, key, {"key": key, "conv_id": conv_id, "question": question["question"],
                            "dimension": question["dimension"], "difficulty": question.get("difficulty"),
                            "ideal_response": question.get("ideal_response"), "answer": answer,
                            "nugget_score": score, "status": "success", **detail})
                    except Exception as exc:
                        with state_lock:
                            stage_failures[stage] += 1
                        save_record(judged, judged_path, key, _fail_record(question, user_id, stage, exc))
                        print(f"[BEAM{scale}] failed {key} at {stage}: {type(exc).__name__}: {str(exc)[:200]}",
                              file=sys.stderr, flush=True)
                    with state_lock:
                        _atomic_json(judged_path, judged)
                        _atomic_json(state_path, state)
                with state_lock:
                    failed_questions = sum(judged.get(q["key"], {}).get("status") != "success" for q in questions)
                update_case(conv_index, status="failed" if failed_questions else "completed",
                            failed_questions=failed_questions, finished_at=_now())
                print(f"[BEAM{scale}] conversation {conv_index - start_index + 1}/"
                      f"{conversation_limit} {conv_id}: "
                      f"{len(questions) - failed_questions}/{len(questions)} valid", flush=True)
            except Exception as exc:
                with state_lock:
                    stage_failures[stage] += 1
                    for question in questions:
                        if judged.get(question["key"], {}).get("status") != "success":
                            judged[question["key"]] = _fail_record(question, user_id, stage, exc)
                    _atomic_json(judged_path, judged)
                update_case(conv_index, status="failed", failed_stage=stage,
                            error_type=type(exc).__name__, error=str(exc)[:800], finished_at=_now())
                print(f"[BEAM{scale}] conversation {conv_id} failed at {stage}: {type(exc).__name__}: {str(exc)[:200]}",
                      file=sys.stderr, flush=True)
            finally:
                try:
                    _post(memory, "/api/v2/memory/clear", {"user_id": user_id})
                except Exception as exc:
                    with state_lock:
                        cleanup_failures += 1
                    update_case(conv_index, cleanup_error=f"{type(exc).__name__}: {str(exc)[:300]}")
                with state_lock:
                    _atomic_json(state_path, state)
                    _atomic_json(judged_path, judged)

        if scale == "100K":
            work = ((conv_index, row, questions) for conv_index, (row, questions) in
                    enumerate(zip(selected_rows, questions_by_conv), start=start_index) if questions)
        else:
            # Never materialize all 10 enormous conversations at once. The
            # streaming iterator and bounded future set cap live rows at workers.
            work = ((conv_index, json.loads(line), questions_by_conv[conv_index - start_index])
                    for conv_index, line in enumerate(_iter_beam10m(data_path))
                    if start_index <= conv_index < start_index + conversation_limit
                    and questions_by_conv[conv_index - start_index])
        with ThreadPoolExecutor(max_workers=min(workers, selected_conversation_count)) as executor:
            if scale == "100K":
                futures = [executor.submit(process_conversation, *item) for item in work]
                for future in as_completed(futures):
                    future.result()
            else:
                pending = set()
                for item in work:
                    pending.add(executor.submit(process_conversation, *item))
                    if len(pending) >= workers:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            future.result()
                for future in as_completed(pending):
                    future.result()

    valid = [r for r in judged.values() if r.get("status") == "success"]
    failed = [r for r in judged.values() if r.get("status") != "success"]
    score = sum(float(r.get("nugget_score", 0)) for r in valid) / len(valid) if valid else None
    by_stage = {stage: sum(r.get("failed_stage") == stage for r in failed)
                for stage in ("clear", "ingest", "search", "answer", "judge")}
    # Includes conversation-level failures which affect many question records.
    by_stage["ingest"] = max(by_stage["ingest"], stage_failures["ingest"])
    all_done = all(case.get("status") in {"completed", "failed"} for case in state.get("cases", {}).values())
    state["status"] = "completed" if all_done and not failed else ("completed_with_failures" if all_done else "running")
    elapsed = round(time.monotonic() - started, 2)
    report = {
        "benchmark": benchmark, "version": version, "scope": config["scope"],
        "metric": "OmniMemEval Nugget Score (valid questions only)", "score": score,
        "valid_questions": len(valid), "total_questions": question_count,
        "valid_conversations": sum(case.get("status") == "completed" for case in state["cases"].values()),
        "total_conversations": selected_conversation_count,
        "start_index": start_index, "dimensions": selected_dimensions,
        "workers": min(workers, selected_conversation_count),
        "failed_questions": len(failed), "failed_by_stage": by_stage,
        "dataset_sha256": digest, "answer_model": answer_model, "eval_model": eval_model,
        "elapsed_seconds": elapsed, "cleanup_failures": cleanup_failures, "updated_at": _now(),
        "dimension_scores": {},
    }
    for dimension in sorted({str(item.get("dimension") or "unknown") for item in valid}):
        values = [float(item["nugget_score"]) for item in valid if item.get("dimension") == dimension]
        report["dimension_scores"][dimension] = {"nugget_score_mean": sum(values) / len(values), "count": len(values)}
    _atomic_json(report_path, report)
    state["summary"] = report
    _atomic_json(state_path, state)
    score_text = f"{score:.4f}" if score is not None else "N/A"
    (output_dir / "exp_report.md").write_text(
        f"# OmniMemEval {benchmark} End-to-End Report\n\n"
        f"- Version: `{version}`\n- Nugget Score (valid questions only): **{score_text}**\n"
        f"- Scope: **{config['scope']}**\n"
        f"- Valid questions: **{len(valid)}/{question_count}**\n- Failed questions: **{len(failed)}**\n"
        f"- Valid conversations: **{report['valid_conversations']}/{conversation_limit}**\n"
        f"- Failures by stage: `{json.dumps(by_stage, sort_keys=True)}`\n"
        f"- Elapsed: **{elapsed:.1f}s**\n", encoding="utf-8",
    )
    print(f"[BEAM{scale}] {state['status']} score={score_text} "
          f"valid={len(valid)}/{question_count} failed={len(failed)}",
          flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[BEAM100K] fatal: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)

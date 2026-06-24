"""Shared search functions for single-user benchmarks.

LoCoMo uses dual-speaker templates and keeps its own search functions in
``locomo_search.py``. LongMemEval uses the functions below.

Typical usage in a benchmark search script::

    from utils.search_helpers import (
        generic_text_search, letta_search, cognee_search,
        hindsight_search, backboard_search, DEFAULT_SEARCH_DISPATCH,
    )
"""

from __future__ import annotations

import os
from typing import Any

from time import time

from utils.prompts import CONTEXT_TEMPLATE

SearchResult = (
    tuple[str, float]
    | tuple[str, float, str | None]
    | tuple[str, float, str | None, str]
)


def _memory_text(results: Any) -> str:
    if results is None:
        return ""
    if isinstance(results, str):
        return results
    return "\n".join(results)


def format_search_context(memories: Any) -> tuple[str, str]:
    """Return ``(prompt_context, raw_context)`` for memory search output."""
    raw_context = _memory_text(memories)
    return CONTEXT_TEMPLATE.format(memories=raw_context), raw_context


def unpack_search_result(result: SearchResult) -> tuple[str, float, str | None, str]:
    """Normalize legacy and current search wrapper return values.

    Current wrappers return ``(context, duration_ms, reflect_answer,
    raw_context)``. The legacy two- and three-element tuples remain accepted so
    tests or external experiments that call custom dispatch functions fail only
    when their shape is truly invalid.
    """
    if not isinstance(result, tuple) or len(result) not in {2, 3, 4}:
        raise ValueError(
            "search result must be (context, duration_ms), "
            "(context, duration_ms, reflect_answer), or "
            "(context, duration_ms, reflect_answer, raw_context)"
        )
    context = result[0] or ""
    duration_ms = float(result[1] or 0.0)
    reflect_answer = result[2] if len(result) >= 3 else None
    raw_context = result[3] if len(result) == 4 else context
    return context, duration_ms, reflect_answer, _memory_text(raw_context)


# ── Core search wrappers ─────────────────────────────────────────────────────


def generic_text_search(
    client: Any,
    query: str,
    user_id: str,
    top_k: int,
    **_kw: Any,
) -> tuple[str, float, None, str]:
    start = time()
    results = client.search(query, user_id, top_k)
    context, raw_context = format_search_context(results)
    duration_ms = (time() - start) * 1000
    return context, duration_ms, None, raw_context


def letta_search(
    client: Any,
    query: str,
    user_id: str,
    top_k: int,
    **_kw: Any,
) -> tuple[str, float, str | None, str]:
    """Letta search with direct/rag dual-mode support."""
    start = time()
    results = client.search(query, user_id, top_k)
    if isinstance(results, dict) and "answer" in results:
        answer = results["answer"]
        raw_context = results.get("context", "")
        context = CONTEXT_TEMPLATE.format(memories=raw_context)
        duration_ms = (time() - start) * 1000
        return context, duration_ms, answer, raw_context
    context, raw_context = format_search_context(results)
    duration_ms = (time() - start) * 1000
    return context, duration_ms, None, raw_context


def cognee_search(
    client: Any,
    query: str,
    user_id: str,
    top_k: int,
    **_kw: Any,
) -> tuple[str, float, str | None, str]:
    """Cognee search with completion/retrieval dual-mode support."""
    start = time()
    results = client.search(query, user_id, top_k)
    context, raw_context = format_search_context(results)
    duration_ms = (time() - start) * 1000

    if hasattr(client, "is_completion_search") and client.is_completion_search and not client._only_context:
        return context, duration_ms, raw_context, raw_context
    return context, duration_ms, None, raw_context


def backboard_search(
    client: Any,
    query: str,
    user_id: str,
    top_k: int,
    *,
    question_date: str | None = None,
    **_kw: Any,
) -> tuple[str, float, str | None, str]:
    """Backboard search with rag/reflect dual-mode.

    When *question_date* is provided (e.g. LME benchmark), a date prefix is
    prepended to the query for temporal reasoning context.
    """
    eval_mode = os.getenv("BACKBOARD_EVAL_MODE", "rag").strip().lower()
    if question_date:
        query = f"Today's date: {question_date}\n\n{query}"
    if eval_mode == "reflect":
        start = time()
        answer, mem_text = client.reflect(query, user_id, top_k)
        context = CONTEXT_TEMPLATE.format(memories=mem_text)
        duration_ms = (time() - start) * 1000
        return context, duration_ms, answer, mem_text
    else:
        return generic_text_search(client, query, user_id, top_k)


def hindsight_search(client: Any, query: str, user_id: str, top_k: int, *,
                     question_date: str | None = None,
                     max_tokens: int | None = None, max_chunk_tokens: int | None = None, **_kw: Any) -> tuple[str, float, str | None, str]:
    """Hindsight search with recall/reflect dual-mode.

    *max_tokens* and *max_chunk_tokens* are optional Hindsight API overrides.
    *question_date* is passed as ``query_timestamp`` when available.
    """
    mode = os.getenv("HINDSIGHT_MODE", "recall").lower()
    start = time()
    if mode == "reflect":
        kw = {}
        if question_date:
            kw["query_timestamp"] = question_date
        answer, based_on = client.reflect(query, user_id, **kw)
        sources = "\n".join(str(s) for s in based_on) if based_on else ""
        context = CONTEXT_TEMPLATE.format(memories=sources)
        duration_ms = (time() - start) * 1000
        return context, duration_ms, answer, sources
    else:
        search_kw = {}
        if max_tokens is not None:
            search_kw["max_tokens"] = max_tokens
        if max_chunk_tokens is not None:
            search_kw["max_chunk_tokens"] = max_chunk_tokens
        if question_date:
            search_kw["query_timestamp"] = question_date
        results = client.search(query, user_id, top_k, **search_kw)
        context, raw_context = format_search_context(results)
        duration_ms = (time() - start) * 1000
        return context, duration_ms, None, raw_context


# ── Default dispatch table ───────────────────────────────────────────────────

DEFAULT_SEARCH_DISPATCH = {
    "zep": generic_text_search,
    "mem0": generic_text_search,
    "memos": generic_text_search,
    "everos": generic_text_search,
    "memu": generic_text_search,
    "supermemory": generic_text_search,
    "letta": letta_search,
    "cognee": cognee_search,
    "hindsight": hindsight_search,
    "graphiti": generic_text_search,
    "backboard": backboard_search,
    "viking": generic_text_search,
    "memori": generic_text_search,
    "memmachine": generic_text_search,
    "memorylake": generic_text_search,
    "mem9": generic_text_search,
}


def dispatch_search(frame: str, client: Any, query: str, user_id: str, top_k: int, *,
                    dispatch: dict[str, Any] | None = None, **extra_kw: Any) -> SearchResult:
    """Look up and call the appropriate search function for *frame*.

    *extra_kw* (e.g. ``question_date``) is forwarded to the search function
    as keyword arguments.

    Returns ``(context, duration_ms, reflect_answer, raw_context)`` for the
    built-in search functions. Legacy two- and three-element tuples remain
    accepted by ``unpack_search_result`` for custom dispatch functions.
    """
    table = dispatch or DEFAULT_SEARCH_DISPATCH
    search_fn = table.get(frame)
    if search_fn is None:
        raise ValueError(f"No search function for lib: {frame!r}")
    return search_fn(client, query, user_id, top_k, **extra_kw)

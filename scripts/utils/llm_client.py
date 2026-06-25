"""Centralized LLM client factory for OmniMemEval.

Each evaluation module (ANSWER, EVAL) can have its own model, API key,
and base URL, configured via environment variables.

Clients returned by create_*_client() are transparently wrapped to
intercept API responses and accumulate token usage via TokenTracker.

Env var naming:
    ANSWER_MODEL   / ANSWER_API_KEY   / ANSWER_BASE_URL
    EVAL_MODEL     / EVAL_API_KEY     / EVAL_BASE_URL
"""

import os
import time
import asyncio
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import tiktoken
from openai import AsyncOpenAI, OpenAI

from client_factory.base_client import env_float, env_int
from utils.token_tracker import get_tracker

_T = TypeVar("_T")

_ENV_KEYS = {
    "ANSWER": {
        "model": "ANSWER_MODEL",
        "api_key": "ANSWER_API_KEY",
        "base_url": "ANSWER_BASE_URL",
    },
    "EVAL": {
        "model": "EVAL_MODEL",
        "api_key": "EVAL_API_KEY",
        "base_url": "EVAL_BASE_URL",
    },
}

_encoding = None


@dataclass(frozen=True)
class LLMRetryConfig:
    module: str
    max_retries: int
    retry_base_seconds: float
    retry_max_seconds: float
    timeout_seconds: float


def _module_env(module: str, suffix: str) -> str:
    return f"{module}_{suffix}"


def _env_int_with_module(module: str, suffix: str, global_name: str, default: int, *, min_value: int) -> int:
    value = env_int(_module_env(module, suffix), None, min_value=min_value)
    if value is not None:
        return value
    return env_int(global_name, default, min_value=min_value)


def _env_float_with_module(module: str, suffix: str, global_name: str, default: float, *, min_value: float) -> float:
    value = env_float(_module_env(module, suffix), None, min_value=min_value)
    if value is not None:
        return value
    return env_float(global_name, default, min_value=min_value)


def get_llm_retry_config(module: str) -> LLMRetryConfig:
    """Return retry/timeout settings for ANSWER or EVAL LLM calls.

    ``*_MAX_RETRIES`` means retries after the initial attempt. For example,
    the default value 4 allows up to 5 total attempts.
    """
    return LLMRetryConfig(
        module=module,
        max_retries=_env_int_with_module(
            module,
            "MAX_RETRIES",
            "LLM_MAX_RETRIES",
            4,
            min_value=0,
        ),
        retry_base_seconds=_env_float_with_module(
            module,
            "RETRY_BASE_SECONDS",
            "LLM_RETRY_BASE_SECONDS",
            1.0,
            min_value=0.0,
        ),
        retry_max_seconds=_env_float_with_module(
            module,
            "RETRY_MAX_SECONDS",
            "LLM_RETRY_MAX_SECONDS",
            60.0,
            min_value=0.0,
        ),
        timeout_seconds=_env_float_with_module(
            module,
            "TIMEOUT_SECONDS",
            "LLM_TIMEOUT_SECONDS",
            600.0,
            min_value=1.0,
        ),
    )


def _retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    header = headers.get("Retry-After") or headers.get("retry-after")
    if header is None:
        return None
    try:
        return max(float(header), 0.0)
    except (TypeError, ValueError):
        return None


def _is_retryable_llm_error(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    if status_code is not None:
        return False

    name = type(exc).__name__
    text = str(exc)
    retryable_markers = (
        "APIConnection",
        "APITimeout",
        "RateLimit",
        "InternalServer",
        "ServiceUnavailable",
        "Timeout",
        "timeout",
        "Connection",
        "connection",
        "temporarily unavailable",
    )
    return any(marker in name or marker in text for marker in retryable_markers)


def _retry_wait_seconds(config: LLMRetryConfig, attempt: int, exc: BaseException) -> float:
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return min(retry_after, config.retry_max_seconds)
    wait = config.retry_base_seconds * (2 ** attempt)
    return min(wait, config.retry_max_seconds)


def _get_encoding():
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text is None and item.get("type") == "text":
                    text = item.get("content")
                if text is not None:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _estimate_prompt_tokens(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int | None:
    messages = kwargs.get("messages")
    if messages is None and len(args) >= 2:
        messages = args[1]
    if not isinstance(messages, list):
        return None

    text_parts = []
    for message in messages:
        if not isinstance(message, dict):
            text_parts.append(str(message))
            continue
        role = message.get("role")
        if role:
            text_parts.append(str(role))
        text_parts.append(_content_text(message.get("content")))
    prompt_text = "\n".join(text_parts)
    if not prompt_text:
        return 0
    return len(_get_encoding().encode(prompt_text, disallowed_special=()))


def _usage_prompt_tokens(usage) -> int:
    value = (
        usage.get("prompt_tokens")
        if isinstance(usage, dict)
        else getattr(usage, "prompt_tokens", None)
    )
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _record_llm_usage(
    tracker,
    module: str,
    resp,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    if isinstance(resp, dict):
        usage = resp.get("usage")
        model = resp.get("model")
    else:
        usage = getattr(resp, "usage", None)
        model = getattr(resp, "model", None)
    prompt_tokens = None
    if usage is None or _usage_prompt_tokens(usage) <= 0:
        prompt_tokens = _estimate_prompt_tokens(args, kwargs)
    if usage is not None or prompt_tokens is not None:
        tracker.record(module, usage=usage, model=model, prompt_tokens=prompt_tokens)
        return


def _call_with_retries(fn: Callable[[], _T], config: LLMRetryConfig) -> _T:
    for attempt in range(config.max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt >= config.max_retries or not _is_retryable_llm_error(exc):
                raise
            wait = _retry_wait_seconds(config, attempt, exc)
            print(
                f"  ⚠ [{config.module}] LLM retry {attempt + 1}/"
                f"{config.max_retries}: {type(exc).__name__}: "
                f"{str(exc)[:200]} (wait {wait:.1f}s)"
            )
            if wait > 0:
                time.sleep(wait)
    raise RuntimeError("unreachable LLM retry state")


async def _acall_with_retries(fn: Callable[[], Any], config: LLMRetryConfig) -> Any:
    for attempt in range(config.max_retries + 1):
        try:
            return await fn()
        except Exception as exc:
            if attempt >= config.max_retries or not _is_retryable_llm_error(exc):
                raise
            wait = _retry_wait_seconds(config, attempt, exc)
            print(
                f"  ⚠ [{config.module}] LLM retry {attempt + 1}/"
                f"{config.max_retries}: {type(exc).__name__}: "
                f"{str(exc)[:200]} (wait {wait:.1f}s)"
            )
            if wait > 0:
                await asyncio.sleep(wait)
    raise RuntimeError("unreachable LLM retry state")


# ── Transparent proxy classes ────────────────────────────────────────────────
# Intercept client.chat.completions.create() to extract response.usage
# without changing any caller code.


class _TrackedSyncCompletions:
    def __init__(self, completions, tracker, retry_config: LLMRetryConfig):
        self._completions = completions
        self._tracker = tracker
        self._retry_config = retry_config

    def create(self, *args, **kwargs):
        resp = _call_with_retries(
            lambda: self._completions.create(*args, **kwargs),
            self._retry_config,
        )
        _record_llm_usage(self._tracker, self._retry_config.module, resp, args, kwargs)
        return resp

    def __getattr__(self, name):
        return getattr(self._completions, name)


class _TrackedSyncChat:
    def __init__(self, chat, tracker, retry_config: LLMRetryConfig):
        self._chat = chat
        self.completions = _TrackedSyncCompletions(chat.completions, tracker, retry_config)

    def __getattr__(self, name):
        return getattr(self._chat, name)


class _TrackedSyncOpenAI:
    def __init__(self, client, tracker, retry_config: LLMRetryConfig):
        self._client = client
        self.chat = _TrackedSyncChat(client.chat, tracker, retry_config)

    def __getattr__(self, name):
        return getattr(self._client, name)


class _TrackedAsyncCompletions:
    def __init__(self, completions, tracker, retry_config: LLMRetryConfig):
        self._completions = completions
        self._tracker = tracker
        self._retry_config = retry_config

    async def create(self, *args, **kwargs):
        resp = await _acall_with_retries(
            lambda: self._completions.create(*args, **kwargs),
            self._retry_config,
        )
        _record_llm_usage(self._tracker, self._retry_config.module, resp, args, kwargs)
        return resp

    def __getattr__(self, name):
        return getattr(self._completions, name)


class _TrackedAsyncChat:
    def __init__(self, chat, tracker, retry_config: LLMRetryConfig):
        self._chat = chat
        self.completions = _TrackedAsyncCompletions(chat.completions, tracker, retry_config)

    def __getattr__(self, name):
        return getattr(self._chat, name)


class _TrackedAsyncOpenAI:
    def __init__(self, client, tracker, retry_config: LLMRetryConfig):
        self._client = client
        self.chat = _TrackedAsyncChat(client.chat, tracker, retry_config)

    def __getattr__(self, name):
        return getattr(self._client, name)


# ── Public API ───────────────────────────────────────────────────────────────


def get_llm_config(module):
    """Return {"model", "api_key", "base_url"} for a given module.

    module: "ANSWER" or "EVAL"
    """
    keys = _ENV_KEYS[module]
    cfg = {
        "model": os.getenv(keys["model"]),
        "api_key": os.getenv(keys["api_key"]),
        "base_url": os.getenv(keys["base_url"]),
    }
    if not cfg["api_key"]:
        raise RuntimeError(
            f"[{module}] API Key not configured. "
            f"Set {keys['api_key']} in .env"
        )
    return cfg


def create_openai_client(module):
    """Create a synchronous OpenAI client for the given module.

    Returns (tracked_client, model_name).
    The client transparently records token usage from every API call.
    """
    cfg = get_llm_config(module)
    retry_config = get_llm_retry_config(module)
    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        timeout=retry_config.timeout_seconds,
        max_retries=0,
    )
    tracked = _TrackedSyncOpenAI(client, get_tracker(), retry_config)
    return tracked, cfg["model"]


def create_async_openai_client(module):
    """Create an asynchronous OpenAI client for the given module.

    Returns (tracked_client, model_name).
    The client transparently records token usage from every API call.
    """
    cfg = get_llm_config(module)
    retry_config = get_llm_retry_config(module)
    client = AsyncOpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        timeout=retry_config.timeout_seconds,
        max_retries=0,
    )
    tracked = _TrackedAsyncOpenAI(client, get_tracker(), retry_config)
    return tracked, cfg["model"]

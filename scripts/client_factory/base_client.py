from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable, Generator
from typing import Any, TypeVar

import requests

_T = TypeVar("_T")


# ── Helpers ──────────────────────────────────────────────────────────────────

DEFAULT_BATCH_SIZE = 20

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?。！？])\s+')
_TRUE_VALUES = frozenset({"1", "true", "yes", "y", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "n", "off"})


def env_str(name: str, default: str | None = None, *, strip: bool = True) -> str | None:
    """Return an environment variable with consistent optional stripping."""
    value = os.getenv(name)
    if value is None:
        value = default
    if value is not None and strip:
        value = value.strip()
    return value


def require_env(name: str) -> str:
    """Return a required env var, raising a clear error if missing or blank."""
    value = env_str(name, "")
    if not value:
        raise ValueError(f"{name} environment variable is not set")
    return value


def env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean env var using strict, documented values."""
    raw = env_str(name)
    if raw in (None, ""):
        return default
    value = raw.lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} must be one of {sorted(_TRUE_VALUES | _FALSE_VALUES)}"
    )


def env_optional_bool(name: str) -> bool | None:
    """Parse an optional boolean env var, returning None when unset or blank."""
    raw = env_str(name)
    if raw in (None, ""):
        return None
    return env_bool(name)


def env_int(
    name: str,
    default: int | None = None,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int | None:
    """Parse an integer env var and optionally validate bounds."""
    raw = env_str(name)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    _validate_bounds(name, value, min_value=min_value, max_value=max_value)
    return value


def env_float(
    name: str,
    default: float | None = None,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    """Parse a float env var and optionally validate bounds."""
    raw = env_str(name)
    if raw in (None, ""):
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    _validate_bounds(name, value, min_value=min_value, max_value=max_value)
    return value


def _validate_bounds(
    name: str,
    value: int | float,
    *,
    min_value: int | float | None,
    max_value: int | float | None,
) -> None:
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value}")


def env_csv(name: str) -> list[str]:
    """Parse comma-separated env values, dropping blank entries."""
    raw = env_str(name, "") or ""
    return [item.strip() for item in raw.split(",") if item.strip()]


def env_json(name: str) -> Any | None:
    """Parse a JSON env var, returning None when unset or blank."""
    raw = env_str(name)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON: {exc}") from exc


def env_max_batch_chars(name: str, default: int = 0) -> int:
    """Parse a client-specific max char budget with MAX_BATCH_CHARS fallback."""
    raw = env_str(name)
    fallback = "MAX_BATCH_CHARS"
    if raw in (None, ""):
        raw = env_str(fallback)
        source = fallback
    else:
        source = name
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{source} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{source} must be >= 0, got {value}")
    return value


def _msg_chars(item: dict | Any) -> int:
    """Return character count of a message item's content."""
    if isinstance(item, dict):
        return len(item.get("content", ""))
    return len(str(item))


def _split_text(text: str, max_chars: int) -> list[str]:
    """Split *text* into chunks ≤ *max_chars* at natural boundaries.

    Priority: paragraph → line → sentence → hard cut.
    """
    if len(text) <= max_chars:
        return [text]

    for sep, rejoin in [("\n\n", "\n\n"), ("\n", "\n")]:
        parts = text.split(sep)
        if len(parts) > 1:
            return _greedy_merge(parts, max_chars, rejoin)

    parts = _SENTENCE_SPLIT_RE.split(text)
    if len(parts) > 1:
        return _greedy_merge(parts, max_chars, " ")

    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def _greedy_merge(parts: list[str], max_chars: int, sep: str) -> list[str]:
    """Greedily merge *parts*, flushing when the next part would exceed *max_chars*."""
    chunks = []
    current = parts[0]
    for part in parts[1:]:
        candidate = current + sep + part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > max_chars:
                chunks.extend(_split_text(part, max_chars))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks


def _split_message(msg: dict | Any, max_chars: int) -> list[dict]:
    """Split one oversized message dict into multiple smaller ones.

    Each piece keeps the original role / metadata and gets a ``[part N/M]``
    prefix so the memory product can reconstruct ordering.
    """
    content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
    chunks = _split_text(content, max_chars)
    if len(chunks) <= 1:
        return [msg]
    result = []
    total = len(chunks)
    for idx, chunk in enumerate(chunks, 1):
        new_msg = {**msg} if isinstance(msg, dict) else {"content": str(msg)}
        new_msg["content"] = f"[part {idx}/{total}] {chunk}"
        result.append(new_msg)
    return result


def iter_batches(items: list, batch_size: int = DEFAULT_BATCH_SIZE, max_chars: int = 0) -> Generator[list, None, None]:
    """Yield non-empty batches of *items*.

    Batching modes (in priority order):

    1. **Character-budget** (``max_chars > 0``): accumulate messages until
       adding the next one would exceed *max_chars* total characters.
       If a single message already exceeds *max_chars* it is automatically
       split at natural text boundaries.  ``batch_size`` still serves as
       an upper count cap per batch.

    2. **Count-based** (``max_chars <= 0``, default): classic fixed-size
       slicing — each batch has at most *batch_size* messages.

    This means users only need ONE parameter (``max_chars``) to control
    the total data volume per API call.
    """
    if max_chars <= 0:
        for start in range(0, max(len(items), 1), batch_size):
            batch = items[start : start + batch_size]
            if batch:
                yield batch
        return

    batch = []
    batch_chars = 0

    for item in items:
        chars = _msg_chars(item)

        if chars > max_chars:
            if batch:
                yield batch
                batch, batch_chars = [], 0
            for sub in _split_message(item, max_chars):
                yield [sub]
            continue

        if batch and (batch_chars + chars > max_chars or len(batch) >= batch_size):
            yield batch
            batch, batch_chars = [], 0

        batch.append(item)
        batch_chars += chars

    if batch:
        yield batch


class _AttrDict(dict):
    """Dict subclass that supports attribute access (backward compat with SDK objects)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


class RateLimitError(Exception):
    """Raised when a 429 response is received, carrying retry-after info."""

    def __init__(self, retry_after: float | None = None, response: requests.Response | None = None):
        self.retry_after = retry_after
        self.response = response
        super().__init__(f"Rate limited (retry_after={retry_after}s)")


class _TokenBucketLimiter:
    """Thread-safe token-bucket rate limiter.

    Call ``acquire()`` before each API request; it blocks until a token is
    available, enforcing at most *qps* requests per second.
    """

    def __init__(self, qps: float):
        self.qps = qps
        self.interval = 1.0 / qps
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            self._next_allowed = max(now, self._next_allowed) + self.interval
        if wait > 0:
            time.sleep(wait)


class BaseApiClient:
    """Base class providing unified HTTP helpers and retry logic.

    All competitor clients inherit from this so that every external call goes
    through ``requests`` — no vendor SDK required.

    Rate-limiting support:
        - Pass ``qps=N`` to __init__ to cap outgoing requests to N per second
          (shared across all threads using this client instance).
        - _retry automatically detects HTTP 429 responses and respects the
          ``Retry-After`` header with jittered exponential backoff.
    """

    DEFAULT_MAX_RETRIES = 8
    DEFAULT_TIMEOUT = 60

    def __init__(self, base_url: str, headers: dict[str, str], qps: float | None = None, timeout: int | None = None):
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.headers = headers
        self._limiter = _TokenBucketLimiter(qps) if qps else None
        self._timeout = timeout or self.DEFAULT_TIMEOUT
        self._max_retries = env_int(
            "OMNIMEMEVAL_MEMORY_MAX_RETRIES",
            self.DEFAULT_MAX_RETRIES,
            min_value=1,
        )
        self._session = requests.Session()
        self._session.headers.update(headers)
        self._session.trust_env = False

    # ── low-level helpers ────────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _throttle(self) -> None:
        """Block until the rate limiter permits the next request."""
        if self._limiter:
            self._limiter.acquire()

    def _apply_timeout(self, kwargs: dict[str, Any]) -> None:
        """Inject default timeout unless the caller explicitly provided one."""
        kwargs.setdefault("timeout", self._timeout)

    def _post(self, path: str, **kwargs: Any) -> requests.Response:
        self._throttle()
        self._apply_timeout(kwargs)
        return self._session.post(self._url(path), **kwargs)

    def _get(self, path: str, **kwargs: Any) -> requests.Response:
        self._throttle()
        self._apply_timeout(kwargs)
        return self._session.get(self._url(path), **kwargs)

    def _put(self, path: str, **kwargs: Any) -> requests.Response:
        self._throttle()
        self._apply_timeout(kwargs)
        return self._session.put(self._url(path), **kwargs)

    def _patch(self, path: str, **kwargs: Any) -> requests.Response:
        self._throttle()
        self._apply_timeout(kwargs)
        return self._session.patch(self._url(path), **kwargs)

    def _delete(self, path: str, **kwargs: Any) -> requests.Response:
        self._throttle()
        self._apply_timeout(kwargs)
        return self._session.delete(self._url(path), **kwargs)

    @staticmethod
    def _parse_retry_after(response: requests.Response) -> float | None:
        """Extract wait seconds from Retry-After header (int or date)."""
        header = response.headers.get("Retry-After") or response.headers.get("retry-after")
        if not header:
            return None
        try:
            return max(float(header), 1.0)
        except (ValueError, TypeError):
            return None

    _RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

    @staticmethod
    def _is_transient(err_str: str) -> bool:
        """Check if an error string indicates a transient / retryable failure."""
        return any(k in err_str for k in (
            "429", "500", "502", "503", "504",
            "rate limit", "Rate limit", "RateLimit",
            "QuotaLimiter", "quota",
            "already exists",
            "Timeout", "timeout",
            "Connection", "connection",
        ))

    @staticmethod
    def sdk_retry(fn: Callable[[], _T], max_retries: int | None = None, base_wait: int = 2) -> _T:
        """Retry *fn()* with exponential backoff for SDK calls (no ``requests``).

        Use this for clients that call vendor SDKs directly instead of
        ``self._post()`` / ``raise_for_status()``.  Retries any exception
        whose ``str()`` contains a known transient keyword.
        """
        if max_retries is None:
            default_retries = env_int(
                "OMNIMEMEVAL_MEMORY_MAX_RETRIES",
                BaseApiClient.DEFAULT_MAX_RETRIES,
                min_value=1,
            )
            max_retries = env_int(
                "OMNIMEMEVAL_MEMORY_SDK_MAX_RETRIES",
                default_retries,
                min_value=1,
            )
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as exc:
                err = str(exc)
                if attempt < max_retries - 1 and BaseApiClient._is_transient(err):
                    wait = base_wait * (2 ** attempt)
                    print(f"  ⚠ SDK retry {attempt + 1}/{max_retries}: "
                          f"{type(exc).__name__}: {err[:200]} (wait {wait}s)")
                    time.sleep(wait)
                else:
                    raise

    def _retry(self, fn: Callable[[], _T], max_retries: int | None = None) -> _T:
        """Execute *fn()* with exponential-backoff retry.

        Retries on:
        - ``RateLimitError`` — explicit rate-limit signal from client code.
        - HTTP 429 / 5xx   — detected via ``raise_for_status()``.
        - Network errors   — connection / timeout / SSL.

        Non-transient errors (e.g. 4xx other than 429) are raised immediately.
        """
        if max_retries is None:
            max_retries = self._max_retries
        for attempt in range(max_retries):
            try:
                return fn()
            except RateLimitError as e:
                if attempt >= max_retries - 1:
                    raise
                base = e.retry_after or 1
                wait = base * (2 ** attempt)
                r = e.response
                if r is not None and r.status_code == 403:
                    tag = "HTTP 403 (throttle)"
                elif r is not None and r.status_code == 429:
                    tag = "HTTP 429 (rate limit)"
                else:
                    tag = "Rate limited"
                print(
                    f"  ⏳ {tag}, waiting {wait:.1f}s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait)
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.SSLError) as e:
                if attempt >= max_retries - 1:
                    raise
                wait = min(2 ** attempt + 1, 60)
                print(f"  ⚠ Connection error, retrying in {wait}s (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(wait)
            except requests.exceptions.HTTPError as e:
                resp = e.response
                if resp is not None and resp.status_code in self._RETRYABLE_STATUS_CODES:
                    if attempt >= max_retries - 1:
                        raise
                    if resp.status_code == 429:
                        retry_after = self._parse_retry_after(resp) or 2
                        wait = retry_after * (2 ** attempt)
                        print(f"  ⏳ HTTP 429, waiting {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
                    else:
                        wait = min(2 ** attempt + 1, 60)
                        print(f"  ⚠ Server error {resp.status_code}, retrying in {wait}s "
                              f"(attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait)
                else:
                    raise

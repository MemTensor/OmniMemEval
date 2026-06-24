"""Per-process LLM token usage tracker.

Accumulates token counts from OpenAI API responses grouped by module
(ANSWER, EVAL, etc.). Each Python process saves its own stats to JSON;
the report generator aggregates them.
"""

import json
import os
import threading
from datetime import datetime


def _usage_value(usage, key: str):
    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key, None)


def _positive_int(value) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


class TokenTracker:
    """Thread-safe accumulator for OpenAI API token usage."""

    def __init__(self):
        self._lock = threading.Lock()
        self._modules: dict[str, dict] = {}

    def record(
        self,
        module: str,
        usage=None,
        model: str | None = None,
        prompt_tokens: int | None = None,
    ):
        """Record token usage for a successful LLM call.

        ``usage`` is preferred when the provider returns OpenAI-compatible
        token accounting. ``prompt_tokens`` is a local fallback estimate used
        when compatible providers omit usage; this keeps answer prompt tokens
        available as the search-result length proxy in reports.
        """
        if usage is None and prompt_tokens is None:
            return
        with self._lock:
            if module not in self._modules:
                self._modules[module] = {
                    "model": model,
                    "call_count": 0,
                    "prompt_tokens": 0,
                    "estimated_prompt_tokens": 0,
                    "estimated_prompt_call_count": 0,
                    "usage_reported_call_count": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            e = self._modules[module]
            e["call_count"] += 1
            if usage is not None:
                usage_prompt_tokens = _positive_int(
                    _usage_value(usage, "prompt_tokens")
                )
                fallback_prompt_tokens = _positive_int(prompt_tokens)
                if usage_prompt_tokens:
                    e["prompt_tokens"] += usage_prompt_tokens
                elif fallback_prompt_tokens:
                    e["prompt_tokens"] += fallback_prompt_tokens
                    e["estimated_prompt_tokens"] += fallback_prompt_tokens
                    e["estimated_prompt_call_count"] += 1
                e["completion_tokens"] += _positive_int(
                    _usage_value(usage, "completion_tokens")
                )
                e["total_tokens"] += _positive_int(_usage_value(usage, "total_tokens"))
                e["usage_reported_call_count"] += 1
            else:
                estimated_prompt_tokens = _positive_int(prompt_tokens)
                e["prompt_tokens"] += estimated_prompt_tokens
                e["estimated_prompt_tokens"] += estimated_prompt_tokens
                e["estimated_prompt_call_count"] += 1
            if model:
                e["model"] = model

    def summary(self) -> dict:
        with self._lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "modules": {k: dict(v) for k, v in self._modules.items()},
            }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2, ensure_ascii=False)
        print(f"Token usage saved → {path}")

    def reset(self):
        with self._lock:
            self._modules.clear()


_global_tracker = TokenTracker()


def get_tracker() -> TokenTracker:
    return _global_tracker

"""Validate optional, question-only pilot selectors for benchmark runners."""

from __future__ import annotations

import os


def selected_ids(env_name: str, available: set[str]) -> set[str] | None:
    """Return an exact selector; reject typos instead of silently running nothing.

    IDs refer to public dataset positions, never answers or rubric contents.
    An unset or blank variable preserves each runner's existing full-range mode.
    """
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return None
    values = [value.strip() for value in raw.split(",")]
    if any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError(f"{env_name} must contain unique, nonempty comma-separated IDs")
    selected = set(values)
    unknown = selected - available
    if unknown:
        raise ValueError(f"{env_name} contains unknown IDs: {', '.join(sorted(unknown))}")
    return selected

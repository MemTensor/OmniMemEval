"""Helpers for duration stats shared by streaming and metric scripts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def duration_values(values: Any) -> list[float]:
    """Return numeric duration values from a scalar, list, or mapping."""
    if isinstance(values, Mapping):
        return flatten_duration_map(values)
    if isinstance(values, (list, tuple)):
        out = []
        for value in values:
            numeric = _coerce_float(value)
            if numeric is not None:
                out.append(numeric)
        return out
    numeric = _coerce_float(values)
    return [numeric] if numeric is not None else []


def flatten_duration_map(duration_map: Any) -> list[float]:
    """Flatten ``{unit_id: duration_or_duration_list}`` to a numeric list."""
    if not isinstance(duration_map, Mapping):
        return []
    out = []
    for values in duration_map.values():
        out.extend(duration_values(values))
    return out


def update_unit_duration_list(
    stats: dict[str, Any],
    unit_key: str | int,
    durations: Any,
    *,
    map_key: str,
    flat_key: str,
) -> None:
    """Update per-unit duration stats and rebuild the compatibility flat list."""
    numeric_values = [round(value, 2) for value in duration_values(durations)]
    by_unit = stats.setdefault(map_key, {})
    by_unit[str(unit_key)] = numeric_values
    stats[flat_key] = flatten_duration_map(by_unit)


def add_duration_values(ingestion_stats: Mapping[str, Any]) -> list[float]:
    """Return add-call durations, preferring restart-safe per-unit stats."""
    by_unit = flatten_duration_map(ingestion_stats.get("add_call_durations_by_unit"))
    if by_unit:
        return by_unit
    flat = duration_values(ingestion_stats.get("add_call_durations_ms"))
    if flat:
        return flat
    return flatten_duration_map(ingestion_stats.get("user_durations_ms"))

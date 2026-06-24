"""Centralized timestamp parsing and conversion utilities.

Each benchmark dataset uses its own raw time format. This module provides
per-dataset parsers that produce timezone-aware ``datetime`` objects (UTC),
plus lightweight converters to the formats most memory-product APIs expect.
"""

from datetime import datetime, timezone

# ── Dataset-specific raw formats ─────────────────────────────────────────────

_LOCOMO_FMT = "%I:%M %p on %d %B, %Y"
_LME_FMT = "%Y/%m/%d (%a) %H:%M"


def parse_locomo_time(raw: str) -> datetime:
    """``'1:56 pm on 8 May, 2023'`` (with or without trailing ``' UTC'``)."""
    raw = raw.strip()
    if raw.endswith(" UTC"):
        raw = raw[:-4]
    return datetime.strptime(raw, _LOCOMO_FMT).replace(tzinfo=timezone.utc)


def parse_lme_time(raw: str) -> datetime:
    """``'2023/05/20 (Sat) 02:21'`` (with or without trailing ``' UTC'``)."""
    raw = raw.strip()
    if not raw.endswith(" UTC"):
        raw += " UTC"
    return datetime.strptime(raw, f"{_LME_FMT} UTC").replace(tzinfo=timezone.utc)


# ── Converters (datetime → target format) ────────────────────────────────────

def to_iso(dt: datetime) -> str:
    """datetime → ISO-8601 string (e.g. ``'2023-05-08T13:56:00+00:00'``)."""
    return dt.isoformat()


def to_unix(dt: datetime) -> int:
    """datetime → unix seconds (int)."""
    return int(dt.timestamp())


def to_unix_ms(dt: datetime) -> int:
    """datetime → unix milliseconds (int)."""
    return int(dt.timestamp() * 1000)


def to_readable(dt: datetime, fmt: str = "%-I:%M %p on %-d %B, %Y") -> str:
    """datetime → human-readable string for display / supermemory."""
    return dt.strftime(fmt)

"""Session date selection for BEAM ingestion (null leading time_anchor)."""

from datetime import datetime, timezone

from beam.beam_ingestion import parse_time_anchor, session_time_anchor


def test_first_message_anchor_is_used():
    msgs = [{"time_anchor": "March-15-2024"}, {"time_anchor": "April-01-2024"}]
    assert session_time_anchor(msgs) == "March-15-2024"


def test_null_leading_anchor_uses_first_non_null():
    msgs = [{"time_anchor": None}] * 3 + [{"time_anchor": "March-05-2020"}, {"time_anchor": "June-01-2020"}]
    assert parse_time_anchor(session_time_anchor(msgs)) == datetime(2020, 3, 5, tzinfo=timezone.utc)


def test_missing_or_empty_anchors_fall_back_to_default():
    assert session_time_anchor([{"role": "user"}, {"time_anchor": ""}]) == "January-01-2024"
    assert session_time_anchor([]) == "January-01-2024"

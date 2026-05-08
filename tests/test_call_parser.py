from __future__ import annotations

from personal_lifelog_rag.line.call_parser import parse_call_duration, parse_line_call_text


def test_parse_completed_call_minutes_seconds() -> None:
    parsed = parse_line_call_text("☎ 通話時間 10:38")

    assert parsed is not None
    assert parsed.call_status == "completed"
    assert parsed.duration_sec == 638


def test_parse_completed_call_hours_minutes_seconds() -> None:
    parsed = parse_line_call_text("☎ 通話時間 1:26:14")

    assert parsed is not None
    assert parsed.call_status == "completed"
    assert parsed.duration_sec == 5174


def test_parse_missed_unanswered_and_canceled_calls() -> None:
    assert parse_line_call_text("☎ 不在着信").call_status == "missed"  # type: ignore[union-attr]
    assert parse_line_call_text("☎ 通話に応答がありませんでした").call_status == "unanswered"  # type: ignore[union-attr]
    assert parse_line_call_text("☎ 通話をキャンセルしました").call_status == "canceled"  # type: ignore[union-attr]


def test_parse_unknown_call_like_message() -> None:
    parsed = parse_line_call_text("☎ LINE通話")

    assert parsed is not None
    assert parsed.call_status == "unknown"
    assert parsed.duration_sec is None


def test_parse_call_duration_rejects_invalid_values() -> None:
    assert parse_call_duration("10:38") == 638
    assert parse_call_duration("1:26:14") == 5174
    assert parse_call_duration("1:99:14") is None

from __future__ import annotations

from datetime import date

from personal_lifelog_rag.retrieval.date_parser import parse_date_query


def test_parse_japanese_exact_date() -> None:
    parsed = parse_date_query("2024年12月24日は何していた？")

    assert parsed is not None
    assert parsed.start_iso == "2024-12-24"
    assert parsed.end_iso == "2024-12-24"


def test_parse_iso_and_slash_dates() -> None:
    hyphen = parse_date_query("2024-12-24は何していた？")
    slash = parse_date_query("2024/12/24は何していた？")

    assert hyphen is not None
    assert slash is not None
    assert hyphen.start_iso == "2024-12-24"
    assert slash.start_iso == "2024-12-24"


def test_parse_last_summer() -> None:
    parsed = parse_date_query("去年の夏に行った場所をまとめて", today=date(2026, 5, 8))

    assert parsed is not None
    assert parsed.start_iso == "2025-06-01"
    assert parsed.end_iso == "2025-08-31"


def test_parse_month_day_uses_default_year_and_marks_ambiguous() -> None:
    parsed = parse_date_query("12月24日は何していた？", default_year=2024)

    assert parsed is not None
    assert parsed.start_iso == "2024-12-24"
    assert parsed.year_was_inferred is True
    assert parsed.ambiguous is True


def test_invalid_date_returns_none() -> None:
    assert parse_date_query("2024年13月99日は何していた？") is None

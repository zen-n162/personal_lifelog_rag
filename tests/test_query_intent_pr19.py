from __future__ import annotations

from datetime import date

from personal_lifelog_rag.retrieval.query_intent import classify_query_intent


def test_classify_date_qa() -> None:
    result = classify_query_intent("2024年12月24日は何していた？")

    assert result.intent == "date_qa"
    assert result.entities["date"] == "2024-12-24"
    assert result.routing_hint == "ask"


def test_classify_place_visit() -> None:
    result = classify_query_intent("新宿に行ったのはいつ？")

    assert result.intent == "place_visit"
    assert result.entities["place"] == "新宿"


def test_classify_food_activity() -> None:
    result = classify_query_intent("ご飯を食べた日は？")

    assert result.intent == "food_activity"
    assert "ご飯" in result.entities["food_terms"]


def test_classify_call_activity_with_person() -> None:
    result = classify_query_intent("友人と通話した日は？")

    assert result.intent == "call_activity"
    assert result.entities["person"] == "友人"
    assert result.entities["call_status"] == "completed"


def test_classify_topic_mention() -> None:
    result = classify_query_intent("アルバムの話をしたのはいつ？")

    assert result.intent == "topic_mention"
    assert result.entities["topic"] == "アルバム"


def test_classify_time_range_summary() -> None:
    result = classify_query_intent("2024年12月の出来事をまとめて")

    assert result.intent == "time_range_summary"
    assert result.entities["date_from"] == "2024-12-01"
    assert result.entities["date_to"] == "2024-12-31"


def test_classify_relative_month_with_injected_today() -> None:
    result = classify_query_intent("去年12月は何してた？", today=date(2026, 5, 9))

    assert result.intent in {"time_range_summary", "date_qa"}
    assert result.entities["date_from"] == "2025-12-01"
    assert result.entities["date_to"] == "2025-12-31"


def test_classify_unknown_does_not_crash() -> None:
    result = classify_query_intent("ふわっといい感じによろしく")

    assert result.intent == "unknown"
    assert result.routing_hint == "unsupported"

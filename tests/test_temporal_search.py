from __future__ import annotations

from personal_lifelog_rag.retrieval.temporal_search import build_timeline_items, extract_keyword


def test_extract_keyword_for_food_question() -> None:
    assert extract_keyword("ラーメンを食べた日はいつ？") == "ラーメン"


def test_extract_keyword_for_date_only_question() -> None:
    assert extract_keyword("2024年12月24日は何していた？") is None


def test_build_timeline_items_orders_line_and_media_by_time() -> None:
    items = build_timeline_items(
        line_messages=[
            {"sent_at": "2024-12-24T17:30:00+09:00", "text": "待ち合わせ"},
        ],
        media_items=[
            {"captured_at": "2024-12-24T19:12:00+09:00", "file_name": "a.jpg"},
        ],
    )

    assert [item["kind"] for item in items] == ["line_message", "media_item"]

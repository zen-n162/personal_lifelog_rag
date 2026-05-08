from __future__ import annotations

from datetime import date

from personal_lifelog_rag.retrieval.answer_builder import build_answer
from personal_lifelog_rag.retrieval.date_parser import DateRange
from personal_lifelog_rag.retrieval.temporal_search import TimelineSearchResult


def test_build_answer_uses_local_records_without_full_dump() -> None:
    result = TimelineSearchResult(
        question="2024年12月24日は何していた？",
        date_range=DateRange(date(2024, 12, 24), date(2024, 12, 24), "2024-12-24"),
        keyword=None,
        events=[],
        media_items=[
            {
                "captured_at": "2024-12-24T11:00:00",
                "file_name": "dummy.jpg",
                "file_path": "/local/photos/dummy.jpg",
                "gps_lat": 35.0,
                "gps_lon": 139.0,
            }
        ],
        line_messages=[
            {
                "sent_at": "2024-12-24T17:30:00",
                "sender": "Me",
                "text": "18時に新宿着く！",
            },
            {
                "sent_at": "2024-12-24T22:10:00",
                "sender": "Me",
                "text": "今日のご飯おいしかったね",
            }
        ],
        timeline_items=[
            {
                "kind": "line_message",
                "at": "2024-12-24T17:30:00",
                "record": {"sent_at": "2024-12-24T17:30:00", "sender": "Me", "text": "18時に新宿着く！"},
            },
            {
                "kind": "media_item",
                "at": "2024-12-24T11:00:00",
                "record": {"captured_at": "2024-12-24T11:00:00", "file_name": "dummy.jpg", "gps_lat": 35.0, "gps_lon": 139.0},
            },
        ],
    )

    answer = build_answer("2024年12月24日は何していた？", result)

    assert "2024年12月24日の記録を確認しました" in answer
    assert "LINEでは" in answer
    assert "写真が1枚" in answer
    assert "推定:" in answer
    assert "可能性があります" in answer
    assert "根拠:" in answer
    assert "LINEメッセージ: 2件" in answer
    assert "写真: 1枚" in answer
    assert "GPS付き写真: 1枚" in answer
    assert "信頼度:" in answer


def test_build_answer_for_empty_day_does_not_overstate() -> None:
    result = TimelineSearchResult(
        question="2024年1月1日は何していた？",
        date_range=DateRange(date(2024, 1, 1), date(2024, 1, 1), "2024-01-01"),
        keyword=None,
        events=[],
        media_items=[],
        line_messages=[],
    )

    answer = build_answer("2024年1月1日は何していた？", result)

    assert "記録が見つかりませんでした" in answer
    assert "可能性があります" not in answer
    assert "何をしていたか: 不明" in answer


def test_build_answer_prefers_events_when_available() -> None:
    result = TimelineSearchResult(
        question="2024年12月24日は何していた？",
        date_range=DateRange(date(2024, 12, 24), date(2024, 12, 24), "2024-12-24"),
        keyword=None,
        events=[
            {
                "id": "event_1",
                "date": "2024-12-24",
                "start_time": "10:21:00",
                "end_time": "11:44:00",
                "title": "通話・連絡",
                "summary": "LINE 4件、写真 0枚から作成したイベント候補です。",
                "confidence": 0.72,
                "event_evidence_count": 4,
            }
        ],
        media_items=[],
        line_messages=[
            {
                "sent_at": "2024-12-24T10:21:00",
                "sender": "Person",
                "text": "この本文はevents優先時には出さない",
            }
        ],
        timeline_items=[
            {
                "kind": "line_message",
                "at": "2024-12-24T10:21:00",
                "record": {
                    "sent_at": "2024-12-24T10:21:00",
                    "sender": "Person",
                    "text": "この本文はevents優先時には出さない",
                },
            },
        ],
    )

    answer = build_answer("2024年12月24日は何していた？", result)

    assert "この日は1件の出来事候補があります" in answer
    assert "1. 10:21〜11:44 通話・連絡 / confidence 0.72" in answer
    assert "根拠: 4件" in answer
    assert "この本文はevents優先時には出さない" not in answer
    assert "信頼度:" in answer
    assert "何をしていたか: 中" in answer


def test_build_answer_does_not_infer_food_or_meeting_without_evidence() -> None:
    result = TimelineSearchResult(
        question="2024年12月24日は何していた？",
        date_range=DateRange(date(2024, 12, 24), date(2024, 12, 24), "2024-12-24"),
        keyword=None,
        events=[],
        media_items=[
            {
                "captured_at": "2024-12-24T11:00:00",
                "file_name": "dummy.jpg",
                "gps_lat": None,
                "gps_lon": None,
            }
        ],
        line_messages=[
            {
                "sent_at": "2024-12-24T12:00:00",
                "sender": "Me",
                "text": "了解です",
            }
        ],
        timeline_items=[],
    )

    answer = build_answer("2024年12月24日は何していた？", result)

    assert "食事" not in answer
    assert "待ち合わせ" not in answer
    assert "断定できません" in answer
    assert "何をしていたか: 低" in answer

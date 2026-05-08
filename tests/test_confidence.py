from __future__ import annotations

from personal_lifelog_rag.retrieval.answer_builder import build_answer
from personal_lifelog_rag.retrieval.confidence import calculate_confidence
from personal_lifelog_rag.retrieval.temporal_search import TimelineSearchResult


def test_gps_photos_alone_do_not_make_activity_high() -> None:
    result = _result(
        media_items=[
            {
                "id": f"photo_{index}",
                "captured_at": f"2024-12-24T10:{index:02d}:00+09:00",
                "gps_lat": 35.0 + index * 0.0001,
                "gps_lon": 139.0 + index * 0.0001,
            }
            for index in range(5)
        ],
    )

    confidence = calculate_confidence(result)

    assert confidence["place"] == "中"
    assert confidence["activity"] == "低"


def test_many_photos_alone_do_not_assert_activity() -> None:
    result = _result(
        media_items=[
            {
                "id": f"photo_{index}",
                "captured_at": f"2024-12-24T12:{index:02d}:00+09:00",
                "file_name": f"dummy_{index}.jpg",
            }
            for index in range(10)
        ],
    )

    answer = build_answer("2024年12月24日は何していた？", result)

    assert "何をしていたか: 低" in answer
    assert "食事" not in answer
    assert "待ち合わせ" not in answer
    assert "断定できません" in answer


def test_line_activity_words_create_candidate_but_not_high_without_image_content() -> None:
    result = _result(
        line_messages=[
            {
                "id": "line_1",
                "sent_at": "2024-12-24T17:30:00+09:00",
                "sender": "自分",
                "text": "18時に新宿着く！",
            },
            {
                "id": "line_2",
                "sent_at": "2024-12-24T17:32:00+09:00",
                "sender": "相手",
                "text": "東口で待ってるね",
            },
        ],
    )

    answer = build_answer("2024年12月24日は何していた？", result)

    assert "待ち合わせや移動に関する会話があった可能性があります" in answer
    assert "何をしていたか: 中" in answer
    assert "確実" not in answer


def test_image_content_can_raise_activity_confidence_when_supported() -> None:
    result = _result(
        media_items=[
            {
                "id": "photo_1",
                "captured_at": "2024-12-24T19:00:00+09:00",
                "caption": "カフェでご飯を食べている写真",
            }
        ],
        line_messages=[
            {
                "id": "line_1",
                "sent_at": "2024-12-24T19:05:00+09:00",
                "sender": "自分",
                "text": "ご飯おいしかった",
            }
        ],
    )

    confidence = calculate_confidence(result)

    assert confidence["activity"] == "高"


def test_empty_result_confidence_is_unknown() -> None:
    confidence = calculate_confidence(_result())

    assert confidence == {
        "date": "不明",
        "contact": "不明",
        "place": "不明",
        "activity": "不明",
    }


def _result(
    *,
    events=None,
    media_items=None,
    line_messages=None,
) -> TimelineSearchResult:
    return TimelineSearchResult(
        question="2024年12月24日は何していた？",
        date_range=None,
        keyword=None,
        events=events or [],
        media_items=media_items or [],
        line_messages=line_messages or [],
        timeline_items=[],
    )


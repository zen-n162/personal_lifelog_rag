from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.line.call_index import build_call_index
from personal_lifelog_rag.retrieval.query_router import route_query


def test_route_date_qa_to_existing_answer_builder(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_router_records(repository)

    result = route_query(repository, "2024年12月24日は何していた？")

    assert result.intent == "date_qa"
    assert result.routing == "ask"
    assert "2024年12月24日の記録を確認しました" in result.answer


def test_route_place_visit_to_search(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_router_records(repository)

    result = route_query(repository, "新宿に行ったのはいつ？")

    assert result.intent == "place_visit"
    assert result.routing == "search"
    assert result.results
    assert result.results[0]["classification"] == "actual_or_likely_action"


def test_route_call_activity_to_call_search_ranking(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_router_records(repository)
    build_call_index(repository)

    result = route_query(repository, "通話した日は？")

    assert result.intent == "call_activity"
    assert result.routing == "search"
    assert result.results[0]["call_summary"]["completed"] == 1


def test_route_unknown_returns_helpful_message(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()

    result = route_query(repository, "ふわっといい感じによろしく")

    assert result.intent == "unknown"
    assert result.routing == "unsupported"
    assert "対応しやすい聞き方" in result.answer


def test_route_monthly_summary_aggregates_period_records(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_monthly_summary_records(repository)

    result = route_query(repository, "2025年1月は何していた？")

    assert result.intent == "monthly_summary"
    assert result.routing == "monthly-summary"
    assert "2025-01の月次要約" in result.answer
    assert "イベント2件" in result.answer
    assert "写真2枚" in result.answer
    assert "画像解析のみの手がかりは「候補」" in result.answer
    assert "代表日 top5" in result.answer
    assert result.results[0]["events_count"] == 2
    assert result.results[0]["media"]["vlm_success_photos"] == 1
    assert result.results[0]["category_counts"]["food_cafe"] >= 1


def _seed_router_records(repository: LifelogRepository) -> None:
    repository.add_line_message(
        id="line_shinjuku",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T17:30:00+09:00",
        sender="自分",
        text="18時に新宿着く！",
    )
    repository.add_line_message(
        id="line_food",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T22:10:00+09:00",
        sender="自分",
        text="今日のご飯おいしかったね",
    )
    repository.add_line_message(
        id="line_call",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T23:00:00+09:00",
        sender="自分",
        text="☎ 通話時間 10:38",
    )
    repository.add_event(
        id="event_shinjuku",
        date="2024-12-24",
        start_time="17:30:00",
        end_time="18:30:00",
        title="移動・待ち合わせの可能性",
        summary="場所候補: 新宿。活動候補: 着く。",
        location_name="新宿",
        confidence=0.9,
    )


def _seed_monthly_summary_records(repository: LifelogRepository) -> None:
    repository.add_line_message(
        id="line_month_food",
        chat_id="chat_month",
        source_file="sample_chat.txt",
        sent_at="2025-01-05T12:00:00+09:00",
        sender="自分",
        text="今日はカフェでご飯の話をした",
    )
    repository.add_line_message(
        id="line_month_call",
        chat_id="chat_month",
        source_file="sample_chat.txt",
        sent_at="2025-01-20T20:00:00+09:00",
        sender="自分",
        text="☎ 通話時間 12:34",
    )
    repository.upsert_line_call_events(
        [
            {
                "message_id": "line_month_call",
                "chat_id": "chat_month",
                "sent_at": "2025-01-20T20:00:00+09:00",
                "sender": "自分",
                "call_status": "completed",
                "duration_sec": 754,
                "raw_text_short": "☎ 通話時間 12:34",
            }
        ]
    )
    repository.add_media_item(
        id="media_food",
        file_path="/local/photos/month_food.jpg",
        media_type="image",
        captured_at="2025-01-05T12:30:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
    )
    repository.add_media_item(
        id="media_stage",
        file_path="/local/photos/month_stage.jpg",
        media_type="image",
        captured_at="2025-01-20T19:30:00+09:00",
    )
    repository.upsert_media_vlm(
        media_id="media_food",
        caption="meal possible at a cafe",
        short_caption="meal possible",
        scene_tags=["cafe_possible"],
        activity_tags=["meal_possible"],
        food_cues=["meal_possible", "rice_possible"],
        vlm_engine="qwen3_vl_transformers",
        model_name="dummy",
        prompt_version="test",
        confidence=0.7,
        status="success",
    )
    repository.upsert_media_ocr(
        media_id="media_food",
        ocr_text="CAFE MENU",
        ocr_text_redacted="CAFE MENU",
        ocr_engine="fake",
        status="success",
    )
    repository.add_event(
        id="event_month_food",
        date="2025-01-05",
        start_time="12:00:00",
        end_time="13:00:00",
        title="食事・カフェの可能性",
        summary="画像解析による食事候補とLINE記録",
        confidence=0.66,
    )
    repository.add_event(
        id="event_month_call",
        date="2025-01-20",
        start_time="20:00:00",
        end_time="20:30:00",
        title="通話・連絡",
        summary="LINE通話ログ",
        confidence=0.7,
    )
    repository.add_event_evidence(event_id="event_month_food", evidence_type="line", evidence_id="line_month_food")
    repository.add_event_evidence(event_id="event_month_food", evidence_type="photo", evidence_id="media_food")
    repository.add_event_evidence(event_id="event_month_food", evidence_type="vlm", evidence_id="media_food")
    repository.add_event_evidence(event_id="event_month_food", evidence_type="ocr", evidence_id="media_food")
    repository.add_event_evidence(event_id="event_month_call", evidence_type="line", evidence_id="line_month_call")

from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import PrivateEvalQuestion, evaluate_private_questions


def test_private_eval_time_range_summary_case(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_range_summary_records(repository)

    report = evaluate_private_questions(
        repository,
        [
            PrivateEvalQuestion(
                id="monthly_202501",
                question="2025年1月は何していた？",
                case_type="time_range_summary",
                min_events=2,
                expected_top_dates=["2025-01-05"],
                expected_evidence_types=["line", "photo", "vlm", "ocr"],
                forbidden_claims=["確実に"],
            )
        ],
    )

    case = report["case_results"][0]
    assert case["status"] == "pass"
    assert case["routing"] == "monthly-summary"
    assert case["events_count"] == 2
    assert case["photo_count"] == 1
    assert "2025-01-05" in case["top_dates"]
    assert not case["missing_expected_evidence_types"]


def test_routed_qa_accepts_legacy_time_range_intent_for_monthly_summary(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_range_summary_records(repository)

    report = evaluate_private_questions(
        repository,
        [
            PrivateEvalQuestion(
                id="legacy_monthly_intent",
                question="2025年1月は何していた？",
                case_type="routed_qa",
                expected_intent="time_range_summary",
            )
        ],
    )

    case = report["case_results"][0]
    assert case["status"] == "pass"
    assert case["intent"] == "monthly_summary"


def _seed_range_summary_records(repository: LifelogRepository) -> None:
    repository.add_line_message(
        id="line_food",
        chat_id="chat_eval",
        source_file="sample_chat.txt",
        sent_at="2025-01-05T12:00:00+09:00",
        sender="自分",
        text="カフェでご飯",
    )
    repository.add_line_message(
        id="line_photo",
        chat_id="chat_eval",
        source_file="sample_chat.txt",
        sent_at="2025-01-20T18:00:00+09:00",
        sender="自分",
        text="写真を見返した",
    )
    repository.add_media_item(
        id="media_food",
        file_path="/local/photos/eval_food.jpg",
        media_type="image",
        captured_at="2025-01-05T12:30:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
    )
    repository.upsert_media_vlm(
        media_id="media_food",
        caption="meal possible",
        short_caption="meal possible",
        scene_tags=["cafe_possible"],
        activity_tags=["meal_possible"],
        food_cues=["meal_possible"],
        vlm_engine="qwen3_vl_transformers",
        model_name="dummy",
        prompt_version="test",
        confidence=0.7,
        status="success",
    )
    repository.upsert_media_ocr(
        media_id="media_food",
        ocr_text="MENU",
        ocr_text_redacted="MENU",
        ocr_engine="fake",
        status="success",
    )
    repository.add_event(
        id="event_food",
        date="2025-01-05",
        start_time="12:00:00",
        end_time="13:00:00",
        title="食事・カフェの可能性",
        summary="LINEと画像解析による候補",
        confidence=0.68,
    )
    repository.add_event(
        id="event_photo",
        date="2025-01-20",
        start_time="18:00:00",
        end_time="19:00:00",
        title="写真撮影の記録",
        summary="写真を見返した記録",
        confidence=0.55,
    )
    repository.add_event_evidence(event_id="event_food", evidence_type="line", evidence_id="line_food")
    repository.add_event_evidence(event_id="event_food", evidence_type="photo", evidence_id="media_food")
    repository.add_event_evidence(event_id="event_food", evidence_type="vlm", evidence_id="media_food")
    repository.add_event_evidence(event_id="event_food", evidence_type="ocr", evidence_id="media_food")
    repository.add_event_evidence(event_id="event_photo", evidence_type="line", evidence_id="line_photo")

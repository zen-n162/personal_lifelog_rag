from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import evaluate_private_questions, load_private_eval_questions


def test_private_eval_event_quality_and_place_assignment(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_event_quality(repository)
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """cases:
  - id: event_quality_001
    type: event_quality
    date: "2024-12-24"
    expected_min_events: 1
    expected_evidence_types:
      - "line"
      - "photo"
    max_line_only_low_value_events: 0
    min_photo_and_line_events: 1
    no_orphan_evidence: true

  - id: place_assignment_001
    type: place_assignment
    date: "2024-12-24"
    expected_any_location_name:
      - "新宿"
""",
        encoding="utf-8",
    )

    report = evaluate_private_questions(repository, load_private_eval_questions(questions_path))

    assert report["case_results"][0]["status"] == "pass"
    assert report["case_results"][1]["status"] == "pass"
    assert report["by_type"]["event_quality"]["passed"] == 1
    assert report["by_type"]["place_assignment"]["passed"] == 1


def test_private_eval_event_override(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_event_quality(repository)
    repository.upsert_event_override("event_quality", title_override="確認済みタイトル", is_verified=True)
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """cases:
  - id: event_override_001
    type: event_override
    event_id: "event_quality"
    expected_hidden: false
    expected_verified: true
    title_override_contains: "確認済み"
""",
        encoding="utf-8",
    )

    report = evaluate_private_questions(repository, load_private_eval_questions(questions_path))

    assert report["case_results"][0]["status"] == "pass"


def test_private_eval_unsupported_case_is_skipped(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """cases:
  - id: future_case
    type: future_eval
    question: "まだ未対応"
""",
        encoding="utf-8",
    )

    report = evaluate_private_questions(repository, load_private_eval_questions(questions_path))

    assert report["case_results"][0]["status"] == "skip"


def _seed_event_quality(repository: LifelogRepository) -> None:
    media_id = repository.add_media_item(
        id="media_quality",
        file_path="/local/photos/quality.jpg",
        file_name="quality.jpg",
        file_hash="hash-quality",
        captured_at="2024-12-24T18:00:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
    )
    line_id = repository.add_line_message(
        id="line_quality",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T17:30:00+09:00",
        sender="自分",
        text="18時に新宿着く！",
    )
    event_id = repository.add_event(
        id="event_quality",
        date="2024-12-24",
        start_time="17:30:00",
        end_time="18:30:00",
        title="移動・待ち合わせの可能性",
        summary="LINE 1件、写真 1枚から作成したイベント候補です。",
        location_name="新宿",
        confidence=0.8,
    )
    repository.add_event_evidence(event_id=event_id, evidence_type="line", evidence_id=line_id, weight=0.8)
    repository.add_event_evidence(event_id=event_id, evidence_type="photo", evidence_id=media_id, weight=0.8)

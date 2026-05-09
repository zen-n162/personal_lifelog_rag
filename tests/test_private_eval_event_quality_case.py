from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import evaluate_private_questions, load_private_eval_questions


def test_event_quality_detects_vlm_only_high_confidence(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_media_item(
        id="media_vlm_only",
        file_path="/local/private/vlm_only.jpg",
        file_name="vlm_only.jpg",
        file_hash="hash-vlm-only",
        captured_at="2024-12-24T12:00:00+09:00",
    )
    event_id = repository.add_event(
        id="event_vlm_only",
        date="2024-12-24",
        start_time="12:00:00",
        end_time="12:30:00",
        title="画像解析による候補",
        summary="VLM-only candidate",
        confidence=0.9,
    )
    repository.add_event_evidence(event_id=event_id, evidence_type="vlm", evidence_id="media_vlm_only", weight=0.2)
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """
cases:
  - id: event_quality_vlm_only
    type: event_quality
    date: "2024-12-24"
    expected_min_events: 1
    max_vlm_only_high_confidence_events: 0
""",
        encoding="utf-8",
    )

    report = evaluate_private_questions(repository, load_private_eval_questions(questions_path))

    case = report["case_results"][0]
    assert case["status"] == "fail"
    assert any("VLM-only high confidence" in issue for issue in case["issues"])


def test_event_quality_excludes_hidden_events(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    visible = repository.add_event(
        id="event_visible",
        date="2024-12-24",
        start_time="10:00:00",
        end_time="10:30:00",
        title="visible",
        summary="visible",
        confidence=0.5,
    )
    hidden = repository.add_event(
        id="event_hidden",
        date="2024-12-24",
        start_time="11:00:00",
        end_time="11:30:00",
        title="hidden",
        summary="hidden",
        confidence=0.5,
    )
    repository.upsert_event_override(hidden, is_hidden=True)
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """
cases:
  - id: event_quality_hidden
    type: event_quality
    date: "2024-12-24"
    expected_min_events: 1
    expected_max_events: 1
""",
        encoding="utf-8",
    )

    report = evaluate_private_questions(repository, load_private_eval_questions(questions_path))

    assert visible
    assert report["case_results"][0]["status"] == "pass"
    assert report["case_results"][0]["events_count"] == 1

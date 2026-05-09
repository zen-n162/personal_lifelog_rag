from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import evaluate_private_questions, load_private_eval_questions


def test_private_eval_event_rebuild_quality_detects_vlm_only_high_confidence(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_event(
        id="event_vlm_only",
        date="2024-12-24",
        start_time="12:00:00",
        end_time="12:10:00",
        title="食事・カフェの可能性",
        summary="画像解析による推定です。",
        confidence=0.9,
    )
    repository.add_media_item(
        id="media_vlm_eval",
        file_path="/local/vlm_eval.jpg",
        file_name="vlm_eval.jpg",
        file_hash="hash-vlm-eval",
        captured_at="2024-12-24T12:00:00+09:00",
    )
    repository.add_event_evidence(event_id="event_vlm_only", evidence_type="photo", evidence_id="media_vlm_eval")
    repository.add_event_evidence(event_id="event_vlm_only", evidence_type="vlm", evidence_id="media_vlm_eval")
    eval_path = tmp_path / "questions.yaml"
    eval_path.write_text(
        """
cases:
  - id: event_rebuild_vlm_001
    type: event_rebuild_quality
    date: "2024-12-24"
    expected_no_overclaim: true
    max_vlm_only_high_confidence_events: 0
    expected_evidence_types_any:
      - "ocr"
      - "vlm"
""",
        encoding="utf-8",
    )

    report = evaluate_private_questions(repository, load_private_eval_questions(eval_path))

    assert report["case_results"][0]["status"] == "fail"
    assert "VLM-only high confidence" in report["case_results"][0]["issues"][0]

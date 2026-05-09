from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.embeddings.engines import FakeEmbeddingEngine
from personal_lifelog_rag.embeddings.repository import MediaEmbeddingRepository
from personal_lifelog_rag.evaluation.private_eval import (
    evaluate_private_questions,
    load_private_eval_questions,
)


def test_private_eval_multimodal_search_case(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_eval_mm",
        file_path=str(tmp_path / "ramen.jpg"),
        file_name="ramen.jpg",
        file_hash="hash-eval-mm",
        media_type="image",
        captured_at="2024-12-24T19:00:00+09:00",
    )
    repository.upsert_media_vlm(
        media_id="media_eval_mm",
        caption="ご飯の可能性がある写真",
        short_caption="ご飯候補",
        food_cues=["meal_possible"],
        status="success",
        vlm_engine="unit_test_vlm",
        model_name="unit-test-vlm",
    )
    MediaEmbeddingRepository(db_path).upsert_embedding(
        media_id="media_eval_mm",
        embedding_type="image",
        embedding_model="unit-test-embedding",
        vector=FakeEmbeddingEngine().embed_text("ご飯").vector,
        status="success",
    )
    event_id = repository.add_event(
        id="event_eval_mm",
        date="2024-12-24",
        start_time="19:00:00",
        end_time="20:00:00",
        title="食事・カフェの可能性",
        summary="画像解析で食事候補",
        confidence=0.6,
    )
    repository.add_event_evidence(event_id=event_id, evidence_type="photo", evidence_id="media_eval_mm")
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """
cases:
  - id: mm_search_food_001
    type: multimodal_search
    query: "ご飯"
    expected_top_dates:
      - "2024-12-24"
    expected_evidence_types_any:
      - "embedding"
      - "vlm"
    max_vlm_only_confidence: "中"
    expected_strength_at_least: "medium"
    should_not_include:
      - "確実に食べた"
""",
        encoding="utf-8",
    )

    report = evaluate_private_questions(repository, load_private_eval_questions(questions_path))

    assert report["summary"]["passed"] == 1

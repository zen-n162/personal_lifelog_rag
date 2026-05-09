from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import evaluate_private_questions, load_private_eval_questions


def test_multimodal_search_private_eval_uses_vlm_sql_fallback(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_food_vlm(repository)
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """
cases:
  - id: mm_food
    type: multimodal_search
    query: "ご飯を食べた写真"
    expected_top_dates:
      - "2024-12-24"
    expected_min_results: 1
    expected_evidence_types_any:
      - "vlm"
    max_vlm_only_confidence: "中"
    should_not_include:
      - "確実に食べた"
""",
        encoding="utf-8",
    )

    report = evaluate_private_questions(repository, load_private_eval_questions(questions_path))

    assert report["summary"]["passed"] == 1
    case = report["case_results"][0]
    assert case["status"] == "pass"
    assert case["matched_dates"] == ["2024-12-24"]


def test_multimodal_search_private_eval_accepts_expected_top_dates_any(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_food_vlm(repository)
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """
cases:
  - id: mm_food_any
    type: multimodal_search
    query: "ご飯を食べた写真"
    expected_top_dates_any:
      - "2024-12-07"
      - "2024-12-24"
    expected_min_results: 1
""",
        encoding="utf-8",
    )

    report = evaluate_private_questions(repository, load_private_eval_questions(questions_path))

    case = report["case_results"][0]
    assert case["status"] == "pass"
    assert "2024-12-24" in case["matched_dates"]
    assert case["expected_top_dates_any"] == ["2024-12-07", "2024-12-24"]


def _seed_food_vlm(repository: LifelogRepository) -> None:
    repository.add_media_item(
        id="media_mm_private_eval_food",
        file_path="/local/private/food.jpg",
        file_name="food.jpg",
        file_hash="hash-mm-private-eval-food",
        captured_at="2024-12-24T15:53:00+09:00",
    )
    repository.upsert_media_vlm(
        media_id="media_mm_private_eval_food",
        caption="A meal possible photo",
        short_caption="meal possible",
        activity_tags=["meal_possible"],
        food_cues=["meal_possible", "rice_possible"],
        status="success",
        vlm_engine="qwen3_vl_transformers",
        confidence=0.6,
    )

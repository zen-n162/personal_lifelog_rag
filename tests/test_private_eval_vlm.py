from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import PrivateEvalQuestion, evaluate_private_questions


def test_private_eval_vlm_quality_and_image_search_cases(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_vlm_eval_record(repository)

    report = evaluate_private_questions(
        repository,
        [
            PrivateEvalQuestion(
                id="vlm_quality_dummy",
                question="2024-12-24",
                case_type="vlm_quality",
                date="2024-12-24",
                expected_min_vlm_success=1,
                forbidden_terms=["家族", "病気"],
            ),
            PrivateEvalQuestion(
                id="image_search_dummy",
                question="ラーメン",
                query="ラーメン",
                case_type="image_search",
                expected_dates=["2024-12-24"],
                expected_min_results=1,
                expected_evidence_types=["vlm"],
            ),
        ],
    )

    assert report["summary"]["passed"] == 2
    assert report["case_results"][0]["vlm_success_count"] == 1
    assert report["case_results"][1]["results_count"] == 1


def _seed_vlm_eval_record(repository: LifelogRepository) -> None:
    repository.add_media_item(
        id="media_vlm_eval",
        file_path="/local/photos/vlm_eval.jpg",
        file_name="vlm_eval.jpg",
        file_hash="hash-vlm-eval",
        media_type="image",
        captured_at="2024-12-24T18:30:00+09:00",
    )
    repository.upsert_media_vlm(
        media_id="media_vlm_eval",
        caption="ラーメンの可能性がある料理写真",
        short_caption="ラーメン写真の可能性",
        food_cues=["ramen_possible"],
        vlm_engine="unit_test_vlm",
        status="success",
        confidence=0.8,
        safety_flags=["low_confidence"],
    )

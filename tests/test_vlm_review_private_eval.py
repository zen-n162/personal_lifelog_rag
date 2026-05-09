from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import PrivateEvalQuestion, evaluate_private_questions


def test_private_eval_vlm_review_case(tmp_path: Path) -> None:
    repository = _seed(tmp_path)
    repository.upsert_media_vlm_override(
        media_id="media_eval_hidden",
        is_hidden=True,
        is_searchable=False,
        review_status="rejected",
    )

    report = evaluate_private_questions(
        repository,
        [
            PrivateEvalQuestion(
                id="vlm_review_hidden",
                question="media_eval_hidden",
                case_type="vlm_review",
                media_id="media_eval_hidden",
                expected_hidden=True,
                expected_searchable=False,
            )
        ],
    )

    assert report["case_results"][0]["status"] == "pass"


def test_private_eval_image_search_excludes_hidden_media(tmp_path: Path) -> None:
    repository = _seed(tmp_path)
    repository.upsert_media_vlm_override(
        media_id="media_eval_hidden",
        is_hidden=True,
        is_searchable=False,
        review_status="rejected",
    )

    report = evaluate_private_questions(
        repository,
        [
            PrivateEvalQuestion(
                id="image_search_hidden",
                question="ご飯",
                query="ご飯",
                case_type="image_search",
                should_exclude_media_ids=["media_eval_hidden"],
            )
        ],
    )

    assert report["case_results"][0]["status"] == "pass"


def _seed(tmp_path: Path) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    for media_id, hour in (("media_eval_visible", "10"), ("media_eval_hidden", "11")):
        repository.add_media_item(
            id=media_id,
            file_path=f"/local/photos/{media_id}.jpg",
            file_name=f"{media_id}.jpg",
            file_hash=f"hash-{media_id}",
            media_type="image",
            captured_at=f"2024-12-24T{hour}:00:00+09:00",
        )
        repository.upsert_media_vlm(
            media_id=media_id,
            caption="ご飯または料理の可能性がある写真",
            short_caption="ご飯候補",
            food_cues=["meal_possible"],
            vlm_engine="fake",
            status="success",
            confidence=0.7,
        )
    return repository

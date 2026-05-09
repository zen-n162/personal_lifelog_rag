from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.vlm.review_service import (
    VlmOverrideUpdate,
    VlmReviewFilters,
    get_effective_vlm_result,
    list_vlm_review_items,
    save_vlm_override,
)


def test_effective_vlm_result_uses_override_caption_and_tags(tmp_path: Path) -> None:
    repository = _seed(tmp_path)

    save_vlm_override(
        repository,
        VlmOverrideUpdate(
            media_id="media_vlm_service_review",
            caption_override="確認済みの料理写真候補",
            food_cues_override=["meal_possible"],
            review_status="accepted",
            is_verified=True,
        ),
    )
    effective = get_effective_vlm_result(repository, "media_vlm_service_review")

    assert effective is not None
    assert effective["caption"] == "確認済みの料理写真候補"
    assert "meal_possible" in effective["food_cues_json"]
    assert effective["review_status"] == "accepted"
    assert effective["is_verified"] == 1


def test_review_queue_filters_unreviewed_and_low_confidence(tmp_path: Path) -> None:
    repository = _seed(tmp_path)
    rows = list_vlm_review_items(
        repository,
        VlmReviewFilters(date="2024-12-24", unreviewed=True, low_confidence=0.6),
    )

    assert [row["media_id"] for row in rows] == ["media_vlm_service_review"]


def _seed(tmp_path: Path) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_media_item(
        id="media_vlm_service_review",
        file_path="/local/photos/review.jpg",
        file_name="review.jpg",
        file_hash="hash-vlm-service-review",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    repository.upsert_media_vlm(
        media_id="media_vlm_service_review",
        caption="カフェのような場所の可能性",
        short_caption="カフェ候補",
        food_cues=["cafe_possible"],
        vlm_engine="fake",
        status="success",
        confidence=0.4,
    )
    return repository

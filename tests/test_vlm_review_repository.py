from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository


def test_media_vlm_override_can_be_saved_and_read(tmp_path: Path) -> None:
    repository = _seed_vlm_review_db(tmp_path)

    repository.upsert_media_vlm_override(
        media_id="media_vlm_review",
        caption_override="修正済みcaption",
        food_cues_override=["meal_possible"],
        is_verified=True,
        review_status="accepted",
    )
    override = repository.get_media_vlm_override("media_vlm_review")
    row = repository.get_media_vlm("media_vlm_review")

    assert override is not None
    assert override["review_status"] == "accepted"
    assert override["is_verified"] == 1
    assert row["caption_override"] == "修正済みcaption"
    assert row["vlm_review_status"] == "accepted"


def _seed_vlm_review_db(tmp_path: Path) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_media_item(
        id="media_vlm_review",
        file_path="/local/photos/vlm_review.jpg",
        file_name="vlm_review.jpg",
        file_hash="hash-vlm-review",
        media_type="image",
        captured_at="2024-12-24T18:00:00+09:00",
    )
    repository.upsert_media_vlm(
        media_id="media_vlm_review",
        caption="ラーメンの可能性がある写真",
        short_caption="ラーメン写真の可能性",
        food_cues=["ramen_possible"],
        vlm_engine="fake",
        status="success",
        confidence=0.8,
    )
    return repository

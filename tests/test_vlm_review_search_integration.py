from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.embeddings.multimodal_search import multimodal_search
from personal_lifelog_rag.embeddings.repository import MediaEmbeddingRepository
from personal_lifelog_rag.embeddings.schemas import MultimodalSearchOptions
from personal_lifelog_rag.vlm.schemas import ImageSearchOptions
from personal_lifelog_rag.vlm.vlm_service import image_search


def test_hidden_and_not_searchable_vlm_result_is_excluded_from_image_search(tmp_path: Path) -> None:
    repository = _seed(tmp_path)
    repository.upsert_media_vlm_override(
        media_id="media_hidden_vlm",
        is_hidden=True,
        is_searchable=False,
        review_status="rejected",
    )

    report = image_search(repository, ImageSearchOptions(query="ご飯", limit=10))
    hidden_report = image_search(repository, ImageSearchOptions(query="ご飯", limit=10, include_hidden=True))

    assert [row["media_id"] for row in report["results"]] == ["media_visible_vlm"]
    assert {row["media_id"] for row in hidden_report["results"]} == {"media_visible_vlm", "media_hidden_vlm"}


def test_accepted_verified_vlm_result_gets_light_search_boost(tmp_path: Path) -> None:
    repository = _seed(tmp_path)
    repository.upsert_media_vlm_override(
        media_id="media_visible_vlm",
        is_verified=True,
        review_status="accepted",
    )
    report = image_search(repository, ImageSearchOptions(query="ご飯", limit=10))

    assert report["results"][0]["media_id"] == "media_visible_vlm"
    assert report["results"][0]["review_status"] == "accepted"


def test_hidden_vlm_result_is_excluded_from_multimodal_search(tmp_path: Path) -> None:
    repository = _seed(tmp_path)
    embedding_repository = MediaEmbeddingRepository(repository.db_path)
    embedding_repository.upsert_embedding(
        media_id="media_hidden_vlm",
        embedding_type="image",
        embedding_model="fake",
        vector=[0.9, 0.1, 0.1],
        source_text=None,
        status="success",
    )
    repository.upsert_media_vlm_override(
        media_id="media_hidden_vlm",
        is_hidden=True,
        is_searchable=False,
        review_status="rejected",
    )

    report = multimodal_search(repository, MultimodalSearchOptions(query="ご飯", backend="hybrid", limit=10))

    assert "media_hidden_vlm" not in {row["media_id"] for row in report["results"]}


def _seed(tmp_path: Path) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    for media_id, hour in (("media_visible_vlm", "10"), ("media_hidden_vlm", "11")):
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
            vlm_engine="unit_test_vlm",
            status="success",
            confidence=0.7,
        )
    return repository

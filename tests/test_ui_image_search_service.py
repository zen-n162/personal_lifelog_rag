from __future__ import annotations

from pathlib import Path

from PIL import Image

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.ui.image_search_service import image_search_for_ui, image_search_gallery_items


def test_image_search_for_ui_returns_rows_and_thumbnail_gallery(tmp_path: Path) -> None:
    repository = _seed_image_search_repository(tmp_path)

    payload = image_search_for_ui(repository, query="ご飯", limit=10)

    assert payload["rows"]
    assert payload["rows"][0][1] in {"media_ui_image_search", "media_ui_image_search_missing_thumb"}
    assert len(payload["gallery"]) == 1
    thumbnail_path, caption = payload["gallery"][0]
    assert Path(thumbnail_path).exists()
    assert "media_ui_image_search" in caption
    assert "Meal candidate" in caption


def test_image_search_gallery_items_are_sorted_by_date(tmp_path: Path) -> None:
    late = tmp_path / "late.jpg"
    early = tmp_path / "early.jpg"
    Image.new("RGB", (8, 8), color=(10, 10, 10)).save(late)
    Image.new("RGB", (8, 8), color=(250, 250, 250)).save(early)
    report = {
        "results": [
            {
                "media_id": "late",
                "date": "2025-01-03",
                "captured_at": "2025-01-03T12:00:00+09:00",
                "thumbnail_path": str(late),
                "caption": "late result",
            },
            {
                "media_id": "early",
                "date": "2025-01-01",
                "captured_at": "2025-01-01T12:00:00+09:00",
                "thumbnail_path": str(early),
                "caption": "early result",
            },
        ]
    }

    gallery = image_search_gallery_items(report)

    assert gallery[0][0] == str(early)
    assert gallery[1][0] == str(late)


def _seed_image_search_repository(tmp_path: Path) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()

    thumbnail_path = tmp_path / "thumb.jpg"
    Image.new("RGB", (8, 8), color=(240, 220, 180)).save(thumbnail_path)

    for media_id, thumb in (
        ("media_ui_image_search", thumbnail_path),
        ("media_ui_image_search_missing_thumb", tmp_path / "missing-thumb.jpg"),
    ):
        repository.add_media_item(
            id=media_id,
            file_path=str(tmp_path / f"{media_id}.jpg"),
            file_name=f"{media_id}.jpg",
            file_hash=f"hash-{media_id}",
            media_type="image",
            captured_at="2024-12-24T15:53:00+09:00",
            thumbnail_path=str(thumb),
        )
        repository.upsert_media_vlm(
            media_id=media_id,
            caption="Meal candidate on a table",
            short_caption="Meal candidate",
            food_cues=["meal_possible", "rice_possible"],
            vlm_engine="qwen3_vl_transformers",
            status="success",
        )

    return repository

from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.vlm.schemas import ImageSearchOptions
from personal_lifelog_rag.vlm.vlm_service import format_image_search, image_search


def test_image_search_returns_vlm_and_ocr_evidence(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_image_search_records(repository)

    report = image_search(repository, ImageSearchOptions(query="ラーメン", limit=5))
    output = format_image_search(report)

    assert report["total"] == 1
    assert report["results"][0]["media_id"] == "media_image_search"
    assert "vlm" in report["results"][0]["evidence_types"]
    assert "ラーメン" in output


def _seed_image_search_records(repository: LifelogRepository) -> None:
    repository.add_media_item(
        id="media_image_search",
        file_path="/local/photos/ramen.jpg",
        file_name="ramen.jpg",
        file_hash="hash-image-search",
        media_type="image",
        captured_at="2024-12-24T18:00:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
    )
    repository.upsert_media_ocr(
        media_id="media_image_search",
        ocr_text="メニュー",
        ocr_text_redacted="メニュー",
        status="success",
    )
    repository.upsert_media_vlm(
        media_id="media_image_search",
        caption="ラーメンの可能性がある料理写真",
        short_caption="ラーメン写真の可能性",
        food_cues=["ramen_possible"],
        vlm_engine="fake",
        status="success",
        confidence=0.8,
    )


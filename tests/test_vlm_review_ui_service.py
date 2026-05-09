from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.ui.vlm_review_service import (
    VlmReviewFilters,
    list_vlm_review_items,
    review_rows_for_dataframe,
)


def test_vlm_review_ui_service_returns_redacted_rows_without_gps(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_media_item(
        id="media_vlm_ui",
        file_path="/local/photos/ui.jpg",
        file_name="ui.jpg",
        file_hash="hash-vlm-ui",
        media_type="image",
        captured_at="2024-12-24T12:00:00+09:00",
        gps_lat=35.123456,
        gps_lon=139.123456,
    )
    repository.upsert_media_vlm(
        media_id="media_vlm_ui",
        caption="看板の文字が写っている可能性",
        short_caption="看板候補",
        location_cues=["station_sign_possible"],
        safety_flags=["low_confidence"],
        vlm_engine="fake",
        status="success",
        confidence=0.5,
    )

    items = list_vlm_review_items(repository, VlmReviewFilters(safety_flags=True))
    rows = review_rows_for_dataframe(items)

    assert rows[0][0] == "media_vlm_ui"
    assert "35.123456" not in str(rows)
    assert "low_confidence" in rows[0][6]

from __future__ import annotations

from pathlib import Path

from PIL import Image

from personal_lifelog_rag.ui.thumbnail_gallery import (
    media_thumbnail_gallery_items,
    sort_media_by_capture_date,
)


def test_media_thumbnail_gallery_items_are_sorted_by_capture_date(tmp_path: Path) -> None:
    late = tmp_path / "late.jpg"
    early = tmp_path / "early.jpg"
    Image.new("RGB", (8, 8), color=(20, 20, 20)).save(late)
    Image.new("RGB", (8, 8), color=(220, 220, 220)).save(early)

    rows = [
        {
            "media_id": "late",
            "captured_at": "2025-01-03T10:00:00+09:00",
            "thumbnail_path": str(late),
            "caption": "late photo",
        },
        {
            "media_id": "early",
            "captured_at": "2025-01-01T10:00:00+09:00",
            "thumbnail_path": str(early),
            "caption": "early photo",
        },
    ]

    gallery = media_thumbnail_gallery_items(rows)

    assert gallery[0][0] == str(early)
    assert "2025-01-01 / early" in gallery[0][1]
    assert gallery[1][0] == str(late)


def test_sort_media_by_capture_date_uses_fallback_date() -> None:
    rows = [
        {"id": "second", "fallback_captured_at": "2025-01-02T00:00:00+09:00"},
        {"id": "first", "date": "2025-01-01"},
    ]

    sorted_rows = sort_media_by_capture_date(rows)

    assert [row["id"] for row in sorted_rows] == ["first", "second"]

from __future__ import annotations

from pathlib import Path

from PIL import Image

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.vlm.pilot import select_vlm_pilot_images


def test_time_spread_selects_existing_images_and_excludes_hidden_only(tmp_path: Path) -> None:
    repository = _seed_pilot_repository(tmp_path)

    rows = select_vlm_pilot_images(repository, date="2024-12-24", limit=3, strategy="time_spread")
    ids = [row["media_id"] for row in rows]

    assert len(rows) == 3
    assert "media_hidden" not in ids
    assert any(row["has_event_evidence"] for row in rows)
    assert any(row["has_ocr"] for row in rows)
    assert any(row["has_gps"] for row in rows)


def test_event_evidence_strategy_prioritizes_event_photos(tmp_path: Path) -> None:
    repository = _seed_pilot_repository(tmp_path)

    rows = select_vlm_pilot_images(repository, date="2024-12-24", limit=2, strategy="event_evidence")

    assert rows[0]["media_id"] == "media_event"
    assert rows[0]["has_event_evidence"] is True


def test_ocr_first_strategy_prioritizes_ocr_images(tmp_path: Path) -> None:
    repository = _seed_pilot_repository(tmp_path)

    rows = select_vlm_pilot_images(repository, date="2024-12-24", limit=2, strategy="ocr_first")

    assert rows[0]["media_id"] == "media_ocr"
    assert rows[0]["has_ocr"] is True


def _seed_pilot_repository(tmp_path: Path) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    for index, media_id in enumerate(["media_plain", "media_event", "media_ocr", "media_gps", "media_hidden"], start=1):
        image_path = tmp_path / f"{media_id}.png"
        Image.new("RGB", (16, 16), "white").save(image_path)
        repository.add_media_item(
            id=media_id,
            file_path=str(image_path),
            file_name=image_path.name,
            file_hash=f"hash-{media_id}",
            media_type="image",
            captured_at=f"2024-12-24T{8 + index:02d}:00:00+09:00",
            gps_lat=35.0 if media_id == "media_gps" else None,
            gps_lon=139.0 if media_id == "media_gps" else None,
        )
    repository.add_event(
        id="event_visible",
        date="2024-12-24",
        start_time="10:00:00",
        end_time="10:30:00",
        title="写真撮影の記録",
        summary="dummy",
        confidence=0.7,
    )
    repository.add_event(
        id="event_hidden",
        date="2024-12-24",
        start_time="13:00:00",
        end_time="13:30:00",
        title="hidden",
        summary="dummy",
        confidence=0.3,
    )
    repository.upsert_event_override("event_hidden", is_hidden=True)
    repository.add_event_evidence(event_id="event_visible", evidence_type="photo", evidence_id="media_event")
    repository.add_event_evidence(event_id="event_hidden", evidence_type="photo", evidence_id="media_hidden")
    repository.upsert_media_ocr(
        media_id="media_ocr",
        ocr_text="新宿",
        ocr_text_redacted="新宿",
        ocr_engine="fake",
        status="success",
    )
    return repository

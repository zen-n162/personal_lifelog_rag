from __future__ import annotations

from pathlib import Path

from PIL import Image

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.ingest.photo_ingest import (
    ingest_photo_directory,
    ingest_photo_directory_with_report,
)


def test_ingest_dummy_photo_saves_media_item_and_thumbnail(tmp_path: Path) -> None:
    photos_dir = tmp_path / "photos"
    thumbnails_dir = tmp_path / "thumbnails"
    photos_dir.mkdir()
    image_path = photos_dir / "dummy.jpg"
    Image.new("RGB", (32, 24), color=(120, 80, 40)).save(image_path)

    repository = LifelogRepository(tmp_path / "lifelog.sqlite")

    inserted = ingest_photo_directory(
        photos_dir,
        repository,
        thumbnails_dir=thumbnails_dir,
    )

    assert inserted == 1
    rows = repository.list_media_items(limit=10)
    assert len(rows) == 1
    assert rows[0]["file_name"] == "dummy.jpg"
    assert rows[0]["file_hash"]
    assert rows[0]["width"] == 32
    assert rows[0]["height"] == 24
    assert rows[0]["thumbnail_path"]
    assert Path(rows[0]["thumbnail_path"]).exists()
    assert rows[0]["captured_at"] is None
    assert rows[0]["fallback_captured_at"]


def test_ingest_dummy_photo_deduplicates_by_hash(tmp_path: Path) -> None:
    photos_dir = tmp_path / "photos"
    thumbnails_dir = tmp_path / "thumbnails"
    photos_dir.mkdir()
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(photos_dir / "a.jpg")
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(photos_dir / "b.jpg")

    repository = LifelogRepository(tmp_path / "lifelog.sqlite")

    inserted = ingest_photo_directory(
        photos_dir,
        repository,
        thumbnails_dir=thumbnails_dir,
    )

    assert inserted == 1
    assert repository.stats()["media_items"] == 1


def test_ingest_exif_datetime_and_gps(tmp_path: Path) -> None:
    photos_dir = tmp_path / "photos"
    thumbnails_dir = tmp_path / "thumbnails"
    photos_dir.mkdir()
    image_path = photos_dir / "exif.jpg"
    image = Image.new("RGB", (20, 12), color=(1, 2, 3))
    exif = Image.Exif()
    exif[36867] = "2024:12:24 19:12:00"
    exif[272] = "Dummy Camera"
    exif[34853] = {
        1: "N",
        2: (35.0, 41.0, 22.0),
        3: "E",
        4: (139.0, 41.0, 30.0),
    }
    image.save(image_path, exif=exif)

    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    report = ingest_photo_directory_with_report(
        photos_dir,
        repository,
        thumbnails_dir=thumbnails_dir,
    )

    assert report.imported == 1
    assert report.skipped == 0
    row = repository.list_media_items(limit=10)[0]
    assert row["captured_at"] == "2024-12-24T19:12:00"
    assert row["camera_model"] == "Dummy Camera"
    assert row["gps_lat"] == 35.68944444444444
    assert row["gps_lon"] == 139.69166666666666


def test_corrupt_image_is_skipped_without_failing_batch(tmp_path: Path) -> None:
    photos_dir = tmp_path / "photos"
    thumbnails_dir = tmp_path / "thumbnails"
    photos_dir.mkdir()
    Image.new("RGB", (16, 16), color=(90, 80, 70)).save(photos_dir / "ok.png")
    (photos_dir / "broken.jpg").write_bytes(b"not an image")

    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    report = ingest_photo_directory_with_report(
        photos_dir,
        repository,
        thumbnails_dir=thumbnails_dir,
    )

    assert report.scanned == 2
    assert report.imported == 1
    assert report.skipped == 1
    assert repository.stats()["media_items"] == 1


def test_ingest_photos_cli_is_idempotent(tmp_path: Path, capsys) -> None:
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    Image.new("RGB", (12, 8), color=(30, 20, 10)).save(photos_dir / "cli.jpeg")
    db_path = tmp_path / "lifelog.sqlite"

    assert main(["--db-path", str(db_path), "ingest-photos", "--path", str(photos_dir)]) == 0
    first_output = capsys.readouterr().out
    assert "Imported media files: 1 new, 0 duplicate, 0 skipped, 1 file(s)" in first_output

    assert main(["--db-path", str(db_path), "ingest-photos", "--path", str(photos_dir)]) == 0
    second_output = capsys.readouterr().out
    assert "Imported media files: 0 new, 1 duplicate, 0 skipped, 1 file(s)" in second_output

    repository = LifelogRepository(db_path)
    assert repository.stats()["media_items"] == 1

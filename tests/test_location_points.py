from __future__ import annotations

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.places.location_store import build_location_points_from_media


def test_build_location_points_from_gps_media_and_deduplicates(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_media_item(
        id="media_gps",
        file_path="/local/fake/photo.jpg",
        file_name="photo.jpg",
        file_hash="hash-gps",
        captured_at="2025-01-02T10:00:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
    )
    repository.add_media_item(
        id="media_no_gps",
        file_path="/local/fake/no_gps.jpg",
        file_name="no_gps.jpg",
        file_hash="hash-no-gps",
        captured_at="2025-01-02T11:00:00+09:00",
    )

    dry = build_location_points_from_media(repository, start_date="2025-01-01", end_date="2025-01-31", dry_run=True)
    first = build_location_points_from_media(repository, start_date="2025-01-01", end_date="2025-01-31", dry_run=False)
    second = build_location_points_from_media(repository, start_date="2025-01-01", end_date="2025-01-31", dry_run=False)

    assert dry.would_create == 1
    assert first.created == 1
    assert second.created == 0
    assert second.updated == 1
    with connect(repository.db_path) as connection:
        rows = connection.execute("SELECT media_id, privacy_level FROM location_points").fetchall()
    assert [(row["media_id"], row["privacy_level"]) for row in rows] == [("media_gps", "exact_private")]


def test_build_location_points_cli_dry_run_does_not_write(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_cli_gps",
        file_path="/local/fake/photo.jpg",
        file_name="photo.jpg",
        file_hash="hash-cli-gps",
        captured_at="2025-01-02T10:00:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
    )

    code = main(["--db-path", str(db_path), "build-location-points", "--from", "2025-01-01", "--to", "2025-01-31", "--dry-run"])
    output = capsys.readouterr().out

    assert code == 0
    assert "Location Points" in output
    with connect(db_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM location_points").fetchone()[0]
    assert count == 0

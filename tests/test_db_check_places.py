from __future__ import annotations

from personal_lifelog_rag.db.checks import run_db_check
import sqlite3

from personal_lifelog_rag.db.repository import LifelogRepository, connect


def test_db_check_detects_invalid_location_point_lat_lon(tmp_path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO location_points (
                id, media_id, captured_at, source, lat, lon, privacy_level
            )
            VALUES ('lp_bad', NULL, '2025-01-01T10:00:00+09:00', 'exif', 999, 139, 'exact_private')
            """
        )
        connection.commit()

    report = run_db_check(db_path)

    assert report["location_places"]["location_points_invalid_lat_lon"] == 1
    assert not report["strict"]["ok"]


def test_db_check_detects_orphan_place_links(tmp_path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            """
            INSERT INTO event_places (event_id, place_id, source, confidence)
            VALUES ('missing_event', 'missing_place', 'gps_cluster', 0.5)
            """
        )
        connection.execute(
            """
            INSERT INTO media_places (media_id, place_id, source, confidence)
            VALUES ('missing_media', 'missing_place', 'gps_cluster', 0.5)
            """
        )
        connection.commit()

    report = run_db_check(db_path)

    assert report["location_places"]["event_places_orphan_event_refs"] == 1
    assert report["location_places"]["event_places_orphan_place_refs"] == 1
    assert report["location_places"]["media_places_orphan_media_refs"] == 1
    assert report["location_places"]["media_places_orphan_place_refs"] == 1
    assert not report["strict"]["ok"]

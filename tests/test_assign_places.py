from __future__ import annotations

import sqlite3

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.places.assignment import assign_places_to_events
from personal_lifelog_rag.places.schemas import Place


def test_assign_places_updates_event_location_name(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_event(
        id="event_with_gps",
        date="2024-12-24",
        start_time="10:00:00",
        title="GPS付きイベント",
        gps_lat=10.0002,
        gps_lon=20.0002,
        confidence=0.6,
    )

    report = assign_places_to_events(
        repository,
        places=[_place()],
        start_date="2024-12-24",
        end_date="2024-12-24",
    )
    event = repository.list_events(start_date="2024-12-24", end_date="2024-12-24")[0]

    assert report.matched == 1
    assert report.updated == 1
    assert event["location_name"] == "テスト場所"


def test_assign_places_dry_run_does_not_update_db(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_event(
        id="event_with_gps",
        date="2024-12-24",
        start_time="10:00:00",
        title="GPS付きイベント",
        gps_lat=10.0002,
        gps_lon=20.0002,
        confidence=0.6,
    )

    report = assign_places_to_events(
        repository,
        places=[_place()],
        start_date="2024-12-24",
        end_date="2024-12-24",
        dry_run=True,
    )
    event = repository.list_events(start_date="2024-12-24", end_date="2024-12-24")[0]

    assert report.matched == 1
    assert report.updated == 0
    assert event["location_name"] is None


def test_assign_places_does_not_overwrite_user_edited_event(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_event(
        id="event_user_edited",
        date="2024-12-24",
        start_time="10:00:00",
        title="手動編集イベント",
        location_name="手動の場所",
        gps_lat=10.0002,
        gps_lon=20.0002,
        confidence=0.6,
        is_user_edited=True,
    )

    report = assign_places_to_events(
        repository,
        places=[_place()],
        start_date="2024-12-24",
        end_date="2024-12-24",
    )
    event = repository.list_events(start_date="2024-12-24", end_date="2024-12-24")[0]

    assert report.skipped_user_edited == 1
    assert report.updated == 0
    assert event["location_name"] == "手動の場所"


def test_assign_places_does_not_overwrite_location_override(tmp_path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_event(
        id="event_overridden",
        date="2024-12-24",
        start_time="10:00:00",
        title="上書き保護イベント",
        location_name="既存の場所",
        gps_lat=10.0002,
        gps_lon=20.0002,
        confidence=0.6,
    )
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO event_overrides (event_id, location_name_override) VALUES (?, ?)",
            ("event_overridden", "手動優先の場所"),
        )

    report = assign_places_to_events(
        repository,
        places=[_place()],
        start_date="2024-12-24",
        end_date="2024-12-24",
    )
    event = repository.list_events(start_date="2024-12-24", end_date="2024-12-24")[0]

    assert report.skipped_overrides == 1
    assert report.updated == 0
    assert event["location_name"] == "手動優先の場所"
    assert event["original_location_name"] == "既存の場所"


def _place() -> Place:
    return Place(
        id="test_place",
        name="テスト場所",
        display_name="テスト場所",
        lat=10.0,
        lon=20.0,
        radius_m=500.0,
        category="test",
    )

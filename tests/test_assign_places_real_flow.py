from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.places.assignment import assign_places_to_events
from personal_lifelog_rag.places.schemas import Place


def test_assign_places_dry_run_does_not_change_events(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    event_id = _seed_event(repository, event_id="event_dry")

    report = assign_places_to_events(repository, places=[_place()], dry_run=True)
    event = repository.get_event(event_id, include_hidden=True)

    assert report.matched == 1
    assert report.updated == 0
    assert event is not None
    assert event.get("location_name") is None


def test_assign_places_updates_event_location_name(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    event_id = _seed_event(repository, event_id="event_assign")

    report = assign_places_to_events(repository, places=[_place()])
    event = repository.get_event(event_id, include_hidden=True)

    assert report.updated == 1
    assert event is not None
    assert event.get("location_name") == "候補地点001"


def test_assign_places_does_not_overwrite_location_override(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    event_id = _seed_event(repository, event_id="event_override")
    repository.upsert_event_override(event_id, location_name_override="手動の場所")

    report = assign_places_to_events(repository, places=[_place()])
    event = repository.get_event(event_id, include_hidden=True)

    assert report.skipped_overrides == 1
    assert report.updated == 0
    assert event is not None
    assert event.get("location_name") == "手動の場所"
    assert event.get("original_location_name") is None


def test_assign_places_includes_hidden_events_but_keeps_hidden_flag(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    event_id = _seed_event(repository, event_id="event_hidden")
    repository.upsert_event_override(event_id, is_hidden=True)

    report = assign_places_to_events(repository, places=[_place()])
    event = repository.get_event(event_id, include_hidden=True)

    assert report.updated == 1
    assert event is not None
    assert event.get("location_name") == "候補地点001"
    assert int(event.get("is_hidden") or 0) == 1


def _seed_event(repository: LifelogRepository, *, event_id: str) -> str:
    return repository.add_event(
        id=event_id,
        date="2024-12-24",
        start_time="10:00:00",
        end_time="10:30:00",
        title="位置情報付き写真の記録",
        summary="ダミーGPSだけを使うテスト用イベント",
        gps_lat=35.0,
        gps_lon=139.0,
        confidence=0.7,
    )


def _place() -> Place:
    return Place(
        id="candidate_place_001",
        name="candidate_place_001",
        display_name="候補地点001",
        lat=35.0,
        lon=139.0,
        radius_m=500,
        category="unknown",
        privacy_level="sensitive",
        show_exact_location=False,
    )

from __future__ import annotations

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.ui.event_review_service import bulk_update_events


def test_bulk_hidden_and_tag_updates_events(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_events(repository)

    report = bulk_update_events(repository, ["event_bulk_1", "event_bulk_2"], hidden=True, add_tags=["要確認"])

    assert report["updated_count"] == 2
    assert repository.get_event("event_bulk_1", include_hidden=True)["is_hidden"] == 1
    assert "要確認" in (repository.get_event("event_bulk_2", include_hidden=True)["tags_json"] or "")


def test_clear_overrides_removes_flags_and_tags(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_events(repository)
    repository.upsert_event_override("event_bulk_1", tags=["重要"], is_hidden=True, is_pinned=True)

    report = bulk_update_events(repository, ["event_bulk_1"], clear_overrides=True)
    event = repository.get_event("event_bulk_1", include_hidden=True)

    assert report["updated_count"] == 1
    assert event["is_hidden"] == 0
    assert event["is_pinned"] == 0
    assert event.get("tags_json") is None


def test_bulk_update_cli_unhide(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_events(repository)
    repository.upsert_event_override("event_bulk_1", is_hidden=True)

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "bulk-update-events",
            "--event-id",
            "event_bulk_1",
            "--unhide",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "updated: 1" in output
    assert repository.get_event("event_bulk_1", include_hidden=True)["is_hidden"] == 0


def _seed_events(repository: LifelogRepository) -> None:
    for index in (1, 2):
        repository.add_event(
            id=f"event_bulk_{index}",
            date="2024-12-24",
            start_time=f"1{index}:00:00",
            end_time=f"1{index}:30:00",
            title=f"bulk event {index}",
            summary="bulk test",
            confidence=0.5,
        )

from __future__ import annotations

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.timeline.event_builder import build_events


def test_build_events_for_date_range_is_idempotent_and_preserves_inputs(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_range_records(repository)
    before = repository.stats()

    first = build_events(repository, start_date="2024-12-24", end_date="2024-12-25")
    after_first = repository.stats()
    second = build_events(repository, start_date="2024-12-24", end_date="2024-12-25")
    after_second = repository.stats()

    assert first.days_scanned == 2
    assert first.events_created >= 2
    assert first.evidence_saved >= 3
    assert after_first["media_items"] == before["media_items"]
    assert after_first["line_messages"] == before["line_messages"]
    assert after_second["events"] == after_first["events"]
    assert after_second["event_evidence"] == after_first["event_evidence"]
    assert after_second["media_items"] == before["media_items"]
    assert after_second["line_messages"] == before["line_messages"]
    assert second.events_deleted == first.events_created


def test_build_events_dry_run_does_not_change_db(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_range_records(repository)
    before = repository.stats()

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "build-events",
            "--from",
            "2024-12-24",
            "--to",
            "2024-12-25",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out
    after = repository.stats()

    assert exit_code == 0
    assert "Dry-run events:" in output
    assert "Day summary:" in output
    assert after == before


def test_build_events_skip_existing_skips_days_with_events(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_range_records(repository)
    build_events(repository, start_date="2024-12-24")
    before = repository.stats()

    report = build_events(
        repository,
        start_date="2024-12-24",
        end_date="2024-12-25",
        skip_existing=True,
        force=False,
    )

    assert report.days_skipped == 1
    assert any(day["date"] == "2024-12-24" and day["action"] == "skipped_existing" for day in report.day_reports)
    assert repository.stats()["events"] >= before["events"]


def test_build_events_force_replaces_generated_events(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_range_records(repository)
    first = build_events(repository, start_date="2024-12-24")

    second = build_events(repository, start_date="2024-12-24", force=True)

    assert second.events_deleted == first.events_created
    assert repository.stats()["events"] == first.events_created


def test_build_events_limit_days_limits_all_mode(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_range_records(repository)

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "build-events",
            "--all",
            "--limit-days",
            "1",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "1 day(s)" in output


def _seed_range_records(repository: LifelogRepository) -> None:
    repository.add_line_message(
        id="line_20241224_1",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T17:30:00+09:00",
        sender="自分",
        text="18時に新宿着く",
    )
    repository.add_line_message(
        id="line_20241224_2",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T17:35:00+09:00",
        sender="相手",
        text="東口で待ってる",
    )
    repository.add_media_item(
        id="media_20241224_1",
        file_path="/local/photos/20241224_1.jpg",
        file_name="20241224_1.jpg",
        file_hash="hash-20241224-1",
        captured_at="2024-12-24T18:05:00+09:00",
        gps_lat=35.69,
        gps_lon=139.70,
    )
    repository.add_line_message(
        id="line_20241225_1",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-25T20:00:00+09:00",
        sender="自分",
        text="☎ 通話時間 1:00",
    )
    repository.add_media_item(
        id="media_20241225_1",
        file_path="/local/photos/20241225_1.jpg",
        file_name="20241225_1.jpg",
        file_hash="hash-20241225-1",
        captured_at="2024-12-25T13:00:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
    )


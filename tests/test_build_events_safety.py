from __future__ import annotations

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository


def test_build_events_cli_reports_input_table_safety(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_safety_records(repository)
    before = repository.stats()

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "build-events",
            "--all",
            "--limit-days",
            "1",
        ]
    )
    output = capsys.readouterr().out
    after = repository.stats()

    assert exit_code == 0
    assert "Input table safety:" in output
    assert "unchanged: True" in output
    assert after["media_items"] == before["media_items"]
    assert after["line_messages"] == before["line_messages"]


def test_build_events_cli_dry_run_preserves_all_tables(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_safety_records(repository)
    before = repository.stats()

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "build-events",
            "--all",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert repository.stats() == before


def test_build_events_cli_skip_existing_skips_existing_day(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_safety_records(repository)

    assert main(["--db-path", str(db_path), "build-events", "--all"]) == 0
    first_stats = repository.stats()
    assert main(["--db-path", str(db_path), "build-events", "--all", "--skip-existing"]) == 0
    output = capsys.readouterr().out

    assert "skipped_existing" in output
    assert repository.stats()["events"] == first_stats["events"]


def _seed_safety_records(repository: LifelogRepository) -> None:
    repository.add_line_message(
        id="line_safety_1",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T17:30:00+09:00",
        sender="自分",
        text="18時に新宿着く",
    )
    repository.add_media_item(
        id="media_safety_1",
        file_path="/local/photos/safety.jpg",
        file_name="safety.jpg",
        file_hash="hash-safety",
        captured_at="2024-12-24T18:00:00+09:00",
        gps_lat=10.0,
        gps_lon=20.0,
    )

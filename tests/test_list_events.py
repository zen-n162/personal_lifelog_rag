from __future__ import annotations

import json

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.timeline.event_builder import build_events
from personal_lifelog_rag.timeline.event_reports import format_event_list, list_events_report


def test_list_events_returns_events_for_date(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_list_records(repository)
    build_events(repository, start_date="2024-12-24")

    rows = list_events_report(repository, start_date="2024-12-24", end_date="2024-12-24")

    assert len(rows) == 1
    assert rows[0]["date"] == "2024-12-24"
    assert rows[0]["event_evidence_count"] == 3
    assert rows[0]["line_evidence_count"] == 2
    assert rows[0]["photo_evidence_count"] == 1


def test_list_events_with_evidence_limits_private_previews(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_list_records(repository)
    build_events(repository, start_date="2024-12-24")

    rows = list_events_report(
        repository,
        start_date="2024-12-24",
        end_date="2024-12-24",
        with_evidence=True,
    )
    output = format_event_list(rows, with_evidence=True)

    assert "line evidence:" in output
    assert "photo evidence:" in output
    assert "list_1.jpg" in output
    assert "これはとても長い本文" in output
    assert len(rows[0]["evidence"]["line"]) <= 5
    assert len(rows[0]["evidence"]["photo"]) <= 5


def test_list_events_cli_json_outputs_valid_json(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_list_records(repository)
    build_events(repository, start_date="2024-12-24")

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "list-events",
            "--date",
            "2024-12-24",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload[0]["date"] == "2024-12-24"
    assert payload[0]["event_evidence_count"] == 3


def _seed_list_records(repository: LifelogRepository) -> None:
    repository.add_line_message(
        id="line_list_1",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T17:30:00+09:00",
        sender="自分",
        text="これはとても長い本文です。新宿に着くので東口で待ってください。" * 3,
    )
    repository.add_line_message(
        id="line_list_2",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T17:40:00+09:00",
        sender="相手",
        text="待っています",
    )
    repository.add_media_item(
        id="media_list_1",
        file_path="/local/photos/list_1.jpg",
        file_name="list_1.jpg",
        file_hash="hash-list-1",
        captured_at="2024-12-24T18:00:00+09:00",
        gps_lat=35.69,
        gps_lon=139.70,
    )


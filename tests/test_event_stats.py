from __future__ import annotations

import json

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.timeline.event_builder import build_events
from personal_lifelog_rag.timeline.event_reports import event_stats, format_event_stats


def test_event_stats_returns_monthly_and_modality_counts(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_stats_records(repository)
    build_events(repository, start_date="2024-12-24", end_date="2025-01-02")

    report = event_stats(repository)

    assert report["total_events"] >= 2
    assert report["total_event_evidence"] >= 3
    assert report["monthly_event_counts"]["2024-12"] >= 1
    assert report["monthly_event_counts"]["2025-01"] >= 1
    assert "line" in report["evidence_type_counts"]
    assert "photo" in report["evidence_type_counts"]
    assert sum(report["confidence_buckets"].values()) == report["total_events"]
    assert report["modality_counts"]["photo_and_line"] >= 1


def test_event_stats_format_contains_required_sections(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_stats_records(repository)
    build_events(repository, start_date="2024-12-24", end_date="2024-12-24")

    output = format_event_stats(event_stats(repository, start_date="2024-12-01", end_date="2024-12-31"))

    assert "Event Stats" in output
    assert "Monthly event counts:" in output
    assert "Title counts:" in output
    assert "Confidence buckets:" in output
    assert "Evidence type counts:" in output


def test_event_stats_cli_json_outputs_valid_json(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_stats_records(repository)
    build_events(repository, start_date="2024-12-24")

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "event-stats",
            "--from",
            "2024-12-01",
            "--to",
            "2024-12-31",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["total_events"] >= 1
    assert payload["monthly_event_counts"]["2024-12"] >= 1


def _seed_stats_records(repository: LifelogRepository) -> None:
    repository.add_line_message(
        id="line_stats_1",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T18:00:00+09:00",
        sender="自分",
        text="新宿に着く",
    )
    repository.add_media_item(
        id="media_stats_1",
        file_path="/local/photos/stats_1.jpg",
        file_name="stats_1.jpg",
        file_hash="hash-stats-1",
        captured_at="2024-12-24T18:10:00+09:00",
        gps_lat=35.69,
        gps_lon=139.70,
    )
    repository.add_line_message(
        id="line_stats_2",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2025-01-02T10:00:00+09:00",
        sender="自分",
        text="☎ 通話時間 2:00",
    )


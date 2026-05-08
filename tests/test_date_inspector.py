from __future__ import annotations

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.retrieval.date_inspector import format_date_inspection, inspect_date


def test_inspect_date_returns_photo_and_line_counts(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_inspection_records(repository)

    inspection = inspect_date(repository, "2024-12-24")

    assert inspection.photo_count == 3
    assert inspection.gps_photo_count == 2
    assert inspection.line_message_count == 3
    assert inspection.event_count == 1
    assert inspection.event_evidence_count == 2
    assert inspection.time_range.first_photo_at == "08:15"
    assert inspection.time_range.last_photo_at == "22:10"
    assert inspection.time_range.first_line_at == "09:00"
    assert inspection.time_range.last_line_at == "21:30"
    assert inspection.photo_hourly_counts[8] == 1
    assert inspection.photo_hourly_counts[22] == 1
    assert inspection.line_hourly_counts[9] == 1
    assert inspection.line_hourly_counts[21] == 1
    assert inspection.gps_summary.lat_min == 35.123
    assert inspection.gps_summary.lat_max == 35.988


def test_inspect_date_handles_empty_date(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()

    inspection = inspect_date(repository, "2024-01-01")

    assert inspection.photo_count == 0
    assert inspection.line_message_count == 0
    assert inspection.event_count == 0
    assert inspection.event_evidence_count == 0
    assert inspection.photo_hourly_counts == [0] * 24
    assert inspection.line_hourly_counts == [0] * 24
    assert "GPS付き写真はありません" in format_date_inspection(inspection)


def test_inspect_date_limit_controls_samples(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_inspection_records(repository)

    inspection = inspect_date(repository, "2024-12-24", limit=2)

    assert len(inspection.line_samples) == 2
    assert len(inspection.photo_samples) == 2


def test_inspect_date_no_snippets_hides_samples(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_inspection_records(repository)

    inspection = inspect_date(repository, "2024-12-24", include_snippets=False)
    output = format_date_inspection(inspection)

    assert inspection.line_samples == []
    assert inspection.photo_samples == []
    assert "--no-snippets により非表示" in output
    assert "新宿で待ち合わせ" not in output
    assert "morning.jpg" not in output


def test_cli_inspect_date_prints_summary(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_inspection_records(repository)

    exit_code = main(["--db-path", str(db_path), "inspect-date", "2024-12-24", "--limit", "1"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Inspect date: 2024-12-24" in output
    assert "写真件数: 3" in output
    assert "LINEメッセージ件数: 3" in output
    assert output.count("新宿で待ち合わせ") == 1


def _seed_inspection_records(repository: LifelogRepository) -> None:
    repository.add_media_item(
        id="media_morning",
        file_path="/local/photos/morning.jpg",
        file_name="morning.jpg",
        file_hash="hash-morning",
        media_type="image",
        captured_at="2024-12-24T08:15:00+09:00",
        gps_lat=35.12345,
        gps_lon=139.12345,
        thumbnail_path="data/thumbnails/morning.jpg",
    )
    repository.add_media_item(
        id="media_noon",
        file_path="/local/photos/noon.jpg",
        file_name="noon.jpg",
        file_hash="hash-noon",
        media_type="image",
        fallback_captured_at="2024-12-24T12:05:00+09:00",
    )
    repository.add_media_item(
        id="media_night",
        file_path="/local/photos/night.jpg",
        file_name="night.jpg",
        file_hash="hash-night",
        media_type="image",
        captured_at="2024-12-24T22:10:00+09:00",
        gps_lat=35.98765,
        gps_lon=139.98765,
    )
    repository.add_media_item(
        id="media_other_day",
        file_path="/local/photos/other.jpg",
        file_name="other.jpg",
        file_hash="hash-other-day",
        media_type="image",
        captured_at="2024-12-25T08:00:00+09:00",
    )
    repository.add_line_message(
        id="line_morning",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T09:00:00+09:00",
        sender="自分",
        text="新宿で待ち合わせ",
    )
    repository.add_line_message(
        id="line_noon",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T12:30:00+09:00",
        sender="相手",
        text="ランチに行こう",
    )
    repository.add_line_message(
        id="line_night",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T21:30:00+09:00",
        sender="自分",
        text="今日のご飯おいしかったね",
    )
    repository.add_line_message(
        id="line_other_day",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-25T09:00:00+09:00",
        sender="自分",
        text="翌日のメッセージ",
    )
    event_id = repository.add_event(
        id="event_dummy",
        date="2024-12-24",
        start_time="09:00:00",
        title="新宿周辺の出来事",
        summary="ダミーイベント",
        confidence=0.7,
    )
    repository.add_event_evidence(
        event_id=event_id,
        evidence_type="line_message",
        evidence_id="line_morning",
        weight=0.8,
    )
    repository.add_event_evidence(
        event_id=event_id,
        evidence_type="media_item",
        evidence_id="media_morning",
        weight=0.8,
    )

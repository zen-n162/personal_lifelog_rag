from __future__ import annotations

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.retrieval.answer_builder import build_answer
from personal_lifelog_rag.retrieval.date_parser import parse_date_query
from personal_lifelog_rag.retrieval.temporal_search import search_timeline
from personal_lifelog_rag.timeline.event_builder import build_all_events, build_events


def test_build_events_persists_events_and_evidence(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_event_records(repository)

    report = build_events(repository, start_date="2024-12-24")

    events = repository.list_events(start_date="2024-12-24", end_date="2024-12-24")
    evidence = repository.list_event_evidence()
    assert report.events_created >= 1
    assert report.evidence_saved >= 3
    assert events
    assert evidence
    assert any(row["evidence_type"] == "line" for row in evidence)
    assert any(row["evidence_type"] == "photo" for row in evidence)
    assert any("待ち合わせ" in (event["title"] or "") or event["location_name"] == "新宿" for event in events)
    assert max(event["confidence"] or 0 for event in events) > 0.5


def test_build_events_generates_event_for_photo_only_day(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_photo_only_records(repository)

    report = build_events(repository, start_date="2024-12-25")

    events = repository.list_events(start_date="2024-12-25", end_date="2024-12-25")
    evidence = repository.list_event_evidence(events[0]["id"])
    assert report.events_created == 1
    assert events[0]["title"] == "位置情報付き写真の記録"
    assert {row["evidence_type"] for row in evidence} == {"photo"}


def test_build_events_generates_event_for_line_only_day(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_line_only_records(repository)

    report = build_events(repository, start_date="2024-12-26")

    events = repository.list_events(start_date="2024-12-26", end_date="2024-12-26")
    evidence = repository.list_event_evidence(events[0]["id"])
    assert report.events_created == 1
    assert events[0]["title"] == "通話・連絡"
    assert {row["evidence_type"] for row in evidence} == {"line"}


def test_build_events_merges_photo_and_line_when_times_are_close(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_event_records(repository)

    build_events(repository, start_date="2024-12-24")

    events = repository.list_events(start_date="2024-12-24", end_date="2024-12-24")
    evidence = repository.list_event_evidence(events[0]["id"])
    assert len(events) == 1
    assert {row["evidence_type"] for row in evidence} == {"line", "photo", "ocr", "vlm"}
    assert len(evidence) == 6


def test_build_events_is_idempotent_for_same_date(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_event_records(repository)

    first = build_events(repository, start_date="2024-12-24")
    second = build_events(repository, start_date="2024-12-24")

    assert first.events_created == second.events_created
    assert repository.stats()["events"] == first.events_created
    assert repository.stats()["event_evidence"] == first.evidence_saved
    assert second.events_deleted == first.events_created


def test_build_events_all_uses_record_dates(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_event_records(repository)
    _seed_line_only_records(repository)

    report = build_all_events(repository)

    assert report.days_scanned == 2
    assert repository.stats()["events"] == 2


def test_ask_prefers_events_when_they_exist(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_event_records(repository)
    build_events(repository, start_date="2024-12-24")

    result = search_timeline(
        repository,
        "2024年12月24日は何していた？",
        date_range=parse_date_query("2024年12月24日は何していた？"),
    )
    answer = build_answer("2024年12月24日は何していた？", result)

    assert "この日は1件の出来事候補があります" in answer
    assert "confidence" in answer
    assert "信頼度:" in answer


def test_build_events_cli_accepts_date(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_event_records(repository)

    assert main(["--db-path", str(db_path), "build-events", "--date", "2024-12-24"]) == 0
    output = capsys.readouterr().out

    assert "Built events:" in output
    assert repository.stats()["events"] >= 1
    assert repository.stats()["event_evidence"] >= 3


def _seed_event_records(repository: LifelogRepository) -> None:
    repository.add_line_message(
        id="line_wait_1",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T17:30:00+09:00",
        sender="自分",
        text="18時に新宿着く！",
    )
    repository.add_line_message(
        id="line_wait_2",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T17:32:00+09:00",
        sender="相手",
        text="じゃあ東口で待ってるね",
    )
    repository.add_media_item(
        id="media_shinjuku_1",
        file_path="/local/photos/shinjuku_1.jpg",
        file_name="shinjuku_1.jpg",
        file_hash="hash-shinjuku-1",
        media_type="image",
        captured_at="2024-12-24T18:05:00+09:00",
        gps_lat=35.6900,
        gps_lon=139.7000,
        caption="新宿駅前の夜景",
    )
    repository.add_media_item(
        id="media_shinjuku_2",
        file_path="/local/photos/shinjuku_2.jpg",
        file_name="shinjuku_2.jpg",
        file_hash="hash-shinjuku-2",
        media_type="image",
        captured_at="2024-12-24T18:20:00+09:00",
        gps_lat=35.6901,
        gps_lon=139.7002,
        ocr_text="東口",
    )


def _seed_photo_only_records(repository: LifelogRepository) -> None:
    for index, minute in enumerate((0, 10, 25), start=1):
        repository.add_media_item(
            id=f"media_photo_only_{index}",
            file_path=f"/local/photos/photo_only_{index}.jpg",
            file_name=f"photo_only_{index}.jpg",
            file_hash=f"hash-photo-only-{index}",
            media_type="image",
            captured_at=f"2024-12-25T13:{minute:02d}:00+09:00",
            gps_lat=35.0 + index * 0.0001,
            gps_lon=139.0 + index * 0.0001,
        )


def _seed_line_only_records(repository: LifelogRepository) -> None:
    for index, minute in enumerate((0, 10, 20), start=1):
        repository.add_line_message(
            id=f"line_call_{index}",
            chat_id="chat_dummy",
            source_file="sample_chat.txt",
            sent_at=f"2024-12-26T20:{minute:02d}:00+09:00",
            sender="自分",
            text=f"☎ 通話時間 {index}:00",
            message_type="text",
        )

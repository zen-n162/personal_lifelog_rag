from __future__ import annotations

import json

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.ui.event_review_service import ReviewQueueFilters, review_queue


def test_review_queue_returns_unverified_events(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_review_queue(repository)
    repository.upsert_event_override("event_verified", is_verified=True)

    report = review_queue(
        repository,
        ReviewQueueFilters(date_from="2024-12-24", date_to="2024-12-24", verified="unverified"),
    )

    assert {row["event_id"] for row in report["rows"]} == {"event_low_line", "event_photo"}


def test_review_queue_confidence_and_line_only_filters(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_review_queue(repository)

    report = review_queue(
        repository,
        ReviewQueueFilters(confidence_lte=0.5, modality="line_only"),
    )

    assert [row["event_id"] for row in report["rows"]] == ["event_low_line"]
    assert report["rows"][0]["modality"] == "line_only"


def test_review_queue_cli_json_outputs_valid_json(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_review_queue(repository)

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "review-queue",
            "--from",
            "2024-12-24",
            "--to",
            "2024-12-24",
            "--low-confidence",
            "0.5",
            "--line-only",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["rows"][0]["event_id"] == "event_low_line"


def _seed_review_queue(repository: LifelogRepository) -> None:
    repository.add_event(
        id="event_low_line",
        date="2024-12-24",
        start_time="09:00:00",
        end_time="09:10:00",
        title="LINEのやりとり",
        summary="短いLINEだけの記録",
        confidence=0.4,
    )
    repository.add_event(
        id="event_photo",
        date="2024-12-24",
        start_time="10:00:00",
        end_time="10:30:00",
        title="写真撮影の記録",
        summary="写真あり",
        confidence=0.7,
    )
    repository.add_event(
        id="event_verified",
        date="2024-12-24",
        start_time="11:00:00",
        end_time="11:30:00",
        title="移動・待ち合わせの可能性",
        summary="確認済み候補",
        confidence=0.8,
    )
    repository.add_line_message(
        id="line_low",
        chat_id="chat",
        source_file="sample.txt",
        sent_at="2024-12-24T09:00:00+09:00",
        sender="自分",
        text="短いLINE",
    )
    repository.add_media_item(
        id="media_photo",
        file_path="/local/photo.jpg",
        file_name="photo.jpg",
        file_hash="hash-review-queue-photo",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    repository.add_event_evidence(event_id="event_low_line", evidence_type="line", evidence_id="line_low")
    repository.add_event_evidence(event_id="event_photo", evidence_type="photo", evidence_id="media_photo")

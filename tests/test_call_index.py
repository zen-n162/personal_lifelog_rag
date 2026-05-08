from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.line.call_index import build_call_index


def test_build_call_index_creates_line_call_events(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_call_messages(repository)

    report = build_call_index(repository)

    assert report.call_events_found == 5
    assert report.call_events_saved == 5
    rows = repository.list_line_call_events(limit=10)
    assert len(rows) == 5
    assert {row["call_status"] for row in rows} >= {"completed", "missed", "unanswered", "canceled"}


def test_build_call_index_is_idempotent(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_call_messages(repository)

    build_call_index(repository)
    build_call_index(repository)

    assert repository.stats()["line_call_events"] == 5


def test_build_call_index_dry_run_does_not_write(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_call_messages(repository)

    report = build_call_index(repository, dry_run=True)

    assert report.call_events_found == 5
    assert report.call_events_saved == 0
    assert repository.stats()["line_call_events"] == 0


def test_build_call_index_force_replaces_range(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_call_messages(repository)
    build_call_index(repository)

    report = build_call_index(repository, start_date="2024-12-24", end_date="2024-12-24", force=True)

    assert report.deleted_existing == 5
    assert repository.stats()["line_call_events"] == 5


def _seed_call_messages(repository: LifelogRepository) -> None:
    rows = [
        ("call_done", "2024-12-24T10:00:00+09:00", "☎ 通話時間 10:38"),
        ("call_long", "2024-12-24T11:00:00+09:00", "☎ 通話時間 1:26:14"),
        ("call_missed", "2024-12-24T12:00:00+09:00", "☎ 不在着信"),
        ("call_unanswered", "2024-12-24T13:00:00+09:00", "☎ 通話に応答がありませんでした"),
        ("call_canceled", "2024-12-24T14:00:00+09:00", "☎ 通話をキャンセルしました"),
        ("not_call", "2024-12-24T15:00:00+09:00", "普通のメッセージ"),
    ]
    for message_id, sent_at, text in rows:
        repository.add_line_message(
            id=message_id,
            chat_id="chat_dummy",
            source_file="sample_chat.txt",
            sent_at=sent_at,
            sender="自分",
            text=text,
        )

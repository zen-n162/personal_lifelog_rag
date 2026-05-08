from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.line.call_index import build_call_index, call_stats, format_call_stats


def test_call_stats_counts_statuses_and_durations(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_messages(repository)
    build_call_index(repository)

    report = call_stats(repository)

    assert report["status_counts"]["completed"] == 2
    assert report["status_counts"]["missed"] == 1
    assert report["status_counts"]["unanswered"] == 1
    assert report["total_completed_duration_sec"] == 638 + 5174
    assert report["daily_completed_counts"] == {"2024-12-24": 2}
    assert report["monthly_completed_counts"] == {"2024-12": 2}
    assert report["longest_calls"][0]["duration_sec"] == 5174


def test_format_call_stats_is_privacy_conscious(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_messages(repository)
    build_call_index(repository)

    output = format_call_stats(call_stats(repository))

    assert "Call Stats" in output
    assert "total completed duration: 1:36:52" in output
    assert "通話時間" not in output


def _seed_messages(repository: LifelogRepository) -> None:
    for message_id, sent_at, text in [
        ("call_done", "2024-12-24T10:00:00+09:00", "☎ 通話時間 10:38"),
        ("call_long", "2024-12-24T11:00:00+09:00", "☎ 通話時間 1:26:14"),
        ("call_missed", "2024-12-24T12:00:00+09:00", "☎ 不在着信"),
        ("call_unanswered", "2024-12-24T13:00:00+09:00", "☎ 通話に応答がありませんでした"),
    ]:
        repository.add_line_message(
            id=message_id,
            chat_id="chat_dummy",
            source_file="sample_chat.txt",
            sent_at=sent_at,
            sender="自分",
            text=text,
        )

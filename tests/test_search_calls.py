from __future__ import annotations

import json

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.line.call_index import build_call_index, search_calls


def test_search_calls_completed_only(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_messages(repository)
    build_call_index(repository)

    report = search_calls(repository, statuses=["completed"], limit=10)

    assert {row["call_status"] for row in report["results"]} == {"completed"}
    assert report["results"][0]["duration_sec"] == 5174


def test_search_calls_min_duration_filter(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_messages(repository)
    build_call_index(repository)

    report = search_calls(repository, statuses=["completed"], min_duration_sec=600, limit=10)

    assert [row["message_id"] for row in report["results"]] == ["call_long", "call_done"]


def test_search_calls_cli_json(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_messages(repository)
    build_call_index(repository)

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "search-calls",
            "--completed",
            "--min-duration-sec",
            "600",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["filters"]["statuses"] == ["completed"]
    assert payload["results"][0]["message_id"] == "call_long"


def test_search_call_ranking_uses_completed_calls(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_messages(repository)
    build_call_index(repository)

    exit_code = main(["--db-path", str(db_path), "search", "通話", "--intent", "call_activity", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["results"][0]["date"] == "2024-12-24"
    assert payload["results"][0]["classification"] == "actual_or_likely_action"
    assert payload["results"][0]["call_summary"]["completed"] == 2


def _seed_messages(repository: LifelogRepository) -> None:
    for message_id, sent_at, text in [
        ("call_done", "2024-12-24T10:00:00+09:00", "☎ 通話時間 10:38"),
        ("call_long", "2024-12-24T11:00:00+09:00", "☎ 通話時間 1:26:14"),
        ("call_missed", "2024-12-25T12:00:00+09:00", "☎ 不在着信"),
        ("call_unanswered", "2024-12-26T13:00:00+09:00", "☎ 通話に応答がありませんでした"),
    ]:
        repository.add_line_message(
            id=message_id,
            chat_id="chat_dummy",
            source_file="sample_chat.txt",
            sent_at=sent_at,
            sender="自分",
            text=text,
        )

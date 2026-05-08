from __future__ import annotations

import json

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.search_snapshot import (
    SearchSnapshotOptions,
    build_search_snapshot,
    write_search_snapshot,
)


def test_search_snapshot_handles_multiple_queries_and_redacts_lines(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_search_snapshot_records(repository)

    snapshot = build_search_snapshot(
        repository,
        SearchSnapshotOptions(queries=["新宿", "ご飯"], limit=5),
    )

    assert len(snapshot["queries"]) == 2
    shinjuku = snapshot["queries"][0]["results"][0]
    assert shinjuku["date"] == "2024-12-24"
    assert shinjuku["counts"]["line"] == 1
    assert len(shinjuku["line_samples_redacted"]) == 1
    assert len(shinjuku["line_samples_redacted"][0]) < 120


def test_search_snapshot_cli_json_outputs_valid_json(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_search_snapshot_records(repository)

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "search-snapshot",
            "--query",
            "新宿",
            "--query",
            "ご飯",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert len(payload["queries"]) == 2
    assert payload["queries"][0]["query"] == "新宿"


def test_search_snapshot_cli_save_writes_eval_output(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_search_snapshot_records(repository)
    output_dir = tmp_path / "eval_outputs"

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "search-snapshot",
            "--query",
            "新宿",
            "--save",
            "--output-dir",
            str(output_dir),
        ]
    )
    output = capsys.readouterr().out
    files = list(output_dir.glob("search_snapshot_*.json"))

    assert exit_code == 0
    assert files
    assert "saved:" in output
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["queries"][0]["query"] == "新宿"


def test_write_search_snapshot_returns_path(tmp_path) -> None:
    path = write_search_snapshot(
        {"created_at": "2026-05-09T00:00:00", "limit": 5, "date_from": None, "date_to": None, "queries": []},
        output_dir=tmp_path / "eval_outputs",
    )

    assert path.exists()
    assert path.name.startswith("search_snapshot_")


def _seed_search_snapshot_records(repository: LifelogRepository) -> None:
    repository.add_line_message(
        id="line_search_1",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T17:30:00+09:00",
        sender="自分",
        text="18時に新宿着く！" + "これは長い本文です。" * 20,
    )
    repository.add_line_message(
        id="line_search_2",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-25T18:00:00+09:00",
        sender="自分",
        text="今日のご飯はおいしかった",
    )
    repository.add_event(
        id="event_search_1",
        date="2024-12-24",
        start_time="17:30:00",
        title="新宿周辺の記録",
        summary="新宿に関するダミーイベント",
        confidence=0.7,
    )


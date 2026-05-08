from __future__ import annotations

import json

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository


def test_search_cli_mode_actual_outputs_actual_section(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_cli_records(repository)

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "search",
            "新宿",
            "--intent",
            "place_visit",
            "--mode",
            "actual",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "実際に行動した可能性が高い日" in output
    assert "classification=actual_or_likely_action" in output
    assert "plan_or_candidate" not in output


def test_search_cli_json_includes_classification_and_score_components(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_cli_records(repository)

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "search",
            "新宿",
            "--intent",
            "place_visit",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["intent"] == "place_visit"
    assert payload["results"][0]["classification"] == "actual_or_likely_action"
    assert "score_components" in payload["results"][0]


def _seed_cli_records(repository: LifelogRepository) -> None:
    repository.add_line_message(
        id="line_cli_actual",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T17:30:00+09:00",
        sender="自分",
        text="新宿着いた！",
    )
    repository.add_line_message(
        id="line_cli_plan",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-25T17:30:00+09:00",
        sender="自分",
        text="新宿かどっか行くかも",
    )

from __future__ import annotations

import json

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.line.call_index import build_call_index


def test_classify_query_cli_json_outputs_valid_json(capsys) -> None:
    exit_code = main(["classify-query", "新宿に行ったのはいつ？", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["intent"] == "place_visit"
    assert payload["entities"]["place"] == "新宿"


def test_qa_cli_routes_date_qa(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_cli_records(repository)

    exit_code = main(["--db-path", str(db_path), "qa", "2024年12月24日は何していた？"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "意図: date_qa" in output
    assert "2024年12月24日の記録を確認しました" in output


def test_qa_cli_routes_place_visit(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_cli_records(repository)

    exit_code = main(["--db-path", str(db_path), "qa", "新宿に行ったのはいつ？"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "意図: place_visit" in output
    assert "classification=actual_or_likely_action" in output


def test_qa_cli_routes_call_activity_json(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_cli_records(repository)
    build_call_index(repository)

    exit_code = main(["--db-path", str(db_path), "qa", "通話した日は？", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["intent"] == "call_activity"
    assert payload["routing"] == "search"
    assert payload["results"][0]["call_summary"]["completed"] == 1


def _seed_cli_records(repository: LifelogRepository) -> None:
    repository.add_line_message(
        id="line_cli_shinjuku",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T17:30:00+09:00",
        sender="自分",
        text="18時に新宿着く！",
    )
    repository.add_line_message(
        id="line_cli_call",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T23:00:00+09:00",
        sender="自分",
        text="☎ 通話時間 10:38",
    )
    repository.add_event(
        id="event_cli_shinjuku",
        date="2024-12-24",
        start_time="17:30:00",
        end_time="18:30:00",
        title="移動・待ち合わせの可能性",
        summary="場所候補: 新宿。活動候補: 着く。",
        location_name="新宿",
        confidence=0.9,
    )

from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.app.cli import main


def test_init_db_and_stats_use_explicit_temp_db(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"

    assert main(["--db-path", str(db_path), "init-db"]) == 0
    init_output = capsys.readouterr().out
    assert str(db_path) in init_output
    assert db_path.exists()

    assert main(["--db-path", str(db_path), "stats"]) == 0
    stats_output = capsys.readouterr().out
    assert "media_items: 0" in stats_output
    assert "line_messages: 0" in stats_output
    assert "events: 0" in stats_output
    assert "event_evidence: 0" in stats_output


def test_db_path_can_be_passed_after_subcommand(tmp_path: Path) -> None:
    db_path = tmp_path / "after-subcommand.sqlite"

    assert main(["init-db", "--db-path", str(db_path)]) == 0

    assert db_path.exists()


def test_ingest_line_and_ask_use_dummy_fixture(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    fixture = Path("tests/fixtures/line/sample_chat.txt")

    assert main(["--db-path", str(db_path), "ingest-line", str(fixture)]) == 0
    ingest_output = capsys.readouterr().out
    assert "Imported LINE messages: 9 new, 0 duplicate, 2 warning(s), 1 file(s)" in ingest_output

    assert main(["--db-path", str(db_path), "ask", "2024年12月24日は何していた？"]) == 0
    answer = capsys.readouterr().out
    assert "2024年12月24日の記録を確認しました" in answer
    assert "LINEメッセージ" in answer
    assert "新宿" in answer
    assert "信頼度:" in answer


def test_ask_empty_date_reports_no_records(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"

    assert main(["--db-path", str(db_path), "init-db"]) == 0
    capsys.readouterr()
    assert main(["--db-path", str(db_path), "ask", "2024年1月1日は何していた？"]) == 0
    answer = capsys.readouterr().out

    assert "記録が見つかりませんでした" in answer

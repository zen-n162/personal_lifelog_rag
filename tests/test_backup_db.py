from __future__ import annotations

from datetime import datetime
import json

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.backup import backup_sqlite_db
from personal_lifelog_rag.db.repository import LifelogRepository


def test_backup_db_copies_sqlite_with_label_and_timestamp(tmp_path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_line_message(
        id="line_backup",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T10:00:00+09:00",
        sender="自分",
        text="バックアップテスト",
    )

    result = backup_sqlite_db(
        db_path,
        label="before_build_all",
        output_dir=tmp_path / "backups",
        now=datetime(2026, 5, 9, 12, 34, 56),
    )

    assert result.backup_path.exists()
    assert result.backup_path.name == "lifelog_before_build_all_20260509_123456.sqlite"
    assert result.size_bytes == result.backup_path.stat().st_size
    assert result.size_bytes > 0


def test_backup_db_cli_outputs_backup_path(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "backup-db",
            "--label",
            "test_label",
            "--output-dir",
            str(tmp_path / "backups"),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Backed up database:" in output
    assert "lifelog_test_label_" in output


def test_backup_db_cli_errors_when_db_missing(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--db-path",
            str(tmp_path / "missing.sqlite"),
            "backup-db",
            "--output-dir",
            str(tmp_path / "backups"),
        ]
    )
    error = capsys.readouterr().err

    assert exit_code == 1
    assert "SQLite database not found" in error


def test_backup_result_is_json_serializable(tmp_path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    LifelogRepository(db_path).initialize()

    result = backup_sqlite_db(db_path, output_dir=tmp_path / "backups")

    assert json.loads(json.dumps(result.to_dict()))["size_bytes"] > 0


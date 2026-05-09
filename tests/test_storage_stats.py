from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.jobs.storage import storage_stats


def test_storage_stats_returns_size_information(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    LifelogRepository(db_path).initialize()

    report = storage_stats(db_path)

    assert report["db_size_bytes"] > 0
    assert "media_items" in report["counts"]
    assert "eval_outputs" in report["directories"]


def test_storage_stats_cli_outputs_summary(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    LifelogRepository(db_path).initialize()

    exit_code = main(["--db-path", str(db_path), "storage-stats"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Storage Stats" in output


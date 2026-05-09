from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository


def test_generate_report_cli_writes_markdown_and_json(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    LifelogRepository(db_path).initialize()
    output = tmp_path / "reports" / "report.md"

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "generate-report",
            "--public",
            "--no-examples",
            "--save-json",
            "--output",
            str(output),
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert output.exists()
    assert output.with_suffix(".json").exists()
    assert "Generated report:" in stdout


def test_reports_is_gitignored() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    assert "reports/" in gitignore


from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.reporting.report_builder import build_report, write_report
from personal_lifelog_rag.reporting.schemas import ReportOptions


def test_report_builder_generates_markdown_and_json(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()

    report = build_report(repository, ReportOptions(mode="public"))
    result = write_report(report, output_path=tmp_path / "report.md", save_json=True)

    assert result.markdown_path.exists()
    assert result.json_path is not None
    assert result.json_path.exists()
    text = result.markdown_path.read_text(encoding="utf-8")
    assert "# Personal LifeLog RAG Evaluation Report" in text
    assert "## 3. Privacy and Safety Design" in text


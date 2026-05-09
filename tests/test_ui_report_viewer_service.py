from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.ui.report_viewer_service import (
    generate_report_for_ui,
    list_reports_for_ui,
    load_report_for_ui,
)


def test_report_viewer_lists_loads_and_generates_reports(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    existing = reports_dir / "existing.md"
    existing.write_text("# Existing\n", encoding="utf-8")
    existing.with_suffix(".json").write_text('{"dataset_summary": {"events_count": 0}}', encoding="utf-8")
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()

    reports = list_reports_for_ui(reports_dir)
    loaded = load_report_for_ui(reports[0])
    generated = generate_report_for_ui(repository, mode="public", save_json=True, reports_dir=reports_dir)

    assert reports[0].endswith("existing.md")
    assert "# Existing" in loaded["markdown"]
    assert "db_summary" in loaded["json_summary"]
    assert generated["markdown_path"].endswith(".md")
    assert generated["json_path"].endswith(".json")
    assert "Personal LifeLog RAG Evaluation Report" in generated["markdown"]

"""Build and write privacy-preserving Markdown reports."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from personal_lifelog_rag.reporting.collectors import collect_report_data
from personal_lifelog_rag.reporting.markdown import render_markdown_report
from personal_lifelog_rag.reporting.schemas import ReportOptions, ReportWriteResult


DEFAULT_REPORTS_DIR = Path("reports")


def build_report(repository, options: ReportOptions) -> dict[str, Any]:
    data = collect_report_data(repository, options)
    markdown = render_markdown_report(data)
    return {"data": data, "markdown": markdown}


def write_report(
    report: dict[str, Any],
    *,
    output_path: str | Path | None = None,
    save_json: bool = False,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> ReportWriteResult:
    if output_path:
        markdown_path = Path(output_path).expanduser()
    else:
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        markdown_path = reports_dir / f"lifelog_rag_eval_{timestamp}.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(str(report["markdown"]), encoding="utf-8")
    json_path = None
    if save_json:
        json_path = markdown_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(report["data"], ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    return ReportWriteResult(markdown_path=markdown_path, json_path=json_path)

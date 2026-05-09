"""Report viewer helpers for the local Gradio UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from personal_lifelog_rag.reporting.report_builder import DEFAULT_REPORTS_DIR, build_report, write_report
from personal_lifelog_rag.reporting.schemas import ReportOptions


def list_reports_for_ui(reports_dir: str | Path = DEFAULT_REPORTS_DIR) -> list[str]:
    root = Path(reports_dir).expanduser()
    if not root.exists():
        return []
    files = sorted(
        [path for path in root.glob("*.md") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return [str(path) for path in files]


def load_report_for_ui(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"markdown": "", "json_summary": "", "path": ""}
    source = Path(path).expanduser()
    if not source.exists() or source.suffix.lower() != ".md":
        return {"markdown": "", "json_summary": "", "path": str(source)}
    markdown = source.read_text(encoding="utf-8")
    json_path = source.with_suffix(".json")
    json_summary = ""
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            json_summary = json.dumps(_compact_json_summary(payload), ensure_ascii=False, sort_keys=True, indent=2)
        except Exception as exc:
            json_summary = f"JSON summary could not be loaded: {exc.__class__.__name__}"
    return {"markdown": markdown, "json_summary": json_summary, "path": str(source)}


def generate_report_for_ui(
    repository,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    mode: str = "public",
    include_examples: bool = False,
    save_json: bool = True,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> dict[str, Any]:
    options = ReportOptions(
        start_date=start_date,
        end_date=end_date,
        mode="private" if mode == "private" else "public",
        include_examples=include_examples,
        save_json=save_json,
    )
    report = build_report(repository, options)
    result = write_report(report, save_json=save_json, reports_dir=Path(reports_dir))
    loaded = load_report_for_ui(result.markdown_path)
    loaded["markdown_path"] = str(result.markdown_path)
    loaded["json_path"] = str(result.json_path) if result.json_path else ""
    return loaded


def _compact_json_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "options": payload.get("options"),
        "db_summary": payload.get("db_summary") or payload.get("dataset_summary"),
        "event_stats": payload.get("event_stats", {}).get("summary")
        if isinstance(payload.get("event_stats"), dict)
        else None,
        "private_eval": payload.get("private_eval", {}).get("summary")
        if isinstance(payload.get("private_eval"), dict)
        else None,
    }

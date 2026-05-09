"""Formatting helpers for analysis job CLI commands."""

from __future__ import annotations

import json
from typing import Any


def format_analysis_plan(plan: dict[str, Any]) -> str:
    lines = [
        "Analysis Plan",
        f"- type: {plan['job_type']}",
        f"- target candidates: {plan['total_candidates']}",
        f"- already success: {plan['already_success']}",
        f"- failed: {plan['failed']}",
        f"- engine_unavailable: {plan['engine_unavailable']}",
        f"- version changed: {plan['version_changed']}",
        f"- selected for run: {plan['selected_count']}",
        f"- estimated storage increase: {_format_bytes(plan['estimated_storage_bytes'])}",
        f"- estimated processing sec: {plan.get('estimated_processing_sec')}",
        f"- example: {plan.get('command_example') or ''}",
    ]
    sample = plan.get("items") or []
    if sample:
        lines.append("sample items:")
        for item in sample[:10]:
            lines.append(
                f"- {item.get('item_id')} status={item.get('existing_status') or 'none'} "
                f"version_changed={item.get('needs_version_update')}"
            )
    return "\n".join(lines)


def format_analysis_run_report(report: dict[str, Any]) -> str:
    if report.get("dry_run"):
        return "\n".join(["Analysis Run (dry-run)", format_analysis_plan(report["plan"])])
    job = report.get("job") or {}
    lines = [
        "Analysis Run",
        f"- job_id: {report.get('job_id')}",
        f"- type: {report.get('job_type')}",
        f"- status: {report.get('status')}",
        f"- total: {job.get('total_items')}",
        f"- processed: {job.get('processed_items')}",
        f"- success: {job.get('success_items')}",
        f"- failed: {job.get('failed_items')}",
        f"- skipped: {job.get('skipped_items')}",
    ]
    paths = report.get("report_paths") or {}
    if paths:
        lines.append("reports:")
        for key, value in paths.items():
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def format_analysis_status(payload: dict[str, Any]) -> str:
    if payload.get("job"):
        jobs = [payload["job"]]
    else:
        jobs = payload.get("jobs") or []
    if not jobs:
        return "Analysis jobs: none"
    lines = ["Analysis jobs"]
    for job in jobs:
        lines.extend(
            [
                "",
                f"- job_id: {job.get('job_id')}",
                f"  type: {job.get('job_type')}",
                f"  status: {job.get('status')}",
                f"  total/processed/success/failed/skipped: "
                f"{job.get('total_items')}/{job.get('processed_items')}/{job.get('success_items')}/"
                f"{job.get('failed_items')}/{job.get('skipped_items')}",
                f"  created_at: {job.get('created_at')}",
            ]
        )
    if payload.get("items"):
        lines.append("")
        lines.append("items:")
        for item in payload["items"][:20]:
            lines.append(f"- {item.get('item_id')}: {item.get('status')}")
    return "\n".join(lines)


def format_analysis_cleanup(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Analysis cleanup",
            f"- dry_run: {report['dry_run']}",
            f"- requires --yes: {report['requires_yes']}",
            f"- jobs targeted: {report['job_count']}",
            f"- items targeted: {report['item_count']}",
        ]
    )


def dumps_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"

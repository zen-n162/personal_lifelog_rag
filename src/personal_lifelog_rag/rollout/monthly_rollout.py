"""Safe month-by-month rollout planning for VLM, embeddings, rebuilds, and reports."""

from __future__ import annotations

import calendar
from contextlib import closing
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import re
from typing import Any

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.db.schema import initialize_schema

DEFAULT_MONTH_LIMIT = 300
DEFAULT_CONFIG_PATH = Path("private_config/model_runtime.yaml")
DEFAULT_EVAL_OUTPUTS_DIR = Path("eval_outputs")
DEFAULT_PRIVATE_EVAL_DIR = Path("private_eval")
DEFAULT_REPORTS_DIR = Path("reports")


@dataclass(frozen=True)
class MonthRange:
    month: str
    start_date: str
    end_date: str


def parse_month(month: str) -> MonthRange:
    """Validate YYYY-MM and return inclusive date bounds."""
    raw = (month or "").strip()
    if not re.fullmatch(r"\d{4}-\d{2}", raw):
        raise ValueError("month must be in YYYY-MM format")
    year_text, month_text = raw.split("-", 1)
    year = int(year_text)
    month_number = int(month_text)
    if month_number < 1 or month_number > 12:
        raise ValueError("month must be in YYYY-MM format with month 01..12")
    last_day = calendar.monthrange(year, month_number)[1]
    return MonthRange(raw, date(year, month_number, 1).isoformat(), date(year, month_number, last_day).isoformat())


def month_plan(
    repository: LifelogRepository,
    *,
    month: str,
    limit: int = DEFAULT_MONTH_LIMIT,
    config_path: Path | None = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Collect month-level counts and recommended safe rollout commands."""
    rng = parse_month(month)
    stats = _month_stats(repository, rng)
    recommendations = _recommendations(rng, stats, limit=limit, config_path=config_path)
    return {
        "month": rng.month,
        "start_date": rng.start_date,
        "end_date": rng.end_date,
        "counts": stats,
        "recommended_limits": recommendations["limits"],
        "recommended_commands": recommendations["commands"],
        "notes": [
            "Run month-run --dry-run first.",
            "Create a DB backup before non-dry-run execution.",
            "Use --yes for real execution so accidental full-month processing is harder.",
        ],
    }


def month_status(
    repository: LifelogRepository,
    *,
    month: str,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    eval_outputs_dir: Path = DEFAULT_EVAL_OUTPUTS_DIR,
    private_eval_dir: Path = DEFAULT_PRIVATE_EVAL_DIR,
) -> dict[str, Any]:
    """Return current rollout status for one calendar month."""
    rng = parse_month(month)
    stats = _month_stats(repository, rng)
    report_paths = _matching_artifact_paths(reports_dir, rng.month, suffixes={".md", ".json"})
    eval_paths = [
        *_matching_artifact_paths(eval_outputs_dir, rng.month, suffixes={".json"}),
        *_matching_artifact_paths(private_eval_dir / "runs", rng.month, suffixes={".json"}),
    ]
    status = {
        "month": rng.month,
        "start_date": rng.start_date,
        "end_date": rng.end_date,
        "vlm": stats["media_vlm"],
        "embeddings": stats["media_embeddings"],
        "ocr": stats["media_ocr"],
        "events_count": stats["events_count"],
        "report_exists": bool(report_paths),
        "report_paths": [str(path) for path in report_paths[:5]],
        "eval_run_exists": bool(eval_paths),
        "eval_run_paths": [str(path) for path in eval_paths[:5]],
        "db_check_hint": "Run `python -m personal_lifelog_rag.app.cli db-check --strict` for full validation.",
    }
    return status


def month_batch_plan(
    repository: LifelogRepository,
    *,
    from_month: str,
    to_month: str,
    limit: int = DEFAULT_MONTH_LIMIT,
    config_path: Path | None = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Build dry-run plans for a sequence of months."""
    start = parse_month(from_month)
    end = parse_month(to_month)
    if start.month > end.month:
        raise ValueError("from-month must be before or equal to to-month")
    months = []
    year, month_number = (int(part) for part in start.month.split("-", 1))
    while f"{year:04d}-{month_number:02d}" <= end.month:
        months.append(month_plan(repository, month=f"{year:04d}-{month_number:02d}", limit=limit, config_path=config_path))
        month_number += 1
        if month_number == 13:
            year += 1
            month_number = 1
    return {
        "from_month": start.month,
        "to_month": end.month,
        "dry_run_only": True,
        "months": months,
    }


def month_run_plan(
    repository: LifelogRepository,
    *,
    month: str,
    limit: int = DEFAULT_MONTH_LIMIT,
    vlm_limit: int | None = None,
    embedding_limit: int | None = None,
    config_path: Path | None = DEFAULT_CONFIG_PATH,
    save_report: bool = False,
    skip_vlm: bool = False,
    skip_embedding: bool = False,
    skip_rebuild: bool = False,
    skip_eval: bool = False,
    skip_report: bool = False,
) -> dict[str, Any]:
    """Describe the exact month-run steps without executing them."""
    plan = month_plan(repository, month=month, limit=limit, config_path=config_path)
    resolved_vlm_limit = vlm_limit if vlm_limit is not None else plan["recommended_limits"]["vlm_limit"]
    resolved_embedding_limit = embedding_limit if embedding_limit is not None else plan["recommended_limits"]["embedding_limit"]
    rng = MonthRange(plan["month"], plan["start_date"], plan["end_date"])
    config_arg = _config_arg(config_path)
    eval_path = _default_eval_path(rng.month)
    steps = [
        {
            "name": "backup-db",
            "enabled": True,
            "command": f"python -m personal_lifelog_rag.app.cli backup-db --label before_month_rollout_{rng.month.replace('-', '_')}",
        },
        {
            "name": "analyze-images",
            "enabled": not skip_vlm,
            "command": (
                "python -m personal_lifelog_rag.app.cli analyze-images "
                f"--from {rng.start_date} --to {rng.end_date} --limit {resolved_vlm_limit} "
                f"{config_arg} --engine qwen3_vl_transformers --prompt-template lifelog_structured_tags_v1 --skip-existing"
            ).strip(),
        },
        {
            "name": "build-image-embeddings",
            "enabled": not skip_embedding,
            "command": (
                "python -m personal_lifelog_rag.app.cli build-image-embeddings "
                f"--from {rng.start_date} --to {rng.end_date} --limit {resolved_embedding_limit} "
                f"{config_arg} --engine qwen3_vl_embedding --skip-existing"
            ).strip(),
        },
        {
            "name": "build-text-embeddings",
            "enabled": not skip_embedding,
            "command": (
                "python -m personal_lifelog_rag.app.cli build-text-embeddings "
                f"--from {rng.start_date} --to {rng.end_date} --type combined_text --limit {resolved_embedding_limit} "
                f"{config_arg} --engine qwen3_vl_embedding --skip-existing"
            ).strip(),
        },
        {
            "name": "rebuild-events-with-analysis",
            "enabled": not skip_rebuild,
            "command": (
                "python -m personal_lifelog_rag.app.cli rebuild-events-with-analysis "
                f"--from {rng.start_date} --to {rng.end_date} --force"
                + (" --save-report" if save_report else "")
            ),
        },
        {
            "name": "db-check",
            "enabled": True,
            "command": "python -m personal_lifelog_rag.app.cli db-check --strict",
        },
        {
            "name": "eval-private",
            "enabled": (not skip_eval) and eval_path.exists(),
            "command": f"python -m personal_lifelog_rag.app.cli eval-private --path {eval_path} --save-run",
            "skip_reason": None if eval_path.exists() else f"{eval_path} not found",
        },
        {
            "name": "generate-report",
            "enabled": not skip_report,
            "command": (
                "python -m personal_lifelog_rag.app.cli generate-report "
                f"--from {rng.start_date} --to {rng.end_date} --public --no-examples"
                + (" --save-json" if save_report else "")
            ),
        },
    ]
    return {
        "month": rng.month,
        "start_date": rng.start_date,
        "end_date": rng.end_date,
        "dry_run_safe": True,
        "limits": {"vlm_limit": resolved_vlm_limit, "embedding_limit": resolved_embedding_limit},
        "steps": steps,
        "counts": plan["counts"],
    }


def format_month_plan(plan: dict[str, Any]) -> str:
    lines = [
        f"Month plan: {plan['month']}",
        f"- period: {plan['start_date']}..{plan['end_date']}",
        "",
        "Input counts:",
        *_count_lines(
            {
                "photos": plan["counts"]["photos_count"],
                "gps_photos": plan["counts"]["gps_photos_count"],
                "line_messages": plan["counts"]["line_messages_count"],
                "call_events": plan["counts"]["call_events_count"],
                "events": plan["counts"]["events_count"],
            }
        ),
        "",
        "Existing analysis:",
        *_nested_status_lines("media_vlm", plan["counts"]["media_vlm"]),
        *_nested_status_lines("media_embeddings", plan["counts"]["media_embeddings"]),
        *_nested_status_lines("media_ocr", plan["counts"]["media_ocr"]),
        "",
        "Recommended limits:",
        *_count_lines(plan["recommended_limits"]),
        "",
        "Recommended commands:",
    ]
    lines.extend(f"- {command}" for command in plan["recommended_commands"])
    lines.extend(["", "Notes:"])
    lines.extend(f"- {note}" for note in plan["notes"])
    return "\n".join(lines)


def format_month_status(status: dict[str, Any]) -> str:
    lines = [
        f"Month status: {status['month']}",
        f"- period: {status['start_date']}..{status['end_date']}",
        "",
        "VLM:",
        *_nested_status_lines("media_vlm", status["vlm"]),
        "Embedding:",
        *_nested_status_lines("media_embeddings", status["embeddings"]),
        "OCR:",
        *_nested_status_lines("media_ocr", status["ocr"]),
        f"- events: {status['events_count']}",
        f"- report exists: {status['report_exists']}",
        f"- eval run exists: {status['eval_run_exists']}",
    ]
    if status["report_paths"]:
        lines.append("- report paths:")
        lines.extend(f"  - {path}" for path in status["report_paths"])
    if status["eval_run_paths"]:
        lines.append("- eval run paths:")
        lines.extend(f"  - {path}" for path in status["eval_run_paths"])
    lines.append(f"- db-check: {status['db_check_hint']}")
    return "\n".join(lines)


def format_month_run_plan(plan: dict[str, Any], *, dry_run: bool) -> str:
    title = "Month run dry-run" if dry_run else "Month run plan"
    lines = [
        f"{title}: {plan['month']}",
        f"- period: {plan['start_date']}..{plan['end_date']}",
        f"- vlm_limit: {plan['limits']['vlm_limit']}",
        f"- embedding_limit: {plan['limits']['embedding_limit']}",
        "",
        "Steps:",
    ]
    for step in plan["steps"]:
        marker = "run" if step["enabled"] else "skip"
        lines.append(f"- [{marker}] {step['name']}: {step['command']}")
        if not step["enabled"] and step.get("skip_reason"):
            lines.append(f"  reason: {step['skip_reason']}")
    if dry_run:
        lines.extend(["", "No DB changes were made. Add --yes without --dry-run to execute."])
    return "\n".join(lines)


def format_month_batch_plan(plan: dict[str, Any]) -> str:
    lines = [
        f"Month batch dry-run: {plan['from_month']}..{plan['to_month']}",
        f"- months: {len(plan['months'])}",
        "",
    ]
    for item in plan["months"]:
        counts = item["counts"]
        limits = item["recommended_limits"]
        lines.append(
            f"- {item['month']}: photos={counts['photos_count']}, "
            f"vlm_success={counts['media_vlm']['success']}, "
            f"embedding_success_media={counts['media_embeddings']['success_media']}, "
            f"events={counts['events_count']}, "
            f"recommended_vlm_limit={limits['vlm_limit']}, "
            f"recommended_embedding_limit={limits['embedding_limit']}"
        )
    lines.append("")
    lines.append("This command is planning-only; execute one month at a time with month-run --yes.")
    return "\n".join(lines)


def _month_stats(repository: LifelogRepository, rng: MonthRange) -> dict[str, Any]:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        params = [rng.start_date, rng.end_date]
        photos_count = _count(
            connection,
            """
            SELECT COUNT(*) FROM media_items
            WHERE substr(COALESCE(captured_at, fallback_captured_at, ''), 1, 10) BETWEEN ? AND ?
              AND LOWER(COALESCE(media_type, 'image')) IN ('image', 'photo')
            """,
            params,
        )
        gps_photos_count = _count(
            connection,
            """
            SELECT COUNT(*) FROM media_items
            WHERE substr(COALESCE(captured_at, fallback_captured_at, ''), 1, 10) BETWEEN ? AND ?
              AND LOWER(COALESCE(media_type, 'image')) IN ('image', 'photo')
              AND gps_lat IS NOT NULL AND gps_lon IS NOT NULL
            """,
            params,
        )
        line_messages_count = _count(
            connection,
            "SELECT COUNT(*) FROM line_messages WHERE substr(COALESCE(sent_at, ''), 1, 10) BETWEEN ? AND ?",
            params,
        )
        call_events_count = _count(
            connection,
            "SELECT COUNT(*) FROM line_call_events WHERE substr(COALESCE(sent_at, ''), 1, 10) BETWEEN ? AND ?",
            params,
        )
        events_count = _count(
            connection,
            """
            SELECT COUNT(*) FROM events
            LEFT JOIN event_overrides ON event_overrides.event_id = events.id
            WHERE substr(COALESCE(events.date, ''), 1, 10) BETWEEN ? AND ?
              AND COALESCE(event_overrides.is_hidden, 0) = 0
            """,
            params,
        )
        return {
            "photos_count": photos_count,
            "gps_photos_count": gps_photos_count,
            "line_messages_count": line_messages_count,
            "call_events_count": call_events_count,
            "events_count": events_count,
            "media_vlm": _status_counts_for_joined_media(connection, "media_vlm", "media_vlm.status", rng),
            "media_ocr": _status_counts_for_joined_media(connection, "media_ocr", "media_ocr.status", rng),
            "media_embeddings": _embedding_counts(connection, rng),
        }


def _status_counts_for_joined_media(connection, table_name: str, status_expr: str, rng: MonthRange) -> dict[str, int]:
    rows = connection.execute(
        f"""
        SELECT COALESCE({status_expr}, 'unknown') AS status, COUNT(*) AS count
        FROM {table_name}
        JOIN media_items ON media_items.id = {table_name}.media_id
        WHERE substr(COALESCE(media_items.captured_at, media_items.fallback_captured_at, ''), 1, 10) BETWEEN ? AND ?
        GROUP BY COALESCE({status_expr}, 'unknown')
        ORDER BY status ASC
        """,
        (rng.start_date, rng.end_date),
    ).fetchall()
    counts = {str(row["status"]): int(row["count"] or 0) for row in rows}
    total = sum(counts.values())
    return {
        "total": total,
        "success": counts.get("success", 0),
        "failed": counts.get("failed", 0),
        "engine_unavailable": counts.get("engine_unavailable", 0),
        "skipped": counts.get("skipped", 0),
        "status_counts": counts,
    }


def _embedding_counts(connection, rng: MonthRange) -> dict[str, int]:
    rows = connection.execute(
        """
        SELECT COALESCE(media_embeddings.status, 'unknown') AS status, COUNT(*) AS count
        FROM media_embeddings
        JOIN media_items ON media_items.id = media_embeddings.media_id
        WHERE substr(COALESCE(media_items.captured_at, media_items.fallback_captured_at, ''), 1, 10) BETWEEN ? AND ?
        GROUP BY COALESCE(media_embeddings.status, 'unknown')
        ORDER BY status ASC
        """,
        (rng.start_date, rng.end_date),
    ).fetchall()
    counts = {str(row["status"]): int(row["count"] or 0) for row in rows}
    success_media = _count(
        connection,
        """
        SELECT COUNT(DISTINCT media_embeddings.media_id)
        FROM media_embeddings
        JOIN media_items ON media_items.id = media_embeddings.media_id
        WHERE substr(COALESCE(media_items.captured_at, media_items.fallback_captured_at, ''), 1, 10) BETWEEN ? AND ?
          AND media_embeddings.status = 'success'
        """,
        [rng.start_date, rng.end_date],
    )
    return {
        "total": sum(counts.values()),
        "success": counts.get("success", 0),
        "success_media": success_media,
        "failed": counts.get("failed", 0),
        "engine_unavailable": counts.get("engine_unavailable", 0),
        "skipped": counts.get("skipped", 0),
        "status_counts": counts,
    }


def _recommendations(
    rng: MonthRange,
    stats: dict[str, Any],
    *,
    limit: int,
    config_path: Path | None,
) -> dict[str, Any]:
    safe_limit = max(0, int(limit or DEFAULT_MONTH_LIMIT))
    vlm_remaining = max(int(stats["photos_count"]) - int(stats["media_vlm"]["success"]), 0)
    embedding_remaining = max(int(stats["photos_count"]) - int(stats["media_embeddings"]["success_media"]), 0)
    vlm_limit = min(vlm_remaining, safe_limit)
    embedding_limit = min(embedding_remaining, safe_limit)
    config_arg = _config_arg(config_path)
    commands = [
        f"python -m personal_lifelog_rag.app.cli month-run --month {rng.month} --limit {safe_limit} --dry-run",
        (
            "python -m personal_lifelog_rag.app.cli month-run "
            f"--month {rng.month} --vlm-limit {vlm_limit} --embedding-limit {embedding_limit} "
            f"{config_arg} --save-report --yes"
        ).strip(),
        f"python -m personal_lifelog_rag.app.cli month-status --month {rng.month}",
        "python -m personal_lifelog_rag.app.cli db-check --strict",
    ]
    return {
        "limits": {
            "vlm_limit": vlm_limit,
            "embedding_limit": embedding_limit,
            "default_limit": safe_limit,
        },
        "commands": commands,
    }


def _matching_artifact_paths(root: Path, month: str, *, suffixes: set[str]) -> list[Path]:
    root = root.expanduser()
    if not root.exists():
        return []
    compact = month.replace("-", "")
    matches: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        name = path.name
        if month in name or compact in name:
            matches.append(path)
    return matches


def _default_eval_path(month: str) -> Path:
    return DEFAULT_PRIVATE_EVAL_DIR / f"questions_{month.replace('-', '')}_month.yaml"


def _config_arg(config_path: Path | None) -> str:
    if config_path is None:
        return ""
    return f"--config {config_path}"


def _count(connection, query: str, params: list[Any]) -> int:
    row = connection.execute(query, params).fetchone()
    return int(row[0] or 0)


def _count_lines(values: dict[str, Any]) -> list[str]:
    return [f"- {key}: {value}" for key, value in values.items()]


def _nested_status_lines(label: str, values: dict[str, Any]) -> list[str]:
    status_counts = values.get("status_counts") or {}
    lines = [
        f"- {label}.total: {values.get('total', 0)}",
        f"- {label}.success: {values.get('success', 0)}",
        f"- {label}.failed: {values.get('failed', 0)}",
        f"- {label}.engine_unavailable: {values.get('engine_unavailable', 0)}",
    ]
    if "success_media" in values:
        lines.append(f"- {label}.success_media: {values.get('success_media', 0)}")
    if status_counts:
        counts = ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items()))
        lines.append(f"- {label}.status_counts: {counts}")
    return lines


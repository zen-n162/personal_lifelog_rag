"""Storage and maintenance helpers for local analysis artifacts."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Any

from personal_lifelog_rag.db.backup import DEFAULT_BACKUP_DIR, backup_sqlite_db
from personal_lifelog_rag.db.repository import connect
from personal_lifelog_rag.db.schema import initialize_schema


WATCHED_DIRS = {
    "thumbnails": Path("data/thumbnails"),
    "eval_outputs": Path("eval_outputs"),
    "backups": Path("backups"),
    "private_eval": Path("private_eval"),
    "models": Path("models"),
}


def storage_stats(db_path: str | Path) -> dict[str, Any]:
    db_path = Path(db_path).expanduser()
    table_counts: dict[str, int] = {}
    blob_bytes = 0
    with closing(connect(db_path)) as connection:
        initialize_schema(connection)
        for table in (
            "media_items",
            "media_ocr",
            "media_vlm",
            "media_embeddings",
            "analysis_jobs",
            "analysis_job_items",
        ):
            table_counts[table] = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] or 0)
        blob_bytes = int(connection.execute("SELECT COALESCE(SUM(length(embedding)), 0) FROM media_embeddings").fetchone()[0] or 0)
    dir_sizes = {name: _path_size(path) for name, path in WATCHED_DIRS.items()}
    db_size = db_path.stat().st_size if db_path.exists() else 0
    largest_tables = sorted(table_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    return {
        "db_path": str(db_path),
        "db_size_bytes": db_size,
        "directories": dir_sizes,
        "counts": table_counts,
        "media_embeddings_blob_bytes": blob_bytes,
        "largest_tables_by_rows": [{"table": table, "rows": rows} for table, rows in largest_tables],
        "vacuum_recommended": db_size > 50 * 1024 * 1024 and blob_bytes > 0,
    }


def run_db_maintenance(
    db_path: str | Path,
    *,
    backup: bool = False,
    vacuum: bool = False,
    yes: bool = False,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
) -> dict[str, Any]:
    result: dict[str, Any] = {"backup": None, "vacuum": "not_requested", "messages": []}
    if backup:
        backup_result = backup_sqlite_db(db_path, label="before_db_maintenance", output_dir=backup_dir)
        result["backup"] = backup_result.to_dict()
    if vacuum:
        if not backup and not yes:
            result["vacuum"] = "skipped"
            result["messages"].append("VACUUM skipped: run with --backup or --yes.")
        else:
            with closing(connect(db_path)) as connection:
                initialize_schema(connection)
                connection.execute("VACUUM")
            result["vacuum"] = "completed"
    return result


def format_storage_stats(report: dict[str, Any]) -> str:
    lines = [
        "Storage Stats",
        f"- db_path: {report['db_path']}",
        f"- DB size: {_format_bytes(report['db_size_bytes'])}",
        f"- media_embeddings BLOB size: {_format_bytes(report['media_embeddings_blob_bytes'])}",
        f"- vacuum recommended: {report['vacuum_recommended']}",
        "directories:",
    ]
    for name, size in report["directories"].items():
        lines.append(f"- {name}: {_format_bytes(size)}")
    lines.append("counts:")
    for name, count in report["counts"].items():
        lines.append(f"- {name}: {count}")
    lines.append("largest tables:")
    for row in report["largest_tables_by_rows"]:
        lines.append(f"- {row['table']}: {row['rows']}")
    return "\n".join(lines)


def format_db_maintenance(report: dict[str, Any]) -> str:
    backup = report.get("backup")
    backup_text = backup.get("backup_path") if isinstance(backup, dict) else (backup or "none")
    lines = ["DB Maintenance", f"- backup: {backup_text}", f"- vacuum: {report.get('vacuum')}"]
    for message in report.get("messages") or []:
        lines.append(f"- {message}")
    return "\n".join(lines)


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"

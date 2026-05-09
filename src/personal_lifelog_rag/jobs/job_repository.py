"""Repository for analysis job metadata."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from personal_lifelog_rag.db.repository import connect
from personal_lifelog_rag.db.schema import initialize_schema
from personal_lifelog_rag.jobs.schemas import JOB_ITEM_STATUSES, JOB_STATUSES, JOB_TYPES


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class AnalysisJobRepository:
    """Persistence boundary for analysis_jobs and analysis_job_items."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()

    def initialize(self) -> None:
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)

    def create_job(
        self,
        *,
        job_id: str,
        job_type: str,
        status: str = "planned",
        target_scope: dict[str, Any] | None = None,
        engine: str | None = None,
        model_name: str | None = None,
        prompt_version: str | None = None,
        analysis_version: str | None = None,
        total_items: int = 0,
        error_message: str | None = None,
    ) -> None:
        if job_type not in JOB_TYPES:
            raise ValueError(f"unknown job_type: {job_type}")
        if status not in JOB_STATUSES:
            raise ValueError(f"unknown job status: {status}")
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            connection.execute(
                """
                INSERT INTO analysis_jobs (
                    job_id,
                    job_type,
                    status,
                    target_scope_json,
                    engine,
                    model_name,
                    prompt_version,
                    analysis_version,
                    total_items,
                    processed_items,
                    success_items,
                    failed_items,
                    skipped_items,
                    created_at,
                    error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, CURRENT_TIMESTAMP, ?)
                """,
                (
                    job_id,
                    job_type,
                    status,
                    json.dumps(target_scope or {}, ensure_ascii=False, sort_keys=True),
                    engine,
                    model_name,
                    prompt_version,
                    analysis_version,
                    total_items,
                    error_message,
                ),
            )
            connection.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            row = connection.execute(
                "SELECT * FROM analysis_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_jobs(self, *, recent: int = 10, job_type: str | None = None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if job_type:
            clauses.append("job_type = ?")
            params.append(job_type)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(recent, 1))
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            rows = connection.execute(
                f"""
                SELECT *
                FROM analysis_jobs
                {where_sql}
                ORDER BY COALESCE(created_at, started_at, finished_at) DESC, job_id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [dict(row) for row in rows]

    def upsert_item(
        self,
        *,
        job_id: str,
        item_id: str,
        item_type: str,
        status: str = "pending",
        error_message: str | None = None,
        latency_sec: float | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> None:
        if status not in JOB_ITEM_STATUSES:
            raise ValueError(f"unknown job item status: {status}")
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            connection.execute(
                """
                INSERT INTO analysis_job_items (
                    job_id,
                    item_id,
                    item_type,
                    status,
                    error_message,
                    started_at,
                    finished_at,
                    latency_sec
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id, item_id) DO UPDATE SET
                    item_type = excluded.item_type,
                    status = excluded.status,
                    error_message = excluded.error_message,
                    started_at = COALESCE(excluded.started_at, analysis_job_items.started_at),
                    finished_at = COALESCE(excluded.finished_at, analysis_job_items.finished_at),
                    latency_sec = COALESCE(excluded.latency_sec, analysis_job_items.latency_sec)
                """,
                (job_id, item_id, item_type, status, error_message, started_at, finished_at, latency_sec),
            )
            connection.commit()

    def list_items(
        self,
        job_id: str,
        *,
        statuses: list[str] | None = None,
        limit: int = 100_000,
    ) -> list[dict[str, Any]]:
        clauses = ["job_id = ?"]
        params: list[Any] = [job_id]
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(statuses)
        params.append(limit)
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            rows = connection.execute(
                f"""
                SELECT *
                FROM analysis_job_items
                WHERE {' AND '.join(clauses)}
                ORDER BY item_id ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [dict(row) for row in rows]

    def update_job_status(
        self,
        job_id: str,
        status: str,
        *,
        error_message: str | None = None,
        mark_started: bool = False,
        mark_finished: bool = False,
    ) -> None:
        if status not in JOB_STATUSES:
            raise ValueError(f"unknown job status: {status}")
        assignments = ["status = ?", "error_message = ?"]
        params: list[Any] = [status, error_message]
        if mark_started:
            assignments.append("started_at = COALESCE(started_at, CURRENT_TIMESTAMP)")
        if mark_finished:
            assignments.append("finished_at = CURRENT_TIMESTAMP")
        params.append(job_id)
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            connection.execute(
                f"UPDATE analysis_jobs SET {', '.join(assignments)} WHERE job_id = ?",
                params,
            )
            connection.commit()

    def recalculate_counts(self, job_id: str) -> dict[str, int]:
        items = self.list_items(job_id)
        total = len(items)
        success = sum(1 for item in items if item.get("status") == "success")
        failed = sum(1 for item in items if item.get("status") == "failed")
        skipped = sum(1 for item in items if item.get("status") in {"skipped", "engine_unavailable"})
        processed = sum(1 for item in items if item.get("status") not in {"pending", "running"})
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            connection.execute(
                """
                UPDATE analysis_jobs
                SET total_items = ?,
                    processed_items = ?,
                    success_items = ?,
                    failed_items = ?,
                    skipped_items = ?
                WHERE job_id = ?
                """,
                (total, processed, success, failed, skipped, job_id),
            )
            connection.commit()
        return {
            "total": total,
            "processed": processed,
            "success": success,
            "failed": failed,
            "skipped": skipped,
        }

    def cleanup(
        self,
        *,
        failed: bool = False,
        engine_unavailable: bool = False,
        old_runs_days: int | None = None,
        dry_run: bool = True,
        yes: bool = False,
    ) -> dict[str, Any]:
        target_jobs: list[str] = []
        target_items: list[tuple[str, str]] = []
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            if failed:
                target_items.extend(
                    (row["job_id"], row["item_id"])
                    for row in connection.execute(
                        "SELECT job_id, item_id FROM analysis_job_items WHERE status = 'failed'"
                    ).fetchall()
                )
            if engine_unavailable:
                target_items.extend(
                    (row["job_id"], row["item_id"])
                    for row in connection.execute(
                        "SELECT job_id, item_id FROM analysis_job_items WHERE status = 'engine_unavailable'"
                    ).fetchall()
                )
            if old_runs_days is not None:
                target_jobs.extend(
                    row["job_id"]
                    for row in connection.execute(
                        """
                        SELECT job_id
                        FROM analysis_jobs
                        WHERE julianday('now') - julianday(COALESCE(finished_at, created_at)) >= ?
                        """,
                        (max(old_runs_days, 0),),
                    ).fetchall()
                )
            if dry_run or not yes:
                return {
                    "dry_run": True,
                    "requires_yes": not yes,
                    "job_ids": sorted(set(target_jobs)),
                    "job_count": len(set(target_jobs)),
                    "item_count": len(set(target_items)),
                }
            for job_id, item_id in set(target_items):
                connection.execute(
                    "DELETE FROM analysis_job_items WHERE job_id = ? AND item_id = ?",
                    (job_id, item_id),
                )
            for job_id in set(target_jobs):
                connection.execute("DELETE FROM analysis_jobs WHERE job_id = ?", (job_id,))
            connection.commit()
            return {
                "dry_run": False,
                "requires_yes": False,
                "job_ids": sorted(set(target_jobs)),
                "job_count": len(set(target_jobs)),
                "item_count": len(set(target_items)),
            }

from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.jobs.job_repository import AnalysisJobRepository


def test_db_maintenance_backup_and_vacuum(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    LifelogRepository(db_path).initialize()

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "db-maintenance",
            "--backup",
            "--vacuum",
            "--backup-dir",
            str(tmp_path / "backups"),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "DB Maintenance" in output
    assert list((tmp_path / "backups").glob("lifelog_before_db_maintenance_*.sqlite"))


def test_analysis_cleanup_dry_run_does_not_delete(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    jobs = AnalysisJobRepository(db_path)
    jobs.initialize()
    jobs.create_job(job_id="job_cleanup", job_type="vlm", status="partial", total_items=1)
    jobs.upsert_item(job_id="job_cleanup", item_id="media_cleanup", item_type="media", status="failed")

    exit_code = main(["--db-path", str(db_path), "analysis-cleanup", "--failed", "--dry-run"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "dry_run: True" in output
    assert jobs.get_job("job_cleanup") is not None

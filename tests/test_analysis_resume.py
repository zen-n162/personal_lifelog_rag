from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.jobs.job_repository import AnalysisJobRepository


def test_analysis_resume_dry_run_from_failed_job(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    jobs = AnalysisJobRepository(db_path)
    jobs.initialize()
    jobs.create_job(
        job_id="job_resume",
        job_type="vlm",
        status="partial",
        target_scope={"job_type": "vlm", "start_date": "2024-12-24", "end_date": "2024-12-24", "engine_name": "fake"},
        total_items=1,
    )
    jobs.upsert_item(job_id="job_resume", item_id="media_missing", item_type="media", status="failed")
    jobs.recalculate_counts("job_resume")

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "analysis-resume",
            "--job-id",
            "job_resume",
            "--failed-only",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Analysis Run (dry-run)" in output


from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.jobs.job_repository import AnalysisJobRepository


def test_analysis_jobs_repository_creates_job_and_items(tmp_path: Path) -> None:
    repository = AnalysisJobRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()

    repository.create_job(
        job_id="job_repo_1",
        job_type="vlm",
        status="planned",
        target_scope={"date": "2024-12-24"},
        engine="fake",
        total_items=2,
    )
    repository.upsert_item(job_id="job_repo_1", item_id="media_1", item_type="media", status="success")
    repository.upsert_item(job_id="job_repo_1", item_id="media_2", item_type="media", status="failed")

    counts = repository.recalculate_counts("job_repo_1")
    job = repository.get_job("job_repo_1")

    assert counts == {"total": 2, "processed": 2, "success": 1, "failed": 1, "skipped": 0}
    assert job is not None
    assert job["success_items"] == 1
    assert len(repository.list_items("job_repo_1")) == 2


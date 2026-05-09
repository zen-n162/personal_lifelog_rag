from __future__ import annotations

from pathlib import Path

from PIL import Image

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.jobs.job_repository import AnalysisJobRepository


def test_analysis_run_dry_run_does_not_create_job(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    _repository_with_image(db_path, tmp_path)

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "analysis-run",
            "--type",
            "vlm",
            "--date",
            "2024-12-24",
            "--engine",
            "fake",
            "--limit",
            "1",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Analysis Run (dry-run)" in output
    assert AnalysisJobRepository(db_path).list_jobs() == []


def test_analysis_run_fake_vlm_creates_job(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = _repository_with_image(db_path, tmp_path)

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "analysis-run",
            "--type",
            "vlm",
            "--date",
            "2024-12-24",
            "--engine",
            "fake",
            "--limit",
            "1",
            "--save-report",
            "--allow-fake-write",
        ]
    )
    output = capsys.readouterr().out
    jobs = AnalysisJobRepository(db_path).list_jobs()

    assert exit_code == 0
    assert "Analysis Run" in output
    assert jobs[0]["status"] == "completed"
    assert repository.get_media_vlm("media_analysis_run")["status"] == "success"


def _repository_with_image(db_path: Path, tmp_path: Path) -> LifelogRepository:
    repository = LifelogRepository(db_path)
    repository.initialize()
    image_path = tmp_path / "analysis_run.png"
    Image.new("RGB", (16, 16), "white").save(image_path)
    repository.add_media_item(
        id="media_analysis_run",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-analysis-run",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    return repository

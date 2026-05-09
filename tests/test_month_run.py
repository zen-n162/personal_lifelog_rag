from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.rollout.monthly_rollout import month_run_plan


def test_month_run_dry_run_does_not_modify_db(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = _repository_with_photo(db_path, tmp_path)
    before = repository.stats()

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "month-run",
            "--month",
            "2025-02",
            "--limit",
            "10",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Month run dry-run: 2025-02" in output
    assert "analyze-images" in output
    assert repository.stats() == before


def test_month_run_plan_orders_pipeline_steps(tmp_path: Path) -> None:
    repository = _repository_with_photo(tmp_path / "lifelog.sqlite", tmp_path)

    plan = month_run_plan(repository, month="2025-02", limit=10, config_path=Path("private_config/model_runtime.yaml"))

    assert [step["name"] for step in plan["steps"]] == [
        "backup-db",
        "analyze-images",
        "build-image-embeddings",
        "build-text-embeddings",
        "rebuild-events-with-analysis",
        "db-check",
        "eval-private",
        "generate-report",
    ]
    assert plan["steps"][1]["enabled"] is True
    assert "--from 2025-02-01 --to 2025-02-28" in plan["steps"][1]["command"]


def _repository_with_photo(db_path: Path, tmp_path: Path) -> LifelogRepository:
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_month_run",
        file_path=str(tmp_path / "month_run.jpg"),
        file_name="month_run.jpg",
        file_hash="hash-month-run",
        media_type="image",
        captured_at="2025-02-03T10:00:00+09:00",
    )
    return repository


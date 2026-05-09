from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.rollout.monthly_rollout import month_batch_plan


def test_month_batch_dry_run_lists_months(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    LifelogRepository(db_path).initialize()

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "month-batch",
            "--from-month",
            "2025-02",
            "--to-month",
            "2025-04",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Month batch dry-run: 2025-02..2025-04" in output
    assert "- 2025-03:" in output


def test_month_batch_plan_builds_multiple_months(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()

    plan = month_batch_plan(repository, from_month="2025-02", to_month="2025-03")

    assert [item["month"] for item in plan["months"]] == ["2025-02", "2025-03"]
    assert plan["dry_run_only"] is True


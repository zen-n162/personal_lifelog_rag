from __future__ import annotations

import json

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.evaluation.private_eval import compare_private_eval_reports


def test_private_eval_compare_detects_improved_and_new_failed_cases(tmp_path) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    _write_report(
        before,
        [
            {"id": "case_a", "status": "fail"},
            {"id": "case_b", "status": "pass"},
        ],
        passed=1,
        failed=1,
    )
    _write_report(
        after,
        [
            {"id": "case_a", "status": "pass"},
            {"id": "case_b", "status": "fail"},
        ],
        passed=1,
        failed=1,
        top1=0.75,
        recall=0.9,
        forbidden=1,
    )

    report = compare_private_eval_reports(before, after)

    assert report["improved_cases"] == ["case_a"]
    assert report["newly_failed_cases"] == ["case_b"]
    assert report["delta"]["forbidden_phrase_violations"] == 1


def test_eval_compare_cli_outputs_json(tmp_path, capsys) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    _write_report(before, [{"id": "case_a", "status": "fail"}], passed=0, failed=1)
    _write_report(after, [{"id": "case_a", "status": "pass"}], passed=1, failed=0)

    exit_code = main(["eval-compare", "--before", str(before), "--after", str(after), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["delta"]["passed"] == 1
    assert payload["improved_cases"] == ["case_a"]


def _write_report(path, cases, *, passed: int, failed: int, top1=None, recall=None, forbidden: int = 0) -> None:
    path.write_text(
        json.dumps(
            {
                "summary": {"cases": len(cases), "total": len(cases), "passed": passed, "failed": failed, "skipped": 0},
                "ranking_metrics": {"top1_accuracy": top1, "expected_date_recall_at_5": recall},
                "safety_metrics": {"forbidden_phrase_violations": forbidden, "overclaim_violations": 0},
                "case_results": cases,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import PrivateEvalQuestion, evaluate_private_questions


def test_privacy_audit_case_detects_forbidden_public_output(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    bad_html = tmp_path / "bad_public.html"
    bad_html.write_text("<html>/home/zennakamura data/raw face crop</html>", encoding="utf-8")

    report = evaluate_private_questions(
        repository,
        [
            PrivateEvalQuestion(
                id="privacy_audit",
                case_type="privacy_audit",
                question=str(bad_html),
                target=str(bad_html),
                forbidden_patterns=["/home/zennakamura", "data/raw", "face crop"],
            )
        ],
    )

    case = report["cases"][0]
    assert case["status"] == "fail"
    assert case["forbidden_claims_found"]

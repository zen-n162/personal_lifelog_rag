from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import PrivateEvalQuestion, evaluate_private_questions
from personal_lifelog_rag.faces.person_service import create_person


def test_export_privacy_case_checks_public_redacted_payload(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    create_person(repository, name="Private Eval Name", public_name="人物A", privacy_level="public_alias")

    report = evaluate_private_questions(
        repository,
        [
            PrivateEvalQuestion(
                id="export_privacy",
                case_type="export_privacy",
                question="person export privacy",
                mode="public_redacted",
                forbidden_fields=["display_name", "face_embedding", "crop_path", "exact_lat", "exact_lon"],
            )
        ],
    )

    case = report["cases"][0]
    assert case["status"] == "pass"
    assert "Private Eval Name" not in case["answer_preview"]

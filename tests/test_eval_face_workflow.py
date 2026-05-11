from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import PrivateEvalQuestion, evaluate_private_questions


def test_face_workflow_quality_allows_zero_when_configured(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()

    report = evaluate_private_questions(
        repository,
        [
            PrivateEvalQuestion(
                id="face_workflow",
                case_type="face_workflow_quality",
                question="face workflow quality",
                expected_min_face_detections=0,
                expected_min_face_clusters=0,
                allow_zero=True,
                require_no_public_face_crops=True,
            )
        ],
    )

    assert report["cases"][0]["status"] == "pass"
    assert report["cases"][0]["face_detections"] == 0

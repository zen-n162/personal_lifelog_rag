from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import PrivateEvalQuestion, evaluate_private_questions


def test_private_eval_vlm_safety_case(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()

    report = evaluate_private_questions(
        repository,
        [
            PrivateEvalQuestion(
                id="vlm_safety_001",
                question="vlm safety",
                case_type="vlm_safety",
                input_text="彼女と楽しそうにご飯を食べている写真です",
                expected_flags=["relationship_inference_removed", "emotion_inference_removed"],
                forbidden_claims=["彼女", "楽しそう"],
            ),
            PrivateEvalQuestion(
                id="vlm_prompt_001",
                question="lifelog_structured_tags_v1",
                case_type="vlm_prompt",
                template="lifelog_structured_tags_v1",
                expected_contains=["Do not identify people", "Return valid JSON only"],
            ),
        ],
    )

    assert report["summary"]["passed"] == 2


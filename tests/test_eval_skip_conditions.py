from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import PrivateEvalQuestion, evaluate_private_questions


def test_person_case_skips_when_no_person_and_allowed(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()

    report = evaluate_private_questions(
        repository,
        [
            PrivateEvalQuestion(
                id="skip_person",
                case_type="person_line_qa",
                question="人物AとLINEした日は？",
                allow_skip_if_no_person=True,
            )
        ],
    )

    assert report["cases"][0]["status"] == "skip"


def test_place_case_skips_when_no_place_and_allowed(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()

    report = evaluate_private_questions(
        repository,
        [
            PrivateEvalQuestion(
                id="skip_place",
                case_type="place_qa",
                question="場所Aに行ったのはいつ？",
                allow_skip_if_no_place=True,
            )
        ],
    )

    assert report["cases"][0]["status"] == "skip"

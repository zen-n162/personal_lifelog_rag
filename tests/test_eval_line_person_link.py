from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import PrivateEvalQuestion, evaluate_private_questions
from personal_lifelog_rag.faces.person_service import create_person
from personal_lifelog_rag.line.person_links import link_line_speaker_to_person


def test_line_person_link_quality_requires_manual_links(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    person = create_person(repository, name="人物リンクA", public_name="人物A", privacy_level="public_alias")
    link_line_speaker_to_person(
        repository,
        chat_id="chat_line_eval",
        speaker_name="SpeakerLineEval",
        person_id=person["id"],
        yes=True,
    )

    report = evaluate_private_questions(
        repository,
        [
            PrivateEvalQuestion(
                id="line_link_quality",
                case_type="line_person_link_quality",
                question="line person link quality",
                allow_zero_links=False,
                require_manual_links_only=True,
            )
        ],
    )

    case = report["cases"][0]
    assert case["status"] == "pass"
    assert case["line_speaker_links"] == 1

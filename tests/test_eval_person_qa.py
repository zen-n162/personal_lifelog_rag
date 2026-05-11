from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import PrivateEvalQuestion, evaluate_private_questions
from personal_lifelog_rag.faces.person_service import create_person
from personal_lifelog_rag.line.person_links import link_line_speaker_to_person


def test_person_line_qa_case_evaluates_manual_line_link(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_line_message(
        id="line_eval_person_1",
        chat_id="chat_eval_person",
        source_file="dummy.txt",
        sent_at="2025-02-03T09:00:00+09:00",
        sender="SpeakerEval",
        text="short dummy",
    )
    person = create_person(repository, name="人物評価A", public_name="人物A", privacy_level="public_alias")
    link_line_speaker_to_person(
        repository,
        chat_id="chat_eval_person",
        speaker_name="SpeakerEval",
        person_id=person["id"],
        yes=True,
    )

    report = evaluate_private_questions(
        repository,
        [
            PrivateEvalQuestion(
                id="person_line_eval",
                case_type="person_line_qa",
                question="人物評価AとLINEした日は？",
                expected_min_results=1,
            )
        ],
    )

    case = report["cases"][0]
    assert case["status"] == "pass"
    assert case["intent"] == "person_line_search"
    assert case["resolved_person_id"] == person["id"]

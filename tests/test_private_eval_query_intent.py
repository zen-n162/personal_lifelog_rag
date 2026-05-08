from __future__ import annotations

import json

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import evaluate_private_questions, load_private_eval_questions


def test_private_eval_query_intent_case_with_entities(tmp_path) -> None:
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """cases:
  - id: intent_place_001
    type: query_intent
    question: "新宿に行ったのはいつ？"
    expected_intent: "place_visit"
    expected_entities:
      place: "新宿"
""",
        encoding="utf-8",
    )
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()

    questions = load_private_eval_questions(questions_path)
    report = evaluate_private_questions(repository, questions)

    assert questions[0].expected_entities == {"place": "新宿"}
    assert report["case_results"][0]["status"] == "pass"
    assert report["case_results"][0]["intent"] == "place_visit"
    json.dumps(report, ensure_ascii=False)


def test_private_eval_query_intent_detects_entity_mismatch(tmp_path) -> None:
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """cases:
  - id: intent_bad
    type: query_intent
    question: "新宿に行ったのはいつ？"
    expected_intent: "place_visit"
    expected_entities:
      place: "渋谷"
""",
        encoding="utf-8",
    )
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()

    report = evaluate_private_questions(repository, load_private_eval_questions(questions_path))

    assert report["case_results"][0]["status"] == "fail"
    assert "entity mismatch" in report["case_results"][0]["issues"][0]

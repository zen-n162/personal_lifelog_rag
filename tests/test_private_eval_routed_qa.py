from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import evaluate_private_questions, load_private_eval_questions


def test_private_eval_routed_qa_checks_top_date_and_classification(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_routed_records(repository)
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """cases:
  - id: routed_place_001
    type: routed_qa
    question: "新宿に行ったのはいつ？"
    expected_intent: "place_visit"
    expected_top_dates:
      - "2024-12-24"
    expected_classification:
      "2024-12-24": "actual_or_likely_action"
    should_not_include:
      - "確実に行った"
""",
        encoding="utf-8",
    )

    report = evaluate_private_questions(repository, load_private_eval_questions(questions_path))
    case = report["case_results"][0]

    assert case["status"] == "pass"
    assert case["intent"] == "place_visit"
    assert case["top_dates"][0] == "2024-12-24"


def test_private_eval_routed_qa_detects_forbidden_phrase(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_routed_records(repository)
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """cases:
  - id: routed_forbidden
    type: routed_qa
    question: "新宿に行ったのはいつ？"
    expected_intent: "place_visit"
    should_not_include:
      - "新宿"
""",
        encoding="utf-8",
    )

    report = evaluate_private_questions(repository, load_private_eval_questions(questions_path))

    assert report["case_results"][0]["status"] == "fail"
    assert report["safety_metrics"]["forbidden_phrase_violations"] >= 1


def _seed_routed_records(repository: LifelogRepository) -> None:
    repository.add_line_message(
        id="line_route_actual",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T17:30:00+09:00",
        sender="自分",
        text="18時に新宿着く！",
    )
    repository.add_event(
        id="event_route_actual",
        date="2024-12-24",
        start_time="17:30:00",
        end_time="18:30:00",
        title="移動・待ち合わせの可能性",
        summary="場所候補: 新宿。活動候補: 着く。",
        location_name="新宿",
        confidence=0.9,
    )

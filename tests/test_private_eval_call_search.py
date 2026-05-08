from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.evaluation.private_eval import evaluate_private_questions, load_private_eval_questions
from personal_lifelog_rag.line.call_index import build_call_index


def test_private_eval_call_search_completed_filter(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_calls(repository)
    build_call_index(repository)
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """cases:
  - id: call_completed_001
    type: call_search
    question: "長電話した日は？"
    filters:
      completed: true
      min_duration_sec: 600
    expected_min_results: 1
    expected_status: "completed"
    should_not_include_status:
      - "missed"
      - "unanswered"
""",
        encoding="utf-8",
    )

    report = evaluate_private_questions(repository, load_private_eval_questions(questions_path))
    case = report["case_results"][0]

    assert case["status"] == "pass"
    assert case["results_count"] == 1
    assert case["statuses"] == ["completed"]


def test_private_eval_call_search_detects_wrong_status(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_calls(repository)
    build_call_index(repository)
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(
        """cases:
  - id: call_bad
    type: call_search
    question: "不在着信の日は？"
    filters:
      missed: true
    expected_status: "completed"
""",
        encoding="utf-8",
    )

    report = evaluate_private_questions(repository, load_private_eval_questions(questions_path))

    assert report["case_results"][0]["status"] == "fail"


def _seed_calls(repository: LifelogRepository) -> None:
    for message_id, sent_at, text in [
        ("call_done", "2024-12-24T10:00:00+09:00", "☎ 通話時間 10:38"),
        ("call_missed", "2024-12-25T10:00:00+09:00", "☎ 不在着信"),
        ("call_unanswered", "2024-12-26T10:00:00+09:00", "☎ 通話に応答がありませんでした"),
    ]:
        repository.add_line_message(
            id=message_id,
            chat_id="chat_dummy",
            source_file="sample_chat.txt",
            sent_at=sent_at,
            sender="自分",
            text=text,
        )

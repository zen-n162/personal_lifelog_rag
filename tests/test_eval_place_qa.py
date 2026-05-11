from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.evaluation.private_eval import PrivateEvalQuestion, evaluate_private_questions
from personal_lifelog_rag.places.location_store import create_place


def test_place_qa_case_evaluates_manual_place_without_gps(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_event(
        id="event_eval_place",
        date="2025-01-10",
        title="場所Aに関連する候補",
        summary="手動場所ラベルに基づく候補。",
        confidence=0.8,
    )
    create_place(
        repository,
        place_id="place_eval_a",
        display_name="場所テストA",
        public_name="場所A",
        privacy_level="public_label",
        aliases=["場所A"],
        manual_verified=True,
    )
    with connect(repository.db_path) as connection:
        connection.execute(
            """
            INSERT INTO event_places (event_id, place_id, source, confidence)
            VALUES ('event_eval_place', 'place_eval_a', 'manual', 0.95)
            """
        )
        connection.commit()

    report = evaluate_private_questions(
        repository,
        [
            PrivateEvalQuestion(
                id="place_eval",
                case_type="place_qa",
                question="場所Aに行ったのはいつ？",
                expected_place_alias="場所A",
                expected_min_results=1,
                forbidden_claims=["緯度", "経度"],
            )
        ],
    )

    case = report["cases"][0]
    assert case["status"] == "pass"
    assert case["intent"] == "place_visit_search"
    assert case["resolved_place_id"] == "place_eval_a"

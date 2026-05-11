from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.faces.person_service import create_person
from personal_lifelog_rag.retrieval.date_parser import parse_date_query
from personal_lifelog_rag.retrieval.query_router import route_query
from personal_lifelog_rag.retrieval.temporal_search import search_timeline
from personal_lifelog_rag.retrieval.answer_builder import build_answer
from personal_lifelog_rag.ui.monthly_summary_service import monthly_summary_for_ui


def test_monthly_summary_private_includes_manual_person_section_and_public_hides_name(tmp_path: Path) -> None:
    repository, _person = _seed_person_event(tmp_path)

    private_payload = monthly_summary_for_ui(repository, month="2025-01", mode="private")
    public_payload = monthly_summary_for_ui(repository, month="2025-01", mode="public")

    assert "手動リンク済み人物" in private_payload["summary_text"]
    assert "人物A" in private_payload["summary_text"]
    assert ["person_related_events", 1] in private_payload["metrics"]
    assert "人物A" not in public_payload["summary_text"]
    assert "public modeでは非表示" in public_payload["summary_text"]


def test_date_answer_includes_related_people_without_relationship_inference(tmp_path: Path) -> None:
    repository, _person = _seed_person_event(tmp_path)
    date_range = parse_date_query("2025年1月5日は何していた？")
    result = search_timeline(repository, "2025年1月5日は何していた？", date_range=date_range)

    answer = build_answer("2025年1月5日は何していた？", result)

    assert "関連人物候補: 人物A" in answer
    assert "関係性" in answer
    assert "恋人" not in answer
    assert "本人確定" not in answer


def test_person_activity_qa_uses_event_people_source_counts(tmp_path: Path) -> None:
    repository, person = _seed_person_event(tmp_path)

    result = route_query(repository, "人物Aとご飯を食べた日は？")

    assert result.intent == "person_activity_search"
    assert result.metadata["resolved_person_id"] == person["id"]
    assert result.results[0]["event_id"] == "event_person_summary"
    assert result.metadata["source_counts"]["event_people"] >= 1


def _seed_person_event(tmp_path: Path) -> tuple[LifelogRepository, dict[str, object]]:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    person = create_person(repository, name="人物A", public_name="人物A", privacy_level="public_alias")
    event_id = repository.add_event(
        id="event_person_summary",
        date="2025-01-05",
        start_time="12:00:00",
        title="食事の候補",
        summary="ご飯に関する候補",
        confidence=0.8,
    )
    with connect(repository.db_path) as connection:
        connection.execute(
            """
            INSERT INTO event_people (
                event_id, person_id, source, confidence, evidence_count, media_count, line_count
            )
            VALUES (?, ?, 'combined', 0.9, 2, 1, 1)
            """,
            [event_id, person["id"]],
        )
        connection.commit()
    return repository, person

from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.faces.person_service import create_person
from personal_lifelog_rag.line.person_links import link_line_speaker_to_person
from personal_lifelog_rag.reporting.report_builder import build_report, write_report
from personal_lifelog_rag.reporting.schemas import ReportOptions
from personal_lifelog_rag.retrieval.query_router import route_query


def test_public_report_does_not_include_private_line_linked_person_name(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_line_message(
        id="line_privacy_1",
        chat_id="chat_privacy",
        source_file="dummy_line.txt",
        sent_at="2025-05-01T10:00:00+09:00",
        sender="SpeakerPrivate",
        text="short dummy",
    )
    person = create_person(repository, name="Private Linked Name", public_name="人物A", privacy_level="public_alias")
    link_line_speaker_to_person(
        repository,
        chat_id="chat_privacy",
        speaker_name="SpeakerPrivate",
        person_id=person["id"],
        yes=True,
        add_alias=True,
    )

    report = build_report(repository, ReportOptions(mode="public"))
    result = write_report(report, output_path=tmp_path / "public_report.md", save_json=True)
    text = result.markdown_path.read_text(encoding="utf-8")

    assert "Private Linked Name" not in text
    assert "SpeakerPrivate" not in text
    assert "人物A" not in text


def test_person_line_qa_does_not_infer_relationship_terms(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_line_message(
        id="line_privacy_qa",
        chat_id="chat_privacy",
        source_file="dummy_line.txt",
        sent_at="2025-05-01T10:00:00+09:00",
        sender="SpeakerPrivate",
        text="short dummy",
    )
    person = create_person(repository, name="人物テストPrivacy", privacy_level="private")
    link_line_speaker_to_person(
        repository,
        chat_id="chat_privacy",
        speaker_name="SpeakerPrivate",
        person_id=person["id"],
        yes=True,
    )

    result = route_query(repository, "人物テストPrivacyとLINEした日は？")

    forbidden = ("恋人", "家族", "友人", "彼女", "彼氏")
    assert all(term not in result.answer for term in forbidden)
    assert "関係性は推定していません" in result.answer

from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.faces.person_service import add_person_alias, create_person
from personal_lifelog_rag.line.person_links import link_line_speaker_to_person
from personal_lifelog_rag.retrieval.person_place_qa import resolve_person


def test_person_resolution_uses_alias_and_manual_line_speaker(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    person = create_person(repository, name="人物テストA", public_name="人物A", privacy_level="public_alias")
    add_person_alias(repository, person_id=person["id"], alias="ニックA")
    link_line_speaker_to_person(
        repository,
        chat_id="chat_a",
        speaker_name="SpeakerA",
        person_id=person["id"],
        yes=True,
    )

    alias_result = resolve_person(repository, "ニックA")
    speaker_result = resolve_person(repository, "SpeakerA")

    assert alias_result.status == "resolved"
    assert alias_result.resolved["id"] == person["id"]
    assert speaker_result.status == "resolved"
    assert speaker_result.resolved["id"] == person["id"]


def test_person_resolution_reports_ambiguous_candidates(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    person_a = create_person(repository, name="人物A", privacy_level="private")
    person_b = create_person(repository, name="人物B", privacy_level="private")
    add_person_alias(repository, person_id=person_a["id"], alias="同名")
    add_person_alias(repository, person_id=person_b["id"], alias="同名")

    result = resolve_person(repository, "同名")

    assert result.status == "ambiguous"
    assert {row["id"] for row in result.candidates} == {person_a["id"], person_b["id"]}

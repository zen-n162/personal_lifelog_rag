from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.faces.person_service import create_person, get_person
from personal_lifelog_rag.line.person_links import link_line_speaker_to_person


def test_link_line_speaker_can_add_person_alias_once(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_line_message(
        id="line_alias_1",
        chat_id="chat_alias",
        source_file="dummy_line.txt",
        sent_at="2025-03-01T12:00:00+09:00",
        sender="AliasSpeaker",
        text="alias test",
    )
    person = create_person(repository, name="人物テストAlias", privacy_level="private")

    link_line_speaker_to_person(
        repository,
        chat_id="chat_alias",
        speaker_name="AliasSpeaker",
        person_id=person["id"],
        yes=True,
        add_alias=True,
    )
    link_line_speaker_to_person(
        repository,
        chat_id="chat_alias",
        speaker_name="AliasSpeaker",
        person_id=person["id"],
        yes=True,
        add_alias=True,
    )

    updated = get_person(repository, person["id"])
    assert updated is not None
    aliases = [alias["alias"] for alias in updated["aliases"]]
    assert aliases.count("AliasSpeaker") == 1

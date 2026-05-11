from __future__ import annotations

import sqlite3

from personal_lifelog_rag.db.checks import run_db_check
from personal_lifelog_rag.db.repository import LifelogRepository


def test_db_check_detects_line_speaker_link_orphan_and_empty_values(tmp_path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    LifelogRepository(db_path).initialize()
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            """
            INSERT INTO line_speaker_links (
                id, chat_id, speaker_name, person_id, source, confidence, verified_by_user
            )
            VALUES ('line_link_bad', '', '', 'missing_person', 'manual', 1.2, 1)
            """
        )
        connection.commit()

    report = run_db_check(db_path)

    assert report["line_person_links"]["line_speaker_links_orphan_person_refs"] == 1
    assert report["line_person_links"]["line_speaker_links_empty_chat_id"] == 1
    assert report["line_person_links"]["line_speaker_links_empty_speaker_name"] == 1
    assert report["line_person_links"]["line_speaker_links_invalid_confidence"] == 1
    assert not report["strict"]["ok"]


def test_db_check_detects_person_line_mentions_orphan_and_invalid_type(tmp_path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    LifelogRepository(db_path).initialize()
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            """
            INSERT INTO person_line_mentions (
                id, person_id, message_id, mention_type, source, confidence
            )
            VALUES ('mention_bad', 'missing_person', 'missing_message', 'auto_relation', 'manual', 0.8)
            """
        )
        connection.commit()

    report = run_db_check(db_path)

    assert report["line_person_links"]["person_line_mentions_orphan_person_refs"] == 1
    assert report["line_person_links"]["person_line_mentions_orphan_message_refs"] == 1
    assert report["line_person_links"]["person_line_mentions_invalid_mention_type"] == 1
    assert not report["strict"]["ok"]

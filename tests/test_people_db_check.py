from __future__ import annotations

import sqlite3

from personal_lifelog_rag.db.checks import run_db_check
from personal_lifelog_rag.db.repository import LifelogRepository


def test_db_check_detects_invalid_media_people_rows(tmp_path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    LifelogRepository(db_path).initialize()
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            """
            INSERT INTO media_people (
                media_id, person_id, source, confidence, face_id, face_cluster_id, verified_by_user
            )
            VALUES ('missing_media', 'missing_person', 'auto_face_guess', 1.2,
                    'missing_face', 'missing_cluster', 1)
            """
        )
        connection.commit()

    report = run_db_check(db_path)
    checks = report["person_event_media"]
    assert checks["media_people_orphan_media_refs"] == 1
    assert checks["media_people_orphan_person_refs"] == 1
    assert checks["media_people_orphan_face_refs"] == 1
    assert checks["media_people_orphan_cluster_refs"] == 1
    assert checks["media_people_invalid_source"] == 1
    assert checks["media_people_invalid_confidence"] == 1
    assert not report["strict"]["ok"]


def test_db_check_detects_invalid_event_people_rows(tmp_path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    LifelogRepository(db_path).initialize()
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            """
            INSERT INTO event_people (
                event_id, person_id, source, confidence, evidence_count, media_count, line_count
            )
            VALUES ('missing_event', 'missing_person', 'auto_relationship', -0.1, -1, 0, 0)
            """
        )
        connection.commit()

    report = run_db_check(db_path)
    checks = report["person_event_media"]
    assert checks["event_people_orphan_event_refs"] == 1
    assert checks["event_people_orphan_person_refs"] == 1
    assert checks["event_people_invalid_source"] == 1
    assert checks["event_people_invalid_confidence"] == 1
    assert not report["strict"]["ok"]

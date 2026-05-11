from __future__ import annotations

import sqlite3

from personal_lifelog_rag.db.checks import run_db_check
from personal_lifelog_rag.db.repository import LifelogRepository


def test_db_check_detects_orphan_face_detection(tmp_path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            """
            INSERT INTO face_detections (
                id, media_id, detected_at, engine, model_name, status,
                bbox_x, bbox_y, bbox_w, bbox_h, privacy_level, review_status
            )
            VALUES ('face_orphan', 'missing_media', '2025-01-01T10:00:00', 'fake', 'fake', 'success',
                    0, 0, 10, 10, 'private', 'unreviewed')
            """
        )
        connection.commit()

    report = run_db_check(db_path)

    assert report["face_detections"]["orphan_media_refs"] == 1
    assert not report["strict"]["ok"]


def test_db_check_detects_invalid_face_bbox(tmp_path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(id="media_face", file_path="/tmp/fake.jpg", file_hash="hash-face", media_type="image")
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO face_detections (
                id, media_id, detected_at, engine, model_name, status,
                bbox_x, bbox_y, bbox_w, bbox_h, image_width, image_height,
                privacy_level, review_status
            )
            VALUES ('face_bad_bbox', 'media_face', '2025-01-01T10:00:00', 'fake', 'fake', 'success',
                    0, 0, -10, 10, 100, 100, 'private', 'unreviewed')
            """
        )
        connection.commit()

    report = run_db_check(db_path)

    assert report["face_detections"]["invalid_bbox"] == 1
    assert not report["strict"]["ok"]


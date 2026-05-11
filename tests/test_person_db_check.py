from __future__ import annotations

import sqlite3

from personal_lifelog_rag.db.checks import run_db_check
from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.db.schema import initialize_schema


def test_db_check_detects_orphan_person_face_cluster_link(tmp_path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    LifelogRepository(db_path).initialize()
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            """
            INSERT INTO person_face_clusters (person_id, face_cluster_id, verified_by_user, source)
            VALUES ('missing_person', 'missing_cluster', 1, 'manual')
            """
        )
        connection.commit()

    report = run_db_check(db_path)

    assert report["persons"]["person_face_clusters_orphan_person_refs"] == 1
    assert report["persons"]["person_face_clusters_orphan_cluster_refs"] == 1
    assert not report["strict"]["ok"]


def test_db_check_detects_person_alias_orphan_and_invalid_source(tmp_path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    LifelogRepository(db_path).initialize()
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            """
            INSERT INTO person_aliases (id, person_id, alias, source)
            VALUES ('alias_bad', 'missing_person', 'Alias', 'auto_guess')
            """
        )
        connection.commit()

    report = run_db_check(db_path)

    assert report["persons"]["person_aliases_orphan_person_refs"] == 1
    assert report["persons"]["person_aliases_invalid_source"] == 1
    assert not report["strict"]["ok"]


def test_db_check_warns_about_rejected_cluster_link(tmp_path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(id="media_face", file_path="/tmp/fake.jpg", file_hash="hash-face", media_type="image")
    with connect(db_path) as connection:
        initialize_schema(connection)
        connection.execute(
            "INSERT INTO persons (id, display_name, aliases_json, privacy_level) VALUES ('person_a', '人物A', '[]', 'private')"
        )
        connection.execute(
            """
            INSERT INTO face_detections (
                id, media_id, detected_at, engine, model_name, status,
                bbox_x, bbox_y, bbox_w, bbox_h, privacy_level, review_status
            )
            VALUES ('face_a', 'media_face', '2025-01-01T10:00:00', 'fake', 'fake', 'success',
                    0, 0, 10, 10, 'private', 'accepted')
            """
        )
        connection.execute(
            """
            INSERT INTO face_clusters (
                id, cluster_label, representative_face_id, face_count,
                clustering_method, status, review_status, privacy_level
            )
            VALUES ('cluster_rejected', 'person_candidate_001', 'face_a', 1,
                    'manual', 'rejected', 'reviewed', 'private')
            """
        )
        connection.execute("INSERT INTO face_cluster_members (cluster_id, face_id) VALUES ('cluster_rejected', 'face_a')")
        connection.execute(
            """
            INSERT INTO person_face_clusters (person_id, face_cluster_id, verified_by_user, source)
            VALUES ('person_a', 'cluster_rejected', 1, 'manual')
            """
        )
        connection.commit()

    report = run_db_check(db_path)

    assert report["persons"]["person_face_clusters_rejected_cluster_links"] == 1
    assert report["strict"]["ok"]

from __future__ import annotations

import sqlite3
from pathlib import Path

from personal_lifelog_rag.db.checks import run_db_check
from personal_lifelog_rag.db.repository import LifelogRepository


def test_db_check_detects_orphan_face_embedding(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            """
            INSERT INTO face_embeddings (
                face_id, embedding_model, embedding_dim, embedding_blob,
                embedding_format, normalized, status
            )
            VALUES ('missing_face', 'test_model', 3, ?, 'float32_numpy', 1, 'success')
            """,
            [b"\x00" * 12],
        )
        connection.commit()

    report = run_db_check(db_path)

    assert report["face_embedding_clusters"]["face_embeddings_orphan_face_refs"] == 1
    assert not report["strict"]["ok"]


def test_db_check_detects_empty_face_embedding_blob(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(id="media_face", file_path="/tmp/fake.jpg", file_hash="hash-face", media_type="image")
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO face_detections (
                id, media_id, detected_at, engine, model_name, status,
                bbox_x, bbox_y, bbox_w, bbox_h, privacy_level, review_status
            )
            VALUES ('face_empty_blob', 'media_face', '2025-01-01T10:00:00', 'fake', 'fake', 'success',
                    0, 0, 10, 10, 'private', 'unreviewed')
            """
        )
        connection.execute(
            """
            INSERT INTO face_embeddings (
                face_id, embedding_model, embedding_dim, embedding_blob,
                embedding_format, normalized, status
            )
            VALUES ('face_empty_blob', 'test_model', 3, X'', 'float32_numpy', 1, 'success')
            """
        )
        connection.commit()

    report = run_db_check(db_path)

    assert report["face_embedding_clusters"]["face_embeddings_success_empty_blob"] == 1
    assert report["face_embedding_clusters"]["face_embeddings_invalid_dim"] == 1
    assert not report["strict"]["ok"]


def test_db_check_detects_invalid_face_cluster_references(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            """
            INSERT INTO face_clusters (
                id, cluster_label, representative_face_id, face_count,
                clustering_method, status, review_status, privacy_level
            )
            VALUES ('cluster_bad', 'person_candidate_001', 'missing_face', 1,
                    'dbscan_cosine', 'unreviewed', 'unreviewed', 'private')
            """
        )
        connection.execute(
            """
            INSERT INTO face_cluster_members (cluster_id, face_id, distance_to_centroid)
            VALUES ('cluster_bad', 'missing_face', -0.1)
            """
        )
        connection.commit()

    report = run_db_check(db_path)

    checks = report["face_embedding_clusters"]
    assert checks["face_clusters_invalid_representative_face"] == 1
    assert checks["face_cluster_members_orphan_face_refs"] == 1
    assert checks["face_cluster_members_invalid_distance"] == 1
    assert not report["strict"]["ok"]

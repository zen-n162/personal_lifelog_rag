from __future__ import annotations

from pathlib import Path

from PIL import Image

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.db.schema import initialize_schema
from personal_lifelog_rag.faces.person_service import (
    create_person,
    link_person_face_cluster,
    persons_for_media,
    unlink_person_face_cluster,
)


def test_manual_person_face_cluster_link_and_unlink(tmp_path: Path) -> None:
    repository = _seed_cluster(tmp_path)
    person = create_person(repository, name="人物テストA", public_name="人物A", privacy_level="public_alias")

    result = link_person_face_cluster(repository, person_id=person["id"], cluster_id="cluster_a", yes=True)
    assert result["person"]["linked_clusters_count"] == 1
    candidates = persons_for_media(repository, "media_face_person", public_mode=True)
    assert candidates[0]["display_name"] == "人物A"

    unlink = unlink_person_face_cluster(repository, person_id=person["id"], cluster_id="cluster_a", yes=True)
    assert unlink["deleted"] == 1
    assert persons_for_media(repository, "media_face_person") == []


def test_persons_for_media_requires_accepted_cluster(tmp_path: Path) -> None:
    repository = _seed_cluster(tmp_path, cluster_status="unreviewed")
    person = create_person(repository, name="人物テストA")
    link_person_face_cluster(repository, person_id=person["id"], cluster_id="cluster_a", yes=True)
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        connection.execute("UPDATE face_clusters SET status = 'rejected' WHERE id = 'cluster_a'")
        connection.commit()

    assert persons_for_media(repository, "media_face_person") == []


def _seed_cluster(tmp_path: Path, *, cluster_status: str = "accepted") -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    image_path = tmp_path / "face.jpg"
    Image.new("RGB", (96, 96), "white").save(image_path)
    repository.add_media_item(
        id="media_face_person",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-face-person",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        connection.execute(
            """
            INSERT INTO face_detections (
                id, media_id, detected_at, engine, model_name, status,
                bbox_x, bbox_y, bbox_w, bbox_h, detection_score,
                image_width, image_height, privacy_level, review_status
            )
            VALUES ('face_a', 'media_face_person', '2024-12-24T10:00:00+09:00', 'fake', 'fake', 'success',
                    10, 10, 32, 32, 0.9, 96, 96, 'private', 'accepted')
            """
        )
        connection.execute(
            """
            INSERT INTO face_clusters (
                id, cluster_label, representative_face_id, face_count,
                first_seen_at, last_seen_at, clustering_method, status,
                review_status, privacy_level
            )
            VALUES ('cluster_a', 'person_candidate_001', 'face_a', 1,
                    '2024-12-24T10:00:00+09:00', '2024-12-24T10:00:00+09:00',
                    'manual', ?, 'reviewed', 'private')
            """,
            [cluster_status],
        )
        connection.execute(
            "INSERT INTO face_cluster_members (cluster_id, face_id) VALUES ('cluster_a', 'face_a')"
        )
        connection.commit()
    return repository

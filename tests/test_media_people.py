from __future__ import annotations

from pathlib import Path

from PIL import Image

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.db.schema import initialize_schema
from personal_lifelog_rag.faces.person_service import create_person, link_person_face_cluster
from personal_lifelog_rag.people.integration import build_media_people, list_media_people


def test_verified_face_cluster_builds_media_people(tmp_path: Path) -> None:
    repository, person_id = _seed_face_person(tmp_path)

    dry = build_media_people(repository, start_date="2025-01-01", end_date="2025-01-31", dry_run=True)
    assert dry["would_insert"] == 1
    assert repository.stats()["media_people"] == 0

    report = build_media_people(
        repository,
        start_date="2025-01-01",
        end_date="2025-01-31",
        dry_run=False,
        yes=True,
    )
    assert report["inserted"] == 1
    rows = list_media_people(repository, date_value="2025-01-10", public_mode=False)
    assert rows[0]["media_id"] == "media_people_face"
    assert rows[0]["person_id"] == person_id
    assert rows[0]["source"] == "face_cluster"
    assert rows[0]["confidence"] == 0.85


def test_unverified_or_rejected_face_clusters_are_excluded_from_media_people(tmp_path: Path) -> None:
    repository, _person_id = _seed_face_person(tmp_path, cluster_status="unreviewed", link_manually=False)
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        connection.execute(
            """
            INSERT INTO person_face_clusters (person_id, face_cluster_id, verified_by_user, source)
            VALUES ('person_unverified', 'cluster_people_face', 0, 'manual')
            """
        )
        connection.commit()

    report = build_media_people(repository, start_date="2025-01-01", end_date="2025-01-31", dry_run=True)
    assert report["would_insert"] == 0

    repository2, _ = _seed_face_person(tmp_path / "rejected", cluster_status="accepted")
    with connect(repository2.db_path) as connection:
        initialize_schema(connection)
        connection.execute("UPDATE face_clusters SET status = 'rejected' WHERE id = 'cluster_people_face'")
        connection.commit()
    rejected = build_media_people(repository2, start_date="2025-01-01", end_date="2025-01-31", dry_run=True)
    assert rejected["would_insert"] == 0


def _seed_face_person(
    tmp_path: Path,
    *,
    cluster_status: str = "accepted",
    link_manually: bool = True,
) -> tuple[LifelogRepository, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    image_path = tmp_path / "face.jpg"
    Image.new("RGB", (96, 96), "white").save(image_path)
    repository.add_media_item(
        id="media_people_face",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash=f"hash-media-people-{cluster_status}",
        media_type="image",
        captured_at="2025-01-10T10:00:00+09:00",
    )
    person = create_person(repository, name="人物テストMedia", public_name="人物A", privacy_level="public_alias")
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        connection.execute(
            """
            INSERT INTO face_detections (
                id, media_id, detected_at, engine, model_name, status,
                bbox_x, bbox_y, bbox_w, bbox_h, privacy_level, review_status
            )
            VALUES ('face_people', 'media_people_face', '2025-01-10T10:00:00+09:00',
                    'fake', 'fake', 'success', 10, 10, 32, 32, 'private', 'accepted')
            """
        )
        connection.execute(
            """
            INSERT INTO face_clusters (
                id, cluster_label, representative_face_id, face_count,
                clustering_method, status, review_status, privacy_level
            )
            VALUES ('cluster_people_face', 'person_candidate_001', 'face_people', 1,
                    'manual', ?, 'reviewed', 'private')
            """,
            [cluster_status],
        )
        connection.execute("INSERT INTO face_cluster_members (cluster_id, face_id) VALUES ('cluster_people_face', 'face_people')")
        if not link_manually:
            connection.execute(
                "INSERT INTO persons (id, display_name, aliases_json, privacy_level) VALUES ('person_unverified', '人物未確認', '[]', 'private')"
            )
        connection.commit()
    if link_manually:
        link_person_face_cluster(repository, person_id=person["id"], cluster_id="cluster_people_face", yes=True)
    return repository, person["id"]

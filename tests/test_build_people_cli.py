from __future__ import annotations

from pathlib import Path

from PIL import Image

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.db.schema import initialize_schema
from personal_lifelog_rag.faces.person_service import create_person, link_person_face_cluster


def test_build_people_cli_dry_runs_do_not_write(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_verified_face_cluster(repository, tmp_path)
    repository.add_event(id="event_cli_people", date="2025-03-02", start_time="10:00:00", title="Dummy")
    repository.add_event_evidence(event_id="event_cli_people", evidence_type="photo", evidence_id="media_cli_people")

    assert main(
        [
            "--db-path",
            str(db_path),
            "build-media-people",
            "--from",
            "2025-03-01",
            "--to",
            "2025-03-31",
            "--dry-run",
        ]
    ) == 0
    assert "would_insert: 1" in capsys.readouterr().out
    assert repository.stats()["media_people"] == 0

    assert main(
        [
            "--db-path",
            str(db_path),
            "build-event-people",
            "--from",
            "2025-03-01",
            "--to",
            "2025-03-31",
            "--dry-run",
        ]
    ) == 0
    assert "would_insert: 0" in capsys.readouterr().out
    assert repository.stats()["event_people"] == 0


def _seed_verified_face_cluster(repository: LifelogRepository, tmp_path: Path) -> None:
    image_path = tmp_path / "face_cli_people.jpg"
    Image.new("RGB", (80, 80), "white").save(image_path)
    repository.add_media_item(
        id="media_cli_people",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-cli-people",
        media_type="image",
        captured_at="2025-03-02T10:00:00+09:00",
    )
    person = create_person(repository, name="人物テストCLI統合", privacy_level="private")
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        connection.execute(
            """
            INSERT INTO face_detections (
                id, media_id, status, bbox_x, bbox_y, bbox_w, bbox_h, privacy_level, review_status
            )
            VALUES ('face_cli_people', 'media_cli_people', 'success', 0, 0, 10, 10, 'private', 'accepted')
            """
        )
        connection.execute(
            """
            INSERT INTO face_clusters (
                id, cluster_label, representative_face_id, face_count, clustering_method,
                status, review_status, privacy_level
            )
            VALUES ('cluster_cli_people', 'person_candidate_001', 'face_cli_people', 1, 'manual',
                    'accepted', 'reviewed', 'private')
            """
        )
        connection.execute("INSERT INTO face_cluster_members (cluster_id, face_id) VALUES ('cluster_cli_people', 'face_cli_people')")
        connection.commit()
    link_person_face_cluster(repository, person_id=person["id"], cluster_id="cluster_cli_people", yes=True)

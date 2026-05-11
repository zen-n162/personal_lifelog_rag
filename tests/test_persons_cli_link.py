from __future__ import annotations

from pathlib import Path

from PIL import Image

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.db.schema import initialize_schema


def test_persons_cli_link_and_unlink_face_cluster(tmp_path: Path, capsys) -> None:
    db_path = _seed_cluster(tmp_path)
    main(["--db-path", str(db_path), "persons", "create", "--name", "人物テストA"])
    capsys.readouterr()
    person_id = LifelogRepository(db_path)._fetch_all("SELECT id FROM persons", [])[0]["id"]

    assert main(
        [
            "--db-path",
            str(db_path),
            "persons",
            "link-face-cluster",
            "--person-id",
            person_id,
            "--cluster-id",
            "cluster_cli",
            "--yes",
        ]
    ) == 0
    assert "linked person" in capsys.readouterr().out

    assert main(
        [
            "--db-path",
            str(db_path),
            "persons",
            "unlink-face-cluster",
            "--person-id",
            person_id,
            "--cluster-id",
            "cluster_cli",
            "--yes",
        ]
    ) == 0
    assert "unlinked person" in capsys.readouterr().out


def _seed_cluster(tmp_path: Path) -> Path:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    image_path = tmp_path / "face.jpg"
    Image.new("RGB", (96, 96), "white").save(image_path)
    repository.add_media_item(
        id="media_face_cli_person",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-face-cli-person",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    with connect(db_path) as connection:
        initialize_schema(connection)
        connection.execute(
            """
            INSERT INTO face_detections (
                id, media_id, detected_at, engine, model_name, status,
                bbox_x, bbox_y, bbox_w, bbox_h, privacy_level, review_status
            )
            VALUES ('face_cli_person', 'media_face_cli_person', '2024-12-24T10:00:00+09:00',
                    'fake', 'fake', 'success', 10, 10, 32, 32, 'private', 'accepted')
            """
        )
        connection.execute(
            """
            INSERT INTO face_clusters (
                id, cluster_label, representative_face_id, face_count,
                clustering_method, status, review_status, privacy_level
            )
            VALUES ('cluster_cli', 'person_candidate_001', 'face_cli_person', 1,
                    'manual', 'accepted', 'reviewed', 'private')
            """
        )
        connection.execute("INSERT INTO face_cluster_members (cluster_id, face_id) VALUES ('cluster_cli', 'face_cli_person')")
        connection.commit()
    return db_path

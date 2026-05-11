from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.db.schema import initialize_schema


def test_face_cluster_cli_dry_run_and_yes(tmp_path: Path, capsys) -> None:
    db_path = _seed_embeddings(tmp_path)

    dry_code = main(
        [
            "--db-path",
            str(db_path),
            "face-cluster",
            "--distance-threshold",
            "0.2",
            "--min-samples",
            "2",
            "--dry-run",
        ]
    )
    dry_output = capsys.readouterr().out
    assert dry_code == 0
    assert "dry_run: True" in dry_output

    run_code = main(
        [
            "--db-path",
            str(db_path),
            "face-cluster",
            "--distance-threshold",
            "0.2",
            "--min-samples",
            "2",
            "--yes",
        ]
    )
    run_output = capsys.readouterr().out
    assert run_code == 0
    assert "cluster candidates: 1" in run_output
    assert "clusters written: 1" in run_output

    stats_code = main(["--db-path", str(db_path), "face-cluster-stats", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert stats_code == 0
    assert payload["clusters_total"] == 1
    assert payload["face_embeddings_total"] == 3

    show_code = main(["--db-path", str(db_path), "face-cluster-show", "--status", "unreviewed", "--limit", "10"])
    show_output = capsys.readouterr().out
    assert show_code == 0
    assert "person_candidate_001" in show_output


def test_update_face_cluster_cli_changes_status(tmp_path: Path, capsys) -> None:
    db_path = _seed_embeddings(tmp_path)
    main(
        [
            "--db-path",
            str(db_path),
            "face-cluster",
            "--distance-threshold",
            "0.2",
            "--min-samples",
            "2",
            "--yes",
        ]
    )
    capsys.readouterr()
    cluster_id = LifelogRepository(db_path)._fetch_all("SELECT id FROM face_clusters", [])[0]["id"]

    assert main(["--db-path", str(db_path), "update-face-cluster", "--cluster-id", cluster_id, "--status", "accepted"]) == 0
    output = capsys.readouterr().out
    row = LifelogRepository(db_path)._fetch_all("SELECT status FROM face_clusters WHERE id = ?", [cluster_id])[0]
    assert "status=accepted" in output
    assert row["status"] == "accepted"


def _seed_embeddings(tmp_path: Path) -> Path:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    image_path = tmp_path / "source.jpg"
    Image.new("RGB", (96, 96), "white").save(image_path)
    repository.add_media_item(
        id="media_face_cluster_cli",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-face-cluster-cli",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    embeddings = [
        ("face_cli_a", [1.0, 0.0, 0.0]),
        ("face_cli_b", [0.99, 0.1, 0.0]),
        ("face_cli_c", [0.0, 1.0, 0.0]),
    ]
    with connect(db_path) as connection:
        initialize_schema(connection)
        for face_id, raw_embedding in embeddings:
            connection.execute(
                """
                INSERT INTO face_detections (
                    id, media_id, detected_at, engine, model_name, status,
                    bbox_x, bbox_y, bbox_w, bbox_h, detection_score,
                    image_width, image_height, privacy_level, review_status
                )
                VALUES (?, 'media_face_cluster_cli', '2024-12-24T10:00:00+09:00', 'fake', 'fake', 'success',
                        10, 10, 32, 32, 0.9, 96, 96, 'private', 'unreviewed')
                """,
                [face_id],
            )
            embedding = _normalized(raw_embedding)
            connection.execute(
                """
                INSERT INTO face_embeddings (
                    face_id, embedding_model, embedding_dim, embedding_blob,
                    embedding_format, normalized, status
                )
                VALUES (?, 'test_model', ?, ?, 'float32_numpy', 1, 'success')
                """,
                [face_id, int(embedding.shape[0]), embedding.astype(np.float32).tobytes()],
            )
        connection.commit()
    return db_path


def _normalized(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / float(np.linalg.norm(array))

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.reporting.report_builder import build_report, write_report
from personal_lifelog_rag.reporting.schemas import ReportOptions


def test_public_report_does_not_include_face_cluster_identifiers(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_media_item(id="media_face", file_path="/tmp/fake.jpg", file_hash="hash-face", media_type="image")
    with sqlite3.connect(repository.db_path) as connection:
        connection.execute(
            """
            INSERT INTO face_detections (
                id, media_id, detected_at, engine, model_name, status,
                bbox_x, bbox_y, bbox_w, bbox_h, privacy_level, review_status
            )
            VALUES ('face_sensitive_001', 'media_face', '2025-01-01T10:00:00', 'fake', 'fake', 'success',
                    0, 0, 10, 10, 'private', 'unreviewed')
            """
        )
        connection.execute(
            """
            INSERT INTO face_clusters (
                id, cluster_label, representative_face_id, face_count,
                clustering_method, status, review_status, privacy_level
            )
            VALUES ('face_cluster_sensitive', 'person_candidate_001', 'face_sensitive_001', 1,
                    'manual', 'unreviewed', 'unreviewed', 'private')
            """
        )
        connection.execute(
            """
            INSERT INTO face_cluster_members (cluster_id, face_id)
            VALUES ('face_cluster_sensitive', 'face_sensitive_001')
            """
        )
        connection.commit()

    report = build_report(repository, ReportOptions(mode="public"))
    result = write_report(report, output_path=tmp_path / "public_report.md", save_json=True)
    markdown = result.markdown_path.read_text(encoding="utf-8")
    json_text = result.json_path.read_text(encoding="utf-8") if result.json_path else json.dumps(report)

    assert "face_sensitive_001" not in markdown
    assert "face_cluster_sensitive" not in markdown
    assert "person_candidate_001" not in markdown
    assert "face_sensitive_001" not in json_text
    assert "face_cluster_sensitive" not in json_text
    assert "person_candidate_001" not in json_text

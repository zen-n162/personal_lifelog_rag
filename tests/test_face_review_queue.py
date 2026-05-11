from __future__ import annotations

from pathlib import Path

from PIL import Image

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.faces.face_service import list_face_detections, run_face_detection, update_face_review_status
from personal_lifelog_rag.faces.schemas import FaceDetectOptions


def test_face_review_queue_and_status_update(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    image_path = tmp_path / "face.jpg"
    Image.new("RGB", (64, 64), "white").save(image_path)
    repository.add_media_item(
        id="media_face",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-face",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    run_face_detection(repository, FaceDetectOptions(date="2024-12-24", engine="fake", limit=1))
    rows = list_face_detections(repository, review_status="unreviewed")
    face_id = rows[0]["id"]

    row = update_face_review_status(repository, face_id=face_id, review_status="bad_detection")

    assert row["review_status"] == "bad_detection"
    assert list_face_detections(repository, review_status="unreviewed") == []


from __future__ import annotations

from pathlib import Path

from PIL import Image

from personal_lifelog_rag.faces.face_service import face_stats, format_face_stats, run_face_detection
from personal_lifelog_rag.faces.schemas import FaceDetectOptions
from personal_lifelog_rag.db.repository import LifelogRepository


def test_face_stats_returns_counts(tmp_path: Path) -> None:
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

    stats = face_stats(repository, date_from="2024-12-24", date_to="2024-12-24")

    assert stats["total"] == 1
    assert "success" in format_face_stats(stats)


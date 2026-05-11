from __future__ import annotations

from pathlib import Path

from PIL import Image

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository


def test_face_detect_dry_run_does_not_write(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    image_path = tmp_path / "face.jpg"
    Image.new("RGB", (80, 80), "white").save(image_path)
    repository.add_media_item(
        id="media_face",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-face",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )

    assert main(["--db-path", str(db_path), "face-detect", "--date", "2024-12-24", "--engine", "fake", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "dry_run: True" in output
    assert repository._fetch_all("SELECT * FROM face_detections", []) == []


def test_face_detect_cli_writes_with_fake_engine(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    image_path = tmp_path / "face.jpg"
    Image.new("RGB", (80, 80), "white").save(image_path)
    repository.add_media_item(
        id="media_face",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-face",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )

    assert main(["--db-path", str(db_path), "face-detect", "--date", "2024-12-24", "--engine", "fake", "--limit", "1"]) == 0
    rows = repository._fetch_all("SELECT status FROM face_detections", [])
    assert rows[0]["status"] == "success"


def test_face_detect_cli_reads_yunet_config_without_writing_on_dry_run(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    image_path = tmp_path / "face.jpg"
    Image.new("RGB", (80, 80), "white").save(image_path)
    repository.add_media_item(
        id="media_face",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-face",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    config_path = tmp_path / "model_runtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                "face_detection:",
                '  engine: "opencv_yunet"',
                f'  model_path: "{tmp_path / "missing_yunet.onnx"}"',
                "  score_threshold: 0.8",
                "  nms_threshold: 0.3",
                "  top_k: 1000",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["--db-path", str(db_path), "face-detect", "--date", "2024-12-24", "--config", str(config_path), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "engine: opencv_yunet" in output
    assert repository._fetch_all("SELECT * FROM face_detections", []) == []


def test_face_diagnostics_cli_reports_yunet_model_from_config(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "model_runtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                "face_detection:",
                '  engine: "opencv_yunet"',
                f'  model_path: "{tmp_path / "missing_yunet.onnx"}"',
            ]
        ),
        encoding="utf-8",
    )

    assert main(["face-diagnostics", "--config", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "yunet model path:" in output
    assert "yunet available: False" in output

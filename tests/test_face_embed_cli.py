from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository


def test_face_embedding_diagnostics_json_reports_fake_engine(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "model_runtime.yaml"
    config_path.write_text(
        """
face_embedding:
  engine: "fake"
  embedding_dim: 8
  local_only: true
""",
        encoding="utf-8",
    )

    assert main(["face-embedding-diagnostics", "--config", str(config_path), "--engine", "fake", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selected_engine"] == "fake"
    assert payload["engine_available"] is True
    assert payload["local_only"] is True


def test_face_embed_cli_dry_run_and_fake_write(tmp_path: Path, capsys) -> None:
    db_path = _seed_face_detection(tmp_path)

    dry_code = main(
        [
            "--db-path",
            str(db_path),
            "face-embed",
            "--date",
            "2024-12-24",
            "--engine",
            "fake",
            "--dry-run",
        ]
    )
    dry_output = capsys.readouterr().out

    run_code = main(
        [
            "--db-path",
            str(db_path),
            "face-embed",
            "--date",
            "2024-12-24",
            "--engine",
            "fake",
            "--skip-existing",
        ]
    )
    run_output = capsys.readouterr().out

    repository = LifelogRepository(db_path)
    rows = repository._fetch_all("SELECT status FROM face_embeddings", [])
    assert dry_code == 0
    assert "dry_run: True" in dry_output
    assert run_code == 0
    assert "success: 1" in run_output
    assert rows[0]["status"] == "success"


def _seed_face_detection(tmp_path: Path) -> Path:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    image_path = tmp_path / "face.jpg"
    Image.new("RGB", (96, 96), "white").save(image_path)
    repository.add_media_item(
        id="media_face_cli",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-face-cli",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    assert main(
        [
            "--db-path",
            str(db_path),
            "face-detect",
            "--date",
            "2024-12-24",
            "--engine",
            "fake",
            "--limit",
            "1",
        ]
    ) == 0
    return db_path

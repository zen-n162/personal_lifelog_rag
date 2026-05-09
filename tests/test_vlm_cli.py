from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from personal_lifelog_rag.app import cli as cli_module
from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.vlm.engines import FakeVlmEngine


def test_analyze_images_vlm_cli_dry_run_and_fake_engine(tmp_path: Path, capsys) -> None:
    db_path = _seed_db(tmp_path)

    dry_code = main(
        [
            "--db-path",
            str(db_path),
            "analyze-images",
            "--date",
            "2024-12-24",
            "--engine",
            "fake",
            "--limit",
            "5",
            "--dry-run",
        ]
    )
    dry_output = capsys.readouterr().out

    run_code = main(
        [
            "--db-path",
            str(db_path),
            "analyze-images",
            "--date",
            "2024-12-24",
            "--engine",
            "fake",
            "--limit",
            "5",
        ]
    )
    run_output = capsys.readouterr().out

    assert dry_code == 0
    assert "- dry_run: True" in dry_output
    assert run_code == 0
    assert "- dry_run: True" in run_output
    assert "- success: 0" in run_output
    assert LifelogRepository(db_path).get_media_vlm("media_vlm_cli") is None


def test_vlm_stats_show_and_image_search_cli(tmp_path: Path, capsys) -> None:
    db_path = _seed_db(tmp_path)
    main(["--db-path", str(db_path), "analyze-images", "--date", "2024-12-24", "--engine", "fake", "--allow-fake-write"])
    capsys.readouterr()
    LifelogRepository(db_path).upsert_media_vlm(
        media_id="media_vlm_cli",
        caption="ラーメンの可能性がある写真",
        short_caption="ラーメン候補",
        food_cues=["ramen_possible"],
        status="success",
        vlm_engine="unit_test_vlm",
        model_name="unit-test-vlm",
    )

    stats_code = main(["--db-path", str(db_path), "vlm-stats", "--json"])
    stats_payload = json.loads(capsys.readouterr().out)
    show_code = main(["--db-path", str(db_path), "vlm-show", "media_vlm_cli"])
    show_output = capsys.readouterr().out
    search_code = main(["--db-path", str(db_path), "image-search", "ラーメン", "--json"])
    search_payload = json.loads(capsys.readouterr().out)

    assert stats_code == 0
    assert stats_payload["status_counts"]["success"] == 1
    assert show_code == 0
    assert "media_vlm_cli" in show_output
    assert search_code == 0
    assert search_payload["results"][0]["media_id"] == "media_vlm_cli"


def test_vlm_show_can_display_error_message(tmp_path: Path, capsys) -> None:
    db_path = _seed_db(tmp_path)
    LifelogRepository(db_path).upsert_media_vlm(
        media_id="media_vlm_cli",
        status="failed",
        vlm_engine="qwen3_vl_transformers",
        model_name="local-qwen",
        error_message="Transformers VLM failed: KeyError: 'caption'\nTraceback details",
    )

    code = main(["--db-path", str(db_path), "vlm-show", "media_vlm_cli", "--show-errors"])
    output = capsys.readouterr().out

    assert code == 0
    assert "error_message:" in output
    assert "prompt_template:" in output
    assert "analyzed_at:" in output
    assert "KeyError" in output


def test_analyze_images_prompt_template_saves_safety_metadata(tmp_path: Path) -> None:
    db_path = _seed_db(tmp_path)

    code = main(
        [
            "--db-path",
            str(db_path),
            "analyze-images",
            "--date",
            "2024-12-24",
            "--engine",
            "fake",
            "--prompt-template",
            "lifelog_structured_tags_v1",
            "--force",
            "--allow-fake-write",
        ]
    )
    repository = LifelogRepository(db_path)
    row = repository.get_media_vlm("media_vlm_cli")

    assert code == 0
    assert row["prompt_version"] == "lifelog_structured_tags_v1"
    assert row["evidence_strength"] == "weak"
    assert "low_confidence" in row["safety_flags_json"]


def test_analyze_images_accepts_model_runtime_config(tmp_path: Path, capsys, monkeypatch) -> None:
    db_path = _seed_db(tmp_path)
    model_path = tmp_path / "models" / "Qwen3-VL-8B-Thinking"
    model_path.mkdir(parents=True)
    config_path = tmp_path / "model_runtime.yaml"
    config_path.write_text(
        f"""
models:
  vlm:
    engine: "qwen3_vl_transformers"
    model_name: "Qwen/Qwen3-VL-8B-Thinking"
    model_path: "{model_path}"
    device: "cuda"
    dtype: "bfloat16"
    local_files_only: true
    prompt_version: "lifelog_safe_caption_v1"
    max_image_size: 768
    max_new_tokens: 256
""",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_get_vlm_engine(name: str | None = None, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return FakeVlmEngine(model_name=str(kwargs.get("model_path") or kwargs.get("model_name") or "fake"))

    monkeypatch.setattr(cli_module, "get_vlm_engine", fake_get_vlm_engine)

    code = main(
        [
            "--db-path",
            str(db_path),
            "analyze-images",
            "--date",
            "2024-12-24",
            "--config",
            str(config_path),
            "--limit",
            "1",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "- dry_run: True" in output
    assert captured["name"] == "qwen3_vl_transformers"
    assert captured["kwargs"] == {
        "model_name": "Qwen/Qwen3-VL-8B-Thinking",
        "model_path": str(model_path),
        "device": "cuda",
        "dtype": "bfloat16",
        "local_files_only": True,
        "max_image_size": 768,
        "max_new_tokens": 256,
    }


def test_analyze_images_cli_engine_and_prompt_override_config(tmp_path: Path, monkeypatch) -> None:
    db_path = _seed_db(tmp_path)
    model_path = tmp_path / "models" / "Qwen3-VL-8B-Thinking"
    model_path.mkdir(parents=True)
    config_path = tmp_path / "model_runtime.yaml"
    config_path.write_text(
        f"""
models:
  vlm:
    engine: "qwen3_vl_transformers"
    model_path: "{model_path}"
    local_files_only: true
    prompt_version: "lifelog_safe_caption_v1"
""",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_get_vlm_engine(name: str | None = None, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return FakeVlmEngine(model_name="override-test")

    monkeypatch.setattr(cli_module, "get_vlm_engine", fake_get_vlm_engine)

    code = main(
        [
            "--db-path",
            str(db_path),
            "analyze-images",
            "--date",
            "2024-12-24",
            "--config",
            str(config_path),
            "--engine",
            "fake",
            "--prompt-template",
            "lifelog_structured_tags_v1",
            "--limit",
            "1",
            "--force",
            "--allow-fake-write",
        ]
    )
    repository = LifelogRepository(db_path)
    row = repository.get_media_vlm("media_vlm_cli")

    assert code == 0
    assert captured["name"] == "fake"
    assert row["prompt_version"] == "lifelog_structured_tags_v1"


def test_retry_vlm_failed_reprocesses_failed_rows_only(tmp_path: Path) -> None:
    db_path = _seed_db(tmp_path)
    repository = LifelogRepository(db_path)
    repository.upsert_media_vlm(
        media_id="media_vlm_cli",
        status="failed",
        vlm_engine="qwen3_vl_transformers",
        model_name="local-qwen",
        error_message="Qwen3-VL JSON parse failed",
    )

    code = main(
        [
            "--db-path",
            str(db_path),
            "retry-vlm-failed",
            "--date",
            "2024-12-24",
            "--engine",
            "fake",
            "--allow-fake-write",
            "--rerun-model",
            "--limit",
            "5",
        ]
    )
    row = LifelogRepository(db_path).get_media_vlm("media_vlm_cli")

    assert code == 0
    assert row["status"] == "success"
    assert row["vlm_engine"] == "fake"


def _seed_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    image_path = tmp_path / "vlm_cli.png"
    Image.new("RGB", (16, 16), "white").save(image_path)
    repository.add_media_item(
        id="media_vlm_cli",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-vlm-cli",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    return db_path

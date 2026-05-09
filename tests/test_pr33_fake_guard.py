from __future__ import annotations

from pathlib import Path

from PIL import Image

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.fake_analysis_cleanup import cleanup_fake_analysis
from personal_lifelog_rag.model_diagnostics import run_model_diagnostics
from personal_lifelog_rag.retrieval.local_search import LocalSearchOptions, local_text_search
from personal_lifelog_rag.retrieval.query_router import route_query
from personal_lifelog_rag.timeline.event_builder import build_events
from personal_lifelog_rag.vlm.schemas import ImageSearchOptions
from personal_lifelog_rag.vlm.status_cleanup import cleanup_vlm_status
from personal_lifelog_rag.vlm.vlm_service import image_search


def test_fake_vlm_cli_does_not_write_without_allow_fake_write(tmp_path: Path, capsys) -> None:
    db_path = _seed_image_db(tmp_path, media_id="media_fake_guard")

    code = main(["--db-path", str(db_path), "analyze-images", "--date", "2024-12-24", "--engine", "fake"])
    output = capsys.readouterr().out

    assert code == 0
    assert "- dry_run: True" in output
    assert LifelogRepository(db_path).get_media_vlm("media_fake_guard") is None


def test_fake_embedding_cli_does_not_write_without_allow_fake_write(tmp_path: Path, capsys) -> None:
    db_path = _seed_image_db(tmp_path, media_id="media_fake_embedding_guard")

    code = main(["--db-path", str(db_path), "build-image-embeddings", "--date", "2024-12-24", "--engine", "fake"])
    capsys.readouterr()
    stats_code = main(["--db-path", str(db_path), "embedding-stats", "--json"])
    output = capsys.readouterr().out

    assert code == 0
    assert stats_code == 0
    assert '"total": 0' in output


def test_engine_unavailable_vlm_is_not_used_in_search_or_events(tmp_path: Path) -> None:
    db_path = _seed_image_db(tmp_path, media_id="media_unavailable")
    repository = LifelogRepository(db_path)
    repository.upsert_media_vlm(
        media_id="media_unavailable",
        caption="ラーメンの可能性がある写真",
        short_caption="ラーメン候補",
        food_cues=["ramen_possible"],
        status="engine_unavailable",
        vlm_engine="qwen3_vl_transformers",
        model_name="local-qwen",
    )

    search_report = image_search(repository, ImageSearchOptions(query="ラーメン"))
    local_report = local_text_search(repository, LocalSearchOptions(query="ラーメン", intent="food_activity"))
    qa_report = route_query(repository, "ラーメンを食べた写真はいつ？", limit=5).to_dict()
    build_events(repository, start_date="2024-12-24")
    events = repository.list_events(start_date="2024-12-24", end_date="2024-12-24")
    evidence = repository.list_event_evidence(str(events[0]["id"]))

    assert search_report["total"] == 0
    assert local_report["results"] == []
    assert qa_report["results"] == []
    assert not any(row["evidence_type"] == "vlm" for row in evidence)
    assert events[0]["title"] != "食事・カフェの可能性"


def test_fake_vlm_mirrored_caption_is_not_used_in_search_or_events(tmp_path: Path) -> None:
    db_path = _seed_image_db(tmp_path, media_id="media_fake_caption")
    repository = LifelogRepository(db_path)
    repository.upsert_media_vlm(
        media_id="media_fake_caption",
        caption="ラーメンやご飯の可能性がある料理写真",
        short_caption="ラーメン候補",
        food_cues=["ramen_possible"],
        status="success",
        vlm_engine="fake",
        model_name="fake-vlm",
    )

    local_report = local_text_search(repository, LocalSearchOptions(query="ラーメン", intent="food_activity"))
    build_events(repository, start_date="2024-12-24")
    events = repository.list_events(start_date="2024-12-24", end_date="2024-12-24")
    evidence = repository.list_event_evidence(str(events[0]["id"]))

    assert local_report["results"] == []
    assert not any(row["evidence_type"] == "vlm" for row in evidence)
    assert events[0]["title"] != "食事・カフェの可能性"


def test_failed_vlm_is_not_used_as_event_evidence(tmp_path: Path) -> None:
    db_path = _seed_image_db(tmp_path, media_id="media_failed_vlm")
    repository = LifelogRepository(db_path)
    repository.upsert_media_vlm(
        media_id="media_failed_vlm",
        caption="カフェのような場所の可能性",
        short_caption="カフェ候補",
        food_cues=["cafe_possible"],
        status="failed",
        vlm_engine="qwen3_vl_transformers",
        model_name="local-qwen",
        error_message="Transformers VLM failed: KeyError: 'caption'",
    )

    build_events(repository, start_date="2024-12-24")
    events = repository.list_events(start_date="2024-12-24", end_date="2024-12-24")
    evidence = repository.list_event_evidence(str(events[0]["id"]))

    assert not any(row["evidence_type"] == "vlm" for row in evidence)
    assert events[0]["title"] != "食事・カフェの可能性"


def test_cleanup_fake_analysis_dry_run_and_yes(tmp_path: Path) -> None:
    db_path = _seed_image_db(tmp_path, media_id="media_fake_cleanup")
    repository = LifelogRepository(db_path)
    repository.upsert_media_vlm(
        media_id="media_fake_cleanup",
        caption="fake caption",
        short_caption="fake caption",
        status="success",
        vlm_engine="fake",
        model_name="fake-vlm",
    )
    event_id = repository.add_event(date="2024-12-24", title="fake event")
    repository.add_event_evidence(event_id=event_id, evidence_type="vlm", evidence_id="media_fake_cleanup")
    main(
        [
            "--db-path",
            str(db_path),
            "build-image-embeddings",
            "--date",
            "2024-12-24",
            "--engine",
            "fake",
            "--allow-fake-write",
        ]
    )

    dry = cleanup_fake_analysis(db_path, dry_run=True)
    assert dry["media_vlm_rows"] == 1
    assert dry["media_embeddings_rows"] == 1
    assert dry["event_evidence_vlm_rows"] == 1
    assert repository.get_media_vlm("media_fake_cleanup") is not None

    deleted = cleanup_fake_analysis(db_path, dry_run=False, yes=True)
    assert deleted["deleted"]["media_vlm_rows"] == 1
    assert deleted["deleted"]["media_embeddings_rows"] == 1
    assert deleted["deleted"]["event_evidence_vlm_rows"] == 1
    assert repository.get_media_vlm("media_fake_cleanup") is None
    assert repository.list_event_evidence(event_id) == []


def test_cleanup_vlm_status_dry_run_and_yes_removes_related_evidence(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    for media_id, status in [("media_failed_cleanup", "failed"), ("media_unavailable_cleanup", "engine_unavailable")]:
        image_path = tmp_path / f"{media_id}.png"
        Image.new("RGB", (16, 16), "white").save(image_path)
        repository.add_media_item(
            id=media_id,
            file_path=str(image_path),
            file_name=image_path.name,
            file_hash=f"hash-{media_id}",
            media_type="image",
            captured_at="2024-12-24T10:00:00+09:00",
        )
        repository.upsert_media_vlm(
            media_id=media_id,
            caption="failed caption",
            short_caption="failed caption",
            status=status,
            vlm_engine="qwen3_vl_transformers",
            model_name="local-qwen",
        )
    event_id = repository.add_event(id="event_status_cleanup", date="2024-12-24", title="status cleanup")
    repository.add_event_evidence(event_id=event_id, evidence_type="vlm", evidence_id="media_failed_cleanup")
    repository.add_event_evidence(event_id=event_id, evidence_type="vlm", evidence_id="media_unavailable_cleanup")

    dry = cleanup_vlm_status(db_path, date="2024-12-24", dry_run=True)

    assert dry["media_vlm_rows"] == 2
    assert dry["event_evidence_vlm_rows"] == 2
    assert repository.get_media_vlm("media_failed_cleanup") is not None

    deleted = cleanup_vlm_status(db_path, date="2024-12-24", dry_run=False, yes=True)

    assert deleted["deleted"]["media_vlm_rows"] == 2
    assert deleted["deleted"]["event_evidence_vlm_rows"] == 2
    assert repository.get_media_vlm("media_failed_cleanup") is None
    assert repository.get_media_vlm("media_unavailable_cleanup") is None
    assert repository.list_event_evidence(event_id) == []


def test_cleanup_vlm_status_cli_dry_run(tmp_path: Path, capsys) -> None:
    db_path = _seed_image_db(tmp_path, media_id="media_failed_cli_cleanup")
    repository = LifelogRepository(db_path)
    repository.upsert_media_vlm(
        media_id="media_failed_cli_cleanup",
        caption="failed caption",
        short_caption="failed caption",
        status="failed",
        vlm_engine="qwen3_vl_transformers",
        model_name="local-qwen",
    )

    code = main(
        [
            "--db-path",
            str(db_path),
            "cleanup-vlm-status",
            "--date",
            "2024-12-24",
            "--status",
            "failed",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "VLM status cleanup" in output
    assert "- media_vlm rows: 1" in output
    assert repository.get_media_vlm("media_failed_cli_cleanup") is not None


def test_model_diagnostics_reports_model_path_and_dependencies(tmp_path: Path, capsys) -> None:
    model_path = tmp_path / "models" / "Qwen3-VL-8B-Thinking"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    (model_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model_path / "model.safetensors").write_bytes(b"dummy")
    config_path = tmp_path / "model_runtime.yaml"
    config_path.write_text(
        f"""
models:
  vlm:
    engine: qwen3_vl_transformers
    model_path: "{model_path}"
    local_files_only: true
  multimodal_embedding:
    engine: qwen3_vl_embedding
    model_path: "{model_path}"
""",
        encoding="utf-8",
    )

    report = run_model_diagnostics(config_path)

    assert report["vlm"]["model_path"] == str(model_path)
    assert report["vlm"]["path"]["exists"] is True
    assert report["vlm"]["path"]["config_json_exists"] is True
    assert report["vlm"]["path"]["processor_or_tokenizer_exists"] is True
    assert report["vlm"]["engine_initialization"]["is_available"] is True
    assert report["embedding"]["engine_initialization"]["engine_name"] == "qwen3_vl_embedding"

    code = main(["model-diagnostics", "--config", str(config_path), "--json"])
    payload = capsys.readouterr().out
    assert code == 0
    assert str(model_path) in payload


def _seed_image_db(tmp_path: Path, *, media_id: str) -> Path:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    image_path = tmp_path / f"{media_id}.png"
    Image.new("RGB", (16, 16), "white").save(image_path)
    repository.add_media_item(
        id=media_id,
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash=f"hash-{media_id}",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    return db_path

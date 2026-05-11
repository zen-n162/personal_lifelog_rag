from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.embeddings.embedding_service import (
    build_image_embeddings,
    build_text_embeddings,
    embedding_stats,
)
from personal_lifelog_rag.embeddings.engines import FakeEmbeddingEngine
from personal_lifelog_rag.embeddings.repository import MediaEmbeddingRepository
from personal_lifelog_rag.embeddings.schemas import BuildMediaEmbeddingsOptions


def test_build_image_embeddings_dry_run_and_skip_existing(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    image_path = tmp_path / "ramen.jpg"
    image_path.write_bytes(b"dummy")
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_image_embedding",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-image-embedding",
        media_type="image",
        captured_at="2024-12-24T12:00:00+09:00",
    )

    dry_report = build_image_embeddings(
        repository,
        BuildMediaEmbeddingsOptions(start_date="2024-12-24", end_date="2024-12-24", dry_run=True),
        engine=FakeEmbeddingEngine(),
    )
    assert dry_report.selected == 1
    assert MediaEmbeddingRepository(db_path).list_embeddings() == []

    report = build_image_embeddings(
        repository,
        BuildMediaEmbeddingsOptions(start_date="2024-12-24", end_date="2024-12-24", force=True),
        engine=FakeEmbeddingEngine(),
    )
    assert report.success == 1

    skip_report = build_image_embeddings(
        repository,
        BuildMediaEmbeddingsOptions(start_date="2024-12-24", end_date="2024-12-24", skip_existing=True),
        engine=FakeEmbeddingEngine(),
    )
    assert skip_report.skipped == 1


def test_build_text_embeddings_from_combined_ocr_and_vlm(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    image_path = tmp_path / "cafe.jpg"
    image_path.write_bytes(b"dummy")
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_text_embedding",
        file_path=str(image_path),
        file_name="cafe.jpg",
        file_hash="hash-text-embedding",
        media_type="image",
        captured_at="2024-12-24T13:00:00+09:00",
    )
    repository.upsert_media_ocr(media_id="media_text_embedding", ocr_text="カフェ メニュー", status="success")
    repository.upsert_media_vlm(
        media_id="media_text_embedding",
        caption="カフェのような場所の可能性",
        short_caption="カフェ候補",
        food_cues=["cafe_possible"],
        status="success",
        vlm_engine="fake",
    )

    report = build_text_embeddings(
        repository,
        BuildMediaEmbeddingsOptions(
            start_date="2024-12-24",
            end_date="2024-12-24",
            embedding_type="combined_text",
            force=True,
        ),
        engine=FakeEmbeddingEngine(),
    )
    stats = embedding_stats(repository)

    assert report.success == 1
    assert stats["by_type"]["combined_text"] == 1


def test_embedding_builders_skip_missing_original_files(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="missing_image_embedding",
        file_path=str(tmp_path / "missing.jpg"),
        file_name="missing.jpg",
        file_hash="hash-missing-image-embedding",
        media_type="image",
        captured_at="2024-12-24T14:00:00+09:00",
    )
    repository.upsert_media_vlm(
        media_id="missing_image_embedding",
        caption="caption that should not be embedded without the file",
        short_caption="caption",
        status="success",
        vlm_engine="qwen3_vl_transformers",
    )

    image_report = build_image_embeddings(
        repository,
        BuildMediaEmbeddingsOptions(start_date="2024-12-24", end_date="2024-12-24", dry_run=True),
        engine=FakeEmbeddingEngine(),
    )
    text_report = build_text_embeddings(
        repository,
        BuildMediaEmbeddingsOptions(
            start_date="2024-12-24",
            end_date="2024-12-24",
            embedding_type="combined_text",
            dry_run=True,
        ),
        engine=FakeEmbeddingEngine(),
    )

    assert image_report.selected == 0
    assert text_report.selected == 0


def test_retry_embedding_failed_cli_dry_run_lists_rows(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    image_path = tmp_path / "failed_embedding.jpg"
    image_path.write_bytes(b"dummy")
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_failed_embedding",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-failed-embedding",
        media_type="image",
        captured_at="2024-12-24T15:00:00+09:00",
    )
    MediaEmbeddingRepository(db_path).upsert_embedding(
        media_id="media_failed_embedding",
        embedding_type="image",
        embedding_model="qwen3_vl_embedding",
        status="engine_unavailable",
    )

    code = main(
        [
            "--db-path",
            str(db_path),
            "retry-embedding-failed",
            "--date",
            "2024-12-24",
            "--type",
            "image",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "Retry embedding failed dry-run" in output
    assert "media_failed_embedding" in output

from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.embeddings.adapter import HashingEmbeddingAdapter
from personal_lifelog_rag.embeddings.vector_search import build_embeddings, semantic_search
from personal_lifelog_rag.ingest.line_parser import parse_line_chat_file


def test_build_embeddings_processes_line_messages(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    messages = parse_line_chat_file(Path("tests/fixtures/line/sample_chat.txt"))
    repository.add_line_messages(messages)

    report = build_embeddings(repository, HashingEmbeddingAdapter())

    assert report.line_messages_seen == 9
    assert report.embedded == 9
    assert len(repository.list_embeddings(model_name="local-hashing-v1")) == 9


def test_semantic_search_returns_candidate_message_and_date(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    messages = parse_line_chat_file(Path("tests/fixtures/line/sample_chat.txt"))
    repository.add_line_messages(messages)
    adapter = HashingEmbeddingAdapter()
    build_embeddings(repository, adapter)

    results = semantic_search(repository, adapter, "新宿でご飯", limit=3)

    assert results
    assert any(result.date == "2024-12-24" for result in results)
    assert any("新宿" in result.text or "ご飯" in result.text for result in results)


def test_media_caption_is_embedding_source(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.add_media_item(
        file_path="/local/photos/shinjuku.jpg",
        file_name="shinjuku.jpg",
        file_hash="media-caption-hash",
        media_type="image",
        fallback_captured_at="2024-12-24T19:12:00+09:00",
        caption="新宿でご飯を食べた写真",
    )
    adapter = HashingEmbeddingAdapter()
    report = build_embeddings(repository, adapter)

    assert report.media_items_seen == 1
    results = semantic_search(repository, adapter, "新宿でご飯", limit=1)
    assert results[0].source_type == "media_item"
    assert results[0].date == "2024-12-24"


def test_cli_build_embeddings_and_search(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"

    assert main(["--db-path", str(db_path), "ingest-line", "--path", "tests/fixtures/line"]) == 0
    capsys.readouterr()
    assert main(["--db-path", str(db_path), "build-embeddings", "--backend", "hash"]) == 0
    build_output = capsys.readouterr().out
    assert "Built embeddings:" in build_output
    assert "line_messages=9" in build_output

    assert main(["--db-path", str(db_path), "search", "新宿でご飯", "--backend", "hash"]) == 0
    search_output = capsys.readouterr().out
    assert "Search Results:" in search_output
    assert "2024-12-24" in search_output

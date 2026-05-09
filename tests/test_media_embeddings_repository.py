from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.embeddings.repository import MediaEmbeddingRepository, embedding_vector


def test_media_embeddings_repository_saves_and_lists_vectors(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_embed_repo",
        file_path=str(tmp_path / "photo.jpg"),
        file_name="photo.jpg",
        file_hash="hash-embed-repo",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )

    embeddings = MediaEmbeddingRepository(db_path)
    embeddings.upsert_embedding(
        media_id="media_embed_repo",
        embedding_type="image",
        embedding_model="fake-qwen3-vl-embedding",
        vector=[1.0, 0.0, 0.0],
        source_text=None,
        status="success",
    )

    rows = embeddings.list_embeddings(embedding_type="image", statuses=["success"])

    assert len(rows) == 1
    assert rows[0]["media_id"] == "media_embed_repo"
    assert rows[0]["embedding_dim"] == 3
    assert embedding_vector(rows[0]) == [1.0, 0.0, 0.0]


"""Local embedding adapters and SQLite vector search."""

from personal_lifelog_rag.embeddings.adapter import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingAdapter,
    HashingEmbeddingAdapter,
    SentenceTransformerEmbeddingAdapter,
    get_embedding_adapter,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "EmbeddingAdapter",
    "HashingEmbeddingAdapter",
    "SentenceTransformerEmbeddingAdapter",
    "get_embedding_adapter",
]

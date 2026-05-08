"""Local-only embedding adapters."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Protocol

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_BACKEND_ENV_VAR = "PERSONAL_LIFELOG_RAG_EMBEDDING_BACKEND"
EMBEDDING_MODEL_ENV_VAR = "PERSONAL_LIFELOG_RAG_EMBEDDING_MODEL"


class EmbeddingAdapter(Protocol):
    model_name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per text."""


@dataclass
class HashingEmbeddingAdapter:
    """Deterministic local fallback that needs no model download.

    This is not as semantically rich as a transformer model, but it keeps
    build/search flows usable when no local model is installed.
    """

    dimensions: int = 384
    model_name: str = "local-hashing-v1"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _char_ngrams(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


@dataclass
class SentenceTransformerEmbeddingAdapter:
    """Adapter for a locally available sentence-transformers model."""

    model_name: str = DEFAULT_EMBEDDING_MODEL
    local_files_only: bool = True

    def __post_init__(self) -> None:
        self._model = self._load_model()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]

    def _load_model(self):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install with "
                '`pip install -e ".[embeddings]"` or use the local hashing fallback.'
            ) from exc

        try:
            return SentenceTransformer(
                self.model_name,
                local_files_only=self.local_files_only,
            )
        except TypeError:
            return SentenceTransformer(self.model_name)


def get_embedding_adapter(
    *,
    backend: str | None = None,
    model_name: str | None = None,
    require_model: bool = False,
) -> EmbeddingAdapter:
    """Create an embedding adapter without using external APIs.

    `auto` tries a local sentence-transformers model first and falls back to the
    deterministic hashing adapter if the model is unavailable.
    """

    resolved_backend = backend or os.getenv(EMBEDDING_BACKEND_ENV_VAR, "auto")
    resolved_model = model_name or os.getenv(EMBEDDING_MODEL_ENV_VAR, DEFAULT_EMBEDDING_MODEL)

    if resolved_backend == "hash":
        return HashingEmbeddingAdapter()
    if resolved_backend in {"sentence-transformers", "sentence_transformers"}:
        return SentenceTransformerEmbeddingAdapter(model_name=resolved_model)
    if resolved_backend != "auto":
        raise ValueError(f"Unknown embedding backend: {resolved_backend}")

    try:
        return SentenceTransformerEmbeddingAdapter(model_name=resolved_model)
    except Exception:
        if require_model:
            raise
        return HashingEmbeddingAdapter()


def _char_ngrams(text: str) -> list[str]:
    normalized = "".join(text.lower().split())
    if not normalized:
        return []
    chars = list(normalized)
    tokens = chars[:]
    for size in (2, 3):
        tokens.extend(normalized[index : index + size] for index in range(max(len(normalized) - size + 1, 0)))
    return tokens

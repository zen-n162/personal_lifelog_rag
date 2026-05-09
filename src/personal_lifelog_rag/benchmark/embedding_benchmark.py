"""Local-only image/text embedding benchmark helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time
from typing import Protocol

from personal_lifelog_rag.benchmark.schemas import BenchmarkCase, ModelSpec
from personal_lifelog_rag.embeddings.engines import get_multimodal_embedding_engine as get_app_embedding_engine


class MultimodalEmbeddingEngine(Protocol):
    name: str
    model_name: str | None

    def is_available(self) -> bool:
        """Return whether this local embedding engine can run."""

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        """Embed local image paths without network access."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed text queries without network access."""


@dataclass
class NoopMultimodalEmbeddingEngine:
    name: str = "noop"
    model_name: str | None = None

    def is_available(self) -> bool:
        return False

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        return []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return []


@dataclass
class FakeMultimodalEmbeddingEngine:
    """Deterministic fake engine for tests and smoke checks."""

    name: str = "fake"
    model_name: str | None = "fake-qwen3-vl-embedding"
    dimensions: int = 8

    def is_available(self) -> bool:
        return True

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        return [_normalize(_concept_vector(_concept_from_text(str(path)), self.dimensions)) for path in image_paths]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_normalize(_concept_vector(_concept_from_text(text), self.dimensions)) for text in texts]


@dataclass
class Qwen3VlEmbeddingSentenceTransformersEngine:
    """Benchmark wrapper around the app's local Qwen3-VL-Embedding adapter."""

    model_name: str | None = None
    model_path: str | None = None
    device: str | None = "auto"
    dtype: str | None = "auto"
    local_files_only: bool | None = True
    embedding_dim: int | None = None
    batch_size: int | None = None
    name: str = "qwen3_vl_embedding_sentence_transformers"

    def is_available(self) -> bool:
        return self._engine().is_available()

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        vectors: list[list[float]] = []
        engine = self._engine()
        for path in image_paths:
            result = engine.embed_image(path)
            if result.status != "success":
                raise RuntimeError(result.error_message or f"image embedding failed with {result.status}")
            vectors.append(result.vector)
        return vectors

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        engine = self._engine()
        for text in texts:
            result = engine.embed_text(text)
            if result.status != "success":
                raise RuntimeError(result.error_message or f"text embedding failed with {result.status}")
            vectors.append(result.vector)
        return vectors

    def _engine(self):
        return get_app_embedding_engine(
            "qwen3_vl_embedding",
            model_name=self.model_name,
            model_path=self.model_path,
            device=self.device,
            dtype=self.dtype,
            local_files_only=self.local_files_only,
            embedding_dim=self.embedding_dim,
            batch_size=self.batch_size,
        )


def get_multimodal_embedding_engine(
    engine_name: str | None = None,
    *,
    model_name: str | None = None,
    model_path: str | None = None,
    device: str | None = "auto",
    dtype: str | None = "auto",
    local_files_only: bool | None = True,
    embedding_dim: int | None = None,
    batch_size: int | None = None,
) -> MultimodalEmbeddingEngine:
    resolved = (engine_name or "noop").strip().lower()
    if resolved in {"", "noop", "none", "disabled", "off"}:
        return NoopMultimodalEmbeddingEngine()
    if resolved == "fake":
        return FakeMultimodalEmbeddingEngine(model_name=model_name or "fake-qwen3-vl-embedding")
    if resolved in {
        "qwen3_vl_embedding_sentence_transformers",
        "qwen3-vl-embedding-sentence-transformers",
        "qwen3_vl_embedding",
        "qwen3-vl-embedding",
        "sentence_transformers",
        "sentence-transformers",
    }:
        return Qwen3VlEmbeddingSentenceTransformersEngine(
            model_name=model_name,
            model_path=model_path,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
        )
    return NoopMultimodalEmbeddingEngine()


def benchmark_image_embedding(
    cases: list[BenchmarkCase],
    *,
    engine: MultimodalEmbeddingEngine,
) -> dict:
    started = time.perf_counter()
    if not engine.is_available():
        return {
            "engine": engine.name,
            "model_name": getattr(engine, "model_name", None),
            "status": "engine_unavailable",
            "latency_sec": round(time.perf_counter() - started, 4),
            "cases": len(cases),
            "case_results": [
                {
                    "case_id": case.id,
                    "status": "engine_unavailable",
                    "query_results": [],
                    "image_embedding_dim": None,
                }
                for case in cases
            ],
            "metrics": {"top1_accuracy": None, "recall_at_3": None, "recall_at_5": None},
        }

    image_paths = [case.image_path for case in cases]
    image_vectors = engine.embed_images(image_paths)
    case_results = []
    query_evaluations = []
    for case_index, case in enumerate(cases):
        query_results = []
        text_vectors = engine.embed_texts(case.query_texts)
        for query, text_vector in zip(case.query_texts, text_vectors, strict=False):
            ranked = _rank_cases(cases, image_vectors, text_vector)
            target_rank = next((index + 1 for index, row in enumerate(ranked) if row["case_id"] == case.id), None)
            target_score = next((row["score"] for row in ranked if row["case_id"] == case.id), None)
            row = {
                "query": query,
                "top1_case_id": ranked[0]["case_id"] if ranked else None,
                "target_rank": target_rank,
                "target_score": target_score,
                "top5_case_ids": [item["case_id"] for item in ranked[:5]],
            }
            query_results.append(row)
            query_evaluations.append(row)
        image_dim = len(image_vectors[case_index]) if case_index < len(image_vectors) else None
        case_results.append(
            {
                "case_id": case.id,
                "status": "success",
                "image_embedding_dim": image_dim,
                "query_results": query_results,
            }
        )

    return {
        "engine": engine.name,
        "model_name": getattr(engine, "model_name", None),
        "status": "success",
        "latency_sec": round(time.perf_counter() - started, 4),
        "cases": len(cases),
        "case_results": case_results,
        "metrics": _ranking_metrics(query_evaluations),
    }


def engine_from_spec(spec: ModelSpec, *, override_engine: str | None = None) -> MultimodalEmbeddingEngine:
    return get_multimodal_embedding_engine(
        override_engine or spec.engine,
        model_name=spec.model_name,
        model_path=spec.model_path,
        device=spec.device,
        dtype=spec.dtype,
        local_files_only=spec.local_files_only,
        embedding_dim=spec.embedding_dim,
        batch_size=spec.batch_size,
    )


def _rank_cases(cases: list[BenchmarkCase], image_vectors: list[list[float]], query_vector: list[float]) -> list[dict]:
    ranked = []
    for case, image_vector in zip(cases, image_vectors, strict=False):
        ranked.append({"case_id": case.id, "score": round(_cosine(query_vector, image_vector), 4)})
    return sorted(ranked, key=lambda row: (-row["score"], row["case_id"]))


def _ranking_metrics(query_results: list[dict]) -> dict[str, float | None]:
    if not query_results:
        return {"top1_accuracy": None, "recall_at_3": None, "recall_at_5": None}
    total = len(query_results)
    top1 = sum(1 for row in query_results if row.get("target_rank") == 1)
    recall3 = sum(1 for row in query_results if row.get("target_rank") is not None and row["target_rank"] <= 3)
    recall5 = sum(1 for row in query_results if row.get("target_rank") is not None and row["target_rank"] <= 5)
    return {
        "top1_accuracy": round(top1 / total, 4),
        "recall_at_3": round(recall3 / total, 4),
        "recall_at_5": round(recall5 / total, 4),
    }


def _concept_from_text(value: str) -> str:
    text = value.lower()
    rules = {
        "food": ("food", "meal", "ramen", "restaurant", "料理", "ご飯", "ラーメン", "食事", "カフェ"),
        "place": ("place", "station", "city", "outdoor", "駅", "街", "場所", "看板"),
        "text": ("text", "screenshot", "ticket", "receipt", "文字", "スクリーンショット", "チケット", "レシート"),
        "indoor": ("indoor", "room", "室内"),
        "outdoor": ("sea", "beach", "park", "outdoor", "海", "公園", "屋外"),
        "unclear": ("unclear", "unknown", "不明"),
    }
    for concept, needles in rules.items():
        if any(needle in text for needle in needles):
            return concept
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return ["food", "place", "text", "indoor", "outdoor", "unclear"][int(digest[:2], 16) % 6]


def _concept_vector(concept: str, dimensions: int) -> list[float]:
    concepts = ["food", "place", "text", "indoor", "outdoor", "unclear"]
    vector = [0.0] * dimensions
    index = concepts.index(concept) if concept in concepts else 0
    vector[index % dimensions] = 1.0
    vector[(index + 3) % dimensions] = 0.15
    return vector


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=False))

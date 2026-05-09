"""Local-only multimodal embedding engines."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import traceback
from typing import Any

from personal_lifelog_rag.embeddings.base import MultimodalEmbeddingEngine
from personal_lifelog_rag.embeddings.schemas import EmbeddingResult
from personal_lifelog_rag.embeddings.similarity import normalize


@dataclass
class NoopEmbeddingEngine:
    name: str = "noop"
    model_name: str | None = None

    def is_available(self) -> bool:
        return False

    def embed_image(self, image_path: Path) -> EmbeddingResult:
        return EmbeddingResult(
            model_name=self.model_name,
            status="engine_unavailable",
            error_message=f"embedding engine '{self.name}' is not available",
        )

    def embed_text(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(
            model_name=self.model_name,
            status="engine_unavailable",
            error_message=f"embedding engine '{self.name}' is not available",
        )


@dataclass
class FakeEmbeddingEngine:
    """Deterministic fake Qwen3-VL-Embedding stand-in for tests and smoke runs."""

    name: str = "fake"
    model_name: str | None = "fake-qwen3-vl-embedding"
    dimensions: int = 16

    def is_available(self) -> bool:
        return True

    def embed_image(self, image_path: Path) -> EmbeddingResult:
        vector = _fake_vector(_concept_from_text(str(image_path)), self.dimensions)
        return EmbeddingResult(
            vector=vector,
            model_name=self.model_name,
            embedding_dim=len(vector),
            status="success",
        )

    def embed_text(self, text: str) -> EmbeddingResult:
        vector = _fake_vector(_concept_from_text(text), self.dimensions)
        return EmbeddingResult(
            vector=vector,
            model_name=self.model_name,
            embedding_dim=len(vector),
            status="success",
        )


@dataclass
class Qwen3VlEmbeddingEngine:
    """Local Qwen3-VL-Embedding runtime.

    The adapter never downloads a model. The default `qwen3_vl_embedding`
    engine uses the official Qwen3VLEmbedder runtime from a local external
    checkout or the model-bundled `scripts/qwen3_vl_embedding.py`. The
    sentence-transformers path is kept only for the explicit
    `qwen3_vl_embedding_sentence_transformers` alias because it can pull in a
    different native stack.
    """

    model_name: str | None = None
    model_path: str | None = None
    device: str | None = "auto"
    dtype: str | None = "auto"
    local_files_only: bool | None = True
    embedding_dim: int | None = None
    batch_size: int | None = None
    name: str = "qwen3_vl_embedding"
    runtime_preference: str = "official"
    _runtime_kind: str | None = field(default=None, init=False, repr=False)
    _runtime_model: Any = field(default=None, init=False, repr=False)
    _runtime_error: str | None = field(default=None, init=False, repr=False)

    @property
    def model_ref(self) -> str | None:
        return self.model_path or self.model_name

    def is_available(self) -> bool:
        return self.availability_error() is None

    def availability_error(self) -> str | None:
        if not self._local_model_exists():
            return "configured Qwen3-VL-Embedding model_path does not exist locally"
        if self.runtime_preference == "sentence_transformers":
            if _has_module("sentence_transformers"):
                return None
            return "sentence_transformers is not installed"

        missing = [
            module_name
            for module_name in ("torch", "transformers", "qwen_vl_utils")
            if not _has_module(module_name)
        ]
        script_path = self._bundled_script_path()
        external_available = self._external_embedder_module_available()
        if missing:
            return "missing local runtime dependencies: " + ", ".join(missing)
        if external_available or script_path:
            return None
        return (
            "Qwen3VLEmbedder runtime not found; expected import "
            "src.models.qwen3_vl_embedding or model-bundled "
            "scripts/qwen3_vl_embedding.py"
        )

    def embed_image(self, image_path: Path) -> EmbeddingResult:
        if not image_path.exists():
            return self._failed("image file does not exist")
        try:
            runtime = self._runtime()
            if self._runtime_kind == "sentence_transformers":
                vector = self._encode_sentence_transformers([{"image": str(image_path.resolve())}])
            else:
                vector = self._embed_image_with_official_runtime(runtime, image_path)
            return self._success(vector)
        except _EmbeddingEngineUnavailable as exc:
            return self._unavailable(str(exc))
        except Exception as exc:  # pragma: no cover - depends on local model runtime
            return self._failed(_exception_message(exc))

    def embed_text(self, text: str) -> EmbeddingResult:
        try:
            runtime = self._runtime()
            if self._runtime_kind == "sentence_transformers":
                vector = self._encode_sentence_transformers([text])
            else:
                output = runtime.process(
                    [
                        {
                            "text": text,
                            "instruction": "Retrieve images relevant to the user's query.",
                        }
                    ]
                )
                vector = self._vector_from_output(output)
            return self._success(vector)
        except _EmbeddingEngineUnavailable as exc:
            return self._unavailable(str(exc))
        except Exception as exc:  # pragma: no cover - depends on local model runtime
            return self._failed(_exception_message(exc))

    def _runtime(self):
        if self._runtime_model is not None:
            return self._runtime_model
        availability_error = self.availability_error()
        if availability_error:
            raise _EmbeddingEngineUnavailable(availability_error)
        if self._local_files_only_enabled():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        errors: list[str] = []
        if self.runtime_preference == "sentence_transformers":
            try:
                self._runtime_model = self._load_sentence_transformers()
                self._runtime_kind = "sentence_transformers"
                return self._runtime_model
            except Exception as exc:  # pragma: no cover - environment dependent
                errors.append(f"sentence-transformers runtime failed: {_exception_message(exc)}")
            self._runtime_error = "; ".join(errors)
            raise _EmbeddingEngineUnavailable(self._runtime_error)

        try:
            self._runtime_model = self._load_external_embedder()
            self._runtime_kind = "official_qwen3_vl_embedding"
            return self._runtime_model
        except Exception as exc:  # pragma: no cover - external checkout dependent
            errors.append(f"external Qwen3VLEmbedder runtime failed: {_exception_message(exc)}")

        script_path = self._bundled_script_path()
        if script_path:
            try:
                self._runtime_model = self._load_bundled_embedder(script_path)
                self._runtime_kind = "bundled_qwen3_vl_embedding"
                return self._runtime_model
            except Exception as exc:  # pragma: no cover - environment dependent
                errors.append(f"bundled Qwen3VLEmbedder runtime failed: {_exception_message(exc)}")
        else:
            errors.append("bundled scripts/qwen3_vl_embedding.py was not found in model_path")

        self._runtime_error = "; ".join(errors) if errors else "no usable Qwen3-VL-Embedding runtime found"
        raise _EmbeddingEngineUnavailable(self._runtime_error)

    def _load_sentence_transformers(self):
        module = importlib.import_module("sentence_transformers")
        constructor = getattr(module, "SentenceTransformer")
        kwargs: dict[str, Any] = {
            "local_files_only": self._local_files_only_enabled(),
            "trust_remote_code": True,
        }
        resolved_device = self._resolved_device()
        if resolved_device:
            kwargs["device"] = resolved_device
        if self.embedding_dim:
            kwargs["truncate_dim"] = self.embedding_dim
        try:
            return constructor(str(self._model_path()), **kwargs)
        except TypeError:
            # Older sentence-transformers builds may not accept all newer args.
            for optional_key in ("truncate_dim", "trust_remote_code", "local_files_only"):
                kwargs.pop(optional_key, None)
            return constructor(str(self._model_path()), **kwargs)

    def _load_external_embedder(self):
        for root in self._external_repo_roots():
            root_str = str(root)
            if root.exists() and root_str not in sys.path:
                sys.path.insert(0, root_str)
        module = importlib.import_module("src.models.qwen3_vl_embedding")
        embedder = getattr(module, "Qwen3VLEmbedder")
        return embedder(model_name_or_path=str(self._model_path()), **self._embedder_kwargs())

    def _load_bundled_embedder(self, script_path: Path):
        module_name = "_personal_lifelog_qwen3_vl_embedding_runtime_" + hashlib.sha1(str(script_path).encode("utf-8")).hexdigest()[:12]
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load {script_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        embedder = getattr(module, "Qwen3VLEmbedder")
        return embedder(model_name_or_path=str(self._model_path()), **self._embedder_kwargs())

    def _embedder_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "local_files_only": self._local_files_only_enabled(),
        }
        torch_dtype = self._torch_dtype()
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        elif _has_module("torch"):
            try:
                torch = importlib.import_module("torch")
                if self.dtype is None or str(self.dtype).lower() == "auto":
                    kwargs["torch_dtype"] = torch.bfloat16
            except Exception:
                pass
        return kwargs

    def _embed_image_with_official_runtime(self, runtime: Any, image_path: Path) -> list[float]:
        image_ref = str(image_path.resolve())
        path_error: Exception | None = None
        try:
            output = runtime.process([{"image": image_ref}])
            return self._vector_from_output(output)
        except Exception as exc:  # pragma: no cover - local runtime dependent
            path_error = exc
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB").copy()
            output = runtime.process([{"image": rgb_image}])
            return self._vector_from_output(output)
        except Exception as pil_error:  # pragma: no cover - local runtime dependent
            message = (
                "Qwen3-VL-Embedding image embedding failed in path and PIL modes; "
                f"path_mode_error={_exception_message(path_error) if path_error else 'none'}; "
                f"pil_mode_error={_exception_message(pil_error)}; "
                f"image_path={_short_path(image_path)}"
            )
            raise RuntimeError(message) from pil_error

    def _encode_sentence_transformers(self, inputs: list[Any]) -> list[float]:
        kwargs: dict[str, Any] = {"normalize_embeddings": True, "show_progress_bar": False}
        if self.batch_size:
            kwargs["batch_size"] = self.batch_size
        try:
            output = self._runtime_model.encode(inputs, **kwargs)
        except TypeError:
            kwargs.pop("show_progress_bar", None)
            output = self._runtime_model.encode(inputs, **kwargs)
        return self._vector_from_output(output)

    def _vector_from_output(self, output: Any) -> list[float]:
        if isinstance(output, tuple):
            output = output[0]
        if hasattr(output, "detach"):
            output = output.detach().cpu().float().tolist()
        elif hasattr(output, "tolist"):
            output = output.tolist()
        if isinstance(output, list) and output and isinstance(output[0], list):
            output = output[0]
        if not isinstance(output, list):
            raise ValueError("embedding runtime returned an unsupported vector type")
        return self._finalize_vector([float(value) for value in output])

    def _finalize_vector(self, vector: list[float]) -> list[float]:
        if self.embedding_dim and self.embedding_dim > 0 and len(vector) > self.embedding_dim:
            vector = vector[: self.embedding_dim]
        return normalize(vector)

    def _success(self, vector: list[float]) -> EmbeddingResult:
        return EmbeddingResult(
            vector=vector,
            model_name=self.model_name or self.model_ref,
            embedding_dim=len(vector),
            status="success",
        )

    def _unavailable(self, message: str) -> EmbeddingResult:
        return EmbeddingResult(
            model_name=self.model_name or self.model_ref,
            status="engine_unavailable",
            error_message=message,
        )

    def _failed(self, message: str) -> EmbeddingResult:
        return EmbeddingResult(
            model_name=self.model_name or self.model_ref,
            status="failed",
            error_message=message,
        )

    def _model_path(self) -> Path:
        return Path(str(self.model_path or self.model_name or "")).expanduser()

    def _local_model_exists(self) -> bool:
        return bool(self.model_path and self._model_path().exists())

    def _bundled_script_path(self) -> Path | None:
        path = self._model_path() / "scripts" / "qwen3_vl_embedding.py"
        return path if path.exists() else None

    def _external_embedder_module_available(self) -> bool:
        try:
            return importlib.util.find_spec("src.models.qwen3_vl_embedding") is not None
        except (ImportError, ValueError):
            return False

    def _external_repo_roots(self) -> list[Path]:
        cwd = Path.cwd()
        return [
            cwd / "external" / "Qwen3-VL-Embedding",
            cwd.parent / "external" / "Qwen3-VL-Embedding",
            Path("/home/zennakamura/MyApplication/external/Qwen3-VL-Embedding"),
        ]

    def _resolved_device(self) -> str | None:
        if not self.device or self.device == "auto":
            if _has_module("torch"):
                try:
                    torch = importlib.import_module("torch")
                    return "cuda" if bool(torch.cuda.is_available()) else "cpu"
                except Exception:
                    return None
            return None
        return self.device

    def _torch_dtype(self):
        if not self.dtype or str(self.dtype).lower() == "auto" or not _has_module("torch"):
            return None
        torch = importlib.import_module("torch")
        mapping = {
            "float16": "float16",
            "fp16": "float16",
            "bfloat16": "bfloat16",
            "bf16": "bfloat16",
            "float32": "float32",
            "fp32": "float32",
        }
        attr_name = mapping.get(str(self.dtype).lower())
        return getattr(torch, attr_name) if attr_name else None

    def _local_files_only_enabled(self) -> bool:
        return self.local_files_only is not False


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
        return NoopEmbeddingEngine(model_name=model_name)
    if resolved == "fake":
        return FakeEmbeddingEngine(model_name=model_name or "fake-qwen3-vl-embedding")
    if resolved in {
        "qwen3_vl_embedding",
        "qwen3-vl-embedding",
    }:
        return Qwen3VlEmbeddingEngine(
            model_name=model_name,
            model_path=model_path,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            runtime_preference="official",
        )
    if resolved in {
        "qwen3_vl_embedding_sentence_transformers",
        "qwen3-vl-embedding-sentence-transformers",
    }:
        return Qwen3VlEmbeddingEngine(
            model_name=model_name,
            model_path=model_path,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            runtime_preference="sentence_transformers",
            name="qwen3_vl_embedding_sentence_transformers",
        )
    return NoopEmbeddingEngine(model_name=model_name)


def infer_query_engine_from_records(rows: list[dict]) -> MultimodalEmbeddingEngine:
    """Choose a local query engine that can compare with existing records."""

    for row in rows:
        model = str(row.get("embedding_model") or "")
        if model.startswith("fake") or "fake" in model:
            continue
    return NoopEmbeddingEngine()


def _concept_from_text(value: str) -> str:
    text = value.lower()
    rules = {
        "food": ("food", "meal", "ramen", "restaurant", "料理", "ご飯", "ラーメン", "食事", "カフェ", "menu"),
        "place": ("place", "station", "city", "outdoor", "駅", "街", "場所", "看板", "新宿", "shinjuku"),
        "text": ("text", "screenshot", "ticket", "receipt", "文字", "スクリーンショット", "チケット", "レシート"),
        "indoor": ("indoor", "room", "室内"),
        "outdoor": ("sea", "beach", "park", "outdoor", "海", "公園", "屋外"),
        "call": ("call", "phone", "通話", "電話"),
    }
    for concept, needles in rules.items():
        if any(needle in text for needle in needles):
            return concept
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return ["food", "place", "text", "indoor", "outdoor", "call"][int(digest[:2], 16) % 6]


def _fake_vector(concept: str, dimensions: int) -> list[float]:
    concepts = ["food", "place", "text", "indoor", "outdoor", "call"]
    vector = [0.0] * dimensions
    index = concepts.index(concept) if concept in concepts else 0
    vector[index % dimensions] = 1.0
    vector[(index + 5) % dimensions] = 0.18
    return normalize(vector)


class _EmbeddingEngineUnavailable(RuntimeError):
    pass


def _exception_message(exc: BaseException | None) -> str:
    if exc is None:
        return "none"
    traceback_tail = "".join(traceback.format_exception(exc.__class__, exc, exc.__traceback__, limit=6)).strip()
    return f"{exc.__class__.__name__}: {exc!r}; traceback_tail={traceback_tail[-1200:]}"


def _short_path(path: Path) -> str:
    text = str(path)
    return text if len(text) <= 160 else "..." + text[-157:]


def _has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return module_name in sys.modules

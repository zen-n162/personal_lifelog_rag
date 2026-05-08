"""Local-only image caption/VLM adapter boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
import base64
import json
import os
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse
from urllib.request import Request, urlopen


VLM_BACKEND_ENV_VAR = "PERSONAL_LIFELOG_RAG_VLM_BACKEND"
VLM_MODEL_ENV_VAR = "PERSONAL_LIFELOG_RAG_VLM_MODEL"
OLLAMA_URL_ENV_VAR = "PERSONAL_LIFELOG_RAG_OLLAMA_URL"
LLAMA_CPP_URL_ENV_VAR = "PERSONAL_LIFELOG_RAG_LLAMA_CPP_URL"


@dataclass(frozen=True)
class VLMAnalysisResult:
    engine: str
    caption: str | None = None
    scene: str | None = None
    objects: tuple[str, ...] = ()
    possible_activity: str | None = None
    text_in_image: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    reason: str | None = None

    def to_analysis_fields(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "objects": list(self.objects),
            "possible_activity": self.possible_activity,
            "text_in_image": self.text_in_image,
            "caption": self.caption,
        }


class LocalVLMAdapter(Protocol):
    engine: str
    available: bool

    def analyze_image(self, image_path: str | Path, *, ocr_text: str | None = None) -> VLMAnalysisResult:
        """Analyze a local image file without using cloud services."""


class UnconfiguredVLMAdapter:
    engine = "none"
    available = False

    def __init__(self, reason: str = "未解析: VLM backend is not configured") -> None:
        self.reason = reason

    def analyze_image(self, image_path: str | Path, *, ocr_text: str | None = None) -> VLMAnalysisResult:
        return VLMAnalysisResult(engine=self.engine, skipped=True, reason=self.reason, text_in_image=ocr_text)


class OllamaVLMAdapter:
    """Adapter for a localhost Ollama vision model."""

    engine = "ollama"
    available = True

    def __init__(self, *, model_name: str, base_url: str | None = None, timeout_seconds: float = 60.0) -> None:
        self.model_name = model_name
        self.base_url = (base_url or os.getenv(OLLAMA_URL_ENV_VAR, "http://127.0.0.1:11434")).rstrip("/")
        self.timeout_seconds = timeout_seconds
        _ensure_local_url(self.base_url)

    def analyze_image(self, image_path: str | Path, *, ocr_text: str | None = None) -> VLMAnalysisResult:
        prompt = (
            "Return compact JSON for this private local lifelog photo with keys: "
            "caption, scene, objects, possible_activity, text_in_image. "
            "Do not include private speculation beyond visible evidence."
        )
        try:
            image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "format": "json",
            }
            request = Request(
                f"{self.base_url}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - localhost-only URL validated above
                body = json.loads(response.read().decode("utf-8"))
            model_response = body.get("response") or ""
            parsed = _json_object(model_response)
        except Exception as exc:  # pragma: no cover - depends on local Ollama setup
            return VLMAnalysisResult(
                engine=self.engine,
                skipped=True,
                reason=f"未解析: Ollama analysis failed with {exc.__class__.__name__}",
                text_in_image=ocr_text,
            )

        result = _result_from_mapping(parsed, engine=self.engine, ocr_text=ocr_text)
        return VLMAnalysisResult(
            engine=result.engine,
            caption=result.caption,
            scene=result.scene,
            objects=result.objects,
            possible_activity=result.possible_activity,
            text_in_image=result.text_in_image,
            raw={"provider": "ollama", "model": self.model_name},
        )


class LlamaCppVLMAdapter:
    """Placeholder boundary for a local llama.cpp vision endpoint."""

    engine = "llama.cpp"
    available = True

    def __init__(self, *, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        _ensure_local_url(self.base_url)

    def analyze_image(self, image_path: str | Path, *, ocr_text: str | None = None) -> VLMAnalysisResult:
        return VLMAnalysisResult(
            engine=self.engine,
            skipped=True,
            reason="未解析: llama.cpp vision endpoint adapter is not configured for this model",
            text_in_image=ocr_text,
        )


class TransformersVLMAdapter:
    """Adapter for a local Hugging Face image-to-text model path."""

    engine = "transformers"
    available = True

    def __init__(self, *, model_name_or_path: str) -> None:
        self.model_name_or_path = model_name_or_path
        self._pipeline: Any | None = None

    def analyze_image(self, image_path: str | Path, *, ocr_text: str | None = None) -> VLMAnalysisResult:
        try:
            from transformers import pipeline
        except ImportError:
            return VLMAnalysisResult(
                engine=self.engine,
                skipped=True,
                reason="未解析: transformers is not installed",
                text_in_image=ocr_text,
            )

        try:
            if self._pipeline is None:
                self._pipeline = pipeline(
                    "image-to-text",
                    model=self.model_name_or_path,
                    local_files_only=True,
                )
            rows = self._pipeline(str(image_path), max_new_tokens=80)
            caption = _caption_from_pipeline(rows)
        except Exception as exc:  # pragma: no cover - depends on local model setup
            return VLMAnalysisResult(
                engine=self.engine,
                skipped=True,
                reason=f"未解析: Transformers analysis failed with {exc.__class__.__name__}",
                text_in_image=ocr_text,
            )

        if not caption:
            return VLMAnalysisResult(
                engine=self.engine,
                skipped=True,
                reason="未解析: no caption generated",
                text_in_image=ocr_text,
            )
        return VLMAnalysisResult(
            engine=self.engine,
            caption=caption,
            scene=caption,
            text_in_image=ocr_text,
            raw={"provider": "transformers", "model": self.model_name_or_path},
        )


def get_vlm_adapter(backend: str | None = None, model_name: str | None = None) -> LocalVLMAdapter:
    resolved = (backend or os.getenv(VLM_BACKEND_ENV_VAR, "none")).strip().lower()
    model = model_name or os.getenv(VLM_MODEL_ENV_VAR)
    if resolved in {"", "none", "disabled", "off"}:
        return UnconfiguredVLMAdapter()
    if resolved == "ollama":
        if not model:
            return UnconfiguredVLMAdapter(reason="未解析: Ollama VLM model is not configured")
        return OllamaVLMAdapter(model_name=model)
    if resolved in {"llama.cpp", "llamacpp", "llama-cpp"}:
        base_url = os.getenv(LLAMA_CPP_URL_ENV_VAR)
        if not base_url:
            return UnconfiguredVLMAdapter(reason="未解析: llama.cpp local endpoint is not configured")
        return LlamaCppVLMAdapter(base_url=base_url)
    if resolved == "transformers":
        if not model:
            return UnconfiguredVLMAdapter(reason="未解析: Transformers VLM model path is not configured")
        return TransformersVLMAdapter(model_name_or_path=model)
    return UnconfiguredVLMAdapter(reason=f"未解析: unsupported VLM backend '{resolved}'")


def _ensure_local_url(raw_url: str) -> None:
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("VLM endpoints must be localhost-only")


def _json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"caption": text.strip()} if text.strip() else {}
    return value if isinstance(value, dict) else {}


def _result_from_mapping(payload: dict[str, Any], *, engine: str, ocr_text: str | None) -> VLMAnalysisResult:
    objects = payload.get("objects") or []
    if isinstance(objects, str):
        objects = [objects]
    if not isinstance(objects, list):
        objects = []
    text_in_image = payload.get("text_in_image") or ocr_text
    return VLMAnalysisResult(
        engine=engine,
        caption=_string_or_none(payload.get("caption")),
        scene=_string_or_none(payload.get("scene")),
        objects=tuple(str(item) for item in objects if str(item).strip()),
        possible_activity=_string_or_none(payload.get("possible_activity")),
        text_in_image=_string_or_none(text_in_image),
    )


def _caption_from_pipeline(rows: Any) -> str | None:
    if isinstance(rows, list) and rows:
        first = rows[0]
        if isinstance(first, dict):
            return _string_or_none(first.get("generated_text") or first.get("caption"))
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


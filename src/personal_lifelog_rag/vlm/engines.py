"""Local VLM engine implementations."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from personal_lifelog_rag.vlm.base import VlmEngine
from personal_lifelog_rag.vlm.image_preprocess import preprocessed_vlm_image_path
from personal_lifelog_rag.vlm.prompts import SAFE_IMAGE_ANALYSIS_PROMPT_VERSION
from personal_lifelog_rag.vlm.safety import result_from_payload, safe_json_object, sanitize_vlm_result
from personal_lifelog_rag.vlm.schemas import VlmResult


VLM_BACKEND_ENV_VAR = "PERSONAL_LIFELOG_RAG_VLM_BACKEND"
VLM_MODEL_ENV_VAR = "PERSONAL_LIFELOG_RAG_VLM_MODEL"
OLLAMA_URL_ENV_VAR = "PERSONAL_LIFELOG_RAG_OLLAMA_URL"
LLAMA_CPP_URL_ENV_VAR = "PERSONAL_LIFELOG_RAG_LLAMA_CPP_URL"


class NoopVlmEngine:
    name = "noop"
    model_name = None

    def is_available(self) -> bool:
        return True

    def analyze_image(self, image_path: Path, prompt: str) -> VlmResult:
        return VlmResult(engine=self.name, status="skipped", error_message="VLM engine is noop")


class FakeVlmEngine:
    """Test-only deterministic VLM engine."""

    name = "fake"

    def __init__(self, *, model_name: str | None = "fake-vlm", caption: str = "ラーメンの可能性がある料理写真") -> None:
        self.model_name = model_name
        self.caption = caption

    def is_available(self) -> bool:
        return True

    def analyze_image(self, image_path: Path, prompt: str) -> VlmResult:
        payload = {
            "caption": self.caption,
            "short_caption": self.caption[:40],
            "scene_tags": ["restaurant", "indoor"],
            "object_tags": ["bowl", "table"],
            "activity_tags": ["meal_possible"],
            "location_cues": ["shop_possible"],
            "food_cues": ["ramen_possible", "meal_possible"],
            "people_count": 0,
            "contains_text_hint": False,
            "confidence": 0.82,
            "safety_flags": ["low_confidence"],
        }
        return result_from_payload(
            payload,
            engine=self.name,
            model_name=self.model_name,
            prompt_version=SAFE_IMAGE_ANALYSIS_PROMPT_VERSION,
        )


class OllamaVlmEngine:
    """Adapter skeleton for localhost Ollama vision models."""

    name = "ollama"

    def __init__(self, *, model_name: str | None, base_url: str | None = None, timeout_sec: int = 90) -> None:
        self.model_name = model_name
        self.base_url = (base_url or os.getenv(OLLAMA_URL_ENV_VAR, "http://127.0.0.1:11434")).rstrip("/")
        self.timeout_sec = timeout_sec
        _ensure_local_url(self.base_url)

    def is_available(self) -> bool:
        return bool(self.model_name)

    def analyze_image(self, image_path: Path, prompt: str) -> VlmResult:
        if not self.is_available():
            return VlmResult(
                engine=self.name,
                model_name=self.model_name,
                status="engine_unavailable",
                error_message="Ollama model is not configured",
            )
        try:
            with preprocessed_vlm_image_path(image_path) as processed_path:
                image_b64 = base64.b64encode(processed_path.read_bytes()).decode("ascii")
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
            with urlopen(request, timeout=self.timeout_sec) as response:  # noqa: S310 - localhost-only URL
                body = json.loads(response.read().decode("utf-8"))
            result = result_from_payload(
                safe_json_object(str(body.get("response") or "")),
                engine=self.name,
                model_name=self.model_name,
                prompt_version=SAFE_IMAGE_ANALYSIS_PROMPT_VERSION,
            )
            return result
        except Exception as exc:  # pragma: no cover - depends on local Ollama setup
            return VlmResult(
                engine=self.name,
                model_name=self.model_name,
                status="failed",
                error_message=f"Ollama VLM failed with {exc.__class__.__name__}",
            )


class TransformersVlmEngine:
    """Adapter skeleton for local Hugging Face image-to-text models."""

    name = "transformers"

    def __init__(self, *, model_name: str | None) -> None:
        self.model_name = model_name
        self._pipeline: Any | None = None

    def is_available(self) -> bool:
        return bool(self.model_name)

    def analyze_image(self, image_path: Path, prompt: str) -> VlmResult:
        if not self.is_available():
            return VlmResult(
                engine=self.name,
                model_name=self.model_name,
                status="engine_unavailable",
                error_message="Transformers local model path is not configured",
            )
        try:
            from transformers import pipeline
        except ImportError:
            return VlmResult(
                engine=self.name,
                model_name=self.model_name,
                status="engine_unavailable",
                error_message="transformers is not installed",
            )
        try:
            if self._pipeline is None:
                self._pipeline = pipeline("image-to-text", model=self.model_name, local_files_only=True)
            with preprocessed_vlm_image_path(image_path) as processed_path:
                rows = self._pipeline(str(processed_path), max_new_tokens=80)
            caption = _caption_from_pipeline(rows)
        except Exception as exc:  # pragma: no cover - depends on local model setup
            return VlmResult(
                engine=self.name,
                model_name=self.model_name,
                status="failed",
                error_message=f"Transformers VLM failed with {exc.__class__.__name__}",
            )
        if not caption:
            return VlmResult(engine=self.name, model_name=self.model_name, status="no_visual_content")
        return sanitize_vlm_result(
            VlmResult(
                caption=f"{caption} の可能性があります",
                short_caption=caption,
                scene_tags=[caption],
                engine=self.name,
                model_name=self.model_name,
                prompt_version=SAFE_IMAGE_ANALYSIS_PROMPT_VERSION,
                status="success",
                confidence=None,
                safety_flags=["low_confidence"],
            )
        )


class LlamaCppVlmEngine:
    """Placeholder boundary for localhost llama.cpp vision endpoints."""

    name = "llama_cpp"

    def __init__(self, *, base_url: str | None) -> None:
        self.base_url = base_url
        self.model_name = None
        if base_url:
            _ensure_local_url(base_url)

    def is_available(self) -> bool:
        return bool(self.base_url)

    def analyze_image(self, image_path: Path, prompt: str) -> VlmResult:
        return VlmResult(
            engine=self.name,
            status="engine_unavailable",
            error_message="llama.cpp vision adapter is a local placeholder",
        )


def get_vlm_engine(name: str | None = None, *, model_name: str | None = None) -> VlmEngine:
    resolved = (name or os.getenv(VLM_BACKEND_ENV_VAR, "noop")).strip().lower()
    model = model_name or os.getenv(VLM_MODEL_ENV_VAR)
    if resolved in {"", "noop", "none", "disabled", "off"}:
        return NoopVlmEngine()
    if resolved == "fake":
        return FakeVlmEngine(model_name=model or "fake-vlm")
    if resolved == "ollama":
        return OllamaVlmEngine(model_name=model)
    if resolved == "transformers":
        return TransformersVlmEngine(model_name=model)
    if resolved in {"llama_cpp", "llama.cpp", "llamacpp"}:
        return LlamaCppVlmEngine(base_url=os.getenv(LLAMA_CPP_URL_ENV_VAR))
    return NoopVlmEngine()


def _ensure_local_url(raw_url: str) -> None:
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("VLM endpoints must be localhost-only")


def _caption_from_pipeline(rows: Any) -> str | None:
    if isinstance(rows, list) and rows:
        first = rows[0]
        if isinstance(first, dict):
            value = first.get("generated_text") or first.get("caption")
            return str(value).strip() if value else None
    return None


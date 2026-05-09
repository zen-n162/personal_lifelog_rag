"""Local VLM engine implementations."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import traceback
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

    def __init__(self, *, model_name: str | None = "fake-vlm", caption: str = "ラーメンやご飯の可能性がある料理写真") -> None:
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
                error_message=_exception_message("Ollama VLM failed", exc),
            )


class TransformersVlmEngine:
    """Adapter skeleton for local Hugging Face image-to-text models."""

    name = "transformers"

    def __init__(
        self,
        *,
        model_name: str | None,
        model_path: str | None = None,
        device: str | None = "auto",
        dtype: str | None = None,
        local_files_only: bool | None = True,
        max_image_size: int | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.device = device or "auto"
        self.dtype = dtype
        self.local_files_only = True if local_files_only is None else bool(local_files_only)
        self.max_image_size = max_image_size or 1024
        self.max_new_tokens = max_new_tokens or 80
        self._pipeline: Any | None = None

    @property
    def model_ref(self) -> str | None:
        return self.model_path or self.model_name

    def is_available(self) -> bool:
        return bool(self.model_ref)

    def analyze_image(self, image_path: Path, prompt: str) -> VlmResult:
        if not self.is_available():
            return VlmResult(
                engine=self.name,
                model_name=self.model_ref,
                status="engine_unavailable",
                error_message="Transformers local model path is not configured",
            )
        try:
            from transformers import pipeline
        except ImportError:
            return VlmResult(
                engine=self.name,
                model_name=self.model_ref,
                status="engine_unavailable",
                error_message="transformers is not installed",
            )
        try:
            if self._pipeline is None:
                self._pipeline = pipeline("image-to-text", model=self.model_ref, local_files_only=self.local_files_only)
            with preprocessed_vlm_image_path(image_path, max_side=self.max_image_size) as processed_path:
                rows = self._pipeline(str(processed_path), max_new_tokens=self.max_new_tokens)
            caption = _caption_from_pipeline(rows)
        except Exception as exc:  # pragma: no cover - depends on local model setup
            return VlmResult(
                engine=self.name,
                model_name=self.model_ref,
                status="failed",
                error_message=_exception_message("Transformers VLM failed", exc),
            )
        if not caption:
            return VlmResult(engine=self.name, model_name=self.model_ref, status="no_visual_content")
        return sanitize_vlm_result(
            VlmResult(
                caption=f"{caption} の可能性があります",
                short_caption=caption,
                scene_tags=[caption],
                engine=self.name,
                model_name=self.model_ref,
                prompt_version=SAFE_IMAGE_ANALYSIS_PROMPT_VERSION,
                status="success",
                confidence=None,
                safety_flags=["low_confidence"],
            )
        )


class Qwen3VlTransformersEngine(TransformersVlmEngine):
    """Local-only Qwen3-VL Transformers adapter.

    Qwen3-VL expects chat-template inputs, not the generic image-to-text
    pipeline path. The first attempt uses a plain absolute image path string.
    If local transformers interprets that badly, the adapter retries with a
    PIL.Image.Image object.
    """

    name = "qwen3_vl_transformers"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model: Any | None = None
        self._processor: Any | None = None

    def is_available(self) -> bool:
        return bool(self.model_ref and Path(self.model_ref).expanduser().exists())

    def analyze_image(self, image_path: Path, prompt: str) -> VlmResult:
        if not self.is_available():
            return VlmResult(
                engine=self.name,
                model_name=self.model_ref,
                status="engine_unavailable",
                error_message="Qwen3-VL local model path is not configured or does not exist",
            )
        try:
            self._ensure_model_loaded()
        except ImportError as exc:
            return VlmResult(
                engine=self.name,
                model_name=self.model_ref,
                status="engine_unavailable",
                error_message=_qwen_exception_message(
                    "Qwen3-VL import failed",
                    exc,
                    image_mode="none",
                    image_path=image_path,
                    prompt=prompt,
                ),
            )
        except Exception as exc:  # pragma: no cover - depends on local model runtime
            return VlmResult(
                engine=self.name,
                model_name=self.model_ref,
                status="failed",
                error_message=_qwen_exception_message(
                    "Qwen3-VL model load failed",
                    exc,
                    image_mode="none",
                    image_path=image_path,
                    prompt=prompt,
                ),
            )

        first_error: Exception | None = None
        try:
            return self._analyze_with_image_content(
                image_path,
                prompt,
                image_content=str(image_path.expanduser().resolve()),
                image_mode="path",
            )
        except Exception as exc:  # pragma: no cover - depends on local model runtime
            first_error = exc

        try:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            try:
                return self._analyze_with_image_content(
                    image_path,
                    prompt,
                    image_content=image,
                    image_mode="pil",
                )
            finally:
                image.close()
        except Exception as exc:  # pragma: no cover - depends on local model runtime
            return VlmResult(
                engine=self.name,
                model_name=self.model_ref,
                status="failed",
                error_message=_qwen_exception_message(
                    "Qwen3-VL inference failed",
                    exc,
                    image_mode="pil",
                    image_path=image_path,
                    prompt=prompt,
                    previous_error=first_error,
                ),
            )

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        # The local conda env can otherwise crash in native MKL/OpenMP loading
        # during Qwen generation before Python can capture an exception.
        os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
        if self.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoModelForImageTextToText, AutoProcessor

        kwargs: dict[str, Any] = {
            "local_files_only": self.local_files_only,
            "trust_remote_code": True,
        }
        torch_dtype = _torch_dtype(self.dtype)
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        model = AutoModelForImageTextToText.from_pretrained(str(self.model_ref), **kwargs)
        device = _resolve_device(self.device)
        if device:
            model = model.to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            str(self.model_ref),
            local_files_only=self.local_files_only,
            trust_remote_code=True,
        )
        self._model = model
        self._processor = processor

    def _analyze_with_image_content(
        self,
        image_path: Path,
        prompt: str,
        *,
        image_content: Any,
        image_mode: str,
    ) -> VlmResult:
        messages = _qwen_messages(image_content=image_content, prompt=prompt)
        prefill_text = ""
        template = _chat_template_without_thinking(getattr(self._processor, "chat_template", None))
        if _should_use_json_prefill(self.model_ref):
            prefill_text = "{"
            messages = [*messages, {"role": "assistant", "content": prefill_text}]
            inputs = self._processor.apply_chat_template(
                messages,
                chat_template=template,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            inputs = self._processor.apply_chat_template(
                messages,
                chat_template=template,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        inputs.pop("token_type_ids", None)
        model_device = _model_device(self._model)
        inputs = inputs.to(model_device) if hasattr(inputs, "to") else {key: value.to(model_device) for key, value in inputs.items()}
        generated_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        input_ids = inputs.input_ids if hasattr(inputs, "input_ids") else inputs["input_ids"]
        generated_ids_trimmed = [
            output_ids[len(input_ids_row) :]
            for input_ids_row, output_ids in zip(input_ids, generated_ids, strict=False)
        ]
        decoded_rows = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        decoded_text = str(decoded_rows[0] if decoded_rows else "").strip()
        raw_text = _restore_prefilled_json(decoded_text, prefill_text)
        if not raw_text:
            return VlmResult(engine=self.name, model_name=self.model_ref, status="no_visual_content")
        parsed = safe_json_object(raw_text)
        if parsed.get("_parse_error"):
            return VlmResult(
                engine=self.name,
                model_name=self.model_ref,
                prompt_version=SAFE_IMAGE_ANALYSIS_PROMPT_VERSION,
                status="failed",
                error_message=_qwen_parse_error_message(
                    raw_text=raw_text,
                    image_mode=image_mode,
                    image_path=image_path,
                    prompt=prompt,
                ),
                safety_flags=["json_parse_failed"],
                raw={"image_loading_mode": image_mode, "raw_text": raw_text[:1000]},
            )
        result = result_from_payload(
            parsed,
            engine=self.name,
            model_name=self.model_ref,
            prompt_version=SAFE_IMAGE_ANALYSIS_PROMPT_VERSION,
        )
        result = sanitize_vlm_result(result)
        return VlmResult(**{**result.to_dict(), "raw": {"image_loading_mode": image_mode, "raw_text": raw_text[:1000]}})


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


def get_vlm_engine(
    name: str | None = None,
    *,
    model_name: str | None = None,
    model_path: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    local_files_only: bool | None = True,
    max_image_size: int | None = None,
    max_new_tokens: int | None = None,
) -> VlmEngine:
    resolved = (name or os.getenv(VLM_BACKEND_ENV_VAR, "noop")).strip().lower()
    model = model_name or os.getenv(VLM_MODEL_ENV_VAR)
    if resolved in {"", "noop", "none", "disabled", "off"}:
        return NoopVlmEngine()
    if resolved == "fake":
        return FakeVlmEngine(model_name=model or "fake-vlm")
    if resolved == "ollama":
        return OllamaVlmEngine(model_name=model)
    if resolved == "transformers":
        return TransformersVlmEngine(
            model_name=model,
            model_path=model_path,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            max_image_size=max_image_size,
            max_new_tokens=max_new_tokens,
        )
    if resolved in {"qwen3_vl_transformers", "qwen3-vl-transformers", "qwen3vl_transformers"}:
        return Qwen3VlTransformersEngine(
            model_name=model,
            model_path=model_path,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            max_image_size=max_image_size,
            max_new_tokens=max_new_tokens,
        )
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


def _exception_message(prefix: str, exc: Exception, *, max_chars: int = 4000) -> str:
    traceback_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    message = f"{prefix}: {exc.__class__.__name__}: {exc}\n{traceback_text}".strip()
    return message[:max_chars]


def _qwen_exception_message(
    prefix: str,
    exc: Exception,
    *,
    image_mode: str,
    image_path: Path,
    prompt: str,
    previous_error: Exception | None = None,
    max_chars: int = 4000,
) -> str:
    parts = [
        prefix,
        f"exception_class: {exc.__class__.__name__}",
        f"exception_repr: {exc!r}",
        f"image_loading_mode: {image_mode}",
        f"image_path: {_short_path(image_path)}",
        f"prompt_template: {_prompt_hint(prompt)}",
    ]
    if previous_error is not None:
        parts.extend(
            [
                "previous_attempt:",
                f"previous_exception_class: {previous_error.__class__.__name__}",
                f"previous_exception_repr: {previous_error!r}",
                "previous_traceback_tail:",
                _traceback_tail(previous_error),
            ]
        )
    parts.extend(["traceback_tail:", _traceback_tail(exc)])
    return "\n".join(parts)[:max_chars]


def _qwen_parse_error_message(
    *,
    raw_text: str,
    image_mode: str,
    image_path: Path,
    prompt: str,
    max_chars: int = 4000,
) -> str:
    try:
        raise ValueError("Qwen3-VL output was not valid JSON")
    except ValueError as exc:
        parts = [
            "Qwen3-VL JSON parse failed",
            f"exception_class: {exc.__class__.__name__}",
            f"exception_repr: {exc!r}",
            f"raw_output_head: {_truncate_for_error(raw_text, 500)}",
            f"prompt_template: {_prompt_hint(prompt)}",
            f"image_input_mode: {image_mode}",
            f"image_path: {_short_path(image_path)}",
            "traceback_tail:",
            _traceback_tail(exc),
        ]
        return "\n".join(parts)[:max_chars]


def _traceback_tail(exc: Exception, *, lines: int = 24) -> str:
    traceback_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return "\n".join(traceback_text.splitlines()[-lines:])


def _short_path(path: Path) -> str:
    resolved = path.expanduser()
    parts = resolved.parts
    if len(parts) <= 3:
        return str(resolved)
    return ".../" + "/".join(parts[-3:])


def _prompt_hint(prompt: str) -> str:
    normalized = " ".join(str(prompt).split())
    return normalized[:240]


def _truncate_for_error(text: str, max_chars: int) -> str:
    normalized = str(text).replace("\x00", "").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "…"


def _chat_template_without_thinking(template: Any) -> str | None:
    if not isinstance(template, str) or "<think>" not in template:
        return template if isinstance(template, str) else None
    replacements = {
        "{{- '<|im_start|>assistant\\n<think>\\n' }}": "{{- '<|im_start|>assistant\\n' }}",
        "{{- '<|im_start|>assistant\\n<think>\\n' -}}": "{{- '<|im_start|>assistant\\n' -}}",
        "<|im_start|>assistant\n<think>\n": "<|im_start|>assistant\n",
        "<|im_start|>assistant\\n<think>\\n": "<|im_start|>assistant\\n",
    }
    cleaned = template
    for before, after in replacements.items():
        cleaned = cleaned.replace(before, after)
    return cleaned


def _qwen_messages(*, image_content: Any, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_content},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _should_use_json_prefill(model_ref: str | None) -> bool:
    return "thinking" in str(model_ref or "").lower()


def _restore_prefilled_json(decoded_text: str, prefill_text: str) -> str:
    stripped = decoded_text.strip()
    if not prefill_text:
        return stripped
    if stripped.startswith(prefill_text):
        return stripped
    return f"{prefill_text}{stripped}"


def _torch_dtype(dtype: str | None) -> Any:
    if not dtype:
        return None
    value = dtype.strip().lower()
    if value in {"auto", "torch.auto"}:
        return "auto"
    try:
        import torch
    except ImportError:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(value)


def _resolve_device(device: str | None) -> str | None:
    if not device or device == "auto":
        try:
            import torch
        except ImportError:
            return None
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return "cpu"

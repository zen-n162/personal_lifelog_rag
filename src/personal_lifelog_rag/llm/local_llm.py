"""Local LLM placeholder.

Cloud LLMs and external APIs are intentionally not used by the MVP. Future
adapters should load models from local files under `models/` and must keep
network access disabled by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LocalLLMNotConfigured(RuntimeError):
    """Raised when a caller asks for a local model before one is configured."""


class LocalLLMAdapter(Protocol):
    """Adapter shape for future Ollama, llama.cpp, or Transformers backends."""

    def generate(self, prompt: str) -> str:
        """Generate text locally from a prompt."""


@dataclass
class LocalLLM:
    model_path: str | None = None
    backend: str | None = None

    def generate(self, prompt: str) -> str:
        if not self.model_path:
            raise LocalLLMNotConfigured(
                "No local model is configured. MVP uses template answers instead."
            )
        raise LocalLLMNotConfigured(
            f"Local backend {self.backend or 'unknown'} is reserved for a later extension."
        )

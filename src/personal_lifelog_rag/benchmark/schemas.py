"""Shared schemas and tiny config readers for local multimodal benchmarks."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    image_path: Path
    query_texts: list[str] = field(default_factory=list)
    expected_tags_any: list[str] = field(default_factory=list)
    forbidden_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["image_path"] = str(self.image_path)
        return row


@dataclass(frozen=True)
class ModelSpec:
    engine: str | None = None
    provider: str | None = None
    model_name: str | None = None
    model_path: str | None = None
    device: str = "auto"
    dtype: str | None = None
    local_files_only: bool | None = None
    max_image_size: int | None = None
    max_new_tokens: int | None = None
    prompt_version: str | None = None
    embedding_dim: int | None = None
    batch_size: int | None = None

    def configured_model_ref(self) -> str | None:
        return self.model_path or self.model_name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelRuntimeConfig:
    vlm: ModelSpec = field(default_factory=ModelSpec)
    multimodal_embedding: ModelSpec = field(default_factory=ModelSpec)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vlm": self.vlm.to_dict(),
            "multimodal_embedding": self.multimodal_embedding.to_dict(),
        }


def load_benchmark_cases(path: str | Path, *, limit: int | None = None) -> list[BenchmarkCase]:
    """Load a small JSON/YAML benchmark case file.

    The parser is intentionally tiny to avoid adding a runtime YAML dependency.
    It supports the simple `cases: - id: ...` shape used by the sample config.
    """

    source = Path(path).expanduser()
    payload = _load_json_or_simple_yaml(source.read_text(encoding="utf-8"))
    rows = payload.get("cases", []) if isinstance(payload, dict) else []
    cases: list[BenchmarkCase] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("id") or f"case_{index:03d}")
        image_path = Path(str(row.get("image_path") or "")).expanduser()
        cases.append(
            BenchmarkCase(
                id=case_id,
                image_path=image_path,
                query_texts=_string_list(row.get("query_texts")),
                expected_tags_any=_string_list(row.get("expected_tags_any")),
                forbidden_terms=_string_list(row.get("forbidden_terms")),
            )
        )
    return cases[: max(limit, 0)] if limit is not None else cases


def load_model_runtime_config(path: str | Path | None = None) -> ModelRuntimeConfig:
    if path is None:
        return ModelRuntimeConfig()
    source = Path(path).expanduser()
    if not source.exists():
        return ModelRuntimeConfig()
    payload = _parse_nested_mapping(source.read_text(encoding="utf-8"))
    models = payload.get("models", {}) if isinstance(payload, dict) else {}
    vlm = _model_spec_from_mapping(models.get("vlm", {}))
    embedding = _model_spec_from_mapping(models.get("multimodal_embedding", {}))
    return ModelRuntimeConfig(vlm=vlm, multimodal_embedding=embedding)


def _model_spec_from_mapping(value: Any) -> ModelSpec:
    row = value if isinstance(value, dict) else {}
    return ModelSpec(
        engine=_none_or_str(row.get("engine")),
        provider=_none_or_str(row.get("provider")),
        model_name=_none_or_str(row.get("model_name")),
        model_path=_none_or_str(row.get("model_path")),
        device=_none_or_str(row.get("device")) or "auto",
        dtype=_none_or_str(row.get("dtype")),
        local_files_only=_none_or_bool(row.get("local_files_only")),
        max_image_size=_none_or_int(row.get("max_image_size")),
        max_new_tokens=_none_or_int(row.get("max_new_tokens")),
        prompt_version=_none_or_str(row.get("prompt_version")),
        embedding_dim=_none_or_int(row.get("embedding_dim")),
        batch_size=_none_or_int(row.get("batch_size")),
    )


def _load_json_or_simple_yaml(raw_text: str) -> Any:
    stripped = raw_text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return json.loads(raw_text)
    return _parse_cases_yaml(raw_text)


def _parse_cases_yaml(raw_text: str) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_list_key: str | None = None
    current_list_indent: int | None = None
    in_cases = False

    for raw_line in raw_text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 0 and stripped == "cases:":
            in_cases = True
            continue
        if not in_cases:
            continue
        if stripped.startswith("- "):
            remainder = stripped[2:].strip()
            if current is not None and current_list_key is not None and current_list_indent is not None and indent > current_list_indent:
                current.setdefault(current_list_key, []).append(_parse_scalar(remainder))
                continue
            if current is not None:
                cases.append(current)
            current = {}
            current_list_key = None
            current_list_indent = None
            if remainder and ":" in remainder:
                key, value = _split_key_value(remainder)
                current[key] = _parse_scalar(value)
            continue
        if current is None or ":" not in stripped:
            continue
        key, value = _split_key_value(stripped)
        if value == "":
            current[key] = []
            current_list_key = key
            current_list_indent = indent
        else:
            current[key] = _parse_scalar(value)
            current_list_key = None
            current_list_indent = None

    if current is not None:
        cases.append(current)
    return {"cases": cases}


def _parse_nested_mapping(raw_text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in raw_text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, value = _split_key_value(line.strip())
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _split_key_value(value: str) -> tuple[str, str]:
    key, raw = value.split(":", 1)
    return key.strip().strip("\"'"), raw.strip()


def _parse_scalar(value: str) -> Any:
    if value == "" or value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = [item.strip().strip("\"'") for item in value[1:-1].split(",") if item.strip()]
        return parsed if isinstance(parsed, list) else []
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value.strip("\"'")
    for parser in (int, float):
        try:
            return parser(value)
        except ValueError:
            continue
    return value


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _none_or_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _none_or_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _none_or_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return None

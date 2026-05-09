"""Small vector helpers without heavy dependencies."""

from __future__ import annotations

import json
import math
import struct
from typing import Any


FLOAT32_FORMAT = "float32_numpy"
JSON_FORMAT = "json"


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else list(vector)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=False))


def serialize_embedding(vector: list[float], *, embedding_format: str = FLOAT32_FORMAT) -> bytes:
    if embedding_format == JSON_FORMAT:
        return json.dumps([float(value) for value in vector], ensure_ascii=False).encode("utf-8")
    return struct.pack(f"<{len(vector)}f", *[float(value) for value in vector])


def deserialize_embedding(raw: bytes | memoryview | str | None, *, embedding_format: str | None, dim: int | None) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, memoryview):
        raw_bytes = raw.tobytes()
    elif isinstance(raw, bytes):
        raw_bytes = raw
    elif isinstance(raw, str):
        raw_bytes = raw.encode("utf-8")
    else:
        return []
    if embedding_format == JSON_FORMAT:
        try:
            parsed = json.loads(raw_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return []
        return [float(value) for value in parsed] if isinstance(parsed, list) else []
    if not raw_bytes:
        return []
    if len(raw_bytes) % 4 != 0:
        return []
    count = len(raw_bytes) // 4
    if dim is not None and dim > 0 and count != dim:
        return []
    return [float(value) for value in struct.unpack(f"<{count}f", raw_bytes)]


def vector_dim_from_blob(raw: Any, *, embedding_format: str | None) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    if embedding_format == JSON_FORMAT:
        vector = deserialize_embedding(raw, embedding_format=embedding_format, dim=None)
        return len(vector) if vector else None
    if isinstance(raw, bytes) and len(raw) % 4 == 0:
        return len(raw) // 4
    return None


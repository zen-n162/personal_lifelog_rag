"""Application configuration with local-first defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

APP_CONFIG_ENV_VAR = "PERSONAL_LIFELOG_RAG_CONFIG"
DB_PATH_ENV_VAR = "PERSONAL_LIFELOG_RAG_DB_PATH"
DEFAULT_YEAR_ENV_VAR = "PERSONAL_LIFELOG_RAG_DEFAULT_YEAR"


@dataclass(frozen=True)
class AppConfig:
    """Resolved paths used by the local MVP."""

    root_dir: Path
    db_path: Path
    raw_photos_dir: Path
    raw_line_dir: Path
    processed_dir: Path
    thumbnails_dir: Path
    models_dir: Path
    default_year: int = 2024
    allow_network: bool = False
    host: str = "127.0.0.1"

    @classmethod
    def default(cls, root_dir: str | Path | None = None) -> "AppConfig":
        root = Path(root_dir) if root_dir is not None else Path.cwd()
        root = root.expanduser()
        db_path = Path(os.getenv(DB_PATH_ENV_VAR, "data/db/lifelog.sqlite"))
        if not db_path.is_absolute():
            db_path = root / db_path

        return cls(
            root_dir=root,
            db_path=db_path,
            raw_photos_dir=root / "data/raw/photos",
            raw_line_dir=root / "data/raw/line",
            processed_dir=root / "data/processed",
            thumbnails_dir=root / "data/thumbnails",
            models_dir=root / "models",
            default_year=int(os.getenv(DEFAULT_YEAR_ENV_VAR, "2024")),
        )


def load_config(root_dir: str | Path | None = None) -> AppConfig:
    """Load MVP config.

    The current implementation intentionally avoids a runtime YAML dependency.
    The example YAML documents the shape, while environment variables provide
    the values needed by the MVP.
    """

    return AppConfig.default(root_dir=root_dir)


def load_event_building_config(root_dir: str | Path | None = None) -> dict[str, Any]:
    """Load the small event_building section without requiring PyYAML."""

    root = Path(root_dir) if root_dir is not None else Path.cwd()
    raw_path = Path(os.getenv(APP_CONFIG_ENV_VAR, "configs/app.example.yaml"))
    config_path = raw_path if raw_path.is_absolute() else root / raw_path
    if not config_path.exists():
        return {}

    values: dict[str, Any] = {}
    in_section = False
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            in_section = line[:-1].strip() == "event_building"
            continue
        if not in_section or not line.startswith("  ") or ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        values[key.strip()] = _parse_scalar(value.strip())
    return values


def _parse_scalar(value: str) -> Any:
    if value == "":
        return None
    for parser in (int, float):
        try:
            return parser(value)
        except ValueError:
            continue
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return value

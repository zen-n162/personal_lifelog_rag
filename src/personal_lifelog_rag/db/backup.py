"""Local SQLite backup helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil


DEFAULT_BACKUP_DIR = Path("backups")


@dataclass(frozen=True)
class BackupResult:
    source_path: Path
    backup_path: Path
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": str(self.source_path),
            "backup_path": str(self.backup_path),
            "size_bytes": self.size_bytes,
        }


def backup_sqlite_db(
    db_path: str | Path,
    *,
    label: str | None = None,
    output_dir: str | Path = DEFAULT_BACKUP_DIR,
    now: datetime | None = None,
) -> BackupResult:
    source = Path(db_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"SQLite database not found: {source}")
    if not source.is_file():
        raise FileNotFoundError(f"SQLite database path is not a file: {source}")

    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    clean_label = _safe_label(label)
    name = f"lifelog_{clean_label}_{timestamp}.sqlite" if clean_label else f"lifelog_backup_{timestamp}.sqlite"
    destination_dir = Path(output_dir).expanduser()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / name
    shutil.copy2(source, destination)
    return BackupResult(
        source_path=source,
        backup_path=destination,
        size_bytes=destination.stat().st_size,
    )


def _safe_label(label: str | None) -> str:
    if not label:
        return ""
    chars = [char if char.isalnum() or char in {"-", "_"} else "_" for char in label.strip()]
    return "".join(chars).strip("_")


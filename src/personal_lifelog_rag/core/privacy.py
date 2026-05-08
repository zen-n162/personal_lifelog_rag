"""Privacy guardrails for local personal data handling."""

from __future__ import annotations

from pathlib import Path

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class PrivacyError(ValueError):
    """Raised when a requested operation breaks local-first constraints."""


def ensure_localhost(host: str) -> str:
    """Validate that a web app binds only to localhost."""

    if host not in LOCAL_HOSTS:
        raise PrivacyError(f"Refusing to bind to non-local host: {host}")
    return host


def redact_text(text: str | None, max_chars: int = 80) -> str:
    """Return a short preview that is safe enough for CLI/debug output."""

    if not text:
        return ""
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars]}..."


def is_inside(path: str | Path, parent: str | Path) -> bool:
    """Return whether path is under parent after resolution."""

    resolved_path = Path(path).expanduser().resolve()
    resolved_parent = Path(parent).expanduser().resolve()
    return resolved_path == resolved_parent or resolved_parent in resolved_path.parents

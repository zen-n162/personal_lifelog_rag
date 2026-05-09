"""Best-effort repair for local VLM JSON-like outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any


@dataclass(frozen=True)
class JsonRepairResult:
    payload: dict[str, Any] | None
    repaired_text: str | None = None
    repaired: bool = False
    notes: list[str] = field(default_factory=list)
    error_message: str | None = None


def parse_json_object_with_repair(text: str) -> JsonRepairResult:
    """Parse a JSON object, repairing common Qwen Thinking output glitches.

    The repair path is intentionally conservative. It handles markdown fences,
    prose before/after JSON, trailing commas, missing closing brackets, and a
    narrow class of unescaped quotes inside string values. If the result still
    cannot be parsed as an object, callers should keep the row failed.
    """

    raw = str(text or "")
    notes: list[str] = []
    if not raw.strip():
        return JsonRepairResult(payload=None, error_message="empty output")

    candidates: list[tuple[str, list[str]]] = []
    candidates.append((raw.strip(), []))
    fenced = _extract_fenced_json(raw)
    if fenced:
        candidates.append((fenced, ["removed_code_fence"]))
    extracted = _extract_object_region(fenced or raw)
    if extracted:
        candidates.append((extracted, ["extracted_json_object"]))

    seen: set[str] = set()
    last_error: str | None = None
    for candidate, candidate_notes in candidates:
        for repaired_text, repair_notes in _repair_candidates(candidate):
            if repaired_text in seen:
                continue
            seen.add(repaired_text)
            try:
                payload = json.loads(repaired_text)
            except json.JSONDecodeError as exc:
                last_error = f"{exc.__class__.__name__}: {exc}"
                continue
            if isinstance(payload, dict):
                notes = [*candidate_notes, *repair_notes]
                return JsonRepairResult(
                    payload=payload,
                    repaired_text=repaired_text,
                    repaired=bool(notes) or repaired_text != raw.strip(),
                    notes=_unique(notes),
                )
            last_error = "parsed JSON was not an object"

    return JsonRepairResult(payload=None, error_message=last_error or "invalid JSON")


def _repair_candidates(text: str) -> list[tuple[str, list[str]]]:
    stripped = text.strip()
    candidates: list[tuple[str, list[str]]] = [(stripped, [])]
    without_trailing = _remove_trailing_commas(stripped)
    if without_trailing != stripped:
        candidates.append((without_trailing, ["removed_trailing_commas"]))
    balanced = _balance_json_delimiters(without_trailing)
    if balanced != without_trailing:
        candidates.append((balanced, ["balanced_delimiters"]))
    escaped = _escape_likely_unescaped_quotes(balanced)
    if escaped != balanced:
        candidates.append((escaped, ["escaped_unescaped_quotes"]))
        escaped_balanced = _balance_json_delimiters(_remove_trailing_commas(escaped))
        if escaped_balanced != escaped:
            candidates.append((escaped_balanced, ["escaped_unescaped_quotes", "balanced_delimiters"]))
    return candidates


def _extract_fenced_json(text: str) -> str | None:
    match = re.search(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else None


def _extract_object_region(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    end = text.rfind("}")
    if end > start:
        return text[start : end + 1].strip()
    return text[start:].strip()


def _remove_trailing_commas(text: str) -> str:
    previous = text
    while True:
        current = re.sub(r",\s*([}\]])", r"\1", previous)
        if current == previous:
            return current
        previous = current


def _balance_json_delimiters(text: str) -> str:
    out: list[str] = []
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        out.append(char)
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in ("}", "]") and stack and stack[-1] == char:
            stack.pop()
    if in_string:
        out.append('"')
    out.extend(reversed(stack))
    return "".join(out)


def _escape_likely_unescaped_quotes(text: str) -> str:
    out: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            out.append(char)
            escaped = False
            continue
        if char == "\\":
            out.append(char)
            escaped = True
            continue
        if char != '"':
            out.append(char)
            continue
        if not in_string:
            in_string = True
            out.append(char)
            continue
        next_non_space = _next_non_space(text, index + 1)
        if next_non_space in {":", ",", "}", "]", ""}:
            in_string = False
            out.append(char)
        else:
            out.append('\\"')
    return "".join(out)


def _next_non_space(text: str, start: int) -> str:
    for char in text[start:]:
        if not char.isspace():
            return char
    return ""


def _unique(values: list[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        if value and value not in unique:
            unique.append(value)
    return unique

# Technical Interview Notes

## Architecture

The app is a local-first multimodal RAG system. It ingests photos, message
exports, GPS-derived metadata, and call logs into SQLite. Local analysis layers
add OCR text, VLM captions/tags, and multimodal embeddings.

## DB Design

- `media_items`: imported image metadata and local thumbnail references.
- `line_messages`: parsed local message records.
- `line_call_events`: structured call events.
- `media_vlm`: local VLM captions, tags, cues, status, and safety flags.
- `media_embeddings`: image and text vectors stored locally.
- `media_ocr`: local OCR text and redacted previews.
- `events`: generated timeline events.
- `event_evidence`: traceable links from events to source evidence.
- `media_vlm_overrides`: human review and search/event usability controls.

## Evidence And Confidence

The ranking system separates score from evidence strength:

- weak: VLM-only, embedding-only, OCR-only, or one isolated mention.
- medium: visual evidence plus event, OCR, or LINE context.
- strong: verified event or multiple independent modalities.

Even strong results are phrased cautiously.

## VLM And Embedding Roles

Qwen3-VL produces human-reviewable descriptions and tags. Qwen3-VL-Embedding is
used for retrieval and candidate generation. They are complementary, not
substitutes.

## Safety

The VLM prompt and safety filter avoid identity, relationship, emotion, and
sensitive attribute inference. Review overrides can remove wrong or private
results from normal search and event generation.

## Failure Recovery

Operational CLIs include missing file checks, failed-row retries, JSON repair,
analysis jobs, DB checks, backups, and release manifests.

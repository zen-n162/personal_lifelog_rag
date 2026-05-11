# Release Notes

## v0.1.1

v0.1.1 is a stabilization patch for the v0.1 freeze. It updates the private
evaluation expectations for the expanded 2025-03 dataset, records maintenance
results, and regenerates the release manifest, reports, and public portfolio
HTML.

Stabilization checks:

- `pytest`: 445 passed.
- `db-check --strict`: ok.
- `eval-private`: 17/17 passed.
- `batch-qa`: 5/5 succeeded.
- Portfolio privacy check: PASS.
- Release manifest: `reports/release_v0_1_manifest.json` with version
  `v0.1.1`.

Maintenance results:

- VLM failed retry selected 20 rows and recovered 4 rows.
- OCR failed retry processed 5 images and saved 5 OCR successes.
- Missing original files remain tracked as unavailable media, not deleted.
- 2025-03 status is confirmed without rerunning the heavy month rollout.

Private eval adjustments:

- The stage-photo case is now scoped to the 2024-12 period so newer 2025-03
  stage/performance photos do not make the regression target unstable.
- `json_repaired` is allowed as a VLM safety/quality flag because it indicates
  successful JSON recovery, not unsafe model output.

## v0.1

v0.1 freezes the first working local-first personal lifelog RAG workflow. The
application ingests local photos, LINE exports, GPS-derived metadata, OCR
results, VLM captions, multimodal embeddings, generated events, review
overrides, private evaluation runs, and public/private reports.

Major capabilities:

- Local SQLite-backed lifelog database.
- Photo, EXIF, GPS, LINE, and call-log ingestion.
- Qwen3-VL image caption and visual tag extraction.
- Qwen3-VL-Embedding image and text retrieval.
- Local OCR engine integration for image text.
- Evidence-linked event generation.
- Date QA, place QA, image search QA, and monthly summaries.
- Human-in-the-loop VLM review controls.
- Private eval, DB checks, batch QA, and report generation.
- Public-safe single-file portfolio HTML generation.

Model roles:

- Qwen3-VL: caption, scene tags, object tags, activity tags, food cues,
  location cues, text cues, safety flags.
- Qwen3-VL-Embedding: text-to-image retrieval, image vectors, combined-text
  vectors, hybrid ranking.
- OCR engine: receipts, signs, menus, labels, screenshots, and other image text.

Privacy stance:

- No external API calls.
- No cloud OCR, VLM, or embedding service.
- Raw private data, reports, and runtime settings are local-only.
- Public reports and portfolio HTML are redacted and checked before sharing.

Known limitations:

- OCR quality varies by image and local language data.
- VLM JSON can still require repair or retry.
- VLM-only and embedding-only matches remain weak evidence.
- Event split/merge review is still future work.

Next roadmap:

- Improve OCR priority and redaction quality.
- Expand monthly rollout after review.
- Add richer event split/merge controls.
- Consider FAISS, Qdrant, or Chroma for larger local indices.
- Continue tightening private eval and portfolio reporting.

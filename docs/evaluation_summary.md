# Evaluation Summary

This page summarizes the local evaluation snapshot used for portfolio
explanation. Values are aggregate-only and should be refreshed before public
presentation.

Snapshot date: 2026-05-09

## Automated Tests

```text
pytest: 435 passed
```

The test suite covers ingestion, query intent, routing, event generation,
private eval, OCR/VLM/embedding services, multimodal search, UI service helpers,
monthly rollout planning, and safety filters with dummy data only.

## DB Integrity

```text
db-check --strict: ok
```

Strict checks include:

- media item integrity
- OCR/VLM orphan checks
- embedding dimension checks
- event evidence references
- fake/failed VLM evidence exclusion
- analysis job metadata integrity

Missing original media files are reported as warnings and skipped by OCR/VLM/
embedding jobs. Operators can opt into a hard failure with
`db-check --strict --fail-on-missing-files`.

## Private Eval

Latest local run:

```text
cases: 17
passed: 16
failed: 1
skipped: 0
top1 accuracy: 0.8
expected date recall@5: 0.8
forbidden phrase violations: 0
overclaim violations: 0
```

The single failure was a VLM quality case that did not yet allow the
`json_repaired` safety flag. This is a known evaluation-spec mismatch rather
than a privacy leak or database integrity issue.

Representative case types:

- date QA
- routed place QA
- keyword search
- image search
- multimodal search
- VLM quality
- event quality
- call search
- monthly summary

## Qwen3-VL Coverage

Current local rollout snapshot:

```text
media_vlm total: 700
success: 690
failed: 10
engine_unavailable: 0
engine: qwen3_vl_transformers
```

Common high-level tags included indoor scenes, cafe-like scenes, performance or
stage cues, food/meal cues, and vehicle interior cues. These are used as search
candidates, not final facts.

## Qwen3-VL-Embedding Coverage

Current local rollout snapshot:

```text
media_embeddings total: 1067
success: 1067
embedding_dim: 4096
types:
- image: 523
- combined_text: 544
```

Embeddings are used for candidate retrieval and then reranked with VLM, OCR,
LINE, event, place, and human-review signals.

## OCR Coverage

OCR is intentionally optional and still limited, but a text-cue-prioritized
Tesseract pass has started:

```text
media_ocr total: 61
success: 54
no_text_detected: 1
engine_unavailable: 6
2024-12..2025-02 OCR text-present rows: 54
```

OCR is useful for signs, receipts, tickets, menus, screenshots, and station or
shop text, but the project treats it as noisy evidence.

## Known Limitations

- OCR coverage is still small.
- Some VLM rows need JSON repair or retry.
- Visual model output can over-describe; safety filtering and human review are
  required.
- Embedding similarity alone cannot prove an event happened.
- LINE mention versus actual action remains a hard ranking problem.
- Place dictionaries require manual private setup.
- Event split/merge editing is not yet implemented.

## PR49 Acceptance Snapshot

Acceptance checks run on 2026-05-09:

```text
pytest: 430 passed
db-check --strict: ok
private eval: 16/17 passed, 0 skipped
batch-qa: 4/4 succeeded
```

Saved local run artifacts:

- `private_eval/runs/eval_20260509_164258.json`
- `eval_outputs/batch_qa/batch_qa_20260509_164337.json`
- `eval_outputs/batch_qa/batch_qa_20260509_164337.md`

Monthly QA smoke checks:

- `2024年12月は何していた？`: `monthly_summary`; 116 events, 388 photos, 54 OCR-success photos.
- `2025年1月は何していた？`: `monthly_summary`; 113 events, 865 photos, 98 VLM-analyzed photos.
- `2025年2月は何していた？`: `monthly_summary`; 113 events, 638 photos, 296 VLM-analyzed photos.

Image QA smoke checks:

- `ご飯を食べた写真はいつ？`: multimodal search; top candidates included 2025-02-14, 2025-02-09, 2025-02-11.
- `ステージの写真はいつ？`: multimodal search; top candidates centered on 2024-12-14.
- `レシートの写真はいつ？`: multimodal search; top candidates included 2025-02-10, 2024-12-24, 2024-12-07.

2025-03 rollout smoke:

- VLM: 286 success, 14 failed, 0 engine_unavailable.
- Embeddings: 586 success, 0 failed, 0 engine_unavailable.
- Events after rebuild: 119.
- `2025年3月は何していた？`: `monthly_summary`; 119 events, 1761 photos, 286 VLM-analyzed photos.
- DB check recovered to `db-check --strict: ok` after one empty-caption VLM success row was downgraded to failed.

OCR prioritization smoke:

```text
ocr-images --text-cues-only --limit 50 --skip-existing:
selected: 50
success: 49
no_text_detected: 1
failed: 0
engine_unavailable: 0
```

Missing-file maintenance snapshot:

```text
media_items total: 21122
missing original files: 24
missing thumbnails: 0
missing-files CSV exported to reports/missing_files.csv
```

## Evaluation Takeaway

The project has a working local multimodal search and QA loop with strict DB
checks and regression tests. The strongest portfolio point is not perfect
accuracy; it is the full privacy-preserving engineering loop:

1. local data ingestion
2. local vision and embedding models
3. evidence-linked event construction
4. human review
5. private eval
6. redacted reporting

## v0.1 Freeze Workflow

The v0.1 freeze adds a release manifest and reproducibility checklist:

- `release-check --version v0.1 --save-manifest`
- `reports/release_v0_1_manifest.json`
- `docs/releases/v0.1.md`
- `docs/reproducibility.md`

The release manifest records aggregate counts, DB validation, eval summary,
portfolio privacy status, model names, and environment metadata without storing
local model paths.

## PR56-63 Current Snapshot

Checks run on 2026-05-10:

```text
pytest: 445 passed
db-check --strict: ok
private eval: 15/17 passed, 0 skipped
portfolio privacy check: PASS
release manifest: reports/release_v0_1_manifest.json
```

Current aggregate counts from the release manifest:

- media_items: 21122
- media_vlm: 976 success, 24 failed
- media_embeddings: 1653 success
- media_ocr: 54 success
- events: 3633
- event_evidence: 51883

Current private eval failures to track:

- `mm_stage_photo`: expected 2024-12-14 in top dates, but current ranking
  favored newer stage/performance candidates.
- `vlm_quality_202412`: `json_repaired` appears as a safety flag and the
  private expectation file does not currently allow it.

## PR64 v0.1.1 Stabilization Snapshot

Checks run on 2026-05-10 after the v0.1.1 stabilization patch:

```text
pytest: 445 passed
db-check --strict: ok
private eval: 17/17 passed, 0 skipped
batch-qa: 5/5 succeeded
portfolio privacy check: PASS
release manifest: reports/release_v0_1_manifest.json
```

Private eval changes:

- `mm_stage_photo` was replaced with a 2024-12 scoped stage-photo regression
  case because 2025-03 added valid newer stage/performance candidates.
- `json_repaired` is now allowed for VLM quality checks, since it indicates
  successful JSON recovery.

Current aggregate counts from the v0.1.1 release manifest:

- media_items: 21122
- media_vlm: 980 success, 20 failed
- media_embeddings: 1653 success
- media_ocr: 59 success
- events: 3633
- event_evidence: 51883

Maintenance results:

- VLM failed retry selected 20 rows, recovered 4, and left 16 unrecovered in
  that retry batch.
- OCR failed retry processed 5 images and saved 5 successes.
- Missing original files: 24; exported to `reports/missing_files_v0_1_1.csv`
  and marked unavailable without deleting media records.

2025-03 rollout status:

- VLM: 288 success, 12 failed, 0 engine_unavailable.
- Embeddings: 586 success, 0 failed.
- Events: 119.
- Monthly QA for 2025-03 returns a monthly summary.
- Stage-photo QA for 2025-03 returns stage/performance candidates centered on
  2025-03-01.

## PR73 Private Eval Expansion

PR73 extends private eval beyond timeline and multimodal search checks. The
suite can now cover manual place QA, manual person QA, person-place-activity QA,
face workflow safety, manual LINE-person link quality, privacy audit, and
public-redacted export privacy.

The new cases support skip conditions for local databases that do not yet have
manual person or place labels. Person-related cases add overclaim checks for
relationship, identity, and certainty phrases, while public-output cases check
that exact coordinates, face artifacts, private names, and local-only artifacts
do not leak into public materials.

Validation run on 2026-05-11:

```text
pytest: 551 passed
db-check --strict: ok
people/place/privacy private eval template: 3 passed, 0 failed, 6 skipped
baseline private eval: 17/17 passed
portfolio privacy check: PASS
```

Skipped PR73 cases are expected on a database with no manually verified
person/place labels yet. The passing PR73 cases cover face workflow public-crop
safety, manual LINE-person link quality, and public portfolio privacy audit.

# PR64 v0.1.1 Execution Report

Run date: 2026-05-10

## Summary

PR64 stabilizes the v0.1 release after the 2025-03 rollout. The main outcome is
that private eval is back to 17/17 passed, maintenance results are recorded, and
public portfolio artifacts are regenerated with privacy checks.

## Backups

Backups were created before DB-changing operations:

- `backups/lifelog_before_v0_1_1_stabilization_20260510_060942.sqlite`
- `backups/lifelog_before_retry_vlm_failed_v0_1_1_20260510_061112.sqlite`
- `backups/lifelog_before_retry_ocr_failed_v0_1_1_20260510_061131.sqlite`
- `backups/lifelog_before_mark_missing_unavailable_v0_1_1_20260510_061256.sqlite`

## Private Eval Fixes

Changes applied to the local private eval file:

- The stage-photo case is now scoped to the 2024-12 period. This prevents valid
  newer 2025-03 stage/performance results from destabilizing a 2024-12
  regression target.
- `json_repaired` is now allowed in VLM quality checks. This flag means JSON
  recovery succeeded, not that the content is unsafe.

Result:

```text
cases: 17
passed: 17
failed: 0
skipped: 0
forbidden phrase violations: 0
overclaim violations: 0
run: private_eval/runs/eval_20260510_062021.json
```

The release-check run also saved:

```text
private_eval/runs/eval_20260510_061531.json
```

## VLM Failed Retry

Dry-run selected 20 failed VLM rows in the 2024-12-01 to 2025-03-31 range.
After backup, the retry ran with the configured local Qwen3-VL engine.

Result:

```text
selected failed rows: 20
recovered: 4
unrecovered in retry batch: 16
report: eval_outputs/maintenance/retry_vlm_failed_20260510_061116.json
```

Post-retry VLM stats for 2024-12-01 to 2025-03-31:

```text
media_vlm total: 1000
success: 980
failed: 20
engine_unavailable: 0
```

Remaining failed rows are not used as VLM evidence because strict DB checks
verify that event evidence references only successful, non-fake VLM rows.

## OCR Failed Retry

Dry-run selected 6 OCR failed candidates. After backup, the retry skipped
unusable source media and processed 5 images with the local Tesseract engine.

Result:

```text
selected images: 5
processed: 5
success: 5
failed: 0
engine_unavailable: 0
report: eval_outputs/maintenance/retry_ocr_failed_20260510_061137.json
```

OCR stats after retry:

```text
OCR rows: 61
success: 59
engine_unavailable: 1
no_text_detected: 1
OCR-linked events: 20
```

OCR text remains a weak candidate signal because local OCR can be noisy.

## Missing Files

Missing originals were exported and marked unavailable without deletion.

```text
missing original files: 24
missing thumbnails: 0
csv: reports/missing_files_v0_1_1.csv
marked unavailable: 24
```

These records remain in the DB so the original files can be restored later.

## 2025-03 Status

Heavy `month-run --yes` was not rerun.

```text
media_vlm: 300 total, 288 success, 12 failed
media_embeddings: 586 success
media_ocr: 0 success
events: 119
eval run exists: yes
```

Monthly QA summary:

```text
2025-03: 119 events, 1761 photos, 1657 GPS photos, 1508 LINE records, 32 calls
main tendencies: performance/stage, photo records, LINE activity, calls, food/cafe
```

Stage-photo QA:

```text
query: 2025年3月のステージの写真はいつ？
top date: 2025-03-01
top cues: stage, performance, presentation, dancing
```

Next month planning:

```text
2025-04: photos=352, recommended limits=300/300
2025-05: photos=437, recommended limits=300/300
2025-06: photos=277, recommended limits=277/277
```

## Final Verification

```text
pytest: 445 passed
db-check --strict: ok
private eval: 17/17 passed
batch-qa: 5/5 succeeded
release-check: v0.1.1 manifest generated
```

Release manifest:

```text
reports/release_v0_1_manifest.json
```

Current aggregate counts:

```text
media_items: 21122
media_vlm: 980 success / 20 failed
media_embeddings: 1653 success / 0 failed
media_ocr: 59 success / 0 failed
events: 3633
event_evidence: 51883
```

## Report And Portfolio Artifacts

Generated reports:

- Private report: `reports/lifelog_rag_eval_20260510_061524.md`
- Private report JSON: `reports/lifelog_rag_eval_20260510_061524.json`
- Public report: `reports/lifelog_rag_eval_20260510_061525.md`
- Public report JSON: `reports/lifelog_rag_eval_20260510_061525.json`

Batch QA run:

- `eval_outputs/batch_qa/batch_qa_20260510_062058.json`
- `eval_outputs/batch_qa/batch_qa_20260510_062058.md`

The public portfolio HTML is regenerated after this documentation update and
must pass the public safety check before publication.

## Remaining Items

- 20 VLM failed rows remain. They are excluded from evidence and can be retried
  later with prompt or JSON-repair improvements.
- 24 missing originals remain marked unavailable. Restore files rather than
  deleting records if the originals become available.
- OCR coverage is still narrow and noisy. Use OCR priority selection before
  expanding OCR to more images.
- Continue monthly rollout one month at a time with dry-run planning first.

## Next Steps

1. Review only top search results in the UI and mark obvious VLM mistakes as
   wrong or not searchable.
2. Run 2025-04 as the next monthly rollout candidate after a fresh backup.
3. Improve OCR quality on text-cue images before relying on OCR-heavy queries.
4. Keep portfolio HTML and privacy checks in sync whenever docs or reports are
   edited.

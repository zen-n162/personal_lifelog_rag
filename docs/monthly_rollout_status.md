# Monthly Rollout Status

This log tracks month-by-month rollout of local VLM, multimodal embedding,
event rebuild, reports, and public portfolio refresh. It intentionally avoids
raw LINE text, exact GPS coordinates, real photo paths, media IDs, and private
configuration details.

## Rollout Overview

| Month | Photos | VLM | Embedding | OCR | Events | db-check | Report | HTML | Status | Next action |
| --- | ---: | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| 2024-10 | 130 | 128/130 | 258 rows | priority only | 56 | PASS | yes | yes | complete | retry VLM failed later |
| 2024-11 | 726 | 290/300 | 590 rows | priority only | 112 | PASS | yes | yes | complete | retry VLM failed later; 426 photos remain outside staged limit |
| 2024-12 | 450 | 296/300 | 509 rows | partial | 116 | PASS | yes | yes | complete | optional incremental completion: VLM 154, embedding 190 |
| 2025-01 | done | done | done | partial | done | PASS | yes | yes | complete | - |
| 2025-02 | done | done | done | partial | done | PASS | yes | yes | complete | - |
| 2025-03 | done | done | done | partial | 119 | PASS | yes | yes | complete | retry VLM failed later |
| 2025-04 | 352 | 293/300 | 593 rows | priority only | 111 | PASS | yes | yes | complete | retry VLM failed later |
| 2025-05 | 437 | 297/300 | 597 rows | priority only | 124 | PASS | yes | yes | complete | retry VLM failed later |
| 2025-06 | 277 | 271/277 | 548 rows | priority only | 121 | PASS | yes | yes | complete | retry VLM failed later |
| 2025-07 | 323 | 297/300 | 597 rows | priority only | 126 | PASS | yes | yes | complete | retry VLM failed later |
| 2025-08 | 599 | 298/300 | 598 rows | priority only | 134 | PASS | yes | yes | complete | retry VLM failed later |
| 2025-09 | 837 | 295/300 | 595 rows | priority only | 125 | PASS | yes | yes | complete | retry VLM failed later |
| 2025-10 | 1244 | 294/300 | 594 rows | priority only | 118 | PASS | yes | yes | complete | retry VLM failed later |
| 2025-11 | 137 | 133/137 | 270 rows | priority only | 116 | PASS | yes | yes | complete | retry VLM failed later |
| 2025-12 | 329 | 294/300 | 594 rows | priority only | 108 | PASS | yes | yes | complete | retry VLM failed later; 29 photos remain outside staged limit |
| 2026-01 | 118 | 117/118 | 235 rows | priority only | 99 | PASS | yes | yes | complete | retry VLM failed later |
| 2026-02 | 83 | 79/83 | 162 rows | priority only | 106 | PASS | yes | yes | complete | retry VLM failed later |
| 2026-03 | 603 | 296/300 | 596 rows | priority only | 125 | PASS | yes | yes | complete | retry VLM failed later; 303 photos remain outside staged limit |
| 2026-04 | 131 | 129/131 | 260 rows | priority only | 101 | PASS | yes | yes | complete | retry VLM failed later |
| 2026-05 | 4985 | 296/300 | 596 rows | priority only | 26 | PASS | yes | yes | complete | retry VLM failed later; 4685 photos remain outside staged limit |
| 2026-06 | 0 | 0 | 0 | 0 | 0 | - | - | - | no source data | dry-run confirmed; no write run needed unless data is added |

## Current Rollouts

### PR64 Location/Place DB Baseline

- Run date: 2026-05-11
- Scope: 2024-12-01 to 2025-03-31
- Policy: exact GPS is stored only in the private SQLite DB. Public reports and
  portfolio HTML must use reviewed place labels, broad categories, or generic
  private-place placeholders.

Execution summary:

| Component | Result |
| --- | --- |
| Backup before write | `backups/lifelog_before_pr64_location_places_20260511_043149.sqlite` |
| build-location-points dry-run | 4,196 media scanned, 3,952 GPS media, 3,952 would create |
| build-location-points write | 3,952 location_points created |
| cluster-places dry-run | 3,952 points scanned, 121 cluster candidates at 100m / min 3 |
| cluster write | Not executed in this pass; clusters remain review candidates |
| place labels | Not created in this pass |
| db-check --strict | PASS |

Notes:

- `place-stats` now reports DB-backed location table counts alongside existing
  event `location_name` statistics.
- Exact cluster centroids are not printed in CLI summaries.
- Next action is manual review of cluster candidates, then optional place label
  creation and `assign-places --dry-run`.

### 2024-11

- Run date: 2026-05-11
- Scope: 2024-11-01 to 2024-11-30
- Policy: one-month write rollout only. 2024-12 was checked with plan only; no
  2024-12 write execution was performed in this session.
- GPU: CUDA was visible through both `nvidia-smi` and PyTorch before the write
  run. The heavy VLM and embedding steps ran with GPU available.

Plan summary:

| Item | Count |
| --- | ---: |
| Photos | 726 |
| GPS photos | 670 |
| LINE messages | 975 |
| Call events | 31 |
| Existing events before rebuild | 107 |
| Existing VLM rows before rollout | 0 |
| Existing embedding rows before rollout | 0 |
| Existing OCR rows before rollout | 0 |
| Rollout VLM limit | 300 |
| Rollout embedding limit | 300 |

Execution summary:

| Component | Result |
| --- | --- |
| Backup before month-run | `backups/lifelog_before_month_run_202411_20260511_033330.sqlite` |
| Internal month-run backup | `backups/lifelog_before_month_rollout_2024_11_20260511_033334.sqlite` |
| Event rebuild backup | `backups/lifelog_before_event_rebuild_20241101_20260511_041357.sqlite` |
| VLM analysis | 290 success, 10 failed, 0 engine_unavailable |
| Image embeddings | 300 success, 0 failed |
| Combined text embeddings | 290 success, 0 failed |
| OCR | Not run in this rollout; priority candidates were listed only |
| Event rebuild | Completed |
| Event count after rebuild | 112 |
| Event count delta | +5 |
| VLM evidence added during rebuild | +290 |
| VLM-only high confidence events | 0 |
| db-check --strict | PASS |

QA smoke checks:

| Query | Result |
| --- | --- |
| 2024年11月は何していた？ | monthly_summary returned 112 events, 726 photos, 670 GPS photos, 975 LINE records, 31 call logs |
| 2024年11月のご飯を食べた写真はいつ？ | batch-qa success; primary candidate dates were 2024-11-06, 2024-11-01, and 2024-11-02 |
| 2024年11月のステージの写真はいつ？ | batch-qa success; primary candidate dates were 2024-11-03 and 2024-11-11 |

Generated reports:

| Mode | Markdown | JSON |
| --- | --- | --- |
| Private | `reports/lifelog_rag_eval_20260511_041719.md` | `reports/lifelog_rag_eval_20260511_041719.json` |
| Public | `reports/lifelog_rag_eval_20260511_041730.md` | `reports/lifelog_rag_eval_20260511_041730.json` |

Portfolio:

| Artifact | Result |
| --- | --- |
| HTML | `reports/portfolio_public.html` |
| Build JSON | `reports/portfolio_public_build.json` |
| Privacy check | PASS |

Notes:

- OCR was intentionally not executed. `ocr-priority` returned 50 text-like
  candidates, mostly package labels, chat or app screenshots, tickets, QR or
  poster images, menu-like images, storefront signs, venue signs, score screens,
  and whiteboard or document images.
- The month contains 726 photos, so this rollout intentionally processed only
  the staged 300-image VLM/embedding batch. 426 photos remain outside this
  rollout batch and can be handled later if needed.
- Ten VLM rows remain failed and should be retried later with the normal retry
  workflow.
- Monthly private eval was skipped because no 2024-11 private eval file exists.
- Public-facing docs and reports must continue to omit raw private data.

Next recommended month/action:

- 2024-12 incremental completion, if desired. The month is already usable and
  marked complete, but `month-plan` still reports remaining analysis work.

2024-12 plan summary:

| Item | Count |
| --- | ---: |
| Photos | 450 |
| GPS photos | 448 |
| LINE messages | 1222 |
| Call events | 33 |
| Existing events | 116 |
| Existing VLM rows | 300 total, 296 success, 4 failed |
| Existing embedding rows | 509 success |
| Existing OCR rows | 59 success, 1 engine_unavailable, 1 no_text_detected |
| Recommended VLM limit | 154 |
| Recommended embedding limit | 190 |

Recommended next command sequence:

```bash
python -m personal_lifelog_rag.app.cli month-plan --month 2024-12
python -m personal_lifelog_rag.app.cli month-run \
  --month 2024-12 \
  --vlm-limit 154 \
  --embedding-limit 190 \
  --config <local model runtime config> \
  --save-report \
  --dry-run
```

## PR75 Face Embedding/Cluster Maintenance

Date: 2026-05-11

Scope:

- Face detection source: YuNet detections
- Date range: 2024-10-01 to 2026-05-31
- Embedding engine: OpenCV SFace
- Clustering scope: `yunet_202410_202605`

Results:

| Item | Result |
| --- | ---: |
| YuNet success detections | 13,914 |
| Old selected embeddings replaced | 48 |
| New embedding success | 13,914 |
| Embedding failed / skipped / unavailable | 0 / 0 / 0 |
| Face clusters written | 130 |
| Cluster members written | 623 |
| Singleton/outlier faces | 13,291 |
| Largest cluster size | 22 |
| db-check --strict | PASS |
| privacy-audit --public | PASS |

Artifacts:

- Embedding report: `reports/face_embed_20260511_133257.json`
- Cluster report: `reports/face_cluster_20260511_133737.json`
- Detailed note: `docs/pr75_yunet_face_embedding_cluster_full_range.md`

Notes:

- Only 4,149 detections had persisted crop paths, so embedding generation
  recreated temporary crops from original local images and bboxes where needed.
- All generated clusters remain unreviewed and are not used by normal QA/search
  or public reports.
- Next recommended action is Face Review of the largest/highest-quality
  clusters before manual person labeling.

### 2024-10

- Run date: 2026-05-11
- Scope: 2024-10-01 to 2024-10-31
- Policy: one-month write rollout only. 2024-11 was checked with dry-run only;
  no 2024-11 write execution was performed in this session.
- GPU: CUDA was visible through both `nvidia-smi` and PyTorch before the write
  run. The heavy VLM and embedding steps ran with GPU available.

Plan summary:

| Item | Count |
| --- | ---: |
| Photos | 130 |
| GPS photos | 103 |
| LINE messages | 338 |
| Call events | 14 |
| Existing events before rebuild | 50 |
| Existing VLM rows before rollout | 0 |
| Existing embedding rows before rollout | 0 |
| Existing OCR rows before rollout | 0 |
| Rollout VLM limit | 130 |
| Rollout embedding limit | 130 |

Execution summary:

| Component | Result |
| --- | --- |
| Backup before month-run | `backups/lifelog_before_month_run_202410_20260511_031038.sqlite` |
| Internal month-run backup | `backups/lifelog_before_month_rollout_2024_10_20260511_031042.sqlite` |
| Event rebuild backup | `backups/lifelog_before_event_rebuild_20241001_20260511_032826.sqlite` |
| VLM analysis | 128 success, 2 failed, 0 engine_unavailable |
| Image embeddings | 130 success, 0 failed |
| Combined text embeddings | 128 success, 0 failed |
| OCR | Not run in this rollout; priority candidates were listed only |
| Event rebuild | Completed |
| Event count after rebuild | 56 |
| Event count delta | +6 |
| VLM evidence added during rebuild | +128 |
| VLM-only high confidence events | 0 |
| db-check --strict | PASS |

QA smoke checks:

| Query | Result |
| --- | --- |
| 2024年10月は何していた？ | monthly_summary returned 56 events, 130 photos, 103 GPS photos, 338 LINE records, 14 call logs |
| 2024年10月のご飯を食べた写真はいつ？ | batch-qa success; primary candidate dates were 2024-10-30, 2024-10-29, and 2024-10-13 |
| 2024年10月のステージの写真はいつ？ | batch-qa success; primary candidate dates were 2024-10-30 and 2024-10-12 |

Generated reports:

| Mode | Markdown | JSON |
| --- | --- | --- |
| Private | `reports/lifelog_rag_eval_20260511_033135.md` | `reports/lifelog_rag_eval_20260511_033135.json` |
| Public | `reports/lifelog_rag_eval_20260511_033145.md` | `reports/lifelog_rag_eval_20260511_033145.json` |

Portfolio:

| Artifact | Result |
| --- | --- |
| HTML | `reports/portfolio_public.html` |
| Build JSON | `reports/portfolio_public_build.json` |
| Privacy check | PASS |

Notes:

- OCR was intentionally not executed. `ocr-priority` returned 50 text-like
  candidates, mostly menu-like images, receipts, screenshots, posters,
  documents, whiteboards, signs, labels, and screen text.
- The month contains 130 photos, so this rollout processed the full planned
  monthly image set under the current safety limit.
- Two VLM rows remain failed and should be retried later with the normal retry
  workflow.
- Monthly private eval was skipped because no 2024-10 private eval file exists.
- Public-facing docs and reports must continue to omit raw private data.

Next recommended month:

- 2024-11

2024-11 plan summary:

| Item | Count |
| --- | ---: |
| Photos | 726 |
| GPS photos | 670 |
| LINE messages | 975 |
| Call events | 31 |
| Existing events | 107 |
| Existing VLM rows | 0 |
| Existing embedding rows | 0 |
| Existing OCR rows | 0 |
| Recommended VLM limit | 300 |
| Recommended embedding limit | 300 |

2024-11 dry-run result:

- No DB changes were made.
- Planned steps: backup, VLM analysis, image embeddings, combined text
  embeddings, event rebuild, strict DB check, public report generation.
- Monthly private eval was skipped because no 2024-11 private eval file exists.
- The month has 726 photos, so the next write run should keep the staged
  300-item limit unless intentionally lowered.

Recommended next command sequence:

```bash
python -m personal_lifelog_rag.app.cli month-plan --month 2024-11
python -m personal_lifelog_rag.app.cli month-run \
  --month 2024-11 \
  --vlm-limit 300 \
  --embedding-limit 300 \
  --config <local model runtime config> \
  --save-report \
  --dry-run
```

### 2026-05

- Run date: 2026-05-11
- Scope: 2026-05-01 to 2026-05-31
- Policy: one-month write rollout only. No 2026-06 write execution was
  performed in this session.
- GPU: CUDA was visible through both `nvidia-smi` and PyTorch before the write
  run. The heavy VLM and embedding steps ran with GPU available.

Plan summary:

| Item | Count |
| --- | ---: |
| Photos | 4985 |
| GPS photos | 2 |
| LINE messages | 269 |
| Call events | 4 |
| Existing events before rebuild | 26 |
| Existing VLM rows before rollout | 0 |
| Existing embedding rows before rollout | 0 |
| Existing OCR rows before rollout | 0 |
| Rollout VLM limit | 300 |
| Rollout embedding limit | 300 |

Execution summary:

| Component | Result |
| --- | --- |
| Backup before month-run | `backups/lifelog_before_month_run_202605_20260511_022413.sqlite` |
| Internal month-run backup | `backups/lifelog_before_month_rollout_2026_05_20260511_022419.sqlite` |
| Event rebuild backup | `backups/lifelog_before_event_rebuild_20260501_20260511_030411.sqlite` |
| VLM analysis | 296 success, 4 failed, 0 engine_unavailable |
| Image embeddings | 300 success, 0 failed |
| Combined text embeddings | 296 success, 0 failed |
| OCR | Not run in this rollout; priority candidates were listed only |
| Event rebuild | Completed |
| Event count after rebuild | 26 |
| Event count delta | +0 |
| VLM evidence added during rebuild | +296 |
| VLM-only high confidence events | 0 |
| db-check --strict | PASS |

QA smoke checks:

| Query | Result |
| --- | --- |
| 2026年5月は何していた？ | monthly_summary returned 26 events, 4985 photos, 2 GPS photos, 269 LINE records, 4 call logs |
| 2026年5月のご飯を食べた写真はいつ？ | batch-qa success; primary candidate dates were 2026-05-09 and 2026-05-01 |
| 2026年5月のステージの写真はいつ？ | batch-qa success; primary candidate date was 2026-05-09 |

Generated reports:

| Mode | Markdown | JSON |
| --- | --- | --- |
| Private | `reports/lifelog_rag_eval_20260511_030820.md` | `reports/lifelog_rag_eval_20260511_030820.json` |
| Public | `reports/lifelog_rag_eval_20260511_030829.md` | `reports/lifelog_rag_eval_20260511_030829.json` |

Portfolio:

| Artifact | Result |
| --- | --- |
| HTML | `reports/portfolio_public.html` |
| Build JSON | `reports/portfolio_public_build.json` |
| Privacy check | PASS |

Notes:

- OCR was intentionally not executed. `ocr-priority` returned 50 text-like
  candidates, mostly ticket or schedule screenshots, real-estate screens,
  documents, code/editor screens, handwritten notes, posters, signs, package
  labels, menu-like images, and stage screens with visible text.
- The month contains 4985 photos, so this rollout intentionally processed only
  the staged 300-image VLM/embedding batch. 4685 photos remain outside this
  rollout batch and can be handled later if needed.
- Four VLM rows remain failed and should be retried later with the normal retry
  workflow.
- Monthly private eval was skipped because no 2026-05 private eval file exists.
- Public-facing docs and reports must continue to omit raw private data.

Next recommended month:

- 2026-06

2026-06 plan summary:

| Item | Count |
| --- | ---: |
| Photos | 0 |
| GPS photos | 0 |
| LINE messages | 0 |
| Call events | 0 |
| Existing events | 0 |
| Existing VLM rows | 0 |
| Existing embedding rows | 0 |
| Existing OCR rows | 0 |
| Recommended VLM limit | 0 |
| Recommended embedding limit | 0 |

2026-06 dry-run result:

- No DB changes were made.
- Planned limits were 0 VLM items and 0 embedding items because the DB currently
  has no source records for 2026-06.
- No 2026-06 `--yes` execution was performed.

Recommended next action:

- Recheck `month-plan --month 2026-06` after new source data is imported. If it
  remains empty, skip write execution and move to the next month with real input
  data.

### 2026-04

- Run date: 2026-05-11
- Scope: 2026-04-01 to 2026-04-30
- Policy: one-month write rollout only. No 2026-05 write execution was
  performed in this session.
- GPU: CUDA was visible through both `nvidia-smi` and PyTorch before the write
  run. The heavy VLM and embedding steps ran with GPU available.

Plan summary:

| Item | Count |
| --- | ---: |
| Photos | 131 |
| GPS photos | 95 |
| LINE messages | 2465 |
| Call events | 45 |
| Existing events before rebuild | 101 |
| Existing VLM rows before rollout | 0 |
| Existing embedding rows before rollout | 0 |
| Existing OCR rows before rollout | 0 |
| Rollout VLM limit | 131 |
| Rollout embedding limit | 131 |

Execution summary:

| Component | Result |
| --- | --- |
| Backup before month-run | `backups/lifelog_before_month_run_202604_20260511_020101.sqlite` |
| Internal month-run backup | `backups/lifelog_before_month_rollout_2026_04_20260511_020105.sqlite` |
| Event rebuild backup | `backups/lifelog_before_event_rebuild_20260401_20260511_021941.sqlite` |
| VLM analysis | 129 success, 2 failed, 0 engine_unavailable |
| Image embeddings | 131 success, 0 failed |
| Combined text embeddings | 129 success, 0 failed |
| OCR | Not run in this rollout; priority candidates were listed only |
| Event rebuild | Completed |
| Event count after rebuild | 101 |
| Event count delta | +0 |
| VLM evidence added during rebuild | +129 |
| VLM-only high confidence events | 0 |
| db-check --strict | PASS |

QA smoke checks:

| Query | Result |
| --- | --- |
| 2026年4月は何していた？ | monthly_summary returned 101 events, 131 photos, 95 GPS photos, 2465 LINE records, 45 call logs |
| 2026年4月のご飯を食べた写真はいつ？ | batch-qa success; primary candidate dates were 2026-04-30, 2026-04-08, and 2026-04-23 |
| 2026年4月のステージの写真はいつ？ | batch-qa success; primary candidate dates were 2026-04-07, 2026-04-30, 2026-04-23, and 2026-04-16 |

Generated reports:

| Mode | Markdown | JSON |
| --- | --- | --- |
| Private | `reports/lifelog_rag_eval_20260511_022159.md` | `reports/lifelog_rag_eval_20260511_022159.json` |
| Public | `reports/lifelog_rag_eval_20260511_022200.md` | `reports/lifelog_rag_eval_20260511_022200.json` |

Portfolio:

| Artifact | Result |
| --- | --- |
| HTML | `reports/portfolio_public.html` |
| Build JSON | `reports/portfolio_public_build.json` |
| Privacy check | PASS |

Notes:

- OCR was intentionally not executed. `ocr-priority` returned 50 text-like
  candidates, including programming screens, login/QR documents, package text,
  park signs, handwritten notes, menu-like images, album art, and bowling venue
  signs.
- The month contains 131 photos, so this rollout processed the full planned
  monthly image set under the current safety limit.
- Two VLM rows remain failed and should be retried later with the normal retry
  workflow.
- Monthly private eval was skipped because no 2026-04 private eval file exists.
- Public-facing docs and reports must continue to omit raw private data.

Next recommended month:

- 2026-05

2026-05 plan summary:

| Item | Count |
| --- | ---: |
| Photos | 4985 |
| GPS photos | 2 |
| LINE messages | 269 |
| Call events | 4 |
| Existing events | 26 |
| Existing VLM rows | 0 |
| Existing embedding rows | 0 |
| Existing OCR rows | 0 |
| Recommended VLM limit | 300 |
| Recommended embedding limit | 300 |

2026-05 dry-run result:

- No DB changes were made.
- Planned steps: backup, VLM analysis, image embeddings, combined text
  embeddings, event rebuild, strict DB check, public report generation.
- Monthly private eval was skipped because no 2026-05 private eval file exists.
- The month has 4985 photos, so the next write run should keep the staged
  300-item limit unless intentionally lowered.

Recommended next command sequence:

```bash
python -m personal_lifelog_rag.app.cli month-plan --month 2026-05
python -m personal_lifelog_rag.app.cli month-run \
  --month 2026-05 \
  --vlm-limit 300 \
  --embedding-limit 300 \
  --config <local model runtime config> \
  --save-report \
  --dry-run
```

### 2026-03

- Run date: 2026-05-11
- Scope: 2026-03-01 to 2026-03-31
- Policy: one-month write rollout only. No 2026-04 write execution was
  performed in this session.
- GPU: CUDA was visible through `nvidia-smi` before the write run. The heavy
  VLM and embedding steps ran with GPU available.

Plan summary:

| Item | Count |
| --- | ---: |
| Photos | 603 |
| GPS photos | 410 |
| LINE messages | 2526 |
| Call events | 37 |
| Existing events before rebuild | 123 |
| Existing VLM rows before rollout | 0 |
| Existing embedding rows before rollout | 0 |
| Existing OCR rows before rollout | 0 |
| Rollout VLM limit | 300 |
| Rollout embedding limit | 300 |

Execution summary:

| Component | Result |
| --- | --- |
| Backup before month-run | `backups/lifelog_before_month_run_202603_20260511_011509.sqlite` |
| Internal month-run backup | `backups/lifelog_before_month_rollout_2026_03_20260511_011513.sqlite` |
| Event rebuild backup | `backups/lifelog_before_event_rebuild_20260301_20260511_015524.sqlite` |
| VLM analysis | 296 success, 4 failed, 0 engine_unavailable |
| Image embeddings | 300 success, 0 failed |
| Combined text embeddings | 296 success, 0 failed |
| OCR | Not run in this rollout; priority candidates were listed only |
| Event rebuild | Completed |
| Event count after rebuild | 125 |
| Event count delta | +2 |
| VLM evidence added during rebuild | +296 |
| VLM-only high confidence events | 0 |
| db-check --strict | PASS |

QA smoke checks:

| Query | Result |
| --- | --- |
| 2026年3月は何していた？ | monthly_summary returned 125 events, 603 photos, 410 GPS photos, 2526 LINE records, 37 call logs |
| 2026年3月のご飯を食べた写真はいつ？ | batch-qa success; primary candidate dates were 2026-03-18, 2026-03-02, and 2026-03-03 |
| 2026年3月のステージの写真はいつ？ | batch-qa success; primary candidate dates were 2026-03-03 and 2026-03-18 |

Generated reports:

| Mode | Markdown | JSON |
| --- | --- | --- |
| Private | `reports/lifelog_rag_eval_20260511_015853.md` | `reports/lifelog_rag_eval_20260511_015853.json` |
| Public | `reports/lifelog_rag_eval_20260511_015852.md` | `reports/lifelog_rag_eval_20260511_015852.json` |

Portfolio:

| Artifact | Result |
| --- | --- |
| HTML | `reports/portfolio_public.html` |
| Build JSON | `reports/portfolio_public_build.json` |
| Privacy check | PASS |

Notes:

- OCR was intentionally not executed. `ocr-priority` returned 50 text-like
  candidates, including tickets, menus, screen text, signs, documents, receipts,
  university/event signage, and administrative forms.
- The month contains 603 photos. This rollout processed the current staged
  300-item limit; 303 photos remain outside this staged pass.
- Four VLM rows remain failed and should be retried later with the normal retry
  workflow.
- Monthly private eval was skipped because no 2026-03 private eval file exists.
- Public-facing docs and reports must continue to omit raw private data.

Next recommended month:

- 2026-04

2026-04 plan summary:

| Item | Count |
| --- | ---: |
| Photos | 131 |
| GPS photos | 95 |
| LINE messages | 2465 |
| Call events | 45 |
| Existing events | 101 |
| Existing VLM rows | 0 |
| Existing embedding rows | 0 |
| Existing OCR rows | 0 |
| Recommended VLM limit | 131 |
| Recommended embedding limit | 131 |

2026-04 dry-run result:

- No DB changes were made.
- Planned steps: backup, VLM analysis, image embeddings, combined text
  embeddings, event rebuild, strict DB check, public report generation.
- Monthly private eval was skipped because no 2026-04 private eval file exists.
- The month has 131 photos, so the next write run can process the full planned
  monthly image set under the current safety limit.

Recommended next command sequence:

```bash
python -m personal_lifelog_rag.app.cli month-plan --month 2026-04
python -m personal_lifelog_rag.app.cli month-run \
  --month 2026-04 \
  --vlm-limit 131 \
  --embedding-limit 131 \
  --config <local model runtime config> \
  --save-report \
  --dry-run
```

### 2026-02

- Run date: 2026-05-11
- Scope: 2026-02-01 to 2026-02-28
- Policy: one-month write rollout only. No 2026-03 write execution was
  performed in this session.
- GPU: CUDA was visible through `nvidia-smi` before the write run. The heavy
  VLM and embedding steps ran with GPU available.

Plan summary:

| Item | Count |
| --- | ---: |
| Photos | 83 |
| GPS photos | 78 |
| LINE messages | 3561 |
| Call events | 18 |
| Existing events before rebuild | 106 |
| Existing VLM rows before rollout | 0 |
| Existing embedding rows before rollout | 0 |
| Existing OCR rows before rollout | 0 |
| Rollout VLM limit | 83 |
| Rollout embedding limit | 83 |

Execution summary:

| Component | Result |
| --- | --- |
| Backup before month-run | `backups/lifelog_before_month_run_202602_20260511_005754.sqlite` |
| Internal month-run backup | `backups/lifelog_before_month_rollout_2026_02_20260511_005758.sqlite` |
| Event rebuild backup | `backups/lifelog_before_event_rebuild_20260201_20260511_011029.sqlite` |
| VLM analysis | 79 success, 4 failed, 0 engine_unavailable |
| Image embeddings | 83 success, 0 failed |
| Combined text embeddings | 79 success, 0 failed |
| OCR | Not run in this rollout; priority candidates were listed only |
| Event rebuild | Completed |
| Event count after rebuild | 106 |
| Event count delta | +0 |
| VLM evidence added during rebuild | +79 |
| VLM-only high confidence events | 0 |
| db-check --strict | PASS |

QA smoke checks:

| Query | Result |
| --- | --- |
| 2026年2月は何していた？ | monthly_summary returned 106 events, 83 photos, 78 GPS photos, 3561 LINE records, 18 call logs |
| 2026年2月のご飯を食べた写真はいつ？ | batch-qa success; primary candidate dates were 2026-02-21 and 2026-02-17 |
| 2026年2月のステージの写真はいつ？ | batch-qa success; primary candidate dates were 2026-02-17, 2026-02-21, and 2026-02-28 |

Generated reports:

| Mode | Markdown | JSON |
| --- | --- | --- |
| Private | `reports/lifelog_rag_eval_20260511_011318.md` | `reports/lifelog_rag_eval_20260511_011318.json` |
| Public | `reports/lifelog_rag_eval_20260511_011319.md` | `reports/lifelog_rag_eval_20260511_011319.json` |

Portfolio:

| Artifact | Result |
| --- | --- |
| HTML | `reports/portfolio_public.html` |
| Build JSON | `reports/portfolio_public_build.json` |
| Privacy check | PASS |

Notes:

- OCR was intentionally not executed. `ocr-priority` returned 50 text-like
  candidates, including ticket booking screens, administrative forms, menus,
  receipts, route/map screens, resort signs, bowling venue banners, and package
  text.
- The month contains 83 photos, so this rollout processed the full planned
  monthly image set under the current safety limit.
- Four VLM rows remain failed and should be retried later with the normal retry
  workflow.
- Monthly private eval was skipped because no 2026-02 private eval file exists.
- Public-facing docs and reports must continue to omit raw private data.

Next recommended month:

- 2026-03

2026-03 plan summary:

| Item | Count |
| --- | ---: |
| Photos | 603 |
| GPS photos | 410 |
| LINE messages | 2526 |
| Call events | 37 |
| Existing events | 123 |
| Existing VLM rows | 0 |
| Existing embedding rows | 0 |
| Existing OCR rows | 0 |
| Recommended VLM limit | 300 |
| Recommended embedding limit | 300 |

2026-03 dry-run result:

- No DB changes were made.
- Planned steps: backup, VLM analysis, image embeddings, combined text
  embeddings, event rebuild, strict DB check, public report generation.
- Monthly private eval was skipped because no 2026-03 private eval file exists.
- The month has 603 photos, so the next write run should use the current staged
  300-item limit unless intentionally lowered.

Recommended next command sequence:

```bash
python -m personal_lifelog_rag.app.cli month-plan --month 2026-03
python -m personal_lifelog_rag.app.cli month-run \
  --month 2026-03 \
  --vlm-limit 300 \
  --embedding-limit 300 \
  --config <local model runtime config> \
  --save-report \
  --dry-run
```

### 2026-01

- Run date: 2026-05-10
- Scope: 2026-01-01 to 2026-01-31
- Policy: one-month write rollout only. No 2026-02 write execution was
  performed in this session.
- GPU: CUDA was visible through both `nvidia-smi` and PyTorch before the write
  run. The heavy VLM and embedding steps ran with GPU available.

Plan summary:

| Item | Count |
| --- | ---: |
| Photos | 118 |
| GPS photos | 73 |
| LINE messages | 4562 |
| Call events | 20 |
| Existing events before rebuild | 98 |
| Existing VLM rows before rollout | 0 |
| Existing embedding rows before rollout | 0 |
| Existing OCR rows before rollout | 0 |
| Rollout VLM limit | 118 |
| Rollout embedding limit | 118 |

Execution summary:

| Component | Result |
| --- | --- |
| Backup before month-run | `backups/lifelog_before_month_run_202601_20260510_215238.sqlite` |
| Internal month-run backup | `backups/lifelog_before_month_rollout_2026_01_20260510_215242.sqlite` |
| Event rebuild backup | `backups/lifelog_before_event_rebuild_20260101_20260510_220921.sqlite` |
| VLM analysis | 117 success, 1 failed, 0 engine_unavailable |
| Image embeddings | 118 success, 0 failed |
| Combined text embeddings | 117 success, 0 failed |
| OCR | Not run in this rollout; priority candidates were listed only |
| Event rebuild | Completed |
| Event count after rebuild | 99 |
| Event count delta | +1 |
| VLM evidence added during rebuild | +117 |
| VLM-only high confidence events | 0 |
| db-check --strict | PASS |

QA smoke checks:

| Query | Result |
| --- | --- |
| 2026年1月は何していた？ | monthly_summary returned 99 events, 118 photos, 73 GPS photos, 4562 LINE records, 20 call logs |
| 2026年1月のご飯を食べた写真はいつ？ | batch-qa success; primary candidate dates were 2026-01-18, 2026-01-25, and 2026-01-22 |
| 2026年1月のステージの写真はいつ？ | batch-qa success; primary candidate dates were 2026-01-12, 2026-01-08, and 2026-01-18 |

Generated reports:

| Mode | Markdown | JSON |
| --- | --- | --- |
| Private | `reports/lifelog_rag_eval_20260510_221206.md` | `reports/lifelog_rag_eval_20260510_221206.json` |
| Public | `reports/lifelog_rag_eval_20260510_221205.md` | `reports/lifelog_rag_eval_20260510_221205.json` |

Portfolio:

| Artifact | Result |
| --- | --- |
| HTML | `reports/portfolio_public.html` |
| Build JSON | `reports/portfolio_public_build.json` |
| Privacy check | PASS |

Notes:

- OCR was intentionally not executed. `ocr-priority` returned 50 text-like
  candidates, including a receipt, route/navigation screens, technical
  documents, registration/login screens, menus, signs, labels, product text,
  and shopping mall map text.
- The month contains 118 photos, so this rollout processed the full planned
  monthly image set under the current safety limit.
- One VLM row remains failed and should be retried later with the normal retry
  workflow.
- Monthly private eval was skipped because no 2026-01 private eval file exists.
- Public-facing docs and reports must continue to omit raw private data.

Next recommended month:

- 2026-02

2026-02 plan summary:

| Item | Count |
| --- | ---: |
| Photos | 70 |
| GPS photos | 65 |
| LINE messages | 3561 |
| Call events | 18 |
| Existing events | 106 |
| Existing VLM rows | 0 |
| Existing embedding rows | 0 |
| Existing OCR rows | 0 |
| Recommended VLM limit | 70 |
| Recommended embedding limit | 70 |

2026-02 dry-run result:

- No DB changes were made.
- Planned steps: backup, VLM analysis, image embeddings, combined text
  embeddings, event rebuild, strict DB check, public report generation.
- Monthly private eval was skipped because no 2026-02 private eval file exists.
- The month has 70 photos, so the next write run can process the full monthly
  image set under the current staged limit.

Recommended next command sequence:

```bash
python -m personal_lifelog_rag.app.cli month-plan --month 2026-02
python -m personal_lifelog_rag.app.cli month-run \
  --month 2026-02 \
  --vlm-limit 70 \
  --embedding-limit 70 \
  --config <local model runtime config> \
  --save-report \
  --dry-run
```

### 2025-12

- Run date: 2026-05-10
- Scope: 2025-12-01 to 2025-12-31
- Policy: one-month write rollout only. No 2026-01, 2024-11, or 2024-10
  write execution was performed in this session.
- GPU: CUDA was visible through both `nvidia-smi` and PyTorch before the write
  run. The heavy VLM and embedding steps ran with GPU available.

Plan summary:

| Item | Count |
| --- | ---: |
| Photos | 329 |
| GPS photos | 309 |
| LINE messages | 4620 |
| Call events | 13 |
| Existing events before rebuild | 108 |
| Existing VLM rows before rollout | 0 |
| Existing embedding rows before rollout | 0 |
| Existing OCR rows before rollout | 0 |
| Rollout VLM limit | 300 |
| Rollout embedding limit | 300 |

Execution summary:

| Component | Result |
| --- | --- |
| Backup before month-run | `backups/lifelog_before_month_run_202512_20260510_210048.sqlite` |
| Internal month-run backup | `backups/lifelog_before_month_rollout_2025_12_20260510_210053.sqlite` |
| Event rebuild backup | `backups/lifelog_before_event_rebuild_20251201_20260510_214120.sqlite` |
| VLM analysis | 294 success, 6 failed, 0 engine_unavailable |
| Image embeddings | 300 success, 0 failed |
| Combined text embeddings | 294 success, 0 failed |
| OCR | Not run in this rollout; priority candidates were listed only |
| Event rebuild | Completed |
| Event count after rebuild | 108 |
| Event count delta | +0 |
| VLM evidence added during rebuild | +294 |
| VLM-only high confidence events | 0 |
| db-check --strict | PASS |

QA smoke checks:

| Query | Result |
| --- | --- |
| 2025年12月は何していた？ | monthly_summary returned 108 events, 329 photos, 309 GPS photos, 4620 LINE records, 13 call logs |
| 2025年12月のご飯を食べた写真はいつ？ | batch-qa success; primary candidate dates were 2025-12-12, 2025-12-20, and 2025-12-24 |
| 2025年12月のステージの写真はいつ？ | batch-qa success; primary candidate dates were 2025-12-21, 2025-12-20, and 2025-12-24 |

Generated reports:

| Mode | Markdown | JSON |
| --- | --- | --- |
| Private | `reports/lifelog_rag_eval_20260510_214536.md` | `reports/lifelog_rag_eval_20260510_214536.json` |
| Public | `reports/lifelog_rag_eval_20260510_214544.md` | `reports/lifelog_rag_eval_20260510_214544.json` |

Portfolio:

| Artifact | Result |
| --- | --- |
| HTML | `reports/portfolio_public.html` |
| Build JSON | `reports/portfolio_public_build.json` |
| Privacy check | PASS |

Notes:

- OCR was intentionally not executed. `ocr-priority` returned 50 text-like
  candidates, including gift tags, coupon/menu screens, documents, signs,
  travel/property screenshots, posters, packages, and route screens.
- The month contains 329 photos, so this rollout processed 300 selected images
  under the current monthly safety limit. The remaining 29 photos can be
  handled by a later targeted continuation pass if needed.
- Six VLM rows remain failed and should be retried later with the normal retry
  workflow.
- Monthly private eval was skipped because no 2025-12 private eval file exists.
- Public-facing docs and reports must continue to omit raw private data.

Forward next recommended month:

- 2026-01

2026-01 plan summary:

| Item | Count |
| --- | ---: |
| Photos | 118 |
| GPS photos | 73 |
| LINE messages | 4562 |
| Call events | 20 |
| Existing events | 98 |
| Existing VLM rows | 0 |
| Existing embedding rows | 0 |
| Existing OCR rows | 0 |
| Recommended VLM limit | 118 |
| Recommended embedding limit | 118 |

2026-01 dry-run result:

- No DB changes were made.
- Planned steps: backup, VLM analysis, image embeddings, combined text
  embeddings, event rebuild, strict DB check, public report generation.
- Monthly private eval was skipped because no 2026-01 private eval file exists.
- The month has 118 photos, so the next forward write run can process the full
  monthly image set under the current staged limit.

Backfill candidates requested by the user:

| Month | Photos | GPS photos | LINE messages | Call events | Existing events | Dry-run result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2024-11 | 621 | 577 | 975 | 31 | 107 | dry-run confirmed with staged 300 VLM / 300 embedding limit |
| 2024-10 | 112 | 87 | 338 | 14 | 50 | dry-run confirmed; full image set fits under staged 300 limit |

Recommended next command sequence:

```bash
python -m personal_lifelog_rag.app.cli month-plan --month 2026-01
python -m personal_lifelog_rag.app.cli month-run \
  --month 2026-01 \
  --vlm-limit 118 \
  --embedding-limit 118 \
  --config <local model runtime config> \
  --save-report \
  --dry-run
```

### 2025-11

- Run date: 2026-05-10
- Scope: 2025-11-01 to 2025-11-30
- Policy: one-month rollout only; no 2025-12 write execution in this session
- GPU: CUDA was visible through both `nvidia-smi` and PyTorch before the write
  run. The heavy VLM and embedding steps ran with GPU available.

Plan summary:

| Item | Count |
| --- | ---: |
| Photos | 137 |
| GPS photos | 135 |
| LINE messages | 736 |
| Call events | 8 |
| Existing events before rebuild | 116 |
| Existing VLM rows before rollout | 0 |
| Existing embedding rows before rollout | 0 |
| Existing OCR rows before rollout | 0 |
| Rollout VLM limit | 137 |
| Rollout embedding limit | 137 |

Execution summary:

| Component | Result |
| --- | --- |
| Backup before month-run | `backups/lifelog_before_month_run_202511_20260510_203503.sqlite` |
| Internal month-run backup | `backups/lifelog_before_month_rollout_2025_11_20260510_203507.sqlite` |
| Event rebuild backup | `backups/lifelog_before_event_rebuild_20251101_20260510_205453.sqlite` |
| VLM analysis | 133 success, 4 failed, 0 engine_unavailable |
| Image embeddings | 137 success, 0 failed |
| Combined text embeddings | 133 success, 0 failed |
| OCR | Not run in this rollout; priority candidates were listed only |
| Event rebuild | Completed |
| Event count after rebuild | 116 |
| Event count delta | +0 |
| VLM evidence added during rebuild | +133 |
| VLM-only high confidence events | 0 |
| db-check --strict | PASS |

QA smoke checks:

| Query | Result |
| --- | --- |
| 2025年11月は何していた？ | monthly_summary returned 116 events, 137 photos, 135 GPS photos, 736 LINE records, 8 call logs |
| 2025年11月のご飯を食べた写真はいつ？ | batch-qa success; primary candidate dates were 2025-11-25 and 2025-11-13 |
| 2025年11月のステージの写真はいつ？ | batch-qa success; primary candidate dates were 2025-11-15, 2025-11-03, and 2025-11-02 |

Generated reports:

| Mode | Markdown | JSON |
| --- | --- | --- |
| Private | `reports/lifelog_rag_eval_20260510_205728.md` | `reports/lifelog_rag_eval_20260510_205728.json` |
| Public | `reports/lifelog_rag_eval_20260510_205737.md` | `reports/lifelog_rag_eval_20260510_205737.json` |

Portfolio:

| Artifact | Result |
| --- | --- |
| HTML | `reports/portfolio_public.html` |
| Build JSON | `reports/portfolio_public_build.json` |
| Privacy check | PASS |

Notes:

- OCR was intentionally not executed. `ocr-priority` returned 50 text-like
  candidates, including tickets, documents, schedules, menus, labels, signs,
  receipts, screenshots, and message-like screens.
- The month contains 137 photos, so this rollout processed the full planned
  monthly image set under the current safety limit.
- Four VLM rows remain failed and should be retried later with the normal retry
  workflow.
- Public-facing docs and reports must continue to omit raw private data.

Next recommended month:

- 2025-12

2025-12 plan summary:

| Item | Count |
| --- | ---: |
| Photos | 329 |
| GPS photos | 309 |
| LINE messages | 4620 |
| Call events | 13 |
| Existing events | 108 |
| Existing VLM rows | 0 |
| Existing embedding rows | 0 |
| Existing OCR rows | 0 |
| Recommended VLM limit | 300 |
| Recommended embedding limit | 300 |

2025-12 dry-run result:

- No DB changes were made.
- Planned steps: backup, VLM analysis, image embeddings, combined text
  embeddings, event rebuild, strict DB check, public report generation.
- Monthly private eval was skipped because no 2025-12 private eval file exists.
- The month has 329 photos, so continue with the 300-item staged limit unless a
  later plan intentionally lowers it.

Recommended next command sequence:

```bash
python -m personal_lifelog_rag.app.cli month-plan --month 2025-12
python -m personal_lifelog_rag.app.cli month-run \
  --month 2025-12 \
  --vlm-limit 300 \
  --embedding-limit 300 \
  --config <local model runtime config> \
  --save-report \
  --dry-run
```

### 2025-10

- Run date: 2026-05-10
- Scope: 2025-10-01 to 2025-10-31
- Policy: one-month rollout only; no 2025-11 write execution in this session
- GPU: CUDA was visible through both `nvidia-smi` and PyTorch before the write
  run. The heavy VLM and embedding steps ran with GPU available.

Plan summary:

| Item | Count |
| --- | ---: |
| Photos | 1244 |
| GPS photos | 1225 |
| LINE messages | 770 |
| Call events | 7 |
| Existing events before rebuild | 117 |
| Existing VLM rows before rollout | 0 |
| Existing embedding rows before rollout | 0 |
| Existing OCR rows before rollout | 0 |
| Rollout VLM limit | 300 |
| Rollout embedding limit | 300 |

Execution summary:

| Component | Result |
| --- | --- |
| Backup before month-run | `backups/lifelog_before_month_run_202510_20260510_194613.sqlite` |
| Internal month-run backup | `backups/lifelog_before_month_rollout_2025_10_20260510_194620.sqlite` |
| Event rebuild backup | `backups/lifelog_before_event_rebuild_20251001_20260510_202742.sqlite` |
| VLM analysis | 294 success, 6 failed, 0 engine_unavailable |
| Image embeddings | 300 success, 0 failed |
| Combined text embeddings | 294 success, 0 failed |
| OCR | Not run in this rollout; priority candidates were listed only |
| Event rebuild | Completed |
| Event count after rebuild | 118 |
| Event count delta | +1 |
| VLM evidence added during rebuild | +294 |
| VLM-only high confidence events | 0 |
| db-check --strict | PASS |

QA smoke checks:

| Query | Result |
| --- | --- |
| 2025年10月は何していた？ | monthly_summary returned 118 events, 1244 photos, 1225 GPS photos, 770 LINE records, 7 call logs |
| 2025年10月のご飯を食べた写真はいつ？ | batch-qa success; primary candidate date was 2025-10-12 |
| 2025年10月のステージの写真はいつ？ | batch-qa success; primary candidate date was 2025-10-12 |

Generated reports:

| Mode | Markdown | JSON |
| --- | --- | --- |
| Private | `reports/lifelog_rag_eval_20260510_203045.md` | `reports/lifelog_rag_eval_20260510_203045.json` |
| Public | `reports/lifelog_rag_eval_20260510_203053.md` | `reports/lifelog_rag_eval_20260510_203053.json` |

Portfolio:

| Artifact | Result |
| --- | --- |
| HTML | `reports/portfolio_public.html` |
| Build JSON | `reports/portfolio_public_build.json` |
| Privacy check | PASS |

Notes:

- OCR was intentionally not executed. `ocr-priority` returned 50 text-like
  candidates, mainly food/drink labels, posters, screens, menus, signs,
  documents, packages, and venue text surfaces.
- The month contains 1244 photos, so this rollout processed 300 selected images
  under the current monthly safety limit. Remaining photos can be handled by a
  later targeted continuation pass if needed.
- Six VLM rows remain failed and should be retried later with the normal retry
  workflow.
- Public-facing docs and reports must continue to omit raw private data.

Next recommended month:

- 2025-11

2025-11 plan summary:

| Item | Count |
| --- | ---: |
| Photos | 137 |
| GPS photos | 135 |
| LINE messages | 736 |
| Call events | 8 |
| Existing events | 116 |
| Existing VLM rows | 0 |
| Existing embedding rows | 0 |
| Existing OCR rows | 0 |
| Recommended VLM limit | 137 |
| Recommended embedding limit | 137 |

2025-11 dry-run result:

- No DB changes were made.
- Planned steps: backup, VLM analysis, image embeddings, combined text
  embeddings, event rebuild, strict DB check, public report generation.
- Monthly private eval was skipped because no 2025-11 private eval file exists.
- The month has 137 photos, so the next write run can use a full-month staged
  limit of 137 for both VLM and embedding.

Recommended next command sequence:

```bash
python -m personal_lifelog_rag.app.cli month-plan --month 2025-11
python -m personal_lifelog_rag.app.cli month-run \
  --month 2025-11 \
  --vlm-limit 137 \
  --embedding-limit 137 \
  --config <local model runtime config> \
  --save-report \
  --dry-run
```

### 2025-09

- Run date: 2026-05-10
- Scope: 2025-09-01 to 2025-09-30
- Policy: one-month rollout only; no 2025-10 write execution in this session
- GPU: CUDA was visible through both `nvidia-smi` and PyTorch before the write
  run. The heavy VLM and embedding steps ran with GPU available.

Plan summary:

| Item | Count |
| --- | ---: |
| Photos | 837 |
| GPS photos | 758 |
| LINE messages | 870 |
| Call events | 7 |
| Existing events before rebuild | 124 |
| Existing VLM rows before rollout | 0 |
| Existing embedding rows before rollout | 0 |
| Existing OCR rows before rollout | 0 |
| Rollout VLM limit | 300 |
| Rollout embedding limit | 300 |

Execution summary:

| Component | Result |
| --- | --- |
| Backup before month-run | `backups/lifelog_before_month_run_202509_20260510_185919.sqlite` |
| Internal month-run backup | `backups/lifelog_before_month_rollout_2025_09_20260510_185924.sqlite` |
| Event rebuild backup | `backups/lifelog_before_event_rebuild_20250901_20260510_194010.sqlite` |
| VLM analysis | 295 success, 5 failed, 0 engine_unavailable |
| Image embeddings | 300 success, 0 failed |
| Combined text embeddings | 295 success, 0 failed |
| OCR | Not run in this rollout; priority candidates were listed only |
| Event rebuild | Completed |
| Event count after rebuild | 125 |
| Event count delta | +1 |
| VLM evidence added during rebuild | +295 |
| VLM-only high confidence events | 0 |
| db-check --strict | PASS |

QA smoke checks:

| Query | Result |
| --- | --- |
| 2025年9月は何していた？ | monthly_summary returned 125 events, 837 photos, 758 GPS photos, 870 LINE records, 7 call logs |
| 2025年9月のご飯を食べた写真はいつ？ | batch-qa success; primary candidate dates were 2025-09-08 and 2025-09-13 |
| 2025年9月のステージの写真はいつ？ | batch-qa success; primary candidate date was 2025-09-19 |

Generated reports:

| Mode | Markdown | JSON |
| --- | --- | --- |
| Private | `reports/lifelog_rag_eval_20260510_194259.md` | `reports/lifelog_rag_eval_20260510_194259.json` |
| Public | `reports/lifelog_rag_eval_20260510_194314.md` | `reports/lifelog_rag_eval_20260510_194314.json` |

Portfolio:

| Artifact | Result |
| --- | --- |
| HTML | `reports/portfolio_public.html` |
| Build JSON | `reports/portfolio_public_build.json` |
| Privacy check | PASS |

Notes:

- OCR was intentionally not executed. `ocr-priority` returned 50 text-like
  candidates, including menus, tickets, documents, screenshots, signs,
  storefront text, receipts, packages, labels, and event-related text surfaces.
- The month contains 837 photos, so this rollout processed 300 selected images
  under the current monthly safety limit. Remaining photos can be handled by a
  later targeted continuation pass if needed.
- Five VLM rows remain failed and should be retried later with the normal retry
  workflow.
- Public-facing docs and reports must continue to omit raw private data.

Next recommended month:

- 2025-10

2025-10 plan summary:

| Item | Count |
| --- | ---: |
| Photos | 1244 |
| GPS photos | 1225 |
| LINE messages | 770 |
| Call events | 7 |
| Existing events | 117 |
| Existing VLM rows | 0 |
| Existing embedding rows | 0 |
| Existing OCR rows | 0 |
| Recommended VLM limit | 300 |
| Recommended embedding limit | 300 |

2025-10 dry-run result:

- No DB changes were made.
- Planned steps: backup, VLM analysis, image embeddings, combined text
  embeddings, event rebuild, strict DB check, public report generation.
- Monthly private eval was skipped because no 2025-10 private eval file exists.
- The month has 1244 photos, so continue with the 300-item staged limit unless
  a later plan intentionally lowers it.

Recommended next command sequence:

```bash
python -m personal_lifelog_rag.app.cli month-plan --month 2025-10
python -m personal_lifelog_rag.app.cli month-run \
  --month 2025-10 \
  --vlm-limit 300 \
  --embedding-limit 300 \
  --config <local model runtime config> \
  --save-report \
  --dry-run
```

### 2025-08

- Run date: 2026-05-10
- Scope: 2025-08-01 to 2025-08-31
- Policy: one-month rollout only; no 2025-09 write execution in this session
- GPU: CUDA was visible through both `nvidia-smi` and PyTorch before the write
  run. The heavy VLM and embedding steps ran with GPU available.

Plan summary:

| Item | Count |
| --- | ---: |
| Photos | 599 |
| GPS photos | 584 |
| LINE messages | 898 |
| Call events | 10 |
| Existing events before rebuild | 132 |
| Existing VLM rows before rollout | 0 |
| Existing embedding rows before rollout | 0 |
| Existing OCR rows before rollout | 0 |
| Rollout VLM limit | 300 |
| Rollout embedding limit | 300 |

Execution summary:

| Component | Result |
| --- | --- |
| Backup before month-run | `backups/lifelog_before_month_run_202508_20260510_180506.sqlite` |
| Internal month-run backup | `backups/lifelog_before_month_rollout_2025_08_20260510_180515.sqlite` |
| Event rebuild backup | `backups/lifelog_before_event_rebuild_20250801_20260510_184509.sqlite` |
| VLM analysis | 298 success, 2 failed, 0 engine_unavailable |
| Image embeddings | 300 success, 0 failed |
| Combined text embeddings | 298 success, 0 failed |
| OCR | Not run in this rollout; priority candidates were listed only |
| Event rebuild | Completed |
| Event count after rebuild | 134 |
| Event count delta | +2 |
| VLM evidence added during rebuild | +298 |
| VLM-only high confidence events | 0 |
| db-check --strict | PASS |

QA smoke checks:

| Query | Result |
| --- | --- |
| 2025年8月は何していた？ | monthly_summary returned 134 events, 599 photos, 584 GPS photos, 898 LINE records, 10 call logs |
| 2025年8月のご飯を食べた写真はいつ？ | batch-qa success; primary candidate dates were 2025-08-12 and 2025-08-11 |
| 2025年8月のステージの写真はいつ？ | batch-qa success; primary candidate dates were 2025-08-12 and 2025-08-03 |

Generated reports:

| Mode | Markdown | JSON |
| --- | --- | --- |
| Private | `reports/lifelog_rag_eval_20260510_184741.md` | `reports/lifelog_rag_eval_20260510_184741.json` |
| Public | `reports/lifelog_rag_eval_20260510_184748.md` | `reports/lifelog_rag_eval_20260510_184748.json` |

Portfolio:

| Artifact | Result |
| --- | --- |
| HTML | `reports/portfolio_public.html` |
| Build JSON | `reports/portfolio_public_build.json` |
| Privacy check | PASS |

Notes:

- OCR was intentionally not executed. `ocr-priority` returned 50 text-like
  candidates, including receipts, maps, signs, menus, screenshots, labels,
  posters, packages, and storefront text. A later OCR-focused pass can use this
  queue without re-running all images.
- The month contains 599 photos, so this rollout processed 300 selected images
  under the current monthly safety limit. Remaining photos can be handled by a
  later targeted continuation pass if needed.
- Two VLM rows remain failed and should be retried later with the normal retry
  workflow.
- Public-facing docs and reports must continue to omit raw private data.

Next recommended month:

- 2025-09

2025-09 plan summary:

| Item | Count |
| --- | ---: |
| Photos | 837 |
| GPS photos | 758 |
| LINE messages | 870 |
| Call events | 7 |
| Existing events | 124 |
| Existing VLM rows | 0 |
| Existing embedding rows | 0 |
| Existing OCR rows | 0 |
| Recommended VLM limit | 300 |
| Recommended embedding limit | 300 |

2025-09 dry-run result:

- No DB changes were made.
- Planned steps: backup, VLM analysis, image embeddings, combined text
  embeddings, event rebuild, strict DB check, public report generation.
- Monthly private eval was skipped because no 2025-09 private eval file exists.
- The month has 837 photos, so continue with the 300-item staged limit unless
  a later plan intentionally lowers it.

Recommended next command sequence:

```bash
python -m personal_lifelog_rag.app.cli month-plan --month 2025-09
python -m personal_lifelog_rag.app.cli month-run \
  --month 2025-09 \
  --vlm-limit 300 \
  --embedding-limit 300 \
  --config <local model runtime config> \
  --save-report \
  --dry-run
```

### 2025-04

- Run date: 2026-05-10
- Scope: 2025-04-01 to 2025-04-30
- Policy: one-month rollout only; no all-period heavy processing
- GPU: CUDA was available and used for the heavy month-run after an initial
  sandboxed run was stopped because it could not use the GPU.

Plan summary:

| Item | Count |
| --- | ---: |
| Photos | 352 |
| GPS photos | 302 |
| LINE messages | 1269 |
| Call events | 12 |
| Existing events before rebuild | 109 |
| Existing VLM rows before rollout | 0 |
| Existing embedding rows before rollout | 0 |
| Existing OCR rows before rollout | 0 |

Execution summary:

| Component | Result |
| --- | --- |
| Backup before month-run | Created |
| VLM analysis | 293 success, 7 failed, 0 engine_unavailable |
| Image embeddings | 300 success, 0 failed |
| Combined text embeddings | 293 success, 0 failed |
| OCR | Not run in this rollout; priority candidates were listed only |
| Event rebuild | Completed |
| Event count after rebuild | 111 |
| Event count delta | +2 |
| VLM evidence added during rebuild | +293 |
| VLM-only high confidence events | 0 |
| db-check --strict | PASS |

QA smoke checks:

| Query | Result |
| --- | --- |
| 2025年4月は何していた？ | monthly_summary returned 111 events, 352 photos, 302 GPS photos, 1269 LINE records, 12 call logs |
| ご飯を食べた写真はいつ？ | batch-qa success; top candidates include 2025-04 among multiple months |
| ステージの写真はいつ？ | batch-qa success; current strongest candidates are in 2025-03 |
| レシートの写真はいつ？ | batch-qa success; OCR is still a future improvement point |

Generated reports:

| Mode | Markdown | JSON |
| --- | --- | --- |
| Private | `reports/lifelog_rag_eval_20260510_154402.md` | `reports/lifelog_rag_eval_20260510_154402.json` |
| Public | `reports/lifelog_rag_eval_20260510_154408.md` | `reports/lifelog_rag_eval_20260510_154408.json` |

Notes:

- The first month-run attempt was stopped after confirming it was running inside
  the sandbox without GPU access. The database still passed strict validation
  afterward.
- The completed run used the local GPU and finished VLM, image embedding,
  combined text embedding, event rebuild, and strict DB validation.
- OCR was not executed for the month. `ocr-priority` returned prioritized text
  candidates, so a later OCR-only pass can focus on screenshots, signs, labels,
  posters, and document-like images.
- Public-facing docs and reports must continue to omit raw private data.

Next recommended month:

- 2025-05

Recommended next command sequence:

```bash
python -m personal_lifelog_rag.app.cli month-plan --month 2025-05
python -m personal_lifelog_rag.app.cli month-run \
  --month 2025-05 \
  --vlm-limit 300 \
  --embedding-limit 300 \
  --config <local model runtime config> \
  --save-report \
  --dry-run
```

### 2025-07

- Run date: 2026-05-10
- Scope: 2025-07-01 to 2025-07-31
- Policy: one-month rollout only; no 2025-08 write execution in this session
- GPU: CUDA was visible through both `nvidia-smi` and PyTorch before the write
  run. The heavy VLM and embedding steps ran with GPU available.

Plan summary:

| Item | Count |
| --- | ---: |
| Photos | 323 |
| GPS photos | 233 |
| LINE messages | 994 |
| Call events | 8 |
| Existing events before rebuild | 123 |
| Existing VLM rows before rollout | 0 |
| Existing embedding rows before rollout | 0 |
| Existing OCR rows before rollout | 0 |
| Rollout VLM limit | 300 |
| Rollout embedding limit | 300 |

Execution summary:

| Component | Result |
| --- | --- |
| Backup before month-run | `backups/lifelog_before_month_run_202507_20260510_171800.sqlite` |
| Internal month-run backup | `backups/lifelog_before_month_rollout_2025_07_20260510_171809.sqlite` |
| Event rebuild backup | `backups/lifelog_before_event_rebuild_20250701_20260510_175843.sqlite` |
| VLM analysis | 297 success, 3 failed, 0 engine_unavailable |
| Image embeddings | 300 success, 0 failed |
| Combined text embeddings | 297 success, 0 failed |
| OCR | Not run in this rollout; priority candidates were listed only |
| Event rebuild | Completed |
| Event count after rebuild | 126 |
| Event count delta | +3 |
| VLM evidence added during rebuild | +297 |
| VLM-only high confidence events | 0 |
| db-check --strict | PASS |

QA smoke checks:

| Query | Result |
| --- | --- |
| 2025年7月は何していた？ | monthly_summary returned 126 events, 323 photos, 233 GPS photos, 994 LINE records, 8 call logs |
| 2025年7月のご飯を食べた写真はいつ？ | batch-qa success; primary candidate dates were 2025-07-03 and 2025-07-25 |
| 2025年7月のステージの写真はいつ？ | batch-qa success; primary candidate date was 2025-07-18 |

Generated reports:

| Mode | Markdown | JSON |
| --- | --- | --- |
| Private | `reports/lifelog_rag_eval_20260510_180111.md` | `reports/lifelog_rag_eval_20260510_180111.json` |
| Public | `reports/lifelog_rag_eval_20260510_180120.md` | `reports/lifelog_rag_eval_20260510_180120.json` |

Portfolio:

| Artifact | Result |
| --- | --- |
| HTML | `reports/portfolio_public.html` |
| Build JSON | `reports/portfolio_public_build.json` |
| Privacy check | PASS |

Notes:

- OCR was intentionally not executed. `ocr-priority` returned 50 text-like
  candidates, especially screenshots, menus, posters, signs, packages, and
  event-related text surfaces. A later OCR-focused pass can use this queue.
- The month contains more than 300 photos, so this rollout processed the first
  300 selected images under the current monthly safety limit. Remaining photos
  can be handled by a later targeted continuation pass if needed.
- Three VLM rows remain failed and should be retried later with the normal
  retry workflow.
- Public-facing docs and reports must continue to omit raw private data.

Next recommended month:

- 2025-08

2025-08 plan summary:

| Item | Count |
| --- | ---: |
| Photos | 599 |
| GPS photos | 584 |
| LINE messages | 898 |
| Call events | 10 |
| Existing events | 132 |
| Existing VLM rows | 0 |
| Existing embedding rows | 0 |
| Existing OCR rows | 0 |
| Recommended VLM limit | 300 |
| Recommended embedding limit | 300 |

2025-08 dry-run result:

- No DB changes were made.
- Planned steps: backup, VLM analysis, image embeddings, combined text
  embeddings, event rebuild, strict DB check, public report generation.
- Monthly private eval was skipped because no 2025-08 private eval file exists.

Recommended next command sequence:

```bash
python -m personal_lifelog_rag.app.cli month-plan --month 2025-08
python -m personal_lifelog_rag.app.cli month-run \
  --month 2025-08 \
  --vlm-limit 300 \
  --embedding-limit 300 \
  --config <local model runtime config> \
  --save-report \
  --dry-run
```

### 2025-06

- Run date: 2026-05-10
- Scope: 2025-06-01 to 2025-06-30
- Policy: one-month rollout only; no 2025-07 write execution in this session
- GPU: CUDA was visible through both `nvidia-smi` and PyTorch before the write
  run. The heavy VLM and embedding steps ran with GPU available.

Plan summary:

| Item | Count |
| --- | ---: |
| Photos | 277 |
| GPS photos | 257 |
| LINE messages | 935 |
| Call events | 9 |
| Existing events before rebuild | 121 |
| Existing VLM rows before rollout | 0 |
| Existing embedding rows before rollout | 0 |
| Existing OCR rows before rollout | 0 |

Execution summary:

| Component | Result |
| --- | --- |
| Backup before month-run | `backups/lifelog_before_month_run_202506_20260510_163237.sqlite` |
| Internal month-run backup | `backups/lifelog_before_month_rollout_2025_06_20260510_163242.sqlite` |
| Event rebuild backup | `backups/lifelog_before_event_rebuild_20250601_20260510_171005.sqlite` |
| VLM analysis | 271 success, 6 failed, 0 engine_unavailable |
| Image embeddings | 277 success, 0 failed |
| Combined text embeddings | 271 success, 0 failed |
| OCR | Not run in this rollout; priority candidates were listed only |
| Event rebuild | Completed |
| Event count after rebuild | 121 |
| Event count delta | +0 |
| VLM evidence added during rebuild | +271 |
| VLM-only high confidence events | 0 |
| db-check --strict | PASS |

QA smoke checks:

| Query | Result |
| --- | --- |
| 2025年6月は何していた？ | monthly_summary returned 121 events, 277 photos, 257 GPS photos, 935 LINE records, 9 call logs |
| 2025年6月のご飯を食べた写真はいつ？ | batch-qa success; primary candidate date was 2025-06-23 |
| 2025年6月のステージの写真はいつ？ | batch-qa success; primary candidate date was 2025-06-11 |

Generated reports:

| Mode | Markdown | JSON |
| --- | --- | --- |
| Private | `reports/lifelog_rag_eval_20260510_171325.md` | `reports/lifelog_rag_eval_20260510_171325.json` |
| Public | `reports/lifelog_rag_eval_20260510_171331.md` | `reports/lifelog_rag_eval_20260510_171331.json` |

Portfolio:

| Artifact | Result |
| --- | --- |
| HTML | `reports/portfolio_public.html` |
| Build JSON | `reports/portfolio_public_build.json` |
| Privacy check | PASS |

Notes:

- OCR was intentionally not executed. `ocr-priority` returned 50 text-like
  candidates, including event signage, tickets, posters, menus, screenshots,
  and labels, so a later OCR-focused pass can stay narrow.
- Month-run generated a public month-local report internally before the broader
  2024-12 to 2025-06 reports were regenerated.
- Six VLM rows remain failed and should be retried later with the normal
  retry workflow.
- Public-facing docs and reports must continue to omit raw private data.

Next recommended month:

- 2025-07

2025-07 plan summary:

| Item | Count |
| --- | ---: |
| Photos | 323 |
| GPS photos | 233 |
| LINE messages | 994 |
| Call events | 8 |
| Existing events | 123 |
| Existing VLM rows | 0 |
| Existing embedding rows | 0 |
| Existing OCR rows | 0 |
| Recommended VLM limit | 300 |
| Recommended embedding limit | 300 |

2025-07 dry-run result:

- No DB changes were made.
- Planned steps: backup, VLM analysis, image embeddings, combined text
  embeddings, event rebuild, strict DB check, public report generation.
- Monthly private eval was skipped because no 2025-07 private eval file exists.

Recommended next command sequence:

```bash
python -m personal_lifelog_rag.app.cli month-plan --month 2025-07
python -m personal_lifelog_rag.app.cli month-run \
  --month 2025-07 \
  --vlm-limit 300 \
  --embedding-limit 300 \
  --config <local model runtime config> \
  --save-report \
  --dry-run
```

### 2025-05

- Run date: 2026-05-10
- Scope: 2025-05-01 to 2025-05-31
- Policy: one-month rollout only; no 2025-06 write execution in this session
- GPU: CUDA was visible through both `nvidia-smi` and PyTorch before the write
  run. The heavy VLM and embedding steps ran with GPU available.

Plan summary:

| Item | Count |
| --- | ---: |
| Photos | 437 |
| GPS photos | 426 |
| LINE messages | 1144 |
| Call events | 14 |
| Existing events before rebuild | 125 |
| Existing VLM rows before rollout | 0 |
| Existing embedding rows before rollout | 0 |
| Existing OCR rows before rollout | 0 |

Execution summary:

| Component | Result |
| --- | --- |
| Backup before month-run | `backups/lifelog_before_month_run_202505_20260510_154912.sqlite` |
| Internal month-run backup | `backups/lifelog_before_month_rollout_2025_05_20260510_154918.sqlite` |
| VLM analysis | 297 success, 3 failed, 0 engine_unavailable |
| Image embeddings | 300 success, 0 failed |
| Combined text embeddings | 297 success, 0 failed |
| OCR | Not run in this rollout; priority candidates were listed only |
| Event rebuild | Completed |
| Event count after rebuild | 124 |
| Event count delta | -1 |
| VLM evidence added during rebuild | +297 |
| VLM-only high confidence events | 0 |
| db-check --strict | PASS |

QA smoke checks:

| Query | Result |
| --- | --- |
| 2025年5月は何していた？ | monthly_summary returned 124 events, 437 photos, 426 GPS photos, 1144 LINE records, 14 call logs |
| 2025年5月のご飯を食べた写真はいつ？ | batch-qa success; primary candidate date was 2025-05-02 |
| 2025年5月のステージの写真はいつ？ | batch-qa success; primary candidate date was 2025-05-02 |

Generated reports:

| Mode | Markdown | JSON |
| --- | --- | --- |
| Private | `reports/lifelog_rag_eval_20260510_163014.md` | `reports/lifelog_rag_eval_20260510_163014.json` |
| Public | `reports/lifelog_rag_eval_20260510_163024.md` | `reports/lifelog_rag_eval_20260510_163024.json` |

Portfolio:

| Artifact | Result |
| --- | --- |
| HTML | `reports/portfolio_public.html` |
| Build JSON | `reports/portfolio_public_build.json` |
| Privacy check | PASS |

Notes:

- OCR was intentionally not executed. `ocr-priority` produced 50 text-like
  candidates, with tickets, signs, date markers, menus, labels, and document-like
  images prominent enough for a later OCR-focused pass.
- Month-run generated a public month-local report internally before the broader
  2024-12 to 2025-05 reports were regenerated.
- Private and public reports were regenerated sequentially after an initial
  same-second parallel generation produced ambiguous output filenames.
- Public-facing docs and reports must continue to omit raw private data.

Next recommended month:

- 2025-06

2025-06 plan summary:

| Item | Count |
| --- | ---: |
| Photos | 277 |
| GPS photos | 257 |
| LINE messages | 935 |
| Call events | 9 |
| Existing events | 121 |
| Existing VLM rows | 0 |
| Existing embedding rows | 0 |
| Existing OCR rows | 0 |
| Recommended VLM limit | 277 |
| Recommended embedding limit | 277 |

2025-06 dry-run result:

- No DB changes were made.
- Planned steps: backup, VLM analysis, image embeddings, combined text
  embeddings, event rebuild, strict DB check, public report generation.
- Monthly private eval was skipped because no 2025-06 private eval file exists.

Recommended next command sequence:

```bash
python -m personal_lifelog_rag.app.cli month-plan --month 2025-06
python -m personal_lifelog_rag.app.cli month-run \
  --month 2025-06 \
  --vlm-limit 277 \
  --embedding-limit 277 \
  --config <local model runtime config> \
  --save-report \
  --dry-run
```

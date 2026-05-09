# Analysis Job Management

This app keeps OCR, VLM, embedding, and event rebuild work local. Heavy analysis should be planned first, run in small batches, and resumed or retried from recorded job state.

## Why Jobs Exist

OCR, Qwen3-VL analysis, and Qwen3-VL-Embedding indexing can be slow and storage-heavy. The job tables keep a privacy-safe record of:

- target scope
- engine, model, prompt, and analysis versions
- total / processed / success / failed / skipped counts
- per-item status and errors

The tables do not store LINE text, exact GPS, or image contents.

## Planning

Use `analysis-plan` before broad runs:

```bash
python -m personal_lifelog_rag.app.cli analysis-plan --type vlm --date 2024-12-24
python -m personal_lifelog_rag.app.cli analysis-plan --type ocr --from 2024-12-01 --to 2024-12-31
python -m personal_lifelog_rag.app.cli analysis-plan --type image_embedding --all --limit 100
```

The plan shows candidate count, already-success rows, failed rows, engine-unavailable rows, estimated storage increase, and a suggested run command.

## Running

Run a limited job first:

```bash
python -m personal_lifelog_rag.app.cli analysis-run --type vlm --date 2024-12-24 --engine fake --limit 5 --save-report
```

Useful options:

- `--dry-run`: do not write job rows or analysis outputs
- `--force`: re-run selected rows
- `--skip-existing`: skip existing successful rows
- `--failed-only`: retry only failed rows in the selected scope
- `--engine-unavailable-only`: retry rows that previously lacked a local engine
- `--version-changed-only`: select rows whose stored engine/model/prompt/version no longer matches
- `--save-report`: write JSON and Markdown under `eval_outputs/analysis_jobs/`

## Status, Resume, Retry

```bash
python -m personal_lifelog_rag.app.cli analysis-status --recent 10
python -m personal_lifelog_rag.app.cli analysis-status --job-id JOB_ID
python -m personal_lifelog_rag.app.cli analysis-resume --job-id JOB_ID
python -m personal_lifelog_rag.app.cli analysis-retry-failed --job-id JOB_ID
```

`analysis-resume` creates a new run from the previous job scope. Use `--engine-unavailable-only` when a local model becomes available after an earlier run.

## Cleanup

Cleanup is dry-run first. Real deletion requires `--yes`.

```bash
python -m personal_lifelog_rag.app.cli analysis-cleanup --failed --dry-run
python -m personal_lifelog_rag.app.cli analysis-cleanup --old-runs 30 --dry-run
python -m personal_lifelog_rag.app.cli analysis-cleanup --old-runs 30 --yes
```

## Storage and Maintenance

```bash
python -m personal_lifelog_rag.app.cli storage-stats
python -m personal_lifelog_rag.app.cli db-maintenance --backup
python -m personal_lifelog_rag.app.cli db-maintenance --backup --vacuum
```

`VACUUM` is skipped unless you also request `--backup` or explicitly pass `--yes`.

## Recommended Full-Period Flow

1. `backup-db --label before_analysis_all`
2. `analysis-plan --type ocr --all`
3. `analysis-run --type ocr --all --limit 500 --skip-existing --save-report`
4. Repeat for `vlm`, `image_embedding`, and `text_embedding`
5. `analysis-status --recent 10`
6. `db-check --strict`
7. `storage-stats`

## Do Not Commit

Keep these private/local paths out of Git:

- `data/`
- `backups/`
- `private_eval/`
- `eval_outputs/`
- `private_config/`
- `models/`

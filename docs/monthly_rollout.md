# Monthly Rollout

This workflow expands VLM, embedding, event rebuild, private eval, and report
generation one month at a time. It is designed for local-only operation over
private photos, LINE messages, GPS, OCR, VLM results, and embeddings.

## Policy

- Do not send photos, LINE text, GPS, OCR, VLM results, or embeddings to any
  external API.
- Start with `month-plan` and `month-run --dry-run`.
- Real month execution requires `--yes`.
- Create a DB backup before any non-dry-run rollout.
- Stop on the first failed step and inspect `month-status` before retrying.
- Keep `data/`, `backups/`, `private_eval/`, `eval_outputs/`,
  `private_config/`, `models/`, and `reports/` out of Git.

## Plan One Month

```bash
python -m personal_lifelog_rag.app.cli month-plan --month 2025-02
```

The plan shows:

- date range
- photo count
- GPS photo count
- LINE message count
- call event count
- existing `media_vlm` status counts
- existing `media_embeddings` status counts
- existing `events` count
- recommended limits and commands

## Dry Run

```bash
python -m personal_lifelog_rag.app.cli month-run \
  --month 2025-02 \
  --limit 100 \
  --dry-run
```

Dry-run prints the ordered pipeline without touching the DB:

1. `backup-db`
2. `analyze-images`
3. `build-image-embeddings`
4. `build-text-embeddings`
5. `rebuild-events-with-analysis`
6. `db-check --strict`
7. optional `eval-private`
8. optional `generate-report`

## Execute One Month

Run only after the dry-run looks right:

```bash
python -m personal_lifelog_rag.app.cli month-run \
  --month 2025-02 \
  --vlm-limit 300 \
  --embedding-limit 300 \
  --config private_config/model_runtime.yaml \
  --save-report \
  --yes
```

Useful safety switches:

- `--skip-vlm`
- `--skip-embedding`
- `--skip-rebuild`
- `--skip-eval`
- `--skip-report`

If a step fails, the command stops and prints a recovery hint. Check:

```bash
python -m personal_lifelog_rag.app.cli month-status --month 2025-02
python -m personal_lifelog_rag.app.cli db-check --strict
```

Then rerun the skipped or failed step directly, or rerun `month-run` with
appropriate skip flags.

## Status

```bash
python -m personal_lifelog_rag.app.cli month-status --month 2025-02
```

The status view shows:

- VLM success / failed / engine unavailable
- embedding success / failed / engine unavailable
- OCR coverage
- event count
- whether a report artifact exists
- whether an eval run artifact exists
- a reminder to run strict DB validation

## Batch Planning

Batch rollout is intentionally conservative. The current command is planning
only:

```bash
python -m personal_lifelog_rag.app.cli month-batch \
  --from-month 2025-02 \
  --to-month 2025-05 \
  --dry-run
```

Execute one month at a time after reviewing each plan.

## Private Eval

If `private_eval/questions_YYYYMM_month.yaml` exists for the target month,
`month-run` runs:

```bash
python -m personal_lifelog_rag.app.cli eval-private \
  --path private_eval/questions_YYYYMM_month.yaml \
  --save-run
```

If it does not exist, eval is skipped with a clear message. Do not commit
private eval files.

## Reports

When reports are enabled, `month-run` generates a public, no-examples report
for the month. Reports are local artifacts under `reports/` and should not be
committed.

## Before All-Period Rollout

For each month:

1. Run `month-plan`.
2. Run `month-run --dry-run`.
3. Execute with a modest `--vlm-limit` and `--embedding-limit`.
4. Run `db-check --strict`.
5. Review monthly summary, multimodal search, and private eval.
6. Increase limits only after failures and overclaims are understood.

## 2025-03 Rollout Snapshot

2025-03 has already completed the heavy VLM and embedding pilot. Do not rerun
the heavy month execution unless there is a specific reason.

Current operational snapshot:

- VLM: 286 success, 14 failed, 0 engine unavailable.
- Embeddings: 586 success.
- Events: 119 after rebuild.
- Monthly QA: `2025年3月は何していた？` routes to monthly summary.
- Strict DB validation recovered to ok after maintenance.

Recommended next checks:

```bash
conda run -n personal_lifelog_rag python -m personal_lifelog_rag.app.cli month-status --month 2025-03
conda run -n personal_lifelog_rag python -m personal_lifelog_rag.app.cli qa "2025年3月は何していた？"
conda run -n personal_lifelog_rag python -m personal_lifelog_rag.app.cli db-check --strict
```

Plan only for following months:

```bash
conda run -n personal_lifelog_rag python -m personal_lifelog_rag.app.cli month-batch --from-month 2025-04 --to-month 2025-06 --dry-run
```

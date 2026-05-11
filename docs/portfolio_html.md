# Portfolio HTML

`reports/portfolio_public.html` is a single-file, offline HTML portfolio for
explaining this local-first lifelog RAG project. It is generated from the
public-safe Markdown docs and aggregate report summaries, not by hand-editing
HTML.

## Purpose

The HTML is meant for research, job-hunting, and portfolio review. It explains:

- what the app does
- the local multimodal RAG architecture
- the roles of Qwen3-VL, Qwen3-VL-Embedding, and OCR
- demo scenarios
- aggregate evaluation results
- privacy and safety design
- engineering highlights and roadmap

It must not contain real photos, raw LINE text, exact location coordinates,
personal names, local absolute paths, local database paths, or private runtime
settings.

## Source Markdown

The default build reads these public-oriented docs when they exist:

- `docs/portfolio_summary.md`
- `docs/system_architecture.md`
- `docs/demo_scenarios.md`
- `docs/privacy_and_safety.md`
- `docs/evaluation_summary.md`
- `docs/roadmap.md`
- `docs/ocr_engine.md`
- `docs/ui_usage.md`
- `docs/monthly_rollout.md`

An optional generated public report can also be passed with `--source-report`.
The builder only extracts aggregate metrics and does not embed raw report text.

## Generate

```bash
python -m personal_lifelog_rag.app.cli build-portfolio-html \
  --output reports/portfolio_public.html \
  --mode public \
  --check-privacy \
  --force
```

This writes:

- `reports/portfolio_public.html`
- `reports/portfolio_public_build.json`

The build JSON records generation time, source files, output path, privacy
check result, blocked patterns, and extracted metrics.

## Privacy Check

Run the standalone checker before sharing:

```bash
python scripts/check_public_portfolio_safety.py reports/portfolio_public.html
```

The checker fails on obvious public-leak patterns such as local home paths,
raw-data paths, private runtime config names, local database filenames, exact
media IDs, coordinate labels, phone numbers, email addresses, public Gradio
sharing, and file URIs.

It also rejects private face-data paths and raw face-data field names. Face
crops, face embeddings, person display names, and exact coordinates should stay
out of the public portfolio.

## Privacy Controls Integration

Before publishing, run:

```bash
python -m personal_lifelog_rag.app.cli privacy-audit --public
```

If a person or place should not appear in public material, use the PR72 privacy
controls to hide, detach, soft-delete, or export a redacted summary before
rebuilding the HTML.

## Editing Rule

When `docs/*.md`, `reports/*.md`, or `README.md` are edited for portfolio or
reporting content, rebuild the public HTML at the same time:

```bash
python -m personal_lifelog_rag.app.cli build-portfolio-html \
  --mode public \
  --check-privacy \
  --force
```

Then confirm:

```bash
python scripts/check_public_portfolio_safety.py reports/portfolio_public.html
```

## Public Checklist

Before sharing the HTML:

- confirm the privacy checker passes
- manually scan the HTML in a browser
- confirm no real photo is embedded
- confirm no raw LINE message body is visible
- confirm no exact location coordinate appears
- confirm no personal name or address appears
- confirm no local absolute path appears
- confirm examples are anonymized or aggregate-only

## Troubleshooting

- If the builder refuses to overwrite the HTML, pass `--force`.
- If the privacy check fails, inspect the reported line and replace the text
  with a generic public-safe description.
- If metrics look stale, refresh `docs/evaluation_summary.md` or generate a new
  public Markdown report first, then rebuild the HTML.
- If a phrase is useful in private docs but unsafe for public output, keep it in
  the Markdown docs and avoid embedding it in the HTML template.

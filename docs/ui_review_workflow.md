# UI Review Workflow

The Gradio UI is for local review only. Launch it on localhost:

```bash
conda run -n personal_lifelog_rag python -m personal_lifelog_rag.app.cli ui
```

Do not enable public sharing.

## Monthly Summary

1. Open the Monthly Summary tab.
2. Select a month such as `2025-01` or `2025-03`.
3. Review event counts, representative dates, photo counts, VLM coverage, OCR
   coverage, and call activity.
4. Treat VLM-only observations as visual candidates, not facts.

## Multimodal Search

1. Search queries such as `ご飯を食べた写真` or `ステージの写真`.
2. Check thumbnails, captions, evidence strength, matched terms, and score
   components.
3. Prioritize review of top-ranked results instead of reviewing every image.

## VLM Review

Use review actions only where they materially affect search or event building:

- `wrong`: visually incorrect analysis.
- `not_searchable`: do not use in normal search.
- `not_event_usable`: do not use for event generation.
- `hidden`: private or not suitable for normal display.
- `accepted` or `verified`: manually checked and useful.

## OCR Review

OCR text can contain private details and recognition errors. Prefer redacted
views, and treat OCR-only matches as candidates.

## After Review

Run:

```bash
conda run -n personal_lifelog_rag python -m personal_lifelog_rag.app.cli db-check --strict
conda run -n personal_lifelog_rag python -m personal_lifelog_rag.app.cli build-portfolio-html --mode public --check-privacy --force
python scripts/check_public_portfolio_safety.py reports/portfolio_public.html
```

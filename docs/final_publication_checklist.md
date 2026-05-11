# Final Publication Checklist

Use this checklist before sharing any public artifact.

## Required Checks

- `pytest` passes.
- `db-check --strict` passes.
- Private eval results are saved and failures are explained.
- Public report is regenerated.
- `reports/portfolio_public.html` is regenerated.
- Public portfolio privacy check passes.

## Do Not Publish

- Real photos.
- Raw LINE text.
- Exact GPS coordinates.
- Personal names.
- Local absolute paths.
- Runtime secrets or private model paths.
- SQLite database files.
- Private eval files.

## Public Artifacts

- Public Markdown report under `reports/`.
- Public portfolio HTML: `reports/portfolio_public.html`.
- Build metadata: `reports/portfolio_public_build.json`.
- Release manifest: `reports/release_v0_1_manifest.json`.

## Recommended Commands

```bash
conda run -n personal_lifelog_rag pytest
conda run -n personal_lifelog_rag python -m personal_lifelog_rag.app.cli db-check --strict
conda run -n personal_lifelog_rag python -m personal_lifelog_rag.app.cli build-portfolio-html --mode public --check-privacy --force
python scripts/check_public_portfolio_safety.py reports/portfolio_public.html
```

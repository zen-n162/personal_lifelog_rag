# Demo Script For Interview

## Preparation

1. Use a local machine only.
2. Confirm the public portfolio HTML passes the privacy check.
3. If showing private data, explain that the demo is local and avoid displaying
   raw messages or exact locations.

## CLI Demo

```bash
conda run -n personal_lifelog_rag python -m personal_lifelog_rag.app.cli qa "2025年1月は何していた？"
conda run -n personal_lifelog_rag python -m personal_lifelog_rag.app.cli qa "ご飯を食べた写真はいつ？"
conda run -n personal_lifelog_rag python -m personal_lifelog_rag.app.cli qa "ステージの写真はいつ？"
```

Explain that each answer is built from local evidence and cautious ranking.

## UI Demo

```bash
conda run -n personal_lifelog_rag python -m personal_lifelog_rag.app.cli ui
```

Show:

- Monthly Summary.
- Multimodal Search.
- VLM Review.
- Report Viewer.

## Portfolio Demo

Open `reports/portfolio_public.html` locally. It is designed to be safe for
public explanation and should not contain real photos, raw private text, or
exact coordinates.

## Talking Points

- Local-first architecture.
- Separate VLM and embedding roles.
- Evidence-linked event generation.
- Conservative confidence and evidence strength.
- Private eval and release reproducibility.

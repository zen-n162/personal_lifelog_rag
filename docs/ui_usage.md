# Local UI Usage

The Gradio UI is a localhost-only operating screen for reviewing the private
lifelog database. It must be launched without `share=True`; the app binds to
`127.0.0.1`.

```bash
python -m personal_lifelog_rag.app.cli ui
```

## Monthly Summary

Use the `Monthly Summary` tab to inspect a month or date range.

- Set `年月` such as `2025-01`, or fill `from date` / `to date`.
- Click `月次要約を表示`.
- Review event/photo/GPS/LINE/call/VLM/OCR counts, title distribution,
  representative days, and representative events.

VLM-only cues are shown as image-analysis candidates. Hidden events are excluded
unless private mode and `include hidden` are enabled.

## Multimodal Search

Use `Multimodal Search` for image/OCR/VLM/embedding/event search.

- Query examples: `ご飯を食べた写真`, `ステージの写真`, `新宿の写真`.
- Backends: `hybrid`, `vlm_sql`, `embedding`, `sql`.
- Results include score, confidence, evidence strength, caption, matched terms,
  cues, related event, thumbnail path, and score components.

Selecting a `media_id` opens the detail panel with thumbnail, caption, OCR
preview, evidence, and review status.

## Search Result Detail

From a selected search result you can mark the corresponding VLM result:

- `Mark accepted`
- `Mark wrong`
- `Hide`
- `Not searchable`
- `Not event usable`

These actions write to VLM overrides and are respected by normal search,
multimodal search, QA, and event generation.

## VLM Review

The `VLM Review / 画像解析レビュー` tab supports:

- month/date/range filtering
- unreviewed and safety flag filters
- food cue and performance/stage filters
- OCR/embedding/event-linked filters
- low-confidence review

Use it to correct captions/tags and decide whether a VLM result should be used
for search or event generation.

## Report Viewer

The `Report Viewer` tab lists local `reports/*.md`, loads Markdown previews, and
can generate a new report with public/private mode and optional examples.

Public mode keeps examples anonymized and does not show exact GPS, raw photo
paths, or full LINE text.

## Safety Notes

- Do not use Gradio `share=True`.
- Keep the UI on `127.0.0.1`.
- Do not expose `data/`, `private_config/`, `private_eval/`, `reports/`, or
  raw photos publicly.
- Review VLM/OCR results before relying on them for important conclusions.

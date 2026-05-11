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

Recommended review workflow:

- Do not review the whole VLM queue.
- Start with top results from important searches such as food, stage, receipts,
  places, and monthly summary outliers.
- Mark only clearly wrong visual interpretations as `wrong`.
- Use `hidden` only for results that are too private for normal browsing.
- Use `not_searchable` when a result should not affect image or multimodal
  search.
- Use `not_event_usable` when a result should not affect event generation.
- Leave low-impact unreviewed results alone until they appear in a useful
  search or QA path.

## Place Review

The `Place Review / 場所レビュー` tab lets you review GPS clusters without
showing exact coordinates by default.

- Filter clusters by date range, status, privacy, category, text, and minimum
  point count.
- Select a cluster to see representative thumbnails, related media, related
  events, and a safe approximate label.
- Create or update a place label with `display_name`, `public_name`, category,
  privacy level, aliases, and notes.
- Link a cluster to a place, mark it accepted/rejected, or reassign places to
  media/events after edits.

Manual labels are preferred in search, QA, monthly summaries, and reports.
Private places are kept out of public report labels.

## Face Review

The `Face Review / 顔検出レビュー` tab is for local face detection and manual
cluster review.

- It shows detection rows, bbox values, detection score, and review status.
- Private local crops can be displayed for review, but they are not public
  report assets.
- Actions are limited to `accepted`, `bad_detection`, and `rejected`.
- The cluster/person section shows a thumbnail grid for face clusters.
- To label a cluster, select it, type a manual name, and click the labeling
  button. If the same display name already exists, that existing person is
  reused so same-name clusters are treated as the same person.
- The LINE speaker link section lets the user manually link a `(chat_id,
  speaker_name)` pair to the selected person and optionally add the speaker name
  as an alias.
- The UI does not infer names, emotions, relationships, or identity from faces.

Unreviewed face detections are not used by normal search, QA, event generation,
or reports. Manual person labels are available only for the narrow LINE speaker
QA path after the user explicitly creates a speaker link. The answer remains
date/count based and does not infer relationships.

After PR70, backend CLI commands can build private `media_people` and
`event_people` links from reviewed face clusters and manual LINE speaker links.
The UI should treat these as private review context; public mode must hide
private person names and face artifacts.

After PR81, Ask, Image Search, Multimodal Search, and Monthly Summary can use
those manually linked person records. Related person columns and score
components are private review signals; public mode hides private names and does
not expose face crops, face embeddings, exact GPS, or raw LINE text.

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

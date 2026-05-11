# personal_lifelog_rag

Local-first personal lifelog search and question-answering application.

This project ingests locally exported photos and LINE chat history text files,
stores them in SQLite, adds local OCR/VLM/embedding analysis, builds
evidence-linked events, and answers natural-language questions over personal
memory data. It is built for private personal data: photos, timestamps,
locations, call records, and chat text should stay on the local machine.

## Project Overview

`personal_lifelog_rag` is a privacy-preserving multimodal RAG app for personal
lifelogs. It combines:

- photo EXIF/GPS metadata
- LINE messages and call-like records
- local OCR
- Qwen3-VL image captions and tags
- Qwen3-VL-Embedding image/text retrieval
- SQLite event and evidence tables
- local QA, image search, monthly summaries, private eval, reports, and review UI

Portfolio summary:

> A local-first multimodal RAG application that integrates personal photos, chat
> exports, location metadata, OCR, and local vision models so past events can be
> searched in natural language without external APIs.

## Features

- Date QA: `qa "2024年12月24日は何していた？"`
- Image-search QA: `qa "ご飯を食べた写真はいつ？"`
- Place QA: `qa "新宿に行ったのはいつ？"`
- Monthly summary: `qa "2025年1月は何していた？"`
- Multimodal search over VLM, embeddings, OCR, LINE, events, and places
- VLM Review UI for accepted/rejected/wrong/hidden/not-searchable controls
- Private eval and eval comparison
- Public/private Markdown report generation
- Month-by-month rollout planning for later analysis expansion

## Quick Start

```bash
cd ~/MyApplication/personal_lifelog_rag
conda activate personal_lifelog_rag
python -m personal_lifelog_rag.app.cli db-check --strict
python -m personal_lifelog_rag.app.cli qa "2025年1月は何していた？"
python -m personal_lifelog_rag.app.cli ui
```

The UI binds to `127.0.0.1` and does not enable Gradio sharing.

## Local Model Setup

Local model runtime is configured outside Git. The current intended roles are:

- Qwen3-VL: caption, tags, visual cues, safety flags
- Qwen3-VL-Embedding: image embeddings, combined-text embeddings, query retrieval

The app must not auto-download models. Use local model paths and
`local_files_only` in private runtime config.

## Key Commands

```bash
python -m personal_lifelog_rag.app.cli qa "ご飯を食べた写真はいつ？"
python -m personal_lifelog_rag.app.cli multimodal-search "ステージの写真" --backend hybrid
python -m personal_lifelog_rag.app.cli generate-report --public --save-json
python -m personal_lifelog_rag.app.cli build-portfolio-html --mode public --check-privacy --force
python -m personal_lifelog_rag.app.cli release-check --version v0.1 --save-manifest
python -m personal_lifelog_rag.app.cli batch-qa --query "2025年1月は何していた？" --query "ステージの写真はいつ？" --save-run
python -m personal_lifelog_rag.app.cli month-plan --month 2025-02
python -m personal_lifelog_rag.app.cli month-run --month 2025-02 --limit 100 --dry-run
```

## Privacy Notice

Do not publish real photos, raw chat exports, exact GPS coordinates, model
runtime config, local databases, backups, private eval files, generated private
reports, or screenshots containing private evidence. Public demos should use
aggregate metrics, anonymized examples, and public report mode.

## Portfolio Docs

- [Portfolio Summary](docs/portfolio_summary.md)
- [System Architecture](docs/system_architecture.md)
- [Demo Scenarios](docs/demo_scenarios.md)
- [Privacy and Safety](docs/privacy_and_safety.md)
- [Location Points and Places](docs/location_places.md)
- [Place Review UI](docs/place_review_ui.md)
- [Face Detection](docs/face_detection.md)
- [Face Embedding and Clustering](docs/face_embedding_clustering.md)
- [Face Review and Manual People Labels](docs/face_review_people.md)
- [LINE Speaker and Person Linking](docs/line_person_linking.md)
- [Person Media and Event Integration](docs/person_event_integration.md)
- [Person and Place QA](docs/person_place_qa.md)
- [Privacy Controls](docs/privacy_controls.md)
- [Person Delete and Export](docs/person_delete_export.md)
- [Private Eval](docs/private_eval.md)
- [Evaluation Summary](docs/evaluation_summary.md)
- [Portfolio HTML](docs/portfolio_html.md)
- [Roadmap](docs/roadmap.md)
- [v0.1 Release](docs/releases/v0.1.md)
- [Reproducibility](docs/reproducibility.md)
- [UI Review Workflow](docs/ui_review_workflow.md)
- [Final Publication Checklist](docs/final_publication_checklist.md)
- [Job Hunting Pitch](docs/job_hunting_pitch.md)
- [Technical Interview Notes](docs/technical_interview_notes.md)
- [ML Learning Takeaways](docs/ml_learning_takeaways.md)
- [ES Self-PR Examples](docs/es_self_pr_examples.md)
- [Interview Demo Script](docs/demo_script_for_interview.md)

## Portfolio Note

This project is useful to discuss as an AI/ML portfolio project because it
shows end-to-end local multimodal engineering: ingestion, model adaptation,
retrieval, safety filtering, human review, evaluation, and reporting.

## Publication Warning

Before sharing any generated report or UI screenshot, manually inspect it for
raw messages, names, exact locations, image paths, and other private details.

## Portfolio HTML

Generate the single-file public portfolio HTML locally:

```bash
python -m personal_lifelog_rag.app.cli build-portfolio-html \
  --output reports/portfolio_public.html \
  --mode public \
  --check-privacy \
  --force
python scripts/check_public_portfolio_safety.py reports/portfolio_public.html
```

The HTML is self-contained, uses inline CSS, avoids external CDN assets, and is
designed for public sharing after privacy review. See
[`docs/portfolio_html.md`](docs/portfolio_html.md) for the generation workflow,
privacy check, and publication checklist.

For person/place privacy operations, see
[`docs/privacy_controls.md`](docs/privacy_controls.md) and
[`docs/person_delete_export.md`](docs/person_delete_export.md). Use dry-run
first, back up the DB before executed changes, and rerun the public audit before
sharing artifacts.

## Privacy Policy

- No external API calls in the MVP.
- No OpenAI API or cloud LLM usage.
- Photos, LINE text, location data, thumbnails, models, and SQLite DBs are not
  tracked by Git.
- Tests use only dummy data under `tests/fixtures/`.
- CLI output prints counts and short answers, not bulk dumps of private data.
- The optional UI binds to `127.0.0.1` by default.

## Setup

```bash
cd ~/MyApplication/personal_lifelog_rag
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For the optional Gradio UI:

```bash
pip install -e ".[ui]"
```

For optional local transformer embeddings:

```bash
pip install -e ".[embeddings]"
```

For optional local OCR or Transformers-based image captions:

```bash
pip install -e ".[ocr]"
pip install -e ".[vlm]"
```

Tesseract itself must be installed locally outside Python. PaddleOCR and
Ollama/llama.cpp can also be used as local backends through environment
variables; no cloud image analysis is used. See `docs/ocr_setup.md` and
`docs/vlm_setup.md` for the privacy-first operating notes. See
`docs/vlm_model_selection.md` for Qwen3-VL and Qwen3-VL-Embedding role
separation and benchmark workflow. See `docs/vlm_prompting_and_safety.md` for
Qwen3-VL prompt templates, safety filtering, and evidence-strength rules. See
`docs/reporting.md` for privacy-preserving Markdown report export. See
`docs/ui_usage.md` for Monthly Summary, Multimodal Search, VLM Review, and
Report Viewer operation. See
`docs/private_eval_20241224.md` for creating a local baseline private eval suite
around an inspected date. See `docs/monthly_rollout.md` for month-by-month
rollout planning from 2025-02 onward.

## Directory Layout

```text
data/raw/photos/      local photos for import
data/raw/line/        exported LINE .txt files
data/processed/       future processed artifacts
data/thumbnails/      generated thumbnails
data/db/              SQLite database
src/                  application code
tests/fixtures/       dummy test data only
```

The default database path is `data/db/lifelog.sqlite`.

## Data Placement

Put local photos in:

```text
data/raw/photos/
```

Put exported LINE talk history files in:

```text
data/raw/line/
```

Supported image extensions in the MVP are `.jpg`, `.jpeg`, and `.png`.
`.heic` is documented as a future extension.

## CLI

Initialize the database:

```bash
python -m personal_lifelog_rag.app.cli init-db
```

Ingest photos:

```bash
python -m personal_lifelog_rag.app.cli ingest-photos --path data/raw/photos
```

Ingest LINE exports:

```bash
python -m personal_lifelog_rag.app.cli ingest-line --path data/raw/line
```

Ask a local extractive question:

```bash
python -m personal_lifelog_rag.app.cli ask "2024年12月24日は何していた？"
```

Classify and route natural-language questions locally:

```bash
python -m personal_lifelog_rag.app.cli classify-query "新宿に行ったのはいつ？"
python -m personal_lifelog_rag.app.cli classify-query "友人と通話した日は？" --json
python -m personal_lifelog_rag.app.cli qa "2024年12月24日は何していた？"
python -m personal_lifelog_rag.app.cli qa "新宿に行ったのはいつ？"
python -m personal_lifelog_rag.app.cli qa "通話した日は？" --json
```

`qa` first runs local rule-based intent classification, then routes to the
existing local path: `ask` for `date_qa`, `search` for place/food/topic/person
queries, structured call search/ranking for `call_activity`, and event/photo
summaries for range or photo-oriented queries. Unsupported queries return
examples of currently supported wording instead of guessing.

Build local embeddings:

```bash
python -m personal_lifelog_rag.app.cli build-embeddings
```

Search local SQLite text records:

```bash
python -m personal_lifelog_rag.app.cli search "新宿"
python -m personal_lifelog_rag.app.cli search "ご飯" --limit 10
python -m personal_lifelog_rag.app.cli search "新宿" --date-from 2024-12-01 --date-to 2024-12-31
python -m personal_lifelog_rag.app.cli search "新宿" --mode actual
python -m personal_lifelog_rag.app.cli search "新宿に行ったのはいつ？" --intent place_visit
python -m personal_lifelog_rag.app.cli search "通話した日は？" --intent call_activity
python -m personal_lifelog_rag.app.cli search "新宿" --json
```

`search` uses local SQLite `LIKE` matching over LINE text, event title/summary/
location, media file names, and optional caption/OCR fields. It groups matches
by date and shows related events, up to five short LINE previews, related photo
counts, confidence, and evidence types. Results are ranked locally with
conservative rule-based labels: `actual_or_likely_action`, `plan_or_candidate`,
`mention_only`, or `unknown`. Use `--mode actual|plan|mention|all` to filter
those labels, and `--intent place_visit|food_activity|call_activity|topic_mention|generic`
to tune the scoring for natural-language questions. No external API or cloud
search is used.

Build and query a structured local index of LINE call logs:

```bash
python -m personal_lifelog_rag.app.cli build-call-index --dry-run
python -m personal_lifelog_rag.app.cli build-call-index --force
python -m personal_lifelog_rag.app.cli call-stats
python -m personal_lifelog_rag.app.cli call-stats --month 2024-12
python -m personal_lifelog_rag.app.cli search-calls --completed --min-duration-sec 600
python -m personal_lifelog_rag.app.cli search-calls --missed --limit 20
python -m personal_lifelog_rag.app.cli search-calls --unanswered --json
```

`build-call-index` reads local `line_messages` and writes only
`line_call_events`. It classifies LINE call logs as `completed`, `missed`,
`unanswered`, `canceled`, or `unknown`, and converts `通話時間 MM:SS` /
`H:MM:SS` into seconds. `--dry-run` does not write to SQLite; `--force`
rebuilds the selected date range. `search "通話"` uses this structured call
summary when available, so completed calls and longer durations can rank above
missed, unanswered, or canceled calls.

Analyze imported images with optional local OCR/VLM backends:

```bash
python -m personal_lifelog_rag.app.cli analyze-images --limit 100
```

Run local OCR over imported photos:

```bash
python -m personal_lifelog_rag.app.cli ocr-images --date 2024-12-24 --dry-run --limit 10
python -m personal_lifelog_rag.app.cli ocr-images --date 2024-12-24 --limit 10 --skip-existing
python -m personal_lifelog_rag.app.cli ocr-images --from 2024-12-01 --to 2024-12-31 --engine tesseract_cli
python -m personal_lifelog_rag.app.cli ocr-diagnostics --config private_config/model_runtime.yaml
python -m personal_lifelog_rag.app.cli ocr-stats
python -m personal_lifelog_rag.app.cli ocr-show --date 2024-12-24 --limit 10
python -m personal_lifelog_rag.app.cli ocr-search "新宿"
```

OCR is optional. If Tesseract or another local OCR engine is not available,
`ocr-images` records `engine_unavailable` and existing search, ask, event, and
UI functions keep working. Search uses `media_ocr` evidence when present, but
CLI/UI previews redact and shorten OCR text because it can contain addresses,
phone numbers, emails, or other private text. See `docs/ocr_setup.md` and
`docs/ocr_engine.md`.

Build local timeline events from LINE, photos, GPS, caption, and OCR evidence:

```bash
python -m personal_lifelog_rag.app.cli build-events --date 2024-12-24
python -m personal_lifelog_rag.app.cli build-events --from 2024-12-01 --to 2024-12-31
python -m personal_lifelog_rag.app.cli build-events --all
python -m personal_lifelog_rag.app.cli build-events --from 2024-12-01 --to 2024-12-31 --dry-run
python -m personal_lifelog_rag.app.cli build-events --from 2024-12-01 --to 2024-12-31 --skip-existing
python -m personal_lifelog_rag.app.cli build-events --all --limit-days 30
```

`build-events` replaces generated event candidates in the target date range so
reruns do not leave stale evidence links behind. It clusters photos by local
date, 90-minute capture gaps, and nearby GPS; clusters LINE by 120-minute gaps;
then merges photo and LINE clusters when their time ranges overlap or are
within 90 minutes. `--dry-run` builds drafts without writing to SQLite.
`--skip-existing` leaves dates that already have events untouched. `--force`
explicitly replaces generated events. Generated rows are marked with
`source=generated` and `generation_method`; future user-edited rows can be
protected with `is_user_edited=1`. `build-events` only writes `events` and
`event_evidence`; it does not modify `media_items` or `line_messages`.

Review generated event coverage:

```bash
python -m personal_lifelog_rag.app.cli event-stats
python -m personal_lifelog_rag.app.cli event-stats --from 2024-12-01 --to 2024-12-31
python -m personal_lifelog_rag.app.cli event-stats --json
python -m personal_lifelog_rag.app.cli list-events --date 2024-12-24
python -m personal_lifelog_rag.app.cli list-events --date 2024-12-24 --with-evidence
python -m personal_lifelog_rag.app.cli list-events --date 2024-12-24 --include-hidden
python -m personal_lifelog_rag.app.cli list-events --from 2024-12-01 --to 2024-12-31 --json
```

`event-stats` shows monthly/day/title counts, confidence buckets, evidence type
counts, and low-confidence events. `list-events --with-evidence` shows at most
five LINE and five photo evidence previews per event, with LINE text shortened
for privacy.

Save manual event corrections without overwriting generated event rows:

```bash
python -m personal_lifelog_rag.app.cli update-event EVENT_ID --title "修正後タイトル"
python -m personal_lifelog_rag.app.cli update-event EVENT_ID --summary "修正後要約"
python -m personal_lifelog_rag.app.cli update-event EVENT_ID --location "新宿駅周辺"
python -m personal_lifelog_rag.app.cli update-event EVENT_ID --tag "旅行" --tag "食事"
python -m personal_lifelog_rag.app.cli update-event EVENT_ID --verified
python -m personal_lifelog_rag.app.cli update-event EVENT_ID --pinned
python -m personal_lifelog_rag.app.cli update-event EVENT_ID --hidden
python -m personal_lifelog_rag.app.cli update-event EVENT_ID --clear-overrides
```

Manual corrections are stored in `event_overrides`. `ask` and `list-events`
prefer override title, summary, and location values; hidden events are excluded
from normal answers; pinned events are shown first for the day; verified events
are labeled as manually checked. Generated `events` and `event_evidence` remain
separate so `build-events` can be rerun without deleting overrides.

Review and operate events in batches:

```bash
python -m personal_lifelog_rag.app.cli review-queue --from 2024-12-01 --to 2024-12-31 --unverified
python -m personal_lifelog_rag.app.cli review-queue --low-confidence 0.5 --line-only
python -m personal_lifelog_rag.app.cli bulk-update-events --event-id EVENT1 --event-id EVENT2 --hidden
python -m personal_lifelog_rag.app.cli bulk-update-events --event-id EVENT1 --unhide
python -m personal_lifelog_rag.app.cli make-eval-case --event-id EVENT_ID
python -m personal_lifelog_rag.app.cli make-eval-case --query "新宿に行ったのはいつ？" --expected-date 2024-12-24
```

`search`, `qa`, and `list-events` exclude hidden events by default. Use
`--include-hidden` only when reviewing hidden records locally. Event override
tags are included in local event search, and pinned/verified events receive a
small ranking boost.

Event builder tuning:

```bash
export PERSONAL_LIFELOG_RAG_EVENT_PHOTO_GAP_MINUTES=90
export PERSONAL_LIFELOG_RAG_EVENT_LINE_GAP_MINUTES=120
export PERSONAL_LIFELOG_RAG_EVENT_MERGE_WINDOW_MINUTES=90
export PERSONAL_LIFELOG_RAG_EVENT_GPS_DISTANCE_METERS=500
```

Inspect one date before judging answer quality:

```bash
python -m personal_lifelog_rag.app.cli inspect-date 2024-12-24
python -m personal_lifelog_rag.app.cli inspect-date 2024-12-24 --limit 50
python -m personal_lifelog_rag.app.cli inspect-date 2024-12-24 --no-snippets
```

`inspect-date` prints counts, time ranges, hourly distributions, limited LINE
and photo samples, and rounded GPS min/max. Use `--no-snippets` when you only
want aggregate diagnostics. If `private_config/places.yaml` exists, registered
place candidates are shown instead of relying only on raw GPS ranges.

Use a fully local place dictionary for GPS-backed events:

```bash
python -m personal_lifelog_rag.app.cli places init-private
python -m personal_lifelog_rag.app.cli places validate --path configs/places.example.yaml
python -m personal_lifelog_rag.app.cli places list --path configs/places.example.yaml
python -m personal_lifelog_rag.app.cli places match --lat 10.0 --lon 20.0 --path configs/places.example.yaml
python -m personal_lifelog_rag.app.cli places redact-preview --path configs/places.example.yaml
python -m personal_lifelog_rag.app.cli cluster-places --from 2024-12-01 --to 2024-12-31 --radius-m 500 --min-points 3
python -m personal_lifelog_rag.app.cli cluster-places --all --output private_config/place_suggestions.yaml
python -m personal_lifelog_rag.app.cli assign-places --date 2024-12-24 --path private_config/places.yaml --dry-run
python -m personal_lifelog_rag.app.cli assign-places --from 2024-12-01 --to 2024-12-31 --path private_config/places.yaml
python -m personal_lifelog_rag.app.cli place-stats
```

Copy `configs/places.example.yaml` to `private_config/places.yaml` and edit it
manually. The dictionary is local-only and uses safe labels such as
`自宅周辺`, `大学周辺`, or `駅周辺`; no reverse geocoding API is called.
`privacy_level=sensitive` with `show_exact_location=false` hides exact
coordinates in CLI display. `cluster-places` suggests candidate entries from
GPS-tagged photos using neutral labels like `candidate_place_001` and
`候補地点001`; it does not infer labels such as home, school, or workplace.
`assign-places` writes matched display names to `events.location_name` without
modifying `media_items` or `line_messages`, and it does not overwrite
`event_overrides.location_name_override`. See `docs/private_places_setup.md`
for the recommended real-data workflow.

For DB-backed location points and reviewed place clusters, use the newer
offline workflow:

```bash
python -m personal_lifelog_rag.app.cli build-location-points --from 2024-12-01 --to 2025-03-31 --dry-run
python -m personal_lifelog_rag.app.cli build-location-points --from 2024-12-01 --to 2025-03-31 --yes
python -m personal_lifelog_rag.app.cli cluster-places --from 2024-12-01 --to 2025-03-31 --eps-meters 100 --min-samples 3 --dry-run
python -m personal_lifelog_rag.app.cli places list-clusters --status unreviewed --limit 20
python -m personal_lifelog_rag.app.cli places show-cluster --cluster-id CLUSTER_ID
python -m personal_lifelog_rag.app.cli places create --name "駅周辺" --public-name "駅周辺" --category station --privacy-level public_label
python -m personal_lifelog_rag.app.cli places link-cluster --place-id PLACE_ID --cluster-id CLUSTER_ID --yes
python -m personal_lifelog_rag.app.cli assign-places --from 2024-12-01 --to 2025-03-31 --dry-run
```

This stores exact coordinates only in the local SQLite DB and uses reviewed
place labels for search, QA, events, monthly summaries, and reports. See
`docs/location_places.md`.

Check SQLite integrity without dumping private text:

```bash
python -m personal_lifelog_rag.app.cli db-check
python -m personal_lifelog_rag.app.cli db-check --json
python -m personal_lifelog_rag.app.cli db-check --strict
```

`db-check` reports counts, duplicate hashes/paths, missing local files,
event-evidence reference problems, and short ID-only anomaly samples.

Run private local evals for answer/event quality:

```bash
python -m personal_lifelog_rag.app.cli private-eval --init-template
python -m personal_lifelog_rag.app.cli eval-private --path configs/private_eval.example.yaml
python -m personal_lifelog_rag.app.cli eval-private --path configs/private_eval.example.yaml --json
python -m personal_lifelog_rag.app.cli eval-private --path private_eval/questions.yaml
python -m personal_lifelog_rag.app.cli eval-private --path private_eval/questions.yaml --json
python -m personal_lifelog_rag.app.cli eval-private --path private_eval/questions.yaml --save-run
python -m personal_lifelog_rag.app.cli eval-private --path private_eval/questions.yaml --case-id date_001
python -m personal_lifelog_rag.app.cli eval-private --path private_eval/questions.yaml --strict
python -m personal_lifelog_rag.app.cli make-private-eval-template --date 2024-12-24 --output private_eval/questions_20241224.yaml
python -m personal_lifelog_rag.app.cli make-private-eval-template --include-people --include-places --include-privacy --output private_eval/questions_people_places.yaml
python -m personal_lifelog_rag.app.cli eval-compare --before eval_outputs/eval_A.json --after eval_outputs/eval_B.json
```

`eval-private` is kept as an alias for `private-eval`. `--path` is an alias for
`--questions`, and `--case-id` runs one case by id. `--save-run` writes a compact
JSON report to `private_eval/runs/eval_YYYYMMDD_HHMMSS.json`; without it, the
command only prints to the terminal. The loader accepts the PR10 `cases:`
mapping, the older `questions:` mapping, or a top-level YAML list.

Supported private eval case types:

- `date_qa`: date extraction, event/evidence presence, forbidden phrases, and activity confidence caps.
- `keyword_search`: top dates, ranking classification, evidence types, score/rank checks, and downrank phrases.
- `query_intent`: natural-language intent and entity checks.
- `routed_qa`: `qa` routing, answer safety, top dates, and search classification checks.
- `call_search`: completed/missed/unanswered/canceled call filters and duration conditions.
- `event_quality`: event count, evidence type, orphan evidence, and low-value event checks.
- `vlm_quality`: VLM success count, engine, forbidden terms, allowed safety flags, and non-empty captions.
- `image_search` / `multimodal_search`: image/VLM/embedding retrieval dates, evidence types, result counts, strength, and overclaim checks.
- `place_assignment`: safe `location_name` checks without exact GPS leakage.
- `event_override`: hidden/pinned/verified and override text checks.
- `place_qa` / `monthly_place_summary`: reviewed place QA without exact coordinate output.
- `person_line_qa` / `person_photo_qa` / `person_place_activity_qa`: manual person-link QA with skip support when no verified person exists.
- `face_workflow_quality` / `line_person_link_quality`: face workflow and manual LINE-person link safety checks.
- `privacy_audit` / `export_privacy`: public artifact and public-redacted export safety checks.

`eval-private` reports summary counts, by-type pass rates, ranking metrics
(`top1 accuracy`, `expected date recall@5`), and safety metrics. `eval-compare`
compares two saved JSON reports and highlights pass/fail deltas, ranking metric
changes, newly failed cases, improved cases, and forbidden phrase changes.

`private_eval/`, `private_config/`, and `eval_outputs/` are ignored by Git.
Reports store counts, pass/fail checks, and short previews rather than full LINE
text or raw personal data. Unsupported eval case types are reported as skipped.

Show DB stats:

```bash
python -m personal_lifelog_rag.app.cli stats
```

Back up the SQLite DB before all-period event rebuilds:

```bash
python -m personal_lifelog_rag.app.cli backup-db
python -m personal_lifelog_rag.app.cli backup-db --label before_build_all
python -m personal_lifelog_rag.app.cli backup-db --label before_build_all --output-dir backups
```

Backups are written under `backups/` by default as
`lifelog_<label>_YYYYMMDD_HHMMSS.sqlite` or
`lifelog_backup_YYYYMMDD_HHMMSS.sqlite`. `backups/` is ignored by Git.

Recommended all-period event rebuild flow:

```bash
python -m personal_lifelog_rag.app.cli backup-db --label before_build_all
python -m personal_lifelog_rag.app.cli build-events --all --dry-run
python -m personal_lifelog_rag.app.cli build-events --all --backup --check-after
python -m personal_lifelog_rag.app.cli db-check --strict
python -m personal_lifelog_rag.app.cli event-stats
python -m personal_lifelog_rag.app.cli search-snapshot --query "新宿" --query "ご飯" --query "通話" --save
```

`build-events --all` derives target dates from `media_items` and
`line_messages`. `--dry-run` previews target days and planned events without
writing to SQLite. `--skip-existing` leaves days that already have events
untouched. `--force` explicitly rebuilds generated events for target days.
`--limit-days 30` is useful for a small safety run before processing the whole
database. CLI execution checks that `media_items` and `line_messages` counts are
unchanged before and after event building.

For a one-command local safety workflow:

```bash
python -m personal_lifelog_rag.app.cli rebuild-events-safe --all
python -m personal_lifelog_rag.app.cli rebuild-events-safe --all --limit-days 30
```

This backs up the DB, runs a dry-run preview, builds events, runs strict DB
checks, prints event stats, and saves a compact search snapshot.

Capture search snapshots for ranking comparison:

```bash
python -m personal_lifelog_rag.app.cli search-snapshot --query "新宿" --query "ご飯" --query "通話"
python -m personal_lifelog_rag.app.cli search-snapshot --query "新宿" --limit 10
python -m personal_lifelog_rag.app.cli search-snapshot --query "新宿" --json
python -m personal_lifelog_rag.app.cli search-snapshot --query "新宿" --save
```

Saved snapshots go to `eval_outputs/search_snapshot_YYYYMMDD_HHMMSS.json`.
They contain compact counts, confidence labels, related event summaries, and at
most five shortened LINE samples per date. `eval_outputs/` is ignored by Git.

Launch the optional localhost UI:

```bash
python -m personal_lifelog_rag.app.cli ui
```

The UI binds to `127.0.0.1` by default and does not enable Gradio sharing.
You can choose another local port:

```bash
python -m personal_lifelog_rag.app.cli ui --port 7861
```

If Gradio is not installed, the command prints an installation hint instead of
starting a server.

## UI

The local UI is localhost-only and includes operational tabs for:

- Home / Stats: shows the DB path, registered photo count, registered LINE
  message count, and registered event count.
- Ingest: imports a photo folder and/or LINE export folder from local paths.
  It also includes a local OCR runner and OCR stats panel.
- Ask: accepts a question, shows the extractive answer, evidence LINE rows, and
  evidence photo rows.
- Monthly Summary, Image Search, Multimodal Search, Report Viewer, VLM Review,
  Place Review, and Face Review.
- Place Review: shows place clusters without exact GPS by default, lets you
  create/update place labels, link clusters, manage `public_name`, category,
  privacy level, aliases, and reassign media/events.
- Face Review: shows local face detection bbox rows and optional private crops
  for review only. It does not identify people or use unreviewed faces in
  search, QA, event generation, or reports.
- Events / Timeline: loads one date's generated events, shows evidence counts,
  provides a Review Queue with confidence/modality/status filters, displays
  short LINE evidence and thumbnail-only photo evidence, and saves manual
  title/summary/location/tags/verified/pinned/hidden overrides.
- The same tab has quick actions, bulk update by event id, location dictionary
  choices from `private_config/places.yaml` when present, and a helper that
  generates private-eval YAML fragments.

The UI is intended for local use only. Do not bind it to a public interface.

## Local Embeddings

Phase 2 adds SQLite-backed semantic search. It embeds `line_messages.text` and
available media text fields. `media_items` has reserved `caption` and
`ocr_text` columns for future OCR/image-caption pipelines.

No external API is used. By default, the app tries to load the lightweight
multilingual sentence-transformers model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Model loading is local-only. If the model or `sentence-transformers` package is
not available, the app falls back to a deterministic local hashing adapter so
existing MVP commands keep working.

Environment variables:

```bash
export PERSONAL_LIFELOG_RAG_EMBEDDING_BACKEND=sentence-transformers
export PERSONAL_LIFELOG_RAG_EMBEDDING_MODEL=/local/path/to/model
```

For a no-download fallback:

```bash
python -m personal_lifelog_rag.app.cli build-embeddings --backend hash
```

The adapter boundary is intentionally small so a local Qwen3-Embedding model can
be used later by pointing `PERSONAL_LIFELOG_RAG_EMBEDDING_MODEL` at a local
model path. The user-facing `search` CLI currently uses local SQLite keyword
search; vector search code remains available internally for future hybrid
ranking.

## Local OCR / VLM

Local OCR is implemented as an optional adapter layer. The dedicated
`media_ocr` table stores OCR text, redacted preview text, engine/language
metadata, confidence, block JSON, status, and analysis version. Successful OCR
is mirrored to `media_items.ocr_text` so existing local search and event
generation can use it.

Local VLM image analysis is also optional. The dedicated `media_vlm` table
stores cautious captions, scene/object/activity tags, location and food cues,
safety flags, engine/model metadata, status, and analysis version. Successful
VLM captions are mirrored to `media_items.caption`, and compact tag metadata is
mirrored to `media_items.analysis_json` so local search and event generation can
use it without requiring the VLM engine at answer time.

VLM results are search hints, not facts. The app uses "可能性" wording and does
not infer relationships, emotions, names, occupation, health, religion,
politics, or other sensitive traits from photos.

By default VLM analysis is disabled. OCR can be run with `ocr-images`; VLM can
be run with `analyze-images`. If the selected local engine is unavailable, the
app records `engine_unavailable` and existing MVP features keep working.

Tesseract OCR:

```bash
pip install -e ".[ocr]"
python -m personal_lifelog_rag.app.cli ocr-images --date 2024-12-24 --engine tesseract_cli --languages jpn+eng
```

See `docs/ocr_engine.md` for local Tesseract diagnostics, `ocr-search`, and
redaction behavior.

PaddleOCR:

```bash
export PERSONAL_LIFELOG_RAG_OCR_BACKEND=paddleocr
python -m personal_lifelog_rag.app.cli analyze-images --limit 100
```

Ollama vision model on localhost:

```bash
export PERSONAL_LIFELOG_RAG_VLM_BACKEND=ollama
export PERSONAL_LIFELOG_RAG_VLM_MODEL=llava
export PERSONAL_LIFELOG_RAG_OLLAMA_URL=http://127.0.0.1:11434
python -m personal_lifelog_rag.app.cli analyze-images --date 2024-12-24 --engine ollama --limit 10
```

Transformers image-to-text model from a local path:

```bash
export PERSONAL_LIFELOG_RAG_VLM_BACKEND=transformers
export PERSONAL_LIFELOG_RAG_VLM_MODEL=/local/path/to/image-to-text-model
python -m personal_lifelog_rag.app.cli analyze-images --date 2024-12-24 --engine transformers --limit 10
```

llama.cpp is represented as a localhost-only adapter boundary for future
model-specific wiring. Endpoints must bind to `127.0.0.1`, `localhost`, or
`::1`; public network endpoints are rejected.

Useful VLM commands:

```bash
python -m personal_lifelog_rag.app.cli vlm-prompt --template lifelog_structured_tags_v1
python -m personal_lifelog_rag.app.cli vlm-safety-check --text "彼女と楽しそうにご飯を食べている写真です"
python -m personal_lifelog_rag.app.cli analyze-images --date 2024-12-24 --dry-run --limit 5
python -m personal_lifelog_rag.app.cli analyze-images --date 2024-12-24 --engine fake --prompt-template lifelog_structured_tags_v1 --limit 5 --force
python -m personal_lifelog_rag.app.cli vlm-stats
python -m personal_lifelog_rag.app.cli vlm-show --date 2024-12-24 --limit 10
python -m personal_lifelog_rag.app.cli image-search "ラーメン"
```

Qwen-style model selection benchmarks keep captioning and retrieval separate:

```bash
python -m personal_lifelog_rag.app.cli vlm-model-info
python -m personal_lifelog_rag.app.cli vlm-model-info --config private_config/model_runtime.yaml
python -m personal_lifelog_rag.app.cli benchmark-vlm --cases configs/vlm_benchmark.example.yaml --engine fake --json
python -m personal_lifelog_rag.app.cli benchmark-image-embedding --cases configs/vlm_benchmark.example.yaml --engine fake --json
python -m personal_lifelog_rag.app.cli benchmark-qwen-multimodal --cases configs/vlm_benchmark.example.yaml --engine fake --save
```

Qwen3-VL is evaluated for caption/tags/event cues. Qwen3-VL-Embedding is
evaluated for text-to-image retrieval and future indexing. Real model settings
belong in `private_config/model_runtime.yaml`; benchmark images and run outputs
belong in ignored `private_eval/` or `eval_outputs/` paths.

Build and search local multimodal embeddings:

```bash
python -m personal_lifelog_rag.app.cli build-image-embeddings --date 2024-12-24 --engine fake --limit 10 --force
python -m personal_lifelog_rag.app.cli build-text-embeddings --date 2024-12-24 --engine fake --type combined_text --force
python -m personal_lifelog_rag.app.cli embedding-stats
python -m personal_lifelog_rag.app.cli multimodal-search "ご飯を食べた写真" --backend hybrid --limit 10
python -m personal_lifelog_rag.app.cli image-search "カフェ" --backend hybrid --limit 10
python -m personal_lifelog_rag.app.cli qa "ご飯を食べた写真はいつ？"
```

`media_embeddings` stores local image/caption/OCR/combined-text vectors. Hybrid
search reranks embedding candidates with OCR, VLM captions/tags, LINE mentions,
events, places, and manual overrides. Embedding-only and VLM-only results are
kept as weak candidates and should not be treated as definitive facts. See
`docs/multimodal_search.md`.

Heavy OCR/VLM/embedding runs can be planned and resumed through the analysis
job manager:

```bash
python -m personal_lifelog_rag.app.cli analysis-plan --type vlm --date 2024-12-24
python -m personal_lifelog_rag.app.cli analysis-run --type vlm --date 2024-12-24 --engine fake --limit 5 --save-report
python -m personal_lifelog_rag.app.cli analysis-status --recent 5
python -m personal_lifelog_rag.app.cli storage-stats
```

See `docs/analysis_job_management.md` for resume/retry/version-change cleanup
and DB maintenance workflow.

## Current Capabilities

- Parse common LINE txt date headers and messages.
- Join multiline LINE messages.
- Classify simple special LINE messages: image, video, sticker, file, system,
  unknown, and text.
- Generate stable `chat_id` and `message_id`.
- Ingest `.jpg`, `.jpeg`, and `.png` with Pillow.
- Extract EXIF `DateTimeOriginal`, GPS, camera model, width, and height when
  available.
- Fall back to file modified time when EXIF capture time is missing.
- Detect duplicate images by sha256 file hash.
- Generate local thumbnails under `data/thumbnails/`.
- Parse simple date expressions such as `2024-12-24`, `2024/12/24`,
  `2024年12月24日`, and `12月24日`.
- Build extractive answers from local SQLite records without an LLM.
- Classify natural-language queries into local intents and route them through
  `qa` without external APIs.
- Search local SQLite records by keyword across LINE, events, and media
  metadata.
- Build structured LINE call logs, distinguish completed/missed/unanswered/
  canceled calls, and summarize call duration locally.
- Validate local place dictionaries, cluster GPS photos into candidate places,
  and assign safe place names to generated events without reverse geocoding.
- Build local SQLite text embeddings and media embeddings for semantic/hybrid
  image search.
- Store local OCR/caption outputs and per-image analysis JSON when optional
  local engines are configured.
- Search image content with local OCR/VLM metadata and optional local
  Qwen3-VL-Embedding style vectors.
- Build rule-based event candidates from photo clusters, LINE clusters, nearby
  GPS, location/activity words, and caption/OCR text.
- Save event evidence links with `photo` and `line` evidence types.

## Tests

```bash
pytest
```

The tests create temporary databases and dummy images. They do not use real
photos or real LINE exports.

## Future Extensions

- HEIC support.
- Video metadata extraction.
- OCR for screenshots and photo text.
- Model-specific llama.cpp VLM integration.
- Higher quality local embedding models such as Qwen3-Embedding.
- Local-only LLM adapter implementations for Ollama, llama.cpp, or
  Transformers.
- Better event clustering and place summarization with richer local models.
- UI controls for place dictionary validation, cluster suggestions, and event
  place assignment.
- Richer thumbnail/photo browsing in the UI.

## Notes

This app is intentionally conservative. Heavy image understanding, OCR, vector
DBs, VLMs, and local LLM model loading are planned as later extensions rather
than MVP requirements.

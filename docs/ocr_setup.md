# Local OCR Setup

This app only supports local OCR. Do not use cloud OCR services such as Google
Vision API or any API that uploads private photos, screenshots, GPS-linked
images, receipts, tickets, or LINE-related content.

## What OCR Adds

OCR extracts text visible in imported photos and stores it in SQLite. Useful
examples include signs, station names, shop names, receipts, tickets, menus,
screenshots, album names, event names, and dates.

OCR results are saved in `media_ocr`:

- `ocr_text`: raw OCR text kept locally in SQLite
- `ocr_text_redacted`: preview text with obvious email, phone, and long number
  patterns masked
- `ocr_engine`, `ocr_languages`, `confidence`, `blocks_json`
- `status`: `pending`, `success`, `skipped`, `failed`, `no_text`, or
  `engine_unavailable`

Successful OCR is mirrored to `media_items.ocr_text` so existing local search
and event generation can use it.

## Engines

Built-in adapters:

- `noop`: safe no-op
- `fake`: test-only deterministic engine
- `tesseract_cli`: calls the local `tesseract` command
- `pytesseract`: optional Python wrapper

Optional dependencies:

```bash
pip install -e ".[ocr]"
```

`pytesseract` still requires the Tesseract system package and language data to
be installed locally. This project does not download OCR models or call any
external service.

## Commands

Check coverage:

```bash
python -m personal_lifelog_rag.app.cli ocr-stats
python -m personal_lifelog_rag.app.cli ocr-stats --from 2024-12-01 --to 2024-12-31
```

Preview a run without writing to SQLite:

```bash
python -m personal_lifelog_rag.app.cli ocr-images --date 2024-12-24 --dry-run --limit 10
```

Run OCR with safe skip behavior:

```bash
python -m personal_lifelog_rag.app.cli ocr-images --date 2024-12-24 --limit 10 --skip-existing
python -m personal_lifelog_rag.app.cli ocr-images --from 2024-12-01 --to 2024-12-31 --engine tesseract_cli --languages jpn+eng
```

Inspect compact results:

```bash
python -m personal_lifelog_rag.app.cli ocr-show --date 2024-12-24 --limit 10
python -m personal_lifelog_rag.app.cli ocr-show MEDIA_ID --full
```

`--full` is the only mode intended to show full OCR text. Default output is
shortened and redacted.

## Search and Events

`search` and `qa` include OCR text when `media_ocr` rows exist:

```bash
python -m personal_lifelog_rag.app.cli search "新宿"
python -m personal_lifelog_rag.app.cli qa "新宿の看板が写っている日は？"
```

OCR evidence is shown as short photo-linked previews. OCR can also lightly
support event summaries, but it is treated as noisy evidence. The app should not
claim a location, activity, relationship, feeling, or person identity from OCR
alone.

## Privacy Notes

- OCR text may contain addresses, phone numbers, emails, account names, receipt
  details, ticket IDs, or private notes.
- Do not commit `data/`, `ocr_outputs/`, `eval_outputs/`, `private_eval/`, or
  `private_config/`.
- CLI and UI previews use redacted or shortened text by default.
- Keep Gradio bound to `127.0.0.1`; never use public sharing for private photo
  analysis.

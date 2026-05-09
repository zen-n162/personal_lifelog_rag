# Local OCR Engine

This app uses local OCR only. It does not call cloud OCR, reverse image APIs, or
external services. OCR text can contain private information, so command output
uses redacted and shortened previews by default.

## Engines

- `tesseract_cli`: runs the local `tesseract` command. Use this first for
  Japanese/English OCR when `jpn` and `eng` traineddata are installed.
- `fake`: deterministic test engine. Do not use it for production data.
- `noop`: skips OCR without failing.
- `paddleocr_local`: optional skeleton. It never downloads models
  automatically; missing dependencies report `engine_unavailable`.

## Config

Add local OCR settings to `private_config/model_runtime.yaml` or app config:

```yaml
ocr:
  engine: "tesseract_cli"
  languages: "jpn+eng"
  tesseract_cmd: "tesseract"
  psm: 6
  oem: 1
  max_text_length: 5000
  redact_sensitive: true
  local_only: true
```

## Diagnostics

```bash
python -m personal_lifelog_rag.app.cli ocr-diagnostics --config private_config/model_runtime.yaml
```

This checks the local command path, `tesseract --version`,
`tesseract --list-langs`, `jpn`/`eng` availability, and optional PaddleOCR
imports.

## Run OCR

```bash
python -m personal_lifelog_rag.app.cli ocr-images --date 2024-12-24 --engine tesseract_cli --limit 10 --dry-run
python -m personal_lifelog_rag.app.cli ocr-images --date 2024-12-24 --engine tesseract_cli --limit 10 --force
python -m personal_lifelog_rag.app.cli ocr-stats --from 2024-12-01 --to 2024-12-31
python -m personal_lifelog_rag.app.cli ocr-show --date 2024-12-24 --limit 10 --show-errors
```

`--dry-run` does not write to the DB. If the engine is missing, rows are marked
`engine_unavailable` instead of crashing the batch.

## Search

```bash
python -m personal_lifelog_rag.app.cli ocr-search "新宿"
python -m personal_lifelog_rag.app.cli search "新宿"
python -m personal_lifelog_rag.app.cli qa "レシートの写真はいつ？"
```

OCR evidence is treated as a local text cue. It can support image search and
event summaries, but OCR-only evidence should remain a candidate, not a
definitive claim.

## Redaction

Preview output redacts obvious sensitive tokens:

- email addresses
- phone numbers
- URLs
- postal codes
- long numeric IDs
- simple address-like strings

The original OCR text may be stored locally in SQLite for search, but UI/CLI
previews should use the redacted field.

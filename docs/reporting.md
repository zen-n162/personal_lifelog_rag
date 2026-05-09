# Reporting

`generate-report` creates a local Markdown evaluation report that can be used
for research notes or portfolio preparation without exposing raw personal data.

## Privacy Model

The default mode is public. Public reports:

- omit LINE message bodies
- hide exact GPS coordinates
- hide file paths
- redact or shorten media IDs
- generalize dates where practical
- generalize location names to `PLACE_1`, `PLACE_2`, or `SENSITIVE_PLACE`
- avoid embedding real photos or thumbnails

Private reports are still local-only and still avoid raw GPS, file paths, full
LINE text, and embedded photos by default. They are meant for local debugging,
not publication.

## Usage

```bash
python -m personal_lifelog_rag.app.cli generate-report --public --no-examples
python -m personal_lifelog_rag.app.cli generate-report --from 2024-12-01 --to 2024-12-31 --public --include-examples --save-json
python -m personal_lifelog_rag.app.cli generate-report --eval-path private_eval/questions.yaml --public --save-json
```

Default output:

```text
reports/lifelog_rag_eval_YYYYMMDD_HHMMSS.md
```

With `--save-json`, a compact machine-readable summary is saved next to the
Markdown report.

## Included Sections

- overview
- system architecture
- privacy and safety design
- dataset summary
- event generation summary
- search / QA evaluation
- OCR / VLM / embedding evaluation
- anonymized example queries
- error analysis
- strengths
- limitations
- roadmap

## Private Eval Integration

You can include a saved eval JSON:

```bash
python -m personal_lifelog_rag.app.cli generate-report --eval-run eval_outputs/eval_xxx.json --public
```

Or run a private eval file during report generation:

```bash
python -m personal_lifelog_rag.app.cli generate-report --eval-path private_eval/questions.yaml --public --save-json
```

The report only includes summary metrics, by-type pass rates, ranking metrics,
and safety metrics. It does not include full answers or raw LINE evidence.

## Do Not Publish

Do not publish:

- `data/`
- `private_config/`
- `private_eval/`
- `eval_outputs/`
- `reports/` if it contains private reports
- raw SQLite DB files
- real photos or thumbnails
- exact GPS coordinates
- full LINE exports

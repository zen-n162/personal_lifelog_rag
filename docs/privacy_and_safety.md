# Privacy and Safety

## Local-Only Design

This project is designed for private lifelog data. The core operating rule is:
photos, chat exports, GPS metadata, OCR text, VLM outputs, embeddings, and
evaluation artifacts stay on the local machine.

The system does not require:

- OpenAI API
- cloud OCR
- cloud VLM
- cloud embedding API
- reverse geocoding API

## Files That Must Not Be Published

Do not publish:

- local photo or chat exports
- local SQLite databases
- generated thumbnails
- model weights
- private configuration files
- private eval files
- generated reports that include private details
- backups
- evaluation outputs

The public repository should contain source code, dummy examples, tests with
synthetic data, and documentation only.

## Location Privacy

GPS coordinates are sensitive. The app avoids exposing exact coordinates in
answers and UI. Place names can be manually assigned through a private place
dictionary, but the app does not use external reverse geocoding.

Sensitive places should be displayed as broad names such as "near a registered
place" or "sensitive place" rather than exact coordinates.

## VLM Safety Filter

Qwen3-VL output is filtered before use. The prompt and safety layer prohibit:

- identifying people
- guessing names
- inferring relationships such as partner, family, friend, or coworker
- inferring emotions
- inferring age, health, disability, religion, politics, nationality,
  sexuality, or other sensitive traits
- claiming that an image proves an activity

If people are present, the output should only record a generic safety flag such
as `people_present` and, when visually obvious, a rough count.

## Evidence Strength

The ranking system distinguishes weak, medium, and strong evidence:

- VLM-only or embedding-only evidence is weak.
- VLM plus event, OCR, or LINE evidence can be medium.
- Multiple independent evidence types, or human verification, can be strong.

Even strong evidence is phrased cautiously in user-facing answers.

## OCR Redaction

OCR can capture private text such as phone numbers, email addresses, URLs,
long numeric IDs, postal-code-like strings, and address-like fragments. CLI/UI
preview paths redact or shorten OCR text. Raw OCR text is a local DB artifact and
should not be copied into public material.

## Report Modes

Public report mode should:

- avoid raw LINE text
- avoid exact GPS
- avoid file paths
- avoid real names
- avoid photos or thumbnails by default
- anonymize example queries and evidence descriptions

Private report mode can include more operational detail for local inspection,
but should still avoid bulk raw text and exact coordinates.

## Human Review

VLM Review lets the user mark image analysis results as:

- accepted
- rejected
- wrong
- hidden
- not searchable
- not usable for event generation

Rejected, wrong, hidden, and not-searchable results are excluded from normal
search. Not-event-usable results are excluded from event generation.

## Publication Checklist

Before using the project in a portfolio:

1. Use public report mode.
2. Check generated Markdown manually.
3. Remove raw chat, exact GPS, file paths, and real names.
4. Do not include screenshots that show private images or message text.
5. Describe results as aggregate metrics and anonymized examples.


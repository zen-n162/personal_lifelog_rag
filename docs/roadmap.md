# Roadmap

## Near Term

### OCR Engine Expansion

Tesseract CLI is available as the first local OCR engine. Next steps:

- improve Japanese/English OCR preprocessing
- add confidence-aware OCR snippets
- review PaddleOCR as an optional local backend
- add OCR-focused private eval cases

### JSON Parse Recovery

Qwen3-VL can occasionally return almost-valid JSON. The project already has a
repair path, but the next iteration should:

- classify repair causes
- add richer retry stats
- update VLM quality eval to allow expected repair flags
- review repaired captions in the UI

### UI Improvements

The Gradio UI now supports monthly summaries, multimodal search, report
viewing, event review, and VLM review. Next UI work:

- event split/merge editing
- side-by-side before/after rebuild diff
- better thumbnail grids
- keyboard-friendly review actions
- private eval case generation from search results

## Month-by-Month Rollout

The rollout CLI supports planning and dry-run execution for each month. The
recommended process is:

1. run `month-plan`
2. run `month-run --dry-run`
3. execute with conservative VLM and embedding limits
4. run strict DB checks
5. review monthly summary and multimodal search
6. add or update private eval cases

## Model Runtime Improvements

Qwen3-VL and Qwen3-VL-Embedding are large local models. Planned improvements:

- reduce repeated model loading in interactive workflows
- cache query embedding model within batch runs
- expose runtime diagnostics in the UI
- record model/runtime versions in reports
- add memory and latency summaries

## Retrieval and Ranking

Hybrid ranking currently combines embeddings, VLM tags, OCR, LINE, events,
places, and overrides. Future improvements:

- learn reranking weights from private eval and review outcomes
- improve actual-action versus mention-only classification
- use OCR and places as stronger confirmation signals
- keep VLM-only and embedding-only evidence conservative

## Vector Index Migration

SQLite BLOB embeddings are sufficient for the current prototype. At larger
scale, evaluate:

- FAISS
- Qdrant
- Chroma

The repository layer should stay abstract so the retrieval backend can change
without rewriting the QA stack.

## Active Learning

Human review data can improve ranking:

- accepted VLM outputs can boost similar results
- wrong/rejected outputs can train penalties
- verified events can become private eval candidates
- hidden/not-searchable items can protect normal search paths

## Reporting

Current reports are Markdown/JSON. Future portfolio outputs:

- HTML report export
- PDF report export
- sanitized diagrams
- anonymized demo screenshots
- one-page project brief for job applications

## Product-Level Gaps

- event split/merge UI
- better schedule/calendar import
- richer place dictionary maintenance
- long-running job dashboard
- more robust OCR engine diagnostics
- month-over-month evaluation dashboard


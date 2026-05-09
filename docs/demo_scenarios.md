# Demo Scenarios

These demos are written for a local environment with private data already
ingested. The examples below are safe presentation scenarios: they describe
evidence types and system behavior without exposing raw chat text, exact GPS, or
real photos.

## Demo 1: Date QA

```bash
python -m personal_lifelog_rag.app.cli qa "2024年12月24日は何していた？"
```

What to show:

- The query is classified as date-oriented QA.
- The answer aggregates generated events for the date.
- Evidence can include LINE, photos, GPS presence, OCR, and VLM cues.
- The answer avoids overclaiming and keeps uncertain activities at medium or
  lower confidence.

Presentation note:

> This demonstrates evidence-backed lifelog QA. The system does not need a cloud
> LLM; it retrieves and summarizes local evidence.

## Demo 2: Image Search QA

```bash
python -m personal_lifelog_rag.app.cli qa "ご飯を食べた写真はいつ？"
```

What to show:

- The query routes to multimodal image search.
- Qwen3-VL captions/tags provide food-related visual cues.
- Qwen3-VL-Embedding retrieves candidate photos.
- Hybrid ranking combines visual match, VLM cues, related events, OCR, and LINE
  evidence.
- The answer says "食事または料理の可能性" rather than claiming certainty.

Presentation note:

> The embedding model finds candidates, while the VLM result explains why they
> match.

## Demo 3: Place QA

```bash
python -m personal_lifelog_rag.app.cli qa "新宿に行ったのはいつ？"
```

What to show:

- Place-like queries route to place visit search.
- The system separates actual/likely actions from mentions or plans.
- GPS-backed photos and event evidence can strengthen the result.
- Sensitive place coordinates are not printed.

Presentation note:

> The ranking tries to distinguish "mentioned a place" from "likely visited a
> place" without reverse geocoding.

## Demo 4: Monthly Summary

```bash
python -m personal_lifelog_rag.app.cli qa "2025年1月は何していた？"
```

What to show:

- The query routes to monthly summary.
- The summary aggregates event counts, title distribution, photo counts, GPS
  photo counts, LINE counts, call counts, VLM coverage, OCR coverage, confidence
  distribution, and representative days.
- The output is a month-level trend summary, not just the first few events.

Presentation note:

> This demo shows the project as a memory analytics system, not only a keyword
> search tool.

## Demo 5: Stage and Performance Photo Search

```bash
python -m personal_lifelog_rag.app.cli qa "ステージの写真はいつ？"
```

Related variants:

```bash
python -m personal_lifelog_rag.app.cli qa "パフォーマンスっぽい写真はいつ？"
python -m personal_lifelog_rag.app.cli qa "ダンスの写真はいつ？"
```

What to show:

- Japanese visual queries expand into English VLM tags such as stage,
  performance, theater, and dancing.
- Multimodal search uses VLM tags and embedding similarity.
- Ranking requires visual match so same-day LINE or event text alone does not
  dominate.

Presentation note:

> This shows query expansion between Japanese user questions and English visual
> model tags.

## UI Demo

```bash
python -m personal_lifelog_rag.app.cli ui
```

Suggested flow:

1. Open Monthly Summary.
2. Search for a food or stage photo in Multimodal Search.
3. Open result detail.
4. Mark a VLM result accepted, wrong, hidden, or not searchable.
5. Open Report Viewer and show a public report preview.

The UI must remain bound to localhost, with sharing disabled.


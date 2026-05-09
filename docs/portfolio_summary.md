# Portfolio Summary

## What This Project Is

`personal_lifelog_rag` is a local-first multimodal lifelog RAG application. It
integrates personal photos, exported chat history, GPS metadata, call records,
OCR text, local VLM captions, and local multimodal embeddings into a SQLite
memory database that can be searched with natural language.

Short portfolio description:

> I developed a privacy-preserving multimodal RAG application that integrates
> personal photos, chat logs, timestamps, location metadata, and local vision
> models, enabling natural-language search over past events without sending
> private data to external APIs.

## Problem

Personal memories are scattered across photos, chat messages, timestamps,
places, screenshots, call logs, and camera metadata. Cloud photo search and
LLM-based assistants are convenient, but they are often unsuitable for private
life data because they may require external processing.

This project explores a local alternative: build a private memory database on
the user's machine, generate evidence-backed events, and answer questions with
cautious wording.

## Main Features

- Photo EXIF/GPS ingestion
- LINE export ingestion
- SQLite local database
- Event generation with evidence links
- Date QA, place QA, image-search QA, and monthly summaries
- Local OCR table and OCR search path
- Qwen3-VL image captions/tags/cues
- Qwen3-VL-Embedding image and combined-text retrieval
- Hybrid multimodal ranking across VLM, embedding, OCR, LINE, events, and places
- Private eval and before/after regression checks
- Gradio UI for event review, VLM review, monthly summary, multimodal search,
  and report viewing
- Public/private report generation with redaction

## Technical Stack

- Python
- SQLite
- Gradio
- Pillow / EXIF metadata extraction
- Local OCR adapters
- Qwen3-VL for image understanding
- Qwen3-VL-Embedding for text-to-image retrieval
- Rule-based query intent classification
- Local evaluation harness
- Markdown/JSON report generation

## Qwen Model Roles

Qwen3-VL is used to turn images into observable text signals:

- caption
- scene tags
- object tags
- activity tags
- food cues
- location cues
- safety flags

Qwen3-VL-Embedding is used for retrieval:

- image embeddings
- combined OCR/caption text embeddings
- text query embeddings
- candidate image retrieval before hybrid reranking

The two models are intentionally not treated as substitutes. The VLM explains
images; the embedding model retrieves candidates.

## Why Local-Only Matters

The app handles photos, location metadata, call records, and chat exports. The
design keeps processing local by default:

- no OpenAI API
- no cloud OCR
- no cloud VLM
- no cloud embedding API
- local DB and local model execution
- ignored private data and model artifacts

This makes the project suitable as a privacy-aware AI portfolio project rather
than a cloud demo with private data risks.

## Future Development

The main next steps are:

- improving OCR coverage
- expanding monthly rollout safely
- building event split/merge UI
- adding active learning from human review
- migrating vector retrieval to FAISS, Qdrant, or Chroma when scale requires it
- exporting polished PDF/HTML reports for portfolio presentation


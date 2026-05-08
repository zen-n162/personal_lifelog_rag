# AGENTS.md

This repository contains a local-first personal lifelog search and question-answering application.

The app ingests locally exported photos and LINE chat histories, extracts timestamps, EXIF GPS, metadata, and message contents, then builds a searchable personal timeline. The final goal is to answer questions such as:

- "What was I doing on 2024-12-24?"
- "When did I go to Shinjuku?"
- "When was the last time I met a specific person?"
- "Summarize my trips from last summer."
- "Find days when I ate ramen."

This app handles highly private personal data, including photos, locations, timestamps, and private messages. Privacy and local-only execution are the most important requirements.

---

## 1. Repository path

The expected project path is:

```bash
~/MyApplication/personal_lifelog_rag/


---

## 21. Conda environment policy

This project must be developed inside a dedicated Conda environment.

Do not install dependencies into the `base` environment.

The default Conda environment name is:

```bash
personal_lifelog_rag

キリのいいところでGit更新はしてください。

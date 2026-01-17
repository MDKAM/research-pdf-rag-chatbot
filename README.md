---
title: Pdf Rag Chatbot
emoji: 📄
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.3.0
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# Research PDF RAG Chatbot (RAG + FAISS + Eval) — Hugging Face Spaces

A fast, end-to-end **PDF RAG chatbot** deployed on **Hugging Face Spaces**.

**Keywords:** RAG, embeddings, FAISS vector DB, retrieval, citations, evaluation, Gradio, Hugging Face, LLM API (Groq/Gemini), rate limits, guardrails.

---

## What this app does

* **Upload PDF(s)** *or* **paste text**
* Extract → chunk → embed → store in **FAISS**
* Ask questions → retrieve top-k chunks → generate answers with **citations** (`[chunk_id]` + page ranges)
* Run a small **golden-set evaluation** to sanity-check retrieval + answers

---

## How to use the app (tabs)

### 1) Upload PDF(s)

Upload one or more PDFs, then click **Extract text**.

* The app extracts text per page and shows a preview + stats (pages, characters, limits).
* If the PDF has many pages, extraction is capped by the guardrails (see below).

### 2) Paste text

Paste any text (paper abstract, notes, copied section of a PDF), then click **Use this text**.

* This is useful when you don’t have a PDF file or want to test quickly.
* The pasted text is treated as a single document (`pasted_text`) with page 1.

### 3) Chunking

Configure chunking parameters and click **Create chunks**.

* **Chunk size**: larger chunks give more context, smaller chunks can improve retrieval precision.
* **Overlap**: helps avoid splitting important information across chunk boundaries.
* Chunks have stable IDs like: `docname:p3-p4:c0007` (used for citations).

### 4) Index

Select an embedding model and click **Build FAISS index**.

* This embeds all chunks and builds a FAISS index (cosine similarity).
* You must build the index before asking questions.

### 5) Ask (RAG)

Enter your question and click **Answer (RAG)**.

* The app retrieves top-k chunks from FAISS and asks the LLM to answer **using only the retrieved sources**.
* Answers include citations like `[docname:p3-p4:c0007]` and a “Sources used” list.

### 6) Evaluation

Click **Run evaluation** to run the golden question set (`eval/golden.json`) end-to-end.

* Reports: citation rate, keyword coverage, expected-page hit rate (optional), and latency.
* Recommended: customize `eval/golden.json` for your specific PDF(s).

---

## Access to LLM features (anti-abuse)

To prevent API abuse and protect free-tier quotas, the **Ask** and **Evaluation** tabs are **access-code gated** (LLM API calls).
**Retrieval-only functionality (chunking + embeddings + FAISS search + retrieval preview) remains public.**

If you would like full access, please email: [mohammad.akhavan999@gmail.com](mailto:mohammad.akhavan999@gmail.com)

---

## Architecture (minimal)

```
PDF/Text
  │
  ├─(PyMuPDF) page-level extraction  →  {doc, page, text}
  │
  ├─Chunking (size/overlap)         →  {chunk_id, pages, text}
  │
  ├─Embeddings (SentenceTransformers)
  │            ↓
  ├─FAISS (IndexFlatIP, cosine)
  │            ↓
  ├─Retrieve top-k chunks
  │            ↓
  └─LLM API (Groq or Gemini) → Answer + citations + Sources list
                ↓
            Eval: golden.json → citation rate, keyword coverage, latency
```

---

## Setup (local)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open the Gradio link shown in the terminal.

---

## Deploy to Hugging Face Spaces (free CPU)

1. Create a Space → **Gradio**
2. Upload/commit this repo files
3. Add API keys in **Space → Settings → Secrets**:

   * `GROQ_API_KEY` (Groq free tier)
   * `GEMINI_API_KEY` (Gemini; quota may vary by account/project)

If using access-code gating, also set:

* `APP_ACCESS_CODE` (shared access code for Ask/Evaluation tabs)

---

## Guardrails (cost + stability)

* Upload limits:

  * max **25 MB per PDF**
  * max **60 pages per PDF**
  * max **800k extracted characters/session**
  * max **1200 chunks/session**
* LLM limits:

  * max **12k source-context characters** in prompt
  * max **2k chars per source chunk**
  * max **~6 LLM calls/min per session**
  * max output tokens (UI slider)
* Evaluation:

  * capped number of eval questions (default 10)

---

## Results (example from my run)

Settings:

* top_k = 5
* Provider: Gemini (evaluation)

Metrics:

* Questions: 5
* Citation rate: **1.00**
* Avg keyword coverage: **0.67**
* Expected-page hit rate: **1.00**
* Avg retrieval latency: **0.50s**
* Avg LLM latency: **1.71s**
* Avg total latency: **2.21s**

---

## Evaluation (golden set)

Edit `eval/golden.json` to match your PDFs (questions + expected keywords + optional expected pages).
Then run the **Evaluation** tab to generate a report.

---

## Notes

* Groq free-tier limits can be tight; evaluation can be configured to **Gemini-only**.
* This project is intentionally lightweight and runs on free CPU.
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

## Demo

* Upload PDF(s) (or paste text)
* Chunk → embed → store in **FAISS**
* Ask questions → retrieve top-k → generate answer with **citations** (`[chunk_id]` + page ranges)
* Run **golden-set evaluation** from the UI

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
3. Add an API key in **Space → Settings → Secrets**

   * `GROQ_API_KEY` (Groq free tier)
   * `GEMINI_API_KEY` (Gemini; quota may vary by account/project)

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

<!-- ## Example Q&As (with citations)

Below are 5 example questions and answers (from your evaluation run).
*(Citations are chunk IDs + page ranges.)*

1. **Q:** What is the main problem this proposal is solving?
   **A:** It addresses infection transmission risk from organic residues on surfaces (especially human-related residues) and aims to detect/quantify contamination reliably. [masterthesisproposal-ma-pdf:p16-p16:c0036]

2. **Q:** What approach is proposed?
   **A:** A fluorescence-imaging based pipeline using UV excitation + imaging + statistical/ML segmentation to detect biological residues on surfaces. [masterthesisproposal-ma-pdf:p2-p3:c0003]

3. **Q:** What key fluorescence/optics detail is used in experiments?
   **A:** Saliva has characteristic UV absorption and emits fluorescence under UV excitation, which the system leverages for detection. [masterthesisproposal-ma-pdf:p3-p4:c0005]

4. **Q:** What hardware setup is used?
   **A:** A modified Sony NEX-6 (APS-C CMOS) and a UV/optical configuration for controlled UV-C excitation and image capture. [masterthesisproposal-ma-pdf:p9-p9:c0020]

5. **Q:** What analysis/modeling is used?
   **A:** Controlled UV-C image capture followed by pixel-wise statistical classification models to segment residue vs background. [masterthesisproposal-ma-pdf:p8-p8:c0016] -->

<!-- --- -->

## Evaluation (golden set)

Edit `eval/golden.json` to match your PDFs (questions + expected keywords + optional expected pages).
Then run **Evaluation** tab to generate a report.

---

## Notes

* Groq free-tier limits can be tight; evaluation can be configured to **Gemini-only**.
* This is intentionally lightweight app
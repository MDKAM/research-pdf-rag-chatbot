---
title: Pdf Rag Chatbot
emoji: 📄
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
python_version: 3.10
app_file: app.py
pinned: false
license: mit
---

# Research PDF RAG Chatbot (v0)

A GenAI project:

* Gradio UI on Hugging Face Spaces
* PDF/text ingestion → chunking → embeddings → FAISS → RAG answers w/ citations
* basic eval with a small golden Q set

## Local run (lightweight)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
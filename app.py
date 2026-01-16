from __future__ import annotations

import os
import time
import uuid
from typing import List, Dict, Any

import gradio as gr

# Make src importable (works on HF + locally)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from ragpdf.ingest import (
    PageRecord,
    extract_pdf_pages,
    validate_uploads,
    IngestError,
)

# ----------------------------
# Guardrails (cost & stability)
# ----------------------------
MAX_FILE_SIZE_MB = 25
MAX_PAGES_PER_PDF = 60
MAX_TOTAL_CHARS = 800_000  # across ALL uploaded PDFs in one session
MAX_PASTE_CHARS = 200_000


def _records_to_preview(records: List[PageRecord], max_chars: int = 4000) -> str:
    """
    Pretty preview in the UI: show a few pages with doc/page headers.
    """
    out = []
    used = 0
    for r in records[:8]:  # preview first few pages
        header = f"\n\n=== {r.doc_name} | page {r.page} ===\n"
        body = (r.text or "").strip()
        snippet = body[:800] + ("…" if len(body) > 800 else "")
        chunk = header + (snippet if snippet else "[no text found on this page]")
        if used + len(chunk) > max_chars:
            break
        out.append(chunk)
        used += len(chunk)

    if not out:
        return "No extracted text to preview yet."
    return "".join(out).strip()


def _records_to_stats(records: List[PageRecord], elapsed_s: float) -> str:
    docs = {}
    total_chars = 0
    nonempty_pages = 0
    for r in records:
        docs.setdefault(r.doc_name, 0)
        docs[r.doc_name] += 1
        tlen = len((r.text or "").strip())
        total_chars += tlen
        if tlen > 0:
            nonempty_pages += 1

    lines = []
    lines.append(f"✅ Extraction complete in {elapsed_s:.2f}s")
    lines.append(f"- PDFs/pages extracted: {sum(docs.values())} pages across {len(docs)} document(s)")
    lines.append(f"- Non-empty pages: {nonempty_pages}")
    lines.append(f"- Total extracted characters: {total_chars:,}")
    lines.append(f"- Guardrails: {MAX_FILE_SIZE_MB}MB/file, {MAX_PAGES_PER_PDF} pages/PDF, {MAX_TOTAL_CHARS:,} chars total")
    lines.append("\nPer-document page counts:")
    for k, v in docs.items():
        lines.append(f"  - {k}: {v} page(s)")
    return "\n".join(lines)


def load_pdfs(filepaths: List[str], existing: List[Dict[str, Any]] | None):
    """
    Extract pages from uploaded PDFs.
    Stores "records" as list of dicts in gr.State for HF compatibility.
    """
    try:
        t0 = time.time()

        validate_uploads(filepaths, max_file_size_mb=MAX_FILE_SIZE_MB)

        # Start from existing records if any
        all_records: List[PageRecord] = []
        if existing:
            for d in existing:
                all_records.append(PageRecord(**d))

        total_chars = sum(len((r.text or "").strip()) for r in all_records)

        # Extract each PDF
        for p in filepaths:
            doc_id = str(uuid.uuid4())[:8]
            recs, total_chars = extract_pdf_pages(
                pdf_path=p,
                doc_id=doc_id,
                max_pages_per_pdf=MAX_PAGES_PER_PDF,
                max_total_chars=MAX_TOTAL_CHARS,
                existing_total_chars=total_chars,
            )
            all_records.extend(recs)

            if total_chars >= MAX_TOTAL_CHARS:
                break

        elapsed = time.time() - t0
        preview = _records_to_preview(all_records)
        stats = _records_to_stats(all_records, elapsed)

        # store as serializable dicts
        stored = [r.__dict__ for r in all_records]
        return stored, stats, preview

    except IngestError as e:
        return existing or [], f"❌ Ingestion error: {e}", "No preview available."
    except Exception as e:
        return existing or [], f"❌ Unexpected error: {e}", "No preview available."


def load_pasted_text(text: str):
    text = (text or "").strip()
    if not text:
        return [], "❌ Please paste some text.", "No preview available."

    if len(text) > MAX_PASTE_CHARS:
        text = text[:MAX_PASTE_CHARS]

    doc_id = "pasted"
    rec = PageRecord(doc_id=doc_id, doc_name="pasted_text", page=1, text=text)
    stored = [rec.__dict__]
    stats = (
        "✅ Text loaded.\n"
        f"- Characters: {len(text):,}\n"
        f"- Guardrail: max {MAX_PASTE_CHARS:,} characters for pasted text"
    )
    preview = _records_to_preview([rec])
    return stored, stats, preview


def clear_session():
    return [], "Cleared.", "No extracted text to preview yet."


with gr.Blocks(title="Research PDF RAG Chatbot (Ticket 1)") as demo:
    gr.Markdown(
        """
# Research PDF RAG Chatbot
**Ticket 1:** PDF/text ingestion with page numbers ✅

Next: chunking → embeddings → FAISS → RAG answers with citations.
"""
    )

    # State: list of {doc_id, doc_name, page, text}
    records_state = gr.State([])

    with gr.Tabs():
        with gr.Tab("Upload PDF(s)"):
            pdfs = gr.File(
                label="Upload one or more PDFs",
                file_types=[".pdf"],
                file_count="multiple",
                type="filepath",
            )
            with gr.Row():
                btn_extract = gr.Button("Extract text from PDFs", variant="primary")
                btn_clear = gr.Button("Clear session")

        with gr.Tab("Paste text"):
            pasted = gr.Textbox(
                label="Paste text",
                lines=12,
                placeholder="Paste paper abstract, notes, or any text here…",
            )
            btn_use_text = gr.Button("Use this text", variant="primary")

    stats = gr.Textbox(label="Status / Stats", lines=10)
    preview = gr.Textbox(label="Preview (first pages)", lines=18)

    btn_extract.click(
        fn=load_pdfs,
        inputs=[pdfs, records_state],
        outputs=[records_state, stats, preview],
    )

    btn_use_text.click(
        fn=load_pasted_text,
        inputs=[pasted],
        outputs=[records_state, stats, preview],
    )

    btn_clear.click(
        fn=clear_session,
        inputs=[],
        outputs=[records_state, stats, preview],
    )

if __name__ == "__main__":
    demo.launch()
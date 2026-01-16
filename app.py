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

from ragpdf.chunking import (
    ChunkRecord,
    chunk_pages,
    ChunkingError,
)

# ----------------------------
# Guardrails (cost & stability)
# ----------------------------
MAX_FILE_SIZE_MB = 25
MAX_PAGES_PER_PDF = 60
MAX_TOTAL_CHARS = 800_000  # across ALL uploaded PDFs in one session
MAX_PASTE_CHARS = 200_000

MAX_CHUNKS = 1200


def _records_to_preview(records: List[PageRecord], max_chars: int = 4000) -> str:
    out = []
    used = 0
    for r in records[:8]:
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
    lines.append(f"- Pages extracted: {sum(docs.values())} across {len(docs)} document(s)")
    lines.append(f"- Non-empty pages: {nonempty_pages}")
    lines.append(f"- Total extracted characters: {total_chars:,}")
    lines.append(f"- Guardrails: {MAX_FILE_SIZE_MB}MB/file, {MAX_PAGES_PER_PDF} pages/PDF, {MAX_TOTAL_CHARS:,} chars total")
    lines.append("\nPer-document page counts:")
    for k, v in docs.items():
        lines.append(f"  - {k}: {v} page(s)")
    return "\n".join(lines)


def _chunks_to_stats(chunks: List[ChunkRecord], elapsed_s: float) -> str:
    docs = {}
    total_chars = 0
    for c in chunks:
        docs.setdefault(c.doc_name, 0)
        docs[c.doc_name] += 1
        total_chars += len(c.text)

    lines = []
    lines.append(f"✅ Chunking complete in {elapsed_s:.2f}s")
    lines.append(f"- Total chunks: {len(chunks)} (max {MAX_CHUNKS})")
    lines.append(f"- Total chunk characters: {total_chars:,}")
    lines.append("\nChunks per document:")
    for k, v in docs.items():
        lines.append(f"  - {k}: {v} chunks")
    return "\n".join(lines)


def _chunks_to_preview(chunks: List[ChunkRecord], max_chars: int = 5000) -> str:
    out = []
    used = 0
    for c in chunks[:8]:
        header = f"\n\n=== {c.chunk_id} | {c.doc_name} | pages {c.page_start}-{c.page_end} ===\n"
        snippet = c.text[:900] + ("…" if len(c.text) > 900 else "")
        block = header + snippet
        if used + len(block) > max_chars:
            break
        out.append(block)
        used += len(block)
    return ("".join(out).strip()) if out else "No chunks to preview yet."


def load_pdfs(filepaths: List[str], existing: List[Dict[str, Any]] | None):
    try:
        t0 = time.time()

        validate_uploads(filepaths, max_file_size_mb=MAX_FILE_SIZE_MB)

        all_records: List[PageRecord] = []
        if existing:
            for d in existing:
                all_records.append(PageRecord(**d))

        total_chars = sum(len((r.text or "").strip()) for r in all_records)

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


def run_chunking(
    records: List[Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
):
    if not records:
        return [], "❌ No extracted records. Upload PDFs or paste text first.", "No preview available."

    try:
        t0 = time.time()
        chunks = chunk_pages(
            page_records=records,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            min_chunk_chars=int(min_chunk_chars),
            max_chunks=MAX_CHUNKS,
        )
        elapsed = time.time() - t0
        stored = [c.__dict__ for c in chunks]
        stats = _chunks_to_stats(chunks, elapsed)
        preview = _chunks_to_preview(chunks)
        return stored, stats, preview

    except ChunkingError as e:
        return [], f"❌ Chunking error: {e}", "No preview available."
    except Exception as e:
        return [], f"❌ Unexpected error: {e}", "No preview available."


def clear_session():
    return [], [], "Cleared.", "No extracted text to preview yet.", "No chunks to preview yet."


with gr.Blocks(title="Research PDF RAG Chatbot (Ticket 2)") as demo:
    gr.Markdown(
        """
# Research PDF RAG Chatbot
✅ Ticket 1: PDF/text ingestion with page numbers  
✅ Ticket 2: Chunking with stable chunk IDs + page ranges  

Next: embeddings → FAISS → retrieval.
"""
    )

    # State: page records and chunk records
    records_state = gr.State([])  # list of {doc_id, doc_name, page, text}
    chunks_state = gr.State([])   # list of {chunk_id, doc_id, doc_name, page_start, page_end, text}

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

        with gr.Tab("Chunking"):
            gr.Markdown("### Chunk settings")
            with gr.Row():
                chunk_size = gr.Slider(600, 2400, value=1200, step=50, label="Chunk size (chars)")
                chunk_overlap = gr.Slider(0, 600, value=200, step=25, label="Chunk overlap (chars)")
                min_chunk_chars = gr.Slider(50, 600, value=200, step=25, label="Min chunk chars (skip tiny chunks)")
            btn_chunk = gr.Button("Create chunks", variant="primary")

    with gr.Row():
        stats_records = gr.Textbox(label="Ingestion status / stats", lines=10)
        stats_chunks = gr.Textbox(label="Chunking status / stats", lines=10)

    with gr.Row():
        preview_records = gr.Textbox(label="Extracted preview (first pages)", lines=18)
        preview_chunks = gr.Textbox(label="Chunk preview (first chunks)", lines=18)

    btn_extract.click(
        fn=load_pdfs,
        inputs=[pdfs, records_state],
        outputs=[records_state, stats_records, preview_records],
    )

    btn_use_text.click(
        fn=load_pasted_text,
        inputs=[pasted],
        outputs=[records_state, stats_records, preview_records],
    )

    btn_chunk.click(
        fn=run_chunking,
        inputs=[records_state, chunk_size, chunk_overlap, min_chunk_chars],
        outputs=[chunks_state, stats_chunks, preview_chunks],
    )

    btn_clear.click(
        fn=clear_session,
        inputs=[],
        outputs=[records_state, chunks_state, stats_records, preview_records, preview_chunks],
    )

if __name__ == "__main__":
    demo.launch()
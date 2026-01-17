from __future__ import annotations

import os
import time
import uuid
from typing import List, Dict, Any, Tuple

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

from ragpdf.vectorstore import (
    build_faiss_index,
    retrieve,
    VectorStoreError,
    RetrievedChunk,
)

from ragpdf.llm_api import (
    call_llm,
    env_has_groq,
    env_has_gemini,
    LLMError,
)

# ----------------------------
# Guardrails (cost & stability)
# ----------------------------
MAX_FILE_SIZE_MB = 25
MAX_PAGES_PER_PDF = 60
MAX_TOTAL_CHARS = 800_000
MAX_PASTE_CHARS = 200_000

MAX_CHUNKS = 1200
MAX_QUERY_CHARS = 2000

# Context and answer limits (cost control)
MAX_CONTEXT_CHARS = 12_000     # concatenated retrieved sources text
MAX_SOURCE_CHARS_EACH = 2_000  # per chunk text cap in prompt
DEFAULT_TOP_K = 5

DEFAULT_MAX_OUTPUT_TOKENS = 450
DEFAULT_TEMPERATURE = 0.2

# Rate limit: per session
MAX_CALLS_PER_MIN = 6

# Embedding model defaults (HF)
EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-small-v2",
]

# LLM defaults
GROQ_DEFAULT_MODEL = "llama-3.1-8b-instant"
GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"


def _records_to_preview(records: List[PageRecord], max_chars: int = 4000) -> str:
    out, used = [], 0
    for r in records[:8]:
        header = f"\n\n=== {r.doc_name} | page {r.page} ===\n"
        body = (r.text or "").strip()
        snippet = body[:800] + ("…" if len(body) > 800 else "")
        block = header + (snippet if snippet else "[no text found on this page]")
        if used + len(block) > max_chars:
            break
        out.append(block)
        used += len(block)
    return ("".join(out).strip()) if out else "No extracted text to preview yet."


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

    lines = [
        f"✅ Extraction complete in {elapsed_s:.2f}s",
        f"- Pages extracted: {sum(docs.values())} across {len(docs)} document(s)",
        f"- Non-empty pages: {nonempty_pages}",
        f"- Total extracted characters: {total_chars:,}",
        f"- Guardrails: {MAX_FILE_SIZE_MB}MB/file, {MAX_PAGES_PER_PDF} pages/PDF, {MAX_TOTAL_CHARS:,} chars total",
        "",
        "Per-document page counts:",
    ]
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

    lines = [
        f"✅ Chunking complete in {elapsed_s:.2f}s",
        f"- Total chunks: {len(chunks)} (max {MAX_CHUNKS})",
        f"- Total chunk characters: {total_chars:,}",
        "",
        "Chunks per document:",
    ]
    for k, v in docs.items():
        lines.append(f"  - {k}: {v} chunks")
    return "\n".join(lines)


def _chunks_to_preview(chunks: List[ChunkRecord], max_chars: int = 5000) -> str:
    out, used = [], 0
    for c in chunks[:8]:
        header = f"\n\n=== {c.chunk_id} | {c.doc_name} | pages {c.page_start}-{c.page_end} ===\n"
        snippet = c.text[:900] + ("…" if len(c.text) > 900 else "")
        block = header + snippet
        if used + len(block) > max_chars:
            break
        out.append(block)
        used += len(block)
    return ("".join(out).strip()) if out else "No chunks to preview yet."


def _retrieval_to_markdown(results: List[RetrievedChunk]) -> str:
    if not results:
        return "No results."

    lines = ["### Top retrieved chunks\n"]
    for r in results:
        snippet = (r.text[:350] + "…") if len(r.text) > 350 else r.text
        snippet_one_line = snippet.replace("\n", " ").replace("\r", " ")
        lines.append(
            f"**{r.rank}.** score={r.score:.4f} | `{r.chunk_id}` | **{r.doc_name}** p{r.page_start}-{r.page_end}\n\n"
            f"> {snippet_one_line}\n"
        )
    return "\n".join(lines).strip()


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

    rec = PageRecord(doc_id="pasted", doc_name="pasted_text", page=1, text=text)
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


def build_index(chunks: List[Dict[str, Any]], model_name: str):
    if not chunks:
        return None, "❌ No chunks found. Run chunking first."

    try:
        t0 = time.time()
        store = build_faiss_index(
            chunks=chunks,
            model_name=model_name,
            batch_size=32,
            max_chunks=MAX_CHUNKS,
        )
        elapsed = time.time() - t0
        msg = (
            "✅ FAISS index built.\n"
            f"- Model: {store['model_name']}\n"
            f"- Dim: {store['dim']}\n"
            f"- Vectors: {store['index'].ntotal}\n"
            f"- Time: {elapsed:.2f}s\n"
            f"- Similarity: cosine (via inner product on L2-normalized vectors)\n"
        )
        return store, msg
    except VectorStoreError as e:
        return None, f"❌ Index build error: {e}"
    except Exception as e:
        return None, f"❌ Unexpected error: {e}"


def run_retrieval(store: Dict[str, Any] | None, query: str, top_k: int):
    query = (query or "").strip()
    if len(query) > MAX_QUERY_CHARS:
        query = query[:MAX_QUERY_CHARS]

    if not query:
        return "❌ Please enter a query."

    try:
        t0 = time.time()
        results = retrieve(store=store or {}, query=query, top_k=int(top_k))
        elapsed = time.time() - t0
        md = _retrieval_to_markdown(results)
        md += f"\n\n_⏱ retrieval time: {elapsed:.2f}s | top_k={int(top_k)}_"
        return md
    except VectorStoreError as e:
        return f"❌ Retrieval error: {e}"
    except Exception as e:
        return f"❌ Unexpected error: {e}"


def _rate_limit_ok(rate_state: List[float]) -> Tuple[bool, List[float], str]:
    now = time.time()
    # Keep last 60s
    rate_state = [t for t in (rate_state or []) if now - t < 60.0]
    if len(rate_state) >= MAX_CALLS_PER_MIN:
        wait_s = int(60 - (now - min(rate_state)))
        return False, rate_state, f"⛔ Rate limit: {MAX_CALLS_PER_MIN}/min. Try again in ~{wait_s}s."
    rate_state.append(now)
    return True, rate_state, ""


def _build_rag_prompts(question: str, results: List[RetrievedChunk]) -> Tuple[str, str, str]:
    """
    Returns (system_prompt, user_prompt, sources_md)
    """
    system_prompt = (
        "You are a research assistant. Answer using ONLY the provided sources. "
        "If the answer is not in the sources, say you don't know. "
        "When you use a source, cite it inline using square brackets with the chunk id, e.g. [mydoc:p2-p3:c0001]. "
        "Be concise and accurate."
    )

    # Build sources block with truncation controls
    blocks = []
    used = 0
    for r in results:
        text = r.text[:MAX_SOURCE_CHARS_EACH]
        block = (
            f"SOURCE: {r.chunk_id}\n"
            f"DOCUMENT: {r.doc_name}\n"
            f"PAGES: {r.page_start}-{r.page_end}\n"
            f"TEXT:\n{text}\n"
        )
        if used + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        used += len(block)

    sources_block = "\n\n".join(blocks).strip()

    user_prompt = (
        f"Sources:\n{sources_block}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- Provide a direct answer.\n"
        "- Include citations like [chunk_id] immediately after the relevant claims.\n"
        "- If sources disagree, mention uncertainty and cite both.\n"
    )

    # Also prepare a clean Sources list for UI
    sources_lines = ["### Sources used"]
    for r in results:
        sources_lines.append(f"- `{r.chunk_id}` — **{r.doc_name}** p{r.page_start}-{r.page_end}")
    sources_md = "\n".join(sources_lines)

    return system_prompt, user_prompt, sources_md


def rag_answer(
    store: Dict[str, Any] | None,
    question: str,
    provider: str,
    groq_model: str,
    gemini_model: str,
    top_k: int,
    temperature: float,
    max_output_tokens: int,
    rate_state: List[float],
):
    question = (question or "").strip()
    if len(question) > MAX_QUERY_CHARS:
        question = question[:MAX_QUERY_CHARS]

    if not question:
        return "❌ Please enter a question.", "No retrieval yet.", rate_state

    ok, rate_state, msg = _rate_limit_ok(rate_state)
    if not ok:
        return msg, "No retrieval yet.", rate_state

    if not store:
        return "❌ Build the FAISS index first (Ticket 3).", "No retrieval yet.", rate_state

    try:
        # 1) Retrieve
        t0 = time.time()
        results = retrieve(store=store, query=question, top_k=int(top_k))
        retr_latency = time.time() - t0
        retrieval_md = _retrieval_to_markdown(results) + f"\n\n_⏱ retrieval time: {retr_latency:.2f}s_"

        if not results:
            return "I couldn't retrieve any relevant sources for that question.", retrieval_md, rate_state

        # 2) Build prompts
        system_prompt, user_prompt, sources_md = _build_rag_prompts(question, results)

        # 3) Choose provider/model
        provider = (provider or "auto").strip().lower()
        if provider == "groq":
            model = (groq_model or GROQ_DEFAULT_MODEL).strip()
        elif provider == "gemini":
            model = (gemini_model or GEMINI_DEFAULT_MODEL).strip()
        else:
            # auto
            if env_has_groq():
                provider, model = "groq", (groq_model or GROQ_DEFAULT_MODEL).strip()
            elif env_has_gemini():
                provider, model = "gemini", (gemini_model or GEMINI_DEFAULT_MODEL).strip()
            else:
                return "❌ No API key found. Set GROQ_API_KEY or GEMINI_API_KEY in Space secrets.", retrieval_md, rate_state

        # 4) Call LLM
        resp = call_llm(
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(temperature),
            max_output_tokens=int(max_output_tokens),
            timeout_s=60,
            retries=2,
        )

        # 5) Ensure we show sources even if the model forgets citations
        answer = resp.text.strip()
        footer = (
            f"\n\n---\n"
            f"_LLM: {resp.provider} / {resp.model} | ⏱ {resp.latency_s:.2f}s | top_k={int(top_k)} | "
            f"max_output_tokens={int(max_output_tokens)}_\n\n"
            f"{sources_md}"
        )
        return answer + footer, retrieval_md, rate_state

    except VectorStoreError as e:
        return f"❌ Retrieval error: {e}", "No retrieval yet.", rate_state
    except LLMError as e:
        return f"❌ LLM error: {e}", "Retrieval may have succeeded above.", rate_state
    except Exception as e:
        return f"❌ Unexpected error: {e}", "No retrieval yet.", rate_state


def clear_session():
    return [], [], None, [], "Cleared.", "No extracted text to preview yet.", "No chunks to preview yet.", "Index cleared.", "No results.", "No answer yet."


def api_key_status() -> str:
    g = "✅" if env_has_groq() else "❌"
    m = "✅" if env_has_gemini() else "❌"
    return (
        "### API key status (Space secrets)\n"
        f"- GROQ_API_KEY: {g}\n"
        f"- GEMINI_API_KEY: {m}\n"
        "\nSet these in **Space → Settings → Secrets**."
    )


with gr.Blocks(title="Research PDF RAG Chatbot (Ticket 4)") as demo:
    gr.Markdown(
        """
# Research PDF RAG Chatbot
✅ Ticket 1: PDF/text ingestion with page numbers  
✅ Ticket 2: Chunking with stable chunk IDs + page ranges  
✅ Ticket 3: Embeddings + FAISS + retrieval UI  
✅ Ticket 4: LLM answering (Groq/Gemini) + citations  

Next: Ticket 5 = evaluation + “golden questions” test set.
"""
    )

    records_state = gr.State([])   # page records
    chunks_state = gr.State([])    # chunk records
    store_state = gr.State(None)   # FAISS store
    rate_state = gr.State([])      # timestamps for rate limiting

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
            pasted = gr.Textbox(label="Paste text", lines=12, placeholder="Paste any text…")
            btn_use_text = gr.Button("Use this text", variant="primary")

        with gr.Tab("Chunking"):
            gr.Markdown("### Chunk settings")
            with gr.Row():
                chunk_size = gr.Slider(600, 2400, value=1200, step=50, label="Chunk size (chars)")
                chunk_overlap = gr.Slider(0, 600, value=200, step=25, label="Chunk overlap (chars)")
                min_chunk_chars = gr.Slider(50, 600, value=200, step=25, label="Min chunk chars")
            btn_chunk = gr.Button("Create chunks", variant="primary")

        with gr.Tab("Index & Retrieve"):
            gr.Markdown("### Build FAISS index (embeddings)")
            model_name = gr.Dropdown(EMBEDDING_MODELS, value=EMBEDDING_MODELS[0], label="Embedding model")
            btn_index = gr.Button("Build FAISS index", variant="primary")
            index_status = gr.Textbox(label="Index status", lines=6)

            gr.Markdown("### Retrieval (sanity check)")
            query = gr.Textbox(label="Query", lines=2, placeholder="Ask something about your PDFs…")
            top_k_retr = gr.Slider(1, 10, value=DEFAULT_TOP_K, step=1, label="Top-k")
            btn_retrieve = gr.Button("Retrieve top-k chunks", variant="secondary")
            retrieval_out = gr.Markdown("No results.")

        with gr.Tab("Ask (RAG Answer)"):
            gr.Markdown(api_key_status())

            with gr.Row():
                provider = gr.Dropdown(
                    ["auto", "groq", "gemini"],
                    value="auto",
                    label="LLM provider",
                )

            with gr.Row():
                groq_model = gr.Textbox(label="Groq model", value=GROQ_DEFAULT_MODEL)
                gemini_model = gr.Textbox(label="Gemini model", value=GEMINI_DEFAULT_MODEL)

            with gr.Row():
                top_k = gr.Slider(1, 10, value=DEFAULT_TOP_K, step=1, label="Top-k sources")
                temperature = gr.Slider(0.0, 1.0, value=DEFAULT_TEMPERATURE, step=0.05, label="Temperature")
                max_output_tokens = gr.Slider(128, 1024, value=DEFAULT_MAX_OUTPUT_TOKENS, step=32, label="Max output tokens")

            question = gr.Textbox(label="Question", lines=2, placeholder="Ask a question about the uploaded PDFs…")
            btn_answer = gr.Button("Answer with RAG + citations", variant="primary")

            answer_out = gr.Markdown("No answer yet.")
            retrieval_out2 = gr.Markdown("No retrieval yet.")

    with gr.Row():
        stats_records = gr.Textbox(label="Ingestion stats", lines=10)
        stats_chunks = gr.Textbox(label="Chunking stats", lines=10)

    with gr.Row():
        preview_records = gr.Textbox(label="Extracted preview", lines=18)
        preview_chunks = gr.Textbox(label="Chunk preview", lines=18)

    # Wiring
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

    btn_index.click(
        fn=build_index,
        inputs=[chunks_state, model_name],
        outputs=[store_state, index_status],
    )

    btn_retrieve.click(
        fn=run_retrieval,
        inputs=[store_state, query, top_k_retr],
        outputs=[retrieval_out],
    )

    btn_answer.click(
        fn=rag_answer,
        inputs=[store_state, question, provider, groq_model, gemini_model, top_k, temperature, max_output_tokens, rate_state],
        outputs=[answer_out, retrieval_out2, rate_state],
    )

    btn_clear.click(
        fn=clear_session,
        inputs=[],
        outputs=[
            records_state, chunks_state, store_state, rate_state,
            stats_records, preview_records, preview_chunks,
            index_status, retrieval_out, answer_out
        ],
    )

if __name__ == "__main__":
    demo.launch()
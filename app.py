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

from ragpdf.eval import (
    load_golden,
    keyword_coverage,
    has_citation,
    pages_hit_expected,
    summarize_rows,
    format_report,
    EvalRow,
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
MAX_CONTEXT_CHARS = 12_000
MAX_SOURCE_CHARS_EACH = 2_000
DEFAULT_TOP_K = 5

DEFAULT_MAX_OUTPUT_TOKENS = 450
DEFAULT_TEMPERATURE = 0.2

# Rate limit: per session
MAX_CALLS_PER_MIN = 6

# Eval guardrails
MAX_EVAL_QUESTIONS = 10

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


def run_chunking(records: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int, min_chunk_chars: int):
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
        store = build_faiss_index(chunks=chunks, model_name=model_name, batch_size=32, max_chunks=MAX_CHUNKS)
        elapsed = time.time() - t0
        msg = (
            "✅ FAISS index built.\n"
            f"- Model: {store['model_name']}\n"
            f"- Dim: {store['dim']}\n"
            f"- Vectors: {store['index'].ntotal}\n"
            f"- Time: {elapsed:.2f}s\n"
            f"- Similarity: cosine (inner product on normalized vectors)\n"
        )
        return store, msg
    except Exception as e:
        return None, f"❌ Index build error: {e}"


def _rate_limit_ok(rate_state: List[float]) -> Tuple[bool, List[float], str]:
    now = time.time()
    rate_state = [t for t in (rate_state or []) if now - t < 60.0]
    if len(rate_state) >= MAX_CALLS_PER_MIN:
        wait_s = int(60 - (now - min(rate_state)))
        return False, rate_state, f"⛔ Rate limit: {MAX_CALLS_PER_MIN}/min. Try again in ~{wait_s}s."
    rate_state.append(now)
    return True, rate_state, ""


def _build_rag_prompts(question: str, results: List[RetrievedChunk]) -> Tuple[str, str, str]:
    system_prompt = (
        "You are a research assistant. Answer using ONLY the provided sources. "
        "If the answer is not in the sources, say you don't know. "
        "Cite sources inline using square brackets with the chunk id, e.g. [mydoc:p2-p3:c0001]."
    )

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
    )

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
    access_code: str,
):
    question = (question or "").strip()
    if len(question) > MAX_QUERY_CHARS:
        question = question[:MAX_QUERY_CHARS]
    if not question:
        return "❌ Please enter a question.", "No retrieval yet.", rate_state
    
    if not _access_ok(access_code):
        return "⛔ Access denied. Ask for the access code to enable LLM answering.", "Retrieval-only is available publicly.", rate_state

    ok, rate_state, msg = _rate_limit_ok(rate_state)
    if not ok:
        return msg, "No retrieval yet.", rate_state

    if not store:
        return "❌ Build the FAISS index first.", "No retrieval yet.", rate_state

    try:
        # Retrieve
        t0 = time.time()
        results = retrieve(store=store, query=question, top_k=int(top_k))
        retr_latency = time.time() - t0
        retrieval_md = _retrieval_to_markdown(results) + f"\n\n_⏱ retrieval time: {retr_latency:.2f}s_"
        if not results:
            return "I couldn't retrieve relevant sources for that question.", retrieval_md, rate_state

        # Prompts
        system_prompt, user_prompt, sources_md = _build_rag_prompts(question, results)

        # Provider/model selection
        provider = (provider or "auto").strip().lower()
        if provider == "groq":
            model = (groq_model or GROQ_DEFAULT_MODEL).strip()
        elif provider == "gemini":
            model = (gemini_model or GEMINI_DEFAULT_MODEL).strip()
        else:
            if env_has_groq():
                provider, model = "groq", (groq_model or GROQ_DEFAULT_MODEL).strip()
            elif env_has_gemini():
                provider, model = "gemini", (gemini_model or GEMINI_DEFAULT_MODEL).strip()
            else:
                return "❌ No API key found. Set GROQ_API_KEY or GEMINI_API_KEY.", retrieval_md, rate_state

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

        answer = resp.text.strip()
        footer = (
            f"\n\n---\n"
            f"_LLM: {resp.provider} / {resp.model} | ⏱ {resp.latency_s:.2f}s | top_k={int(top_k)} | max_output_tokens={int(max_output_tokens)}_\n\n"
            f"{sources_md}"
        )
        return answer + footer, retrieval_md, rate_state

    except Exception as e:
        return f"❌ Error: {e}", "Retrieval may have succeeded above.", rate_state


# ----------------------------
# Evaluation runner
# ----------------------------
def run_eval(
    store: Dict[str, Any] | None,
    provider: str,
    groq_model: str,
    gemini_model: str,
    top_k: int,
    temperature: float,
    max_output_tokens: int,
    golden_path: str,
    rate_state: List[float],
    access_code: str,
):
    if not store:
        return "❌ Build the FAISS index first.", rate_state
    
    if not _access_ok(access_code):
        return "⛔ Access denied. Evaluation is gated to prevent API abuse.", rate_state


    golden_path = (golden_path or "").strip()
    if not golden_path:
        golden_path = "eval/golden.json"

    if not os.path.exists(golden_path):
        return f"❌ Golden set not found at: {golden_path}", rate_state

    items = load_golden(golden_path)[:MAX_EVAL_QUESTIONS]
    if not items:
        return "❌ No eval items found in golden set.", rate_state

    rows: List[EvalRow] = []
    for it in items:
        # Rate limit (counts as an LLM call)
        ok, rate_state, msg = _rate_limit_ok(rate_state)
        if not ok:
            return f"{msg}\n\nStopped early after {len(rows)} questions.", rate_state

        # Retrieval
        t0 = time.time()
        results = retrieve(store=store, query=it.question, top_k=int(top_k))
        retr_latency = time.time() - t0

        # Prompts + call LLM
        system_prompt, user_prompt, _sources_md = _build_rag_prompts(it.question, results)

        # Force evaluation to use Gemini only (avoids Groq free-tier limits)
        if not env_has_gemini():
            return "❌ Evaluation is set to Gemini-only, but GEMINI_API_KEY is missing.", rate_state

        provider2 = "gemini"
        model = (gemini_model or GEMINI_DEFAULT_MODEL).strip()

        llm_t0 = time.time()
        resp = call_llm(
            provider=provider2,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(temperature),
            max_output_tokens=int(max_output_tokens),
            timeout_s=60,
            retries=2,
        )
        llm_latency = time.time() - llm_t0
        total = retr_latency + llm_latency

        # if it != items[-1]:
        #     time.sleep(1.2)  # small delay between calls

        # Metrics
        ans = resp.text or ""
        kw = keyword_coverage(ans, it.expected_keywords)
        cite = has_citation(ans)

        retrieved_chunk_ids = [r.chunk_id for r in results]
        retrieved_pages = [(r.doc_name, r.page_start, r.page_end) for r in results]
        hit_pages = pages_hit_expected(retrieved_pages, it.expected_pages)

        preview = ans.strip().replace("\n", " ")
        preview = preview[:160] + ("…" if len(preview) > 160 else "")

        rows.append(
            EvalRow(
                id=it.id,
                question=it.question,
                top_k=int(top_k),
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieved_pages=retrieved_pages,
                retrieval_hits_expected_pages=hit_pages,
                keyword_coverage=kw,
                has_citation=cite,
                retrieval_latency_s=retr_latency,
                llm_latency_s=llm_latency,
                total_latency_s=total,
                answer_preview=preview,
            )
        )

    summary = summarize_rows(rows)
    report = format_report(summary, rows)
    return report, rate_state


def clear_session():
    return [], [], None, [], "Cleared.", "No extracted text to preview yet.", "No chunks to preview yet.", "Index cleared.", "No results.", "No answer yet.", "No eval run yet."


def api_key_status() -> str:
    g = "✅" if env_has_groq() else "❌"
    m = "✅" if env_has_gemini() else "❌"
    return (
        "### API key status (Space secrets)\n"
        f"- GROQ_API_KEY: {g}\n"
        f"- GEMINI_API_KEY: {m}\n"
        "\nSet these in **Space → Settings → Secrets**."
    )

def _access_ok(user_code: str) -> bool:
    secret = os.getenv("APP_ACCESS_CODE", "").strip()
    # If no secret is set, we treat it as "open"
    if not secret:
        return True
    return (user_code or "").strip() == secret

with gr.Blocks(title="Research PDF RAG Chatbot (Ticket 5)") as demo:
    gr.Markdown(
        """
# Research PDF RAG Chatbot
"""
    )

    records_state = gr.State([])
    chunks_state = gr.State([])
    store_state = gr.State(None)
    rate_state = gr.State([])

    access_code = gr.Textbox(
        label="Access code (required for LLM features)",
        type="password",
        placeholder="Ask Mohammad for the code",
    )


    with gr.Tabs():
        with gr.Tab("Upload PDF(s)"):
            pdfs = gr.File(label="Upload PDFs", file_types=[".pdf"], file_count="multiple", type="filepath")
            with gr.Row():
                btn_extract = gr.Button("Extract text", variant="primary")
                btn_clear = gr.Button("Clear session")

        with gr.Tab("Paste text"):
            pasted = gr.Textbox(label="Paste text", lines=12)
            btn_use_text = gr.Button("Use this text", variant="primary")

        with gr.Tab("Chunking"):
            with gr.Row():
                chunk_size = gr.Slider(600, 2400, value=1200, step=50, label="Chunk size")
                chunk_overlap = gr.Slider(0, 600, value=200, step=25, label="Chunk overlap")
                min_chunk_chars = gr.Slider(50, 600, value=200, step=25, label="Min chunk chars")
            btn_chunk = gr.Button("Create chunks", variant="primary")

        with gr.Tab("Index"):
            model_name = gr.Dropdown(EMBEDDING_MODELS, value=EMBEDDING_MODELS[0], label="Embedding model")
            btn_index = gr.Button("Build FAISS index", variant="primary")
            index_status = gr.Textbox(label="Index status", lines=6)

        with gr.Tab("Ask"):
            gr.Markdown(api_key_status())
            provider = gr.Dropdown(["auto", "groq", "gemini"], value="auto", label="LLM provider")
            with gr.Row():
                groq_model = gr.Textbox(label="Groq model", value=GROQ_DEFAULT_MODEL)
                gemini_model = gr.Textbox(label="Gemini model", value=GEMINI_DEFAULT_MODEL)
            with gr.Row():
                top_k = gr.Slider(1, 10, value=DEFAULT_TOP_K, step=1, label="Top-k sources")
                temperature = gr.Slider(0.0, 1.0, value=DEFAULT_TEMPERATURE, step=0.05, label="Temperature")
                max_output_tokens = gr.Slider(128, 1024, value=DEFAULT_MAX_OUTPUT_TOKENS, step=32, label="Max output tokens")
            question = gr.Textbox(label="Question", lines=2)
            btn_answer = gr.Button("Answer (RAG)", variant="primary")
            answer_out = gr.Markdown("No answer yet.")
            retrieval_out = gr.Markdown("No retrieval yet.")

        with gr.Tab("Evaluation"):
            gr.Markdown(
                "Runs a small golden test set through retrieval + LLM.\n\n"
                "**Evaluation uses Gemini-only** to avoid Groq free-tier rate limits.\n\n"
                "Edit `eval/golden.json` to match your uploaded PDFs for meaningful results."
            )
            golden_path = gr.Textbox(label="Golden set path", value="eval/golden.json")
            btn_eval = gr.Button("Run evaluation", variant="primary")
            eval_out = gr.Markdown("No eval run yet.")

    with gr.Row():
        stats_records = gr.Textbox(label="Ingestion stats", lines=10)
        stats_chunks = gr.Textbox(label="Chunking stats", lines=10)

    with gr.Row():
        preview_records = gr.Textbox(label="Extracted preview", lines=18)
        preview_chunks = gr.Textbox(label="Chunk preview", lines=18)

    # Wire up actions
    btn_extract.click(fn=load_pdfs, inputs=[pdfs, records_state], outputs=[records_state, stats_records, preview_records])
    btn_use_text.click(fn=load_pasted_text, inputs=[pasted], outputs=[records_state, stats_records, preview_records])
    btn_chunk.click(fn=run_chunking, inputs=[records_state, chunk_size, chunk_overlap, min_chunk_chars], outputs=[chunks_state, stats_chunks, preview_chunks])
    btn_index.click(fn=build_index, inputs=[chunks_state, model_name], outputs=[store_state, index_status])

    btn_answer.click(
        fn=rag_answer,
        inputs=[store_state, question, provider, groq_model, gemini_model, top_k, temperature, max_output_tokens, rate_state, access_code],
        outputs=[answer_out, retrieval_out, rate_state],
    )

    btn_eval.click(
        fn=run_eval,
        inputs=[store_state, provider, groq_model, gemini_model, top_k, temperature, max_output_tokens, golden_path, rate_state, access_code],
        outputs=[eval_out, rate_state],
    )

    btn_clear.click(
        fn=clear_session,
        inputs=[],
        outputs=[
            records_state, chunks_state, store_state, rate_state,
            stats_records, preview_records, preview_chunks,
            index_status, retrieval_out, answer_out, eval_out
        ],
    )

if __name__ == "__main__":
    demo.launch()
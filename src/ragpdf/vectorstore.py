from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    rank: int
    score: float
    chunk_id: str
    doc_name: str
    page_start: int
    page_end: int
    text: str


class VectorStoreError(Exception):
    pass


def _to_float32(a: np.ndarray) -> np.ndarray:
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    return a


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # normalize row-wise
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def build_faiss_index(
    chunks: List[Dict[str, Any]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    max_chunks: int = 1200,
) -> Dict[str, Any]:
    """
    Builds a cosine-similarity FAISS index using IndexFlatIP (inner product on normalized vectors).
    Returns a dict suitable for storing in gr.State (contains the FAISS index object and metadata).
    """
    if not chunks:
        raise VectorStoreError("No chunks provided. Run Ticket 2 chunking first.")

    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]

    # Load embedding model (downloads once & cached by HF/transformers)
    model = SentenceTransformer(model_name)

    texts = [c.get("text", "") or "" for c in chunks]
    if not any(t.strip() for t in texts):
        raise VectorStoreError("All chunks are empty. Cannot build index.")

    # Embed in batches
    embs_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embs_list.append(emb)

    embs = np.vstack(embs_list)
    embs = _to_float32(embs)
    embs = _l2_normalize(embs)

    dim = embs.shape[1]

    # Cosine similarity = inner product on normalized vectors
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    # Store chunk metadata in same order as vectors
    meta = []
    for c in chunks:
        meta.append(
            {
                "chunk_id": str(c.get("chunk_id", "")),
                "doc_name": str(c.get("doc_name", "")),
                "page_start": int(c.get("page_start", 1)),
                "page_end": int(c.get("page_end", 1)),
                "text": str(c.get("text", "")),
            }
        )

    return {
        "model_name": model_name,
        "dim": dim,
        "index": index,
        "meta": meta,
    }


def retrieve(
    store: Dict[str, Any],
    query: str,
    top_k: int = 5,
) -> List[RetrievedChunk]:
    if not store or "index" not in store or "meta" not in store:
        raise VectorStoreError("Index not built yet. Click 'Build FAISS index' first.")

    query = (query or "").strip()
    if not query:
        return []

    index: faiss.Index = store["index"]
    meta: List[Dict[str, Any]] = store["meta"]
    model_name: str = store["model_name"]

    # Reuse model for query embedding (load from cache)
    model = SentenceTransformer(model_name)
    q = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
    q = _to_float32(q)
    q = _l2_normalize(q)

    top_k = int(top_k)
    top_k = max(1, min(top_k, 10))

    scores, ids = index.search(q, top_k)
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    results: List[RetrievedChunk] = []
    rank = 1
    for score, idx in zip(scores, ids):
        if idx < 0 or idx >= len(meta):
            continue
        m = meta[idx]
        results.append(
            RetrievedChunk(
                rank=rank,
                score=float(score),
                chunk_id=m["chunk_id"],
                doc_name=m["doc_name"],
                page_start=int(m["page_start"]),
                page_end=int(m["page_end"]),
                text=m["text"],
            )
        )
        rank += 1

    return results
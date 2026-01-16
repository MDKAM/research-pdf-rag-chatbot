from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    doc_name: str
    page_start: int
    page_end: int
    text: str


class ChunkingError(Exception):
    pass


_whitespace_re = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = s or ""
    s = s.replace("\u00a0", " ")  # non-breaking space
    s = _whitespace_re.sub(" ", s) # Replace any sequence of whitespace with a single space
    return s.strip()


def chunk_pages(
    page_records: List[Dict[str, Any]],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    min_chunk_chars: int = 200,
    max_chunks: int = 1200,
) -> List[ChunkRecord]:
    """
    Build chunks from page-level records.
    - Keeps page ranges for citations.
    - Chunking is by characters, with overlap.
    """
    if chunk_size <= 0:
        raise ChunkingError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ChunkingError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ChunkingError("chunk_overlap must be smaller than chunk_size")

    # Group pages by doc_name (and doc_id)
    docs: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
    for r in page_records:
        doc_id = str(r.get("doc_id", "unknown"))
        doc_name = str(r.get("doc_name", "unknown"))
        page = int(r.get("page", 1))
        text = normalize_text(r.get("text", ""))

        docs.setdefault((doc_id, doc_name), [])
        docs[(doc_id, doc_name)].append((page, text))

    chunks: List[ChunkRecord] = []

    for (doc_id, doc_name), pages in docs.items():
        # sort by page
        pages.sort(key=lambda x: x[0])

        # Build a single stream but keep mapping from character offsets to pages
        # We'll store tuples: (page_num, start_char_in_stream, end_char_in_stream)
        stream_parts: List[str] = []
        page_spans: List[Tuple[int, int, int]] = []

        cursor = 0
        for page_num, page_text in pages:
            page_text = normalize_text(page_text)
            if not page_text:
                # still create a span of length 0; it won't contribute
                page_spans.append((page_num, cursor, cursor))
                continue

            # Add a clear separator between pages to avoid merging words
            prefix = f"\n\n[PAGE {page_num}]\n"
            part = prefix + page_text

            start = cursor
            stream_parts.append(part)
            cursor += len(part)
            end = cursor
            page_spans.append((page_num, start, end))

        stream = "".join(stream_parts)
        stream = normalize_text(stream)

        if not stream:
            continue

        # Chunk by character window with overlap
        step = chunk_size - chunk_overlap
        start = 0
        local_idx = 0

        while start < len(stream):
            end = min(len(stream), start + chunk_size)
            text_chunk = normalize_text(stream[start:end])

            if len(text_chunk) >= min_chunk_chars:
                # Determine page_start/page_end by overlap with original page spans
                p_start, p_end = _page_range_for_span(page_spans, start, end)

                chunk_id = f"{_slug(doc_name)}:p{p_start}-p{p_end}:c{local_idx:04d}"
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        page_start=p_start,
                        page_end=p_end,
                        text=text_chunk,
                    )
                )
                local_idx += 1

                if len(chunks) >= max_chunks:
                    return chunks

            if end == len(stream):
                break
            start += step

    return chunks


def _page_range_for_span(
    page_spans: List[Tuple[int, int, int]],
    span_start: int,
    span_end: int,
) -> Tuple[int, int]:
    """
    Given page spans in the concatenated stream, find which pages overlap the chunk span.
    """
    overlapping = []
    for page_num, p_start, p_end in page_spans:
        # overlap condition
        if p_end <= span_start:
            continue
        if p_start >= span_end:
            continue
        overlapping.append(page_num)

    if not overlapping:
        # fallback: unknown, but keep 1..1
        return 1, 1
    return min(overlapping), max(overlapping)


def _slug(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:40] if s else "doc"
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import fitz  # PyMuPDF


@dataclass
class PageRecord:
    doc_id: str
    doc_name: str
    page: int  # 1-indexed
    text: str


class IngestError(Exception):
    pass


def _safe_filename(path: str) -> str:
    base = os.path.basename(path)
    # Keep it simple; later we can do better normalization
    return base if base else "document.pdf"


def extract_pdf_pages(
    pdf_path: str,
    doc_id: str,
    max_pages_per_pdf: int,
    max_total_chars: int,
    existing_total_chars: int = 0,
) -> Tuple[List[PageRecord], int]:
    """
    Extract text per page from a PDF, keeping page numbers.
    Returns: (records, updated_total_chars)
    """
    if not os.path.exists(pdf_path):
        raise IngestError(f"File not found: {pdf_path}")

    doc_name = _safe_filename(pdf_path)
    records: List[PageRecord] = []

    try:
        pdf = fitz.open(pdf_path)
    except Exception as e:
        raise IngestError(f"Could not open PDF '{doc_name}': {e}") from e

    total_chars = existing_total_chars

    n_pages = pdf.page_count
    to_read = min(n_pages, max_pages_per_pdf)

    for i in range(to_read):
        page = pdf.load_page(i)
        text = page.get_text("text") or ""
        text = text.strip()

        if text:
            # Guardrail: cap total extracted chars across all files
            if total_chars + len(text) > max_total_chars:
                remaining = max_total_chars - total_chars
                if remaining <= 0:
                    break
                text = text[:remaining]
            total_chars += len(text)

        records.append(
            PageRecord(
                doc_id=doc_id,
                doc_name=doc_name,
                page=i + 1,  # 1-indexed
                text=text,
            )
        )

        if total_chars >= max_total_chars:
            break

    pdf.close()
    return records, total_chars


def validate_uploads(
    filepaths: List[str],
    max_file_size_mb: int,
) -> None:
    """
    Guardrail: file size check.
    """
    if not filepaths:
        raise IngestError("No files provided.")

    max_bytes = max_file_size_mb * 1024 * 1024
    for p in filepaths:
        try:
            size = os.path.getsize(p)
        except OSError:
            raise IngestError(f"Could not read uploaded file size: {p}")

        if size > max_bytes:
            raise IngestError(
                f"File too large: {os.path.basename(p)} is {size/1024/1024:.1f} MB. "
                f"Max allowed is {max_file_size_mb} MB."
            )
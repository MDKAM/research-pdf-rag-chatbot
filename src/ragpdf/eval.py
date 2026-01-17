from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz


@dataclass
class EvalItem:
    id: str
    question: str
    expected_keywords: List[str]
    expected_pages: List[int]  # optional


@dataclass
class EvalRow:
    id: str
    question: str
    top_k: int
    retrieved_chunk_ids: List[str]
    retrieved_pages: List[Tuple[str, int, int]]  # (doc_name, p_start, p_end)
    retrieval_hits_expected_pages: Optional[bool]
    keyword_coverage: float
    has_citation: bool
    retrieval_latency_s: float
    llm_latency_s: float
    total_latency_s: float
    answer_preview: str


_cite_re = re.compile(r"\[[a-z0-9\-]+:p\d+\-p\d+:c\d{4}\]")


def load_golden(path: str) -> List[EvalItem]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    for it in data.get("items", []):
        items.append(
            EvalItem(
                id=str(it.get("id", "")),
                question=str(it.get("question", "")),
                expected_keywords=[str(x) for x in it.get("expected_keywords", [])],
                expected_pages=[int(x) for x in it.get("expected_pages", []) if str(x).strip().isdigit()],
            )
        )
    return items


def keyword_coverage(answer: str, expected_keywords: List[str]) -> float:
    """
    Simple heuristic: check if answer roughly contains expected keywords
    using fuzzy token matching.
    """
    if not expected_keywords:
        return 1.0
    ans = (answer or "").lower()
    if not ans.strip():
        return 0.0

    hits = 0
    for kw in expected_keywords:
        kw = (kw or "").lower().strip()
        if not kw:
            continue

        # exact substring hit is best
        if kw in ans:
            hits += 1
            continue

        # fuzzy fallback: compare keyword to best-matching window
        # cheap approach: fuzz partial ratio to entire answer
        score = fuzz.partial_ratio(kw, ans)
        if score >= 80:
            hits += 1

    denom = max(1, len([k for k in expected_keywords if (k or "").strip()]))
    return hits / denom


def has_citation(answer: str) -> bool:
    return bool(_cite_re.search(answer or ""))


def pages_hit_expected(
    retrieved_pages: List[Tuple[str, int, int]],
    expected_pages: List[int],
) -> Optional[bool]:
    """
    If expected_pages is empty, return None (not applicable).
    Otherwise return True if any retrieved range overlaps any expected page.
    """
    if not expected_pages:
        return None
    exp = set(expected_pages)
    for _, p1, p2 in retrieved_pages:
        for p in range(p1, p2 + 1):
            if p in exp:
                return True
    return False


def summarize_rows(rows: List[EvalRow]) -> Dict[str, Any]:
    if not rows:
        return {"error": "no rows"}

    n = len(rows)
    citation_rate = sum(1 for r in rows if r.has_citation) / n
    avg_kw = sum(r.keyword_coverage for r in rows) / n

    # only count expected-page hits where applicable
    applicable = [r for r in rows if r.retrieval_hits_expected_pages is not None]
    if applicable:
        page_hit_rate = sum(1 for r in applicable if r.retrieval_hits_expected_pages) / len(applicable)
    else:
        page_hit_rate = None

    avg_retr = sum(r.retrieval_latency_s for r in rows) / n
    avg_llm = sum(r.llm_latency_s for r in rows) / n
    avg_total = sum(r.total_latency_s for r in rows) / n

    return {
        "n": n,
        "citation_rate": citation_rate,
        "avg_keyword_coverage": avg_kw,
        "expected_page_hit_rate": page_hit_rate,
        "avg_retrieval_latency_s": avg_retr,
        "avg_llm_latency_s": avg_llm,
        "avg_total_latency_s": avg_total,
    }


def format_report(summary: Dict[str, Any], rows: List[EvalRow]) -> str:
    lines = []
    lines.append("## Evaluation Report")
    lines.append("")
    lines.append(f"- Questions: {summary.get('n')}")
    lines.append(f"- Citation rate: {summary.get('citation_rate', 0):.2f}")
    lines.append(f"- Avg keyword coverage: {summary.get('avg_keyword_coverage', 0):.2f}")
    ehr = summary.get("expected_page_hit_rate")
    if ehr is None:
        lines.append("- Expected-page hit rate: N/A (no expected_pages provided)")
    else:
        lines.append(f"- Expected-page hit rate: {ehr:.2f}")

    lines.append(f"- Avg retrieval latency: {summary.get('avg_retrieval_latency_s', 0):.2f}s")
    lines.append(f"- Avg LLM latency: {summary.get('avg_llm_latency_s', 0):.2f}s")
    lines.append(f"- Avg total latency: {summary.get('avg_total_latency_s', 0):.2f}s")
    lines.append("")
    lines.append("### Per-question details")
    for r in rows:
        lines.append(f"- **{r.id}**: cite={r.has_citation} | kw={r.keyword_coverage:.2f} | total={r.total_latency_s:.2f}s")
        lines.append(f"  - Retrieved: {', '.join(r.retrieved_chunk_ids[:3])}{'...' if len(r.retrieved_chunk_ids) > 3 else ''}")
        lines.append(f"  - Answer preview: {r.answer_preview}")
    return "\n".join(lines)
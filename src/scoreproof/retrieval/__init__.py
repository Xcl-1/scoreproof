"""检索层：结构化（主）+ 原文兜底 + 双通道调度。"""

from .router import (
    REFUSAL_MESSAGE,
    ChannelName,
    Clause,
    LexicalRetriever,
    RetrievalHit,
    RetrievalResult,
    Retriever,
    Router,
    StructuredChannel,
    VectorChannel,
    clauses_from_pdf_pages,
    extract_score_candidates,
)

__all__ = [
    "REFUSAL_MESSAGE",
    "ChannelName",
    "Clause",
    "LexicalRetriever",
    "RetrievalHit",
    "RetrievalResult",
    "Retriever",
    "Router",
    "StructuredChannel",
    "VectorChannel",
    "clauses_from_pdf_pages",
    "extract_score_candidates",
]

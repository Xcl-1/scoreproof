"""检索层：结构化（主）+ 原文兜底 + 双通道调度。"""

from .hybrid import BM25Retriever, HybridRetriever, reciprocal_rank_fusion
from .query import DEFAULT_QUERY_ALIASES, QueryRewritingRetriever, rewrite_retrieval_query
from .rerank import FastEmbedReranker, Reranker, RerankingRetriever
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
    "BM25Retriever",
    "DEFAULT_QUERY_ALIASES",
    "FastEmbedReranker",
    "HybridRetriever",
    "QueryRewritingRetriever",
    "LexicalRetriever",
    "RetrievalHit",
    "RetrievalResult",
    "Retriever",
    "Reranker",
    "RerankingRetriever",
    "Router",
    "StructuredChannel",
    "VectorChannel",
    "clauses_from_pdf_pages",
    "extract_score_candidates",
    "reciprocal_rank_fusion",
    "rewrite_retrieval_query",
]

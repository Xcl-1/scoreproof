"""Manifest 感知的 BM25 + Chroma 双路召回与 RRF 融合。"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence

from ..indexing.hybrid import BM25IndexDocument, HybridIndexManifestStore
from ..schema import SourceRef
from ..tokenize import tokenize_for_search
from .router import Clause, RetrievalHit


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievalHit]],
    *,
    rrf_k: int = 60,
    top_k: int = 5,
    weights: Sequence[float] | None = None,
) -> list[RetrievalHit]:
    """按条款 ID 融合多路排名；分数尺度不参与融合。"""
    if rrf_k < 1 or top_k < 1:
        raise ValueError("rrf_k 与 top_k 必须大于 0")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings) or any(weight <= 0 for weight in weights):
        raise ValueError("weights 必须与召回通道一一对应且全部大于 0")

    scores: dict[str, float] = {}
    clauses: dict[str, Clause] = {}
    component_ranks: dict[str, dict[str, int]] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for fallback_rank, hit in enumerate(ranking, start=1):
            rank = hit.rank if hit.rank > 0 else fallback_rank
            clause_id = hit.clause.id
            scores[clause_id] = scores.get(clause_id, 0.0) + weight / (rrf_k + rank)
            clauses[clause_id] = hit.clause
            if hit.channel:
                component_ranks.setdefault(clause_id, {})[hit.channel] = rank

    ordered = sorted(scores, key=lambda clause_id: (-scores[clause_id], clause_id))[:top_k]
    return [
        RetrievalHit(
            clause=clauses[clause_id],
            score=scores[clause_id],
            rank=rank,
            channel="rrf",
            component_ranks=component_ranks.get(clause_id, {}),
        )
        for rank, clause_id in enumerate(ordered, start=1)
    ]


class HybridRetriever:
    """只查询活动 Manifest 所指向的完整 BM25/向量批次。"""

    def __init__(
        self,
        store: HybridIndexManifestStore,
        *,
        academic_year: str | None = None,
        college: str | None = None,
        doc_id: str | None = None,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
        query_rewriter: Callable[[str], str] | None = None,
    ) -> None:
        if bm25_weight <= 0 or vector_weight <= 0:
            raise ValueError("召回权重必须大于 0")
        self.store = store
        self.academic_year = academic_year
        self.college = college
        self.doc_id = doc_id
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.query_rewriter = query_rewriter

    def available(self) -> bool:
        return bool(self.store.active_batches(doc_id=self.doc_id))

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        if not query.strip() or top_k < 1 or not self.available():
            return []
        fetch_k = max(top_k * 3, top_k)
        bm25 = self.search_bm25(query, top_k=fetch_k)
        vector = self.search_vector(query, top_k=fetch_k)
        return reciprocal_rank_fusion(
            [bm25, vector],
            rrf_k=self.rrf_k,
            top_k=top_k,
            weights=[self.bm25_weight, self.vector_weight],
        )

    def search_bm25(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        documents = [
            document
            for document in self.store.active_bm25_documents(doc_id=self.doc_id)
            if self._eligible(document.metadata)
        ]
        lexical_query = self.query_rewriter(query) if self.query_rewriter else query
        tokens = tokenize_for_search(lexical_query)
        if not documents or not tokens:
            return []
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover - 最小安装环境
            raise RuntimeError("BM25 不可用：请安装 scoreproof[retrieval]") from exc

        bm25 = BM25Okapi([document.tokens for document in documents])
        scores = bm25.get_scores(tokens)
        ordered = sorted(
            range(len(documents)),
            key=lambda index: (-float(scores[index]), _clause_id(documents[index])),
        )
        hits: list[RetrievalHit] = []
        for index in ordered:
            score = float(scores[index])
            if score <= 0:
                continue
            rank = len(hits) + 1
            hits.append(
                RetrievalHit(
                    clause=_bm25_clause(documents[index]),
                    score=score,
                    rank=rank,
                    channel="bm25",
                    component_ranks={"bm25": rank},
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def search_vector(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        candidates: list[tuple[Clause, float]] = []
        for batch in self.store.active_batches(doc_id=self.doc_id):
            if batch.embedding_model != self.store.embedding_model:
                continue
            vector_store = self.store.chroma_store(batch)
            count = batch.vector_doc_count
            if not count:
                continue
            for document, distance in vector_store.similarity_search_with_score(query, k=count):
                metadata = document.metadata
                extra = _metadata_dict(metadata.get("metadata"))
                if not self._eligible(extra):
                    continue
                source = SourceRef.model_validate_json(str(metadata["source"]))
                clause = Clause(
                    id=f"{metadata['doc_id']}:{metadata['logical_key']}",
                    text=document.page_content,
                    source=source,
                    academic_year=_optional_text(extra.get("academic_year")),
                    college=_optional_text(extra.get("college")),
                    meta={**extra, "manifest_id": batch.manifest_id},
                )
                similarity = 1.0 / (1.0 + max(float(distance), 0.0))
                candidates.append((clause, similarity))
        candidates.sort(key=lambda item: (-item[1], item[0].id))
        return [
            RetrievalHit(
                clause=clause,
                score=score,
                rank=rank,
                channel="vector",
                component_ranks={"vector": rank},
            )
            for rank, (clause, score) in enumerate(candidates[:top_k], start=1)
        ]

    def _eligible(self, metadata: Mapping[str, object]) -> bool:
        year = _optional_text(metadata.get("academic_year"))
        college = _optional_text(metadata.get("college"))
        if self.academic_year and year and year != self.academic_year:
            return False
        return not (self.college and college and college != self.college)


class BM25Retriever:
    """消融实验 A 档：复用同一活动批次，但只启用 BM25。"""

    def __init__(self, hybrid: HybridRetriever) -> None:
        self.hybrid = hybrid

    def available(self) -> bool:
        return self.hybrid.available()

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        return self.hybrid.search_bm25(query, top_k=top_k)


def _bm25_clause(document: BM25IndexDocument) -> Clause:
    return Clause(
        id=_clause_id(document),
        text=document.text,
        source=document.source,
        academic_year=_optional_text(document.metadata.get("academic_year")),
        college=_optional_text(document.metadata.get("college")),
        meta={**document.metadata, "manifest_id": document.manifest_id},
    )


def _clause_id(document: BM25IndexDocument) -> str:
    return f"{document.doc_id}:{document.logical_key}"


def _metadata_dict(value: object) -> dict[str, object]:
    if isinstance(value, str):
        decoded = json.loads(value)
        return decoded if isinstance(decoded, dict) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_text(value: object) -> str | None:
    return str(value) if value not in (None, "") else None


__all__ = ["BM25Retriever", "HybridRetriever", "reciprocal_rank_fusion"]

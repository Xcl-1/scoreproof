"""混合召回后的可替换精排层。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from .router import RetrievalHit, Retriever


class Reranker(Protocol):
    model_version: str

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class FastEmbedReranker:
    """使用 FastEmbed ONNX CrossEncoder 对 query/document 对打分。"""

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-reranker-base",
        cache_dir: str | Path | None = None,
        threads: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_version = f"fastembed:{model_name}"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.threads = threads
        self._model = None

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        return [float(score) for score in self._load().rerank(query, list(documents))]

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:  # pragma: no cover - 最小安装环境
            raise RuntimeError("Rerank 不可用：请安装 scoreproof[retrieval]") from exc
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = TextCrossEncoder(
            model_name=self.model_name,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            threads=self.threads,
        )
        return self._model


class RerankingRetriever:
    """对基础召回 Top-N 做 CrossEncoder 精排。"""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        *,
        candidate_k: int = 20,
        base_rank_weight: float = 4.0,
        rerank_rank_weight: float = 1.0,
        fusion_k: int = 60,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k 必须大于 0")
        if base_rank_weight <= 0 or rerank_rank_weight <= 0 or fusion_k < 1:
            raise ValueError("精排融合权重与 fusion_k 必须大于 0")
        self.retriever = retriever
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.base_rank_weight = base_rank_weight
        self.rerank_rank_weight = rerank_rank_weight
        self.fusion_k = fusion_k
        self.last_pair_count = 0

    def available(self) -> bool:
        return self.retriever.available()

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        if top_k < 1 or not query.strip() or not self.available():
            self.last_pair_count = 0
            return []
        candidates = self.retriever.search(query, top_k=max(top_k, self.candidate_k))
        scores = self.reranker.score(query, [hit.clause.text for hit in candidates])
        if len(scores) != len(candidates):
            raise ValueError("Reranker 返回数量与候选数量不一致")
        self.last_pair_count = len(candidates)
        cross_encoder_ranking = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-item[1], item[0].rank, item[0].clause.id),
        )
        rerank_ranks = {
            hit.clause.id: rank
            for rank, (hit, _) in enumerate(cross_encoder_ranking, start=1)
        }
        score_by_id = {hit.clause.id: score for hit, score in cross_encoder_ranking}
        ranked = sorted(
            candidates,
            key=lambda hit: (
                -(
                    self.base_rank_weight / (self.fusion_k + hit.rank)
                    + self.rerank_rank_weight
                    / (self.fusion_k + rerank_ranks[hit.clause.id])
                ),
                hit.clause.id,
            ),
        )[:top_k]
        return [
            RetrievalHit(
                clause=hit.clause,
                score=(
                    self.base_rank_weight / (self.fusion_k + hit.rank)
                    + self.rerank_rank_weight
                    / (self.fusion_k + rerank_ranks[hit.clause.id])
                ),
                rank=rank,
                channel="rerank",
                component_ranks={
                    **hit.component_ranks,
                    "rrf": hit.rank,
                    "rerank": rerank_ranks[hit.clause.id],
                },
                rerank_score=score_by_id[hit.clause.id],
            )
            for rank, hit in enumerate(ranked, start=1)
        ]


__all__ = ["FastEmbedReranker", "Reranker", "RerankingRetriever"]

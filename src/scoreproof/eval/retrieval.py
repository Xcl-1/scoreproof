"""检索冻结集评测：A/B/C 三档指标、置信区间、延迟与调用成本。"""

from __future__ import annotations

import math
import random
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..retrieval.router import Retriever


class RetrievalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_ids: list[str] = Field(min_length=1)
    intent: str | None = None


class MetricEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float = Field(ge=0.0, le=1.0)
    ci95_low: float = Field(ge=0.0, le=1.0)
    ci95_high: float = Field(ge=0.0, le=1.0)


class RetrievalVariantReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    variant: str
    sample_size: int = Field(ge=1)
    hit_at_5: MetricEstimate
    mrr_at_10: MetricEstimate
    ndcg_at_10: MetricEstimate
    latency_p50_ms: float = Field(ge=0)
    latency_p95_ms: float = Field(ge=0)
    embedding_query_count: int = Field(default=0, ge=0)
    rerank_pair_count: int = Field(default=0, ge=0)
    failures: list[str] = Field(default_factory=list)


class RetrievalAblationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_version: str
    dataset_kind: str
    variants: list[RetrievalVariantReport] = Field(min_length=3, max_length=3)
    deltas: dict[str, float]
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_variants(self) -> RetrievalAblationReport:
        if [variant.variant for variant in self.variants] != ["A", "B", "C"]:
            raise ValueError("三档报告顺序必须为 A/B/C")
        return self


def load_retrieval_cases(path: str | Path) -> tuple[str, str, list[RetrievalCase]]:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases: list[RetrievalCase] = []
    for group in payload["groups"]:
        for index, item in enumerate(group["queries"], start=1):
            query = item["text"] if isinstance(item, dict) else item
            relevant_ids = (
                item.get("relevant_ids", group["relevant_ids"])
                if isinstance(item, dict)
                else group["relevant_ids"]
            )
            cases.append(
                RetrievalCase(
                    id=f"{group['id']}-{index:02d}",
                    query=query,
                    relevant_ids=relevant_ids,
                    intent=group.get("intent"),
                )
            )
    return str(payload["dataset_version"]), str(payload["dataset_kind"]), cases


def evaluate_retriever(
    retriever: Retriever,
    cases: Sequence[RetrievalCase],
    *,
    variant: str,
    embedding_count: Callable[[], int] | None = None,
    rerank_pair_count: Callable[[], int] | None = None,
) -> RetrievalVariantReport:
    if not cases:
        raise ValueError("检索评测集不能为空")
    hit_values: list[float] = []
    rr_values: list[float] = []
    ndcg_values: list[float] = []
    latencies: list[float] = []
    failures: list[str] = []
    initial_embeddings = embedding_count() if embedding_count else 0
    total_rerank_pairs = 0
    for case in cases:
        started = time.perf_counter()
        hits = retriever.search(case.query, top_k=10)
        latencies.append((time.perf_counter() - started) * 1000)
        if rerank_pair_count:
            total_rerank_pairs += rerank_pair_count()
        ranked_ids = [hit.clause.id for hit in hits]
        relevant = set(case.relevant_ids)
        ranks = [rank for rank, clause_id in enumerate(ranked_ids, start=1) if clause_id in relevant]
        first_rank = min(ranks, default=None)
        hit_values.append(float(first_rank is not None and first_rank <= 5))
        rr_values.append(1.0 / first_rank if first_rank is not None and first_rank <= 10 else 0.0)
        ndcg_values.append(_ndcg(ranked_ids[:10], relevant))
        if first_rank is None or first_rank > 5:
            failures.append(case.id)
    embeddings = (embedding_count() - initial_embeddings) if embedding_count else 0
    return RetrievalVariantReport(
        variant=variant,
        sample_size=len(cases),
        hit_at_5=_wilson(hit_values),
        mrr_at_10=_bootstrap(rr_values),
        ndcg_at_10=_bootstrap(ndcg_values),
        latency_p50_ms=round(statistics.median(latencies), 3),
        latency_p95_ms=round(_percentile(latencies, 0.95), 3),
        embedding_query_count=embeddings,
        rerank_pair_count=total_rerank_pairs,
        failures=failures,
    )


def build_ablation_report(
    *,
    dataset_version: str,
    dataset_kind: str,
    variants: Sequence[RetrievalVariantReport],
    notes: Sequence[str] = (),
) -> RetrievalAblationReport:
    if len(variants) != 3:
        raise ValueError("消融实验必须包含 A/B/C 三档")
    a, b, c = variants
    return RetrievalAblationReport(
        dataset_version=dataset_version,
        dataset_kind=dataset_kind,
        variants=list(variants),
        deltas={
            "hit_at_5_B_minus_A": round(b.hit_at_5.value - a.hit_at_5.value, 6),
            "hit_at_5_C_minus_B": round(c.hit_at_5.value - b.hit_at_5.value, 6),
            "mrr_at_10_B_minus_A": round(b.mrr_at_10.value - a.mrr_at_10.value, 6),
            "mrr_at_10_C_minus_B": round(c.mrr_at_10.value - b.mrr_at_10.value, 6),
            "ndcg_at_10_B_minus_A": round(b.ndcg_at_10.value - a.ndcg_at_10.value, 6),
            "ndcg_at_10_C_minus_B": round(c.ndcg_at_10.value - b.ndcg_at_10.value, 6),
        },
        notes=list(notes),
    )


def _ndcg(ranked_ids: Sequence[str], relevant: set[str]) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, clause_id in enumerate(ranked_ids, start=1)
        if clause_id in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), 10) + 1))
    return dcg / ideal if ideal else 0.0


def _wilson(values: Sequence[float]) -> MetricEstimate:
    n = len(values)
    successes = sum(values)
    rate = successes / n
    z = 1.959963984540054
    denominator = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return MetricEstimate(
        value=round(rate, 6),
        ci95_low=round(max(0.0, centre - margin), 6),
        ci95_high=round(min(1.0, centre + margin), 6),
    )


def _bootstrap(values: Sequence[float], *, samples: int = 2000) -> MetricEstimate:
    rng = random.Random(20250912)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(samples))
    return MetricEstimate(
        value=round(statistics.mean(values), 6),
        ci95_low=round(_percentile(means, 0.025), 6),
        ci95_high=round(_percentile(means, 0.975), 6),
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return float(ordered[index])


__all__ = [
    "MetricEstimate",
    "RetrievalAblationReport",
    "RetrievalCase",
    "RetrievalVariantReport",
    "build_ablation_report",
    "evaluate_retriever",
    "load_retrieval_cases",
]

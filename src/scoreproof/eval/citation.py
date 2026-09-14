"""引用定位与拒答成对评测，防止用“一律拒答”虚高指标。"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..retrieval.citation import verify_text_citations
from ..retrieval.router import Retriever
from .gateway import wilson_interval
from .retrieval import MetricEstimate, RetrievalCase


class RefusalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    reason: str | None = None


class CitationRefusalReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    positive_dataset_version: str
    positive_dataset_kind: str
    negative_dataset_version: str
    negative_dataset_kind: str
    positive_sample_size: int = Field(ge=1)
    negative_sample_size: int = Field(ge=1)
    citation_location_accuracy: MetricEstimate
    correct_refusal_rate: MetricEstimate
    false_refusal_rate: MetricEstimate
    positive_citation_failures: list[str] = Field(default_factory=list)
    negative_refusal_failures: list[str] = Field(default_factory=list)
    false_refusal_failures: list[str] = Field(default_factory=list)
    latency_p50_ms: float = Field(ge=0)
    latency_p95_ms: float = Field(ge=0)
    embedding_query_count: int = Field(default=0, ge=0)
    passed: bool
    notes: list[str] = Field(default_factory=list)


def load_refusal_cases(path: str | Path) -> tuple[str, str, list[RefusalCase]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [RefusalCase.model_validate(item) for item in payload["cases"]]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("拒答评测集 case id 不得重复")
    return str(payload["dataset_version"]), str(payload["dataset_kind"]), cases


def evaluate_citation_refusal(
    retriever: Retriever,
    positive_cases: Sequence[RetrievalCase],
    negative_cases: Sequence[RefusalCase],
    *,
    positive_dataset_version: str,
    positive_dataset_kind: str,
    negative_dataset_version: str,
    negative_dataset_kind: str,
    embedding_count: Callable[[], int] | None = None,
    notes: Sequence[str] = (),
) -> CitationRefusalReport:
    if not positive_cases or not negative_cases:
        raise ValueError("引用/拒答成对评测的正负样本均不能为空")
    initial_embeddings = embedding_count() if embedding_count else 0
    citation_values: list[bool] = []
    refusal_values: list[bool] = []
    false_refusal_values: list[bool] = []
    citation_failures: list[str] = []
    negative_failures: list[str] = []
    false_refusal_failures: list[str] = []
    latencies: list[float] = []

    for case in positive_cases:
        started = time.perf_counter()
        hits = retriever.search(case.query, top_k=5)
        check = verify_text_citations(case.query, hits)
        latencies.append((time.perf_counter() - started) * 1000)
        false_refused = check.disposition == "refuse"
        citation_correct = bool(set(check.citation_ids) & set(case.relevant_ids))
        false_refusal_values.append(false_refused)
        citation_values.append(citation_correct)
        if false_refused:
            false_refusal_failures.append(case.id)
        if not citation_correct:
            citation_failures.append(case.id)

    for case in negative_cases:
        started = time.perf_counter()
        hits = retriever.search(case.query, top_k=5)
        check = verify_text_citations(case.query, hits)
        latencies.append((time.perf_counter() - started) * 1000)
        correct = check.disposition == "refuse"
        refusal_values.append(correct)
        if not correct:
            negative_failures.append(case.id)

    citation_metric = _binary_metric(citation_values)
    refusal_metric = _binary_metric(refusal_values)
    false_refusal_metric = _binary_metric(false_refusal_values)
    embeddings = embedding_count() - initial_embeddings if embedding_count else 0
    return CitationRefusalReport(
        positive_dataset_version=positive_dataset_version,
        positive_dataset_kind=positive_dataset_kind,
        negative_dataset_version=negative_dataset_version,
        negative_dataset_kind=negative_dataset_kind,
        positive_sample_size=len(positive_cases),
        negative_sample_size=len(negative_cases),
        citation_location_accuracy=citation_metric,
        correct_refusal_rate=refusal_metric,
        false_refusal_rate=false_refusal_metric,
        positive_citation_failures=citation_failures,
        negative_refusal_failures=negative_failures,
        false_refusal_failures=false_refusal_failures,
        latency_p50_ms=round(statistics.median(latencies), 3),
        latency_p95_ms=round(_percentile(latencies, 0.95), 3),
        embedding_query_count=embeddings,
        passed=(
            citation_metric.value >= 0.95
            and refusal_metric.value >= 0.98
            and false_refusal_metric.value <= 0.05
        ),
        notes=list(notes),
    )


def _binary_metric(values: Sequence[bool]) -> MetricEstimate:
    successes = sum(values)
    low, high = wilson_interval(successes, len(values))
    return MetricEstimate(
        value=round(successes / len(values), 6),
        ci95_low=round(low, 6),
        ci95_high=round(high, 6),
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * probability + 0.999999)))
    return ordered[index]


__all__ = [
    "CitationRefusalReport",
    "RefusalCase",
    "evaluate_citation_refusal",
    "load_refusal_cases",
]

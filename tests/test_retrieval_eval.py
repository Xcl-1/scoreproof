"""预训练模型适配、Rerank 与检索消融评测测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scoreproof.eval.retrieval import (
    RetrievalCase,
    build_ablation_report,
    evaluate_retriever,
    load_retrieval_cases,
)
from scoreproof.indexing import FastEmbedEmbeddings
from scoreproof.retrieval import RetrievalHit
from scoreproof.retrieval.query import QueryRewritingRetriever, rewrite_retrieval_query
from scoreproof.retrieval.rerank import FastEmbedReranker, RerankingRetriever
from scoreproof.retrieval.router import Clause
from scoreproof.schema import SourceRef


class FakeRetriever:
    def __init__(self, ranked: list[str]) -> None:
        self.ranked = ranked

    def available(self) -> bool:
        return True

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        return [
            RetrievalHit(
                clause=Clause(id=clause_id, text=f"document {clause_id}", source=SourceRef(doc="x")),
                score=1 / rank,
                rank=rank,
                channel="rrf",
            )
            for rank, clause_id in enumerate(self.ranked[:top_k], start=1)
        ]


class FakeFastEmbedModel:
    def embed(self, texts):
        return iter(np.array([index, 1.0]) for index, _ in enumerate(texts, start=1))

    def query_embed(self, text):
        return iter([np.array([2.0, 3.0])])


def test_fastembed_adapter_uses_document_and_query_paths() -> None:
    adapter = FastEmbedEmbeddings(model_name="fake")
    adapter._model = FakeFastEmbedModel()
    assert adapter.embed_documents(["a", "b"]) == [[1.0, 1.0], [2.0, 1.0]]
    assert adapter.embed_query("q") == [2.0, 3.0]
    assert adapter.document_count == 2 and adapter.query_count == 1
    assert adapter.model_version == "fastembed:fake"


class ReverseReranker:
    model_version = "fake-reranker"

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [float(index) for index in range(len(documents))]


def test_query_rewrite_keeps_original_and_adds_canonical_terms() -> None:
    rewritten = rewrite_retrieval_query("CET四级成绩用于免试研究生遴选")
    assert rewritten.startswith("CET四级成绩用于免试研究生遴选")
    assert "大学英语四级" in rewritten and "推免" in rewritten
    assert rewrite_retrieval_query("  ") == ""


def test_query_rewriting_retriever_passes_expanded_query() -> None:
    delegate = FakeRetriever(["a"])
    retriever = QueryRewritingRetriever(delegate, aliases={"志愿服务": ("公益",)})
    assert retriever.search("公益经历")[0].clause.id == "a"
    assert retriever.last_query == "公益经历 志愿服务"


def test_reranking_retriever_fuses_base_and_cross_encoder_ranks() -> None:
    retriever = RerankingRetriever(
        FakeRetriever(["a", "b", "c"]),
        ReverseReranker(),
        candidate_k=3,
        base_rank_weight=1.0,
        rerank_rank_weight=4.0,
    )
    hits = retriever.search("query", top_k=2)
    assert [hit.clause.id for hit in hits] == ["c", "b"]
    assert hits[0].channel == "rerank" and hits[0].rerank_score == 2.0
    assert hits[0].component_ranks["rrf"] == 3
    assert hits[0].component_ranks["rerank"] == 1
    assert retriever.last_pair_count == 3


def test_reranking_rejects_wrong_score_count() -> None:
    class BrokenReranker:
        model_version = "broken"

        def score(self, query: str, documents: list[str]) -> list[float]:
            return []

    retriever = RerankingRetriever(FakeRetriever(["a"]), BrokenReranker())
    with pytest.raises(ValueError, match="数量"):
        retriever.search("query")


def test_fastembed_reranker_uses_cross_encoder_order() -> None:
    class FakeCrossEncoder:
        def rerank(self, query: str, documents: list[str]):
            assert query == "q" and documents == ["a", "b"]
            return iter([0.25, 0.75])

    reranker = FastEmbedReranker(model_name="fake")
    reranker._model = FakeCrossEncoder()
    assert reranker.score("q", ["a", "b"]) == [0.25, 0.75]


def test_frozen_fixture_has_exactly_100_unique_cases() -> None:
    path = Path(__file__).parent / "fixtures" / "retrieval_cases_v1.json"
    version, kind, cases = load_retrieval_cases(path)
    assert version == "fzu-ccds-retrieval-validation-v1"
    assert "not production user logs" in kind
    assert len(cases) == len({case.id for case in cases}) == 100


def test_regression_fixture_is_disjoint_and_has_exactly_100_cases() -> None:
    fixtures = Path(__file__).parent / "fixtures"
    _, _, validation = load_retrieval_cases(fixtures / "retrieval_cases_v1.json")
    version, kind, test_cases = load_retrieval_cases(fixtures / "retrieval_test_v1.json")
    assert version == "fzu-ccds-retrieval-regression-v1"
    assert "regression queries" in kind and "not production user logs" in kind
    assert len(test_cases) == len({case.id for case in test_cases}) == 100
    assert {case.query for case in validation}.isdisjoint(case.query for case in test_cases)


def test_held_out_v2_fixture_is_disjoint_and_has_exactly_100_cases() -> None:
    fixtures = Path(__file__).parent / "fixtures"
    _, _, validation = load_retrieval_cases(fixtures / "retrieval_cases_v1.json")
    _, _, regression = load_retrieval_cases(fixtures / "retrieval_test_v1.json")
    version, kind, test_cases = load_retrieval_cases(fixtures / "retrieval_test_v2.json")
    assert version == "fzu-ccds-retrieval-test-v2"
    assert "held-out test" in kind and "not production user logs" in kind
    assert len(test_cases) == len({case.id for case in test_cases}) == 100
    prior_queries = {case.query for case in [*validation, *regression]}
    assert prior_queries.isdisjoint(case.query for case in test_cases)
    competition = [case for case in test_cases if case.id.startswith("competition-points-v2")]
    assert competition[1].relevant_ids == ["fzu-ccds-2026-rules:page:5:block:26"]


def test_evaluate_retriever_computes_metrics_and_failures() -> None:
    cases = [
        RetrievalCase(id="hit", query="q1", relevant_ids=["a"]),
        RetrievalCase(id="miss", query="q2", relevant_ids=["z"]),
    ]
    report = evaluate_retriever(FakeRetriever(["a", "b"]), cases, variant="A")
    assert report.sample_size == 2
    assert report.hit_at_5.value == 0.5
    assert report.mrr_at_10.value == 0.5
    assert report.ndcg_at_10.value == 0.5
    assert report.failures == ["miss"]
    assert report.hit_at_5.ci95_low <= 0.5 <= report.hit_at_5.ci95_high


def test_ablation_report_discloses_negative_deltas() -> None:
    cases = [RetrievalCase(id="x", query="q", relevant_ids=["a"])]
    good = evaluate_retriever(FakeRetriever(["a"]), cases, variant="A")
    bad = evaluate_retriever(FakeRetriever(["b"]), cases, variant="B")
    final = evaluate_retriever(FakeRetriever(["a"]), cases, variant="C")
    report = build_ablation_report(
        dataset_version="v1", dataset_kind="synthetic", variants=[good, bad, final]
    )
    assert report.deltas["hit_at_5_B_minus_A"] == -1.0
    assert report.deltas["hit_at_5_C_minus_B"] == 1.0


@pytest.mark.parametrize("candidate_k", [0, -1])
def test_reranking_rejects_invalid_candidate_count(candidate_k: int) -> None:
    with pytest.raises(ValueError):
        RerankingRetriever(FakeRetriever([]), ReverseReranker(), candidate_k=candidate_k)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_rank_weight": 0},
        {"rerank_rank_weight": 0},
        {"fusion_k": 0},
    ],
)
def test_reranking_rejects_invalid_fusion_configuration(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="融合"):
        RerankingRetriever(FakeRetriever([]), ReverseReranker(), **kwargs)


def test_ablation_requires_exactly_three_variants() -> None:
    with pytest.raises(ValueError, match="A/B/C"):
        build_ablation_report(dataset_version="v", dataset_kind="x", variants=[])

"""阶段 4.5：结构化引用门禁、文本候选核查与成对拒答评测。"""

from __future__ import annotations

from scoreproof.agent import OrchestrationRequest, ScoreProofOrchestrator
from scoreproof.calc.engine import compute_claims
from scoreproof.eval.citation import (
    RefusalCase,
    evaluate_citation_refusal,
    load_refusal_cases,
)
from scoreproof.eval.retrieval import RetrievalCase
from scoreproof.retrieval.citation import (
    has_precise_locator,
    verify_score_breakdown,
    verify_text_citations,
)
from scoreproof.retrieval.router import Clause, RetrievalHit
from scoreproof.schema import Ruleset, SourceRef

from .conftest import make_claim, make_rule


def _hybrid_hit(clause_id: str = "rules:page:4", text: str = "省级二等奖计 8 分") -> RetrievalHit:
    return RetrievalHit(
        clause=Clause(
            id=clause_id,
            text=text,
            source=SourceRef(doc="细则.pdf", page=4, text=text),
        ),
        score=1.0,
        rank=1,
        channel="rrf",
        component_ranks={"bm25": 1, "vector": 1},
    )


def test_precise_locator_requires_more_than_document_name() -> None:
    assert not has_precise_locator(SourceRef(doc="细则.pdf", text="一段文字"))
    assert has_precise_locator(SourceRef(doc="细则.pdf", page=2))
    assert has_precise_locator(SourceRef(doc="细则.xlsx", table="加分表", row=7))


def test_structured_ledger_is_auto_scored_only_when_rule_and_source_agree() -> None:
    ruleset = Ruleset(rules=[make_rule("省级二等奖", 8, rule_id="r-citation")])
    ledger = compute_claims(
        [make_claim(level="省级二等奖", claim_id="c-citation")],
        ruleset,
        academic_year="2025-2026",
    )
    accepted = verify_score_breakdown(ledger, ruleset)
    assert accepted.supported and accepted.disposition == "auto_score"
    assert accepted.citation_ids == ["r-citation"]

    tampered = ledger.model_copy(deep=True)
    tampered.matches[0].raw_score = 99
    rejected = verify_score_breakdown(tampered, ruleset)
    assert not rejected.supported and rejected.disposition == "refuse"
    assert rejected.unsupported_items == ["c-citation:raw_score"]


def test_orchestrator_blocks_rule_without_precise_source_locator() -> None:
    rule = make_rule("省级二等奖", 8, rule_id="r-no-locator")
    rule.source = SourceRef(doc="细则.pdf", text="省级二等奖计 8 分")
    request = OrchestrationRequest(
        query="学科竞赛省级二等奖如何加分",
        claims=[make_claim(level="省级二等奖")],
        academic_year="2025-2026",
    )
    result = ScoreProofOrchestrator(Ruleset(rules=[rule])).run(request)
    assert result.outcome == "refusal" and result.ledger is None
    assert result.citation_check is not None
    assert result.citation_check.unsupported_items == [
        f"{request.claims[0].id}:source_locator"
    ]
    assert "citation_block" in result.state_trace


def test_text_hit_is_manual_review_and_unquoted_number_is_blocked() -> None:
    hit = _hybrid_hit()
    accepted = verify_text_citations("学科竞赛省级二等奖如何加分", [hit])
    assert accepted.supported and accepted.disposition == "manual_review"
    assert accepted.citation_ids == ["rules:page:4"]

    blocked = verify_text_citations(
        "学科竞赛省级二等奖如何加分", [hit], conclusion="可以加 9 分"
    )
    assert blocked.disposition == "refuse"
    assert blocked.unsupported_items == ["number:9"]


def test_out_of_domain_query_refuses_even_when_vector_search_returns_a_hit() -> None:
    result = verify_text_citations("宿舍空调坏了找谁维修", [_hybrid_hit()])
    assert not result.supported and result.disposition == "refuse"


def test_refusal_fixture_has_required_fifty_unique_cases() -> None:
    version, kind, cases = load_refusal_cases("tests/fixtures/refusal_cases_v1.json")
    assert version == "scoreproof-refusal-boundary-v1"
    assert "not production user logs" in kind
    assert len(cases) == 50 and len({case.id for case in cases}) == 50


def test_pair_evaluation_reports_both_sides_instead_of_rewarding_blanket_refusal() -> None:
    class FakeRetriever:
        def available(self) -> bool:
            return True

        def search(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
            return [_hybrid_hit()] if "竞赛" in query else []

    report = evaluate_citation_refusal(
        FakeRetriever(),
        [RetrievalCase(id="p1", query="学科竞赛省级二等奖如何加分", relevant_ids=["rules:page:4"])],
        [RefusalCase(id="n1", query="宿舍空调坏了找谁维修")],
        positive_dataset_version="positive-v1",
        positive_dataset_kind="unit",
        negative_dataset_version="negative-v1",
        negative_dataset_kind="unit",
    )
    assert report.passed
    assert report.citation_location_accuracy.value == 1
    assert report.correct_refusal_rate.value == 1
    assert report.false_refusal_rate.value == 0

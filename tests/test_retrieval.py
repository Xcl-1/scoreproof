"""检索层测试：双通道调度、置信度、未命中拒答、原文分值候选抽取。"""

from __future__ import annotations

import pytest

from scoreproof.calc.engine import EngineConfig
from scoreproof.retrieval.router import (
    REFUSAL_MESSAGE,
    Clause,
    LexicalRetriever,
    Router,
    VectorChannel,
    extract_score_candidates,
)
from scoreproof.schema import Ruleset, SourceRef

from .conftest import make_claim

# ======================================================================
# 原文分值候选（正则，不让模型心算）
# ======================================================================


class TestScoreCandidates:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("省级二等奖 8 分", [8.0]),
            ("国家级一等奖加 15 分", [15.0]),
            ("获省赛二等奖，计 8 分", [8.0]),
            ("第三章第7条 二等奖 8分", [8.0]),
            ("没有分值描述", []),
            ("", []),
            ("参赛人数 200 人", []),  # 没有"分"字，不应误抽
        ],
    )
    def test_extract(self, text: str, expected: list[float]) -> None:
        assert extract_score_candidates(text) == expected

    def test_no_absurd_numbers(self) -> None:
        """超过 100 的数字（页码/编号）不作为分值候选。"""
        assert extract_score_candidates("第 999 分") == []

    def test_dedup(self) -> None:
        assert extract_score_candidates("8分 8分") == [8.0]


# ======================================================================
# 结构化通道
# ======================================================================


class TestStructuredChannel:
    def test_exact_hit(self, base_ruleset: Ruleset) -> None:
        router = Router(base_ruleset, academic_year="2025-2026")
        res = router.route(make_claim("省二等奖", level="省级二等奖"))
        assert res.channel == "structured" and res.matched
        assert res.rule is not None and res.rule.score == 8
        assert res.confidence == 1.0 and res.needs_review is False
        assert res.source is not None and res.source.doc == "合成细则.pdf"

    def test_category_fallback_attaches_candidates(self, base_ruleset: Ruleset) -> None:
        router = Router(base_ruleset, academic_year="2025-2026")
        res = router.route(make_claim("参加了比赛", level=None))
        assert res.matched and res.needs_review
        assert res.confidence < EngineConfig().review_threshold
        assert "降级匹配" in " ".join(res.notes)

    def test_structured_disabled(self, base_ruleset: Ruleset) -> None:
        router = Router(base_ruleset, academic_year="2025-2026", structured_first=False)
        res = router.route(make_claim("省二等奖", level="省级二等奖"))
        assert res.channel == "none" and res.refused


# ======================================================================
# 拒答（简历指标：未找到规则时正确拒答率 100%）
# ======================================================================


class TestRefusal:
    def test_refuse_without_corpus(self, base_ruleset: Ruleset) -> None:
        router = Router(base_ruleset, academic_year="2025-2026")
        res = router.route(make_claim("宿舍卫生优秀", level=None, category="宿舍管理"))
        assert res.refused
        assert res.reason == REFUSAL_MESSAGE
        assert res.channel == "none"
        assert res.score_candidates == []

    def test_refuse_is_not_zero_score_silently(self, base_ruleset: Ruleset) -> None:
        res = Router(base_ruleset).route(make_claim("完全无关的内容", level=None, category="其它"))
        assert res.refused and res.reason and "辅导员" in res.reason


# ======================================================================
# 兜底通道
# ======================================================================


def _clauses() -> list[Clause]:
    return [
        Clause(
            id="c1",
            text="第三章第7条 学科竞赛：省级二等奖计 8 分，省级一等奖计 10 分。",
            source=SourceRef(doc="2025综测细则.pdf", page=4, clause="第三章第7条"),
        ),
        Clause(
            id="c2",
            text="第四章第2条 文体活动：校级一等奖计 3 分。",
            source=SourceRef(doc="2025综测细则.pdf", page=9, clause="第四章第2条"),
        ),
        Clause(
            id="c3",
            text="第五章 志愿服务：每满 20 小时计 1 分，最高 5 分。",
            source=SourceRef(doc="2025综测细则.pdf", page=12, clause="第五章"),
        ),
    ]


class TestFallbackChannel:
    def test_lexical_retriever_finds_clause(self) -> None:
        r = LexicalRetriever(_clauses())
        if not r.available():  # pragma: no cover - 未装 rank_bm25 时跳过
            pytest.skip("rank-bm25 未安装")
        hits = r.search("省级二等奖", top_k=1)
        assert hits and hits[0].clause.id == "c1"

    def test_fallback_returns_candidates_but_not_counted(self, base_ruleset: Ruleset) -> None:
        """兜底命中的分值只是候选，绝不直接计分。"""
        r = LexicalRetriever(_clauses())
        if not r.available():  # pragma: no cover
            pytest.skip("rank-bm25 未安装")
        router = Router(base_ruleset, clauses=_clauses(), academic_year="2025-2026")
        res = router.route(make_claim("志愿服务满40小时", level=None, category="志愿服务"))
        assert res.channel == "vector"
        assert res.matched is False  # 关键：兜底不算命中
        assert res.needs_review is True
        assert res.score_candidates  # 但给出候选
        assert res.clause_text and res.source is not None
        assert res.source.page == 12

    def test_router_without_corpus_refuses(self, base_ruleset: Ruleset) -> None:
        router = Router(base_ruleset, academic_year="2025-2026")
        assert router.route(make_claim("志愿服务满40小时", level=None, category="志愿服务")).refused

    def test_vector_channel_is_explicit_placeholder(self) -> None:
        vc = VectorChannel()
        assert vc.available() is False
        with pytest.raises(NotImplementedError):
            vc.search("任意")

    def test_route_many(self, base_ruleset: Ruleset) -> None:
        router = Router(base_ruleset, academic_year="2025-2026")
        results = router.route_many([
            make_claim("省二等奖", level="省级二等奖", claim_id="c1"),
            make_claim("没有的东西", level=None, category="其它", claim_id="c2"),
        ])
        assert len(results) == 2 and results[0].matched and results[1].refused

    def test_explain_payload_shape(self, base_ruleset: Ruleset) -> None:
        payload = Router(base_ruleset, academic_year="2025-2026").explain(
            make_claim("省二等奖", level="省级二等奖")
        )
        for key in ("channel", "matched", "rule_id", "source", "confidence", "claim"):
            assert key in payload
        assert payload["claim"]["canonical_level"] == "省级二等奖"
        assert payload["source"]["page"] == 4


class TestRetrieverProtocol:
    def test_custom_retriever_is_used(self, base_ruleset: Ruleset) -> None:
        """任何实现 search/available 的对象都能当兜底通道（可替换为向量检索）。"""

        class Fake:
            def available(self) -> bool:
                return True

            def search(self, query: str, *, top_k: int = 5):
                from scoreproof.retrieval.router import RetrievalHit

                return [RetrievalHit(clause=_clauses()[0], score=1.0, rank=1)]

        router = Router(base_ruleset, retriever=Fake(), academic_year="2025-2026")
        res = router.route(make_claim("志愿者", level=None, category="志愿服务"))
        assert res.channel == "vector" and res.clause_text

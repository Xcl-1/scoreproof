"""检索层：双通道调度。

    精确通道（优先）：规则库结构化查表 -> 命中即给分（100% 可复现）
            ↓ 未命中
    兜底通道：细则原文检索 -> 正则抽分值候选 -> 标注"低置信，需人工确认"
            ↓ 仍未命中
    拒答："未在细则中找到，建议咨询辅导员"（不瞎给分）

**关键纪律**：兜底通道抽出来的分值**不直接计入总分**，只作为候选值返回，
必须经人工确认后固化为规则。这样"模型/正则心算分值"的风险为零。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from ..calc.engine import EngineConfig, MatchOutcome, RuleIndex, match_claim
from ..normalize import normalize_level
from ..schema import Claim, Rule, Ruleset, SourceRef
from ..tokenize import tokenize_for_search

ChannelName = Literal["structured", "vector", "none"]

REFUSAL_MESSAGE = "未在细则中找到对应条款，建议咨询辅导员或补充材料（系统不做无依据给分）。"


# ======================================================================
# 原文片段与检索器接口
# ======================================================================


@dataclass
class Clause:
    """细则原文片段，带出处。"""

    id: str
    text: str
    source: SourceRef
    academic_year: str | None = None
    college: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class RetrievalHit:
    clause: Clause
    score: float
    rank: int = 0
    channel: Literal["bm25", "vector", "rrf"] | None = None
    component_ranks: dict[str, int] = field(default_factory=dict)


class Retriever(Protocol):
    """检索器协议：结构化索引与词法索引都实现它。"""

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]: ...

    def available(self) -> bool: ...


@dataclass
class RetrievalResult:
    """统一检索结果，供 API / 引用面板直接渲染。"""

    channel: ChannelName
    matched: bool
    rule: Rule | None = None
    clause_text: str | None = None
    source: SourceRef | None = None
    confidence: float = 0.0
    needs_review: bool = True
    reason: str | None = None
    candidates: list[RetrievalHit] = field(default_factory=list)
    score_candidates: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return not self.matched

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "matched": self.matched,
            "rule_id": self.rule.id if self.rule else None,
            "level": self.rule.level if self.rule else None,
            "score": self.rule.score if self.rule else None,
            "clause_text": self.clause_text,
            "source": self.source.model_dump(mode="json") if self.source else None,
            "confidence": self.confidence,
            "needs_review": self.needs_review,
            "reason": self.reason,
            "score_candidates": self.score_candidates,
            "notes": self.notes,
        }


# ======================================================================
# 通道 1：结构化查表（主）
# ======================================================================


class StructuredChannel:
    """精确查表：命中即给确定分值。"""

    name: ChannelName = "structured"

    def __init__(self, ruleset: Ruleset, *, academic_year: str | None = None,
                 college: str | None = None, config: EngineConfig | None = None) -> None:
        self.ruleset = ruleset
        self.config = config or EngineConfig()
        self.index = RuleIndex(ruleset, academic_year=academic_year, college=college)

    def search_claim(self, claim: Claim) -> MatchOutcome:
        return match_claim(claim, self.index, config=self.config)

    def __len__(self) -> int:
        return len(self.index)


# ======================================================================
# 通道 2：原文检索（兜底）
# ======================================================================


def extract_score_candidates(text: str, *, limit: int = 5) -> list[float]:
    """从原文片段里用**正则**抽分值候选（不让模型心算）。

    只在"等级词"附近取数字，避免把条款编号、页码当成分值。
    """
    if not text:
        return []
    number = r"(\d+(?:\.\d+)?)"
    patterns = [
        rf"{number}\s*分",
        rf"加\s*{number}\s*分",
        rf"计\s*{number}\s*分",
        rf"加\s*{number}",
    ]
    out: list[float] = []
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            try:
                val = float(m.group(1))
            except (TypeError, ValueError):  # pragma: no cover
                continue
            if 0 < val <= 100 and val not in out:
                out.append(val)
            if len(out) >= limit:
                return out
    return out


class LexicalRetriever:
    """词法检索兜底（BM25，需 ``rank-bm25``）。

    这是"向量通道"的可运行占位：接口与向量检索一致，装上 embedding 后端即可替换。
    语义检索对数字/表格天然不可靠（"二等奖 8 分"与"三等奖 5 分"向量距离很近），
    所以它只做兜底 + 溯源，不参与确定性给分。
    """

    name: ChannelName = "vector"

    def __init__(self, clauses: Sequence[Clause], *, academic_year: str | None = None,
                 college: str | None = None, top_k: int = 5) -> None:
        self.clauses = list(clauses)
        self.academic_year = academic_year
        self.college = college
        self.top_k = top_k
        self._bm25 = None
        self._corpus_tokens: list[list[str]] = []

    def available(self) -> bool:
        if not self.clauses:
            return False
        try:
            import rank_bm25  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """中文按字符 bigram + 英文/数字按词切分（无需分词器依赖）。"""
        return tokenize_for_search(text)

    def _build(self) -> None:
        if self._bm25 is not None:
            return
        from rank_bm25 import BM25Okapi

        self._corpus_tokens = [self._tokenize(c.text) for c in self.clauses]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, *, top_k: int | None = None) -> list[RetrievalHit]:
        if not self.available():
            return []
        self._build()
        assert self._bm25 is not None
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        k = top_k or self.top_k
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], self.clauses[i].id))[:k]
        hits: list[RetrievalHit] = []
        for rank, i in enumerate(ranked):
            if scores[i] <= 0:
                continue
            hits.append(
                RetrievalHit(
                    clause=self.clauses[i],
                    score=float(scores[i]),
                    rank=rank + 1,
                    channel="bm25",
                    component_ranks={"bm25": rank + 1},
                )
            )
        return hits


class VectorChannel:
    """真正的向量通道占位（V3.0 阶段 4：langchain-chroma）。

    未接入前保持显式失败，避免"以为有语义检索其实没有"的隐性错误。
    """

    name: ChannelName = "vector"

    def __init__(self, *_, **__) -> None:
        self._backend = None

    def available(self) -> bool:
        return self._backend is not None

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:  # pragma: no cover - 占位
        raise NotImplementedError(
            "向量通道尚未接入：V3.0 阶段 4 计划用 langchain-chroma 实现。"
            "在此之前兜底由 LexicalRetriever（BM25）承担。"
        )


# ======================================================================
# 调度
# ======================================================================


class Router:
    """双通道调度 + 置信度 + 未命中拒答。"""

    def __init__(
        self,
        ruleset: Ruleset,
        *,
        clauses: Sequence[Clause] = (),
        academic_year: str | None = None,
        college: str | None = None,
        config: EngineConfig | None = None,
        retriever: Retriever | None = None,
        structured_first: bool = True,
    ) -> None:
        self.config = config or EngineConfig()
        self.structured = StructuredChannel(
            ruleset, academic_year=academic_year, college=college, config=self.config
        )
        self.retriever: Retriever | None = retriever
        if self.retriever is None and clauses:
            self.retriever = LexicalRetriever(
                clauses, academic_year=academic_year, college=college
            )
        self.structured_first = structured_first
        self.academic_year = academic_year
        self.college = college

    # ---------- 主入口 ----------

    def route(self, claim: Claim, *, top_k: int = 5) -> RetrievalResult:
        """先精确查表；未命中再走兜底；仍未命中则拒答。"""
        if self.structured_first:
            outcome = self.structured.search_claim(claim)
            if outcome.matched and outcome.rule is not None:
                high_confidence = outcome.strategy in ("exact_level", "synonym")
                result = RetrievalResult(
                    channel="structured",
                    matched=True,
                    rule=outcome.rule,
                    clause_text=outcome.rule.source.text,
                    source=outcome.rule.source,
                    confidence=outcome.confidence,
                    needs_review=outcome.confidence < self.config.review_threshold,
                    reason=outcome.reason,
                )
                if high_confidence:
                    return result
                # 降级命中：同时给出原文候选，便于人工快速确认
                result.candidates = self._fallback_search(claim, top_k=top_k)
                result.notes.append("结构化通道为降级匹配，附原文候选供人工确认")
                return result
            return self._fallback(claim, structured_reason=outcome.reason, top_k=top_k)
        return self._fallback(claim, structured_reason="结构化通道已关闭", top_k=top_k)

    def route_many(self, claims: Iterable[Claim], *, top_k: int = 5) -> list[RetrievalResult]:
        return [self.route(c, top_k=top_k) for c in claims]

    # ---------- 兜底 ----------

    def _fallback_search(self, claim: Claim, *, top_k: int) -> list[RetrievalHit]:
        if self.retriever is None:
            return []
        query = " ".join(x for x in (claim.level or "", claim.raw_text, claim.category) if x)
        if not query.strip():
            return []
        try:
            return self.retriever.search(query, top_k=top_k)
        except NotImplementedError:  # 通道未接入时安静降级为"没有兜底结果"
            return []

    def _fallback(self, claim: Claim, *, structured_reason: str | None, top_k: int) -> RetrievalResult:
        hits = self._fallback_search(claim, top_k=top_k)
        if not hits:
            return RetrievalResult(
                channel="none",
                matched=False,
                confidence=0.0,
                needs_review=True,
                reason=REFUSAL_MESSAGE,
                notes=[structured_reason] if structured_reason else [],
            )
        best = hits[0]
        candidates = extract_score_candidates(best.clause.text)
        return RetrievalResult(
            channel="vector",
            matched=False,  # 兜底不算"命中"：必须人工确认后才能计分
            clause_text=best.clause.text,
            source=best.clause.source,
            confidence=0.4,
            needs_review=True,
            reason="命中细则原文，但未结构化：分值候选仅供参考，需人工确认后才可计分",
            candidates=hits,
            score_candidates=candidates,
            notes=[structured_reason] if structured_reason else [],
        )

    def explain(self, claim: Claim, *, top_k: int = 5) -> dict:
        """给人看的解释（引用面板 / CLI）。"""
        res = self.route(claim, top_k=top_k)
        payload = res.to_dict()
        payload["claim"] = {
            "id": claim.id,
            "raw_text": claim.raw_text,
            "level": claim.level,
            "canonical_level": normalize_level(claim.raw_text).canonical if claim.raw_text else None,
            "category": claim.category,
        }
        return payload


def clauses_from_pdf_pages(pages: Iterable, *, min_chars: int = 8) -> list[Clause]:
    """把 PDF 页面切成片段，构造兜底检索语料。"""
    clauses: list[Clause] = []
    for pg in pages:
        for i, block in enumerate(getattr(pg, "blocks", []) or []):
            text = str(block.get("text", "")).strip()
            if len(text) < min_chars:
                continue
            doc = (getattr(pg, "meta", {}) or {}).get("doc", "")
            bbox = block.get("bbox")
            clauses.append(
                Clause(
                    id=f"{doc}#p{pg.page}#b{i}",
                    text=text,
                    source=SourceRef(
                        doc=doc,
                        page=pg.page,
                        text=text[:200],
                        bbox=tuple(bbox) if bbox else None,
                    ),
                )
            )
    return clauses


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

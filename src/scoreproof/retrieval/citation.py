"""代码级引用核查：结构化账本可自动计分，原文命中只供人工确认。"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..schema import Ruleset, ScoreBreakdown, SourceRef
from ..tokenize import tokenize_for_search

if TYPE_CHECKING:
    from .router import RetrievalHit

CitationDisposition = Literal["auto_score", "manual_review", "refuse"]

_GENERIC_QUERY_TOKENS = frozenset(
    {
        "什么",
        "怎么",
        "如何",
        "多少",
        "可以",
        "是否",
        "哪里",
        "哪个",
        "需要",
        "应该",
        "规定",
        "相关",
        "进行",
        "通过",
        "获得",
        "处理",
        "申请",
        "学院",
        "学校",
        "学生",
        "项目",
        "要求",
        "规则",
        "请问",
        "旗山",
        "山校",
        "校区",
        "大学",
        "校内",
        "校园",
    }
)
_POLICY_QUERY_ANCHORS = frozenset(
    {
        "推免",
        "推荐免试",
        "保研",
        "英语",
        "四级",
        "六级",
        "外语",
        "cet",
        "科创",
        "创新大赛",
        "挑战杯",
        "国创",
        "竞赛",
        "比赛",
        "赛事",
        "获奖",
        "特等奖",
        "学术专长",
        "论文",
        "科研",
        "sci",
        "一作",
        "二作",
        "职称",
        "成果",
        "刊物",
        "专利",
        "知识产权",
        "综合测评",
        "综合素质",
        "德育",
        "志愿服务",
        "志愿",
        "国际组织",
        "实习",
        "参军",
        "服役",
        "加分",
        "原始分",
        "原始成绩",
        "换算",
        "折算",
        "归一化",
        "百分制",
        "权重",
        "总分",
        "分值",
        "得分",
        "成绩",
        "排名",
        "推荐名单",
        "入围",
        "团队",
        "队员",
        "队长",
        "系数",
        "贡献",
        "名次",
        "等级",
    }
)
_POLICY_INTENT_ANCHORS = frozenset(
    {
        "加分",
        "计分",
        "记几分",
        "分数",
        "分值",
        "得分",
        "成绩",
        "原始分",
        "换算",
        "折算",
        "归一化",
        "百分制",
        "公式",
        "分式",
        "比例",
        "权重",
        "基础分",
        "最高分",
        "系数",
        "资格",
        "条件",
        "认定",
        "审核",
        "推荐",
        "推免",
        "保研",
        "要求",
        "规定",
        "依据",
        "门槛",
        "符合",
        "通过",
        "认可",
        "范围",
        "计入",
        "计算",
        "核算",
        "处理",
        "占用",
        "占",
        "名额",
        "口径",
        "证明",
        "材料",
        "贡献",
        "限制",
        "标准",
        "排名",
        "排位",
        "结果",
        "采用",
        "参照",
        "收集",
        "还要",
        "达到",
        "乘",
        "相除",
        "分母",
        "分子",
        "基准",
        "统一",
        "百分点",
        "最高值",
        "只取",
        "最高奖",
        "取得",
        "记",
        "对应",
        "哪些",
    }
)
_OUT_OF_SCOPE_MARKERS = frozenset(
    {
        "打印",
        "维修",
        "报修",
        "收费",
        "购买",
        "快递",
        "天气",
        "电影",
        "营业",
        "预约",
        "借阅",
        "宿舍",
        "食堂",
        "校园卡",
        "宽带",
        "投影仪",
    }
)
_POLICY_TOPIC_GROUPS = (
    frozenset({"推免", "推荐免试", "保研", "推荐名单", "入围"}),
    frozenset({"英语", "外语", "四级", "六级", "cet"}),
    frozenset({"科创", "创新大赛", "挑战杯", "国创", "竞赛", "比赛", "赛事", "获奖", "特等奖"}),
    frozenset({"学术专长", "论文", "科研", "sci", "一作", "二作", "职称", "成果", "刊物"}),
    frozenset({"专利", "知识产权"}),
    frozenset({"综合测评", "综合素质", "德育", "权重", "总分", "百分制", "归一化"}),
    frozenset({"志愿服务", "志愿"}),
    frozenset({"国际组织", "实习"}),
    frozenset({"参军", "服役"}),
)
_NUMBER_RE = re.compile(r"(?<![\d.])[-+]?\d+(?:\.\d+)?(?![\d.])")


class CitationCheck(BaseModel):
    """引用门禁的机器可读结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    supported: bool
    disposition: CitationDisposition
    reason: str
    citation_ids: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    unsupported_items: list[str] = Field(default_factory=list)


def has_precise_locator(source: SourceRef | None) -> bool:
    """出处必须有文档名，且至少能定位到页、条款、表格行或文本坐标。"""
    if source is None or not source.doc.strip():
        return False
    return any(
        (
            source.page is not None,
            bool(source.clause),
            bool(source.table) and source.row is not None,
            bool(source.chunk_hash)
            and source.char_start is not None
            and source.char_end is not None,
        )
    )


def verify_score_breakdown(ledger: ScoreBreakdown, ruleset: Ruleset) -> CitationCheck:
    """核对所有实际计入项都来自当前规则集，且分值与出处未被改写。"""
    unsupported: list[str] = []
    sources: list[SourceRef] = []
    cited_rule_ids: list[str] = []
    manual_review = False
    counted = [match for match in ledger.matches if match.counted]
    if not counted:
        return CitationCheck(
            supported=False,
            disposition="refuse",
            reason="账本没有可由规则出处支撑的计入项，禁止自动给分。",
            unsupported_items=["counted_matches"],
        )

    for match in counted:
        item_id = match.claim_id
        rule = ruleset.by_id(match.rule_id or "")
        if rule is None:
            unsupported.append(f"{item_id}:rule_id")
            continue
        if match.channel != "structured":
            unsupported.append(f"{item_id}:channel")
        if not has_precise_locator(match.source):
            unsupported.append(f"{item_id}:source_locator")
        elif match.source != rule.source:
            unsupported.append(f"{item_id}:source_mismatch")
        if abs(match.raw_score - rule.score) > 1e-9:
            unsupported.append(f"{item_id}:raw_score")
        if match.needs_review or match.confidence < 1.0:
            manual_review = True
        cited_rule_ids.append(rule.id)
        sources.append(rule.source)

    if unsupported:
        return CitationCheck(
            supported=False,
            disposition="refuse",
            reason="账本结论与当前结构化规则或精确出处不一致，已阻断自动答复。",
            citation_ids=list(dict.fromkeys(cited_rule_ids)),
            sources=_deduplicate_sources(sources),
            unsupported_items=unsupported,
        )
    if manual_review:
        return CitationCheck(
            supported=True,
            disposition="manual_review",
            reason="规则出处可定位，但存在低置信匹配，必须人工确认后才能计分。",
            citation_ids=list(dict.fromkeys(cited_rule_ids)),
            sources=_deduplicate_sources(sources),
        )
    return CitationCheck(
        supported=True,
        disposition="auto_score",
        reason="所有计入项均与当前结构化规则的分值和精确出处一致。",
        citation_ids=list(dict.fromkeys(cited_rule_ids)),
        sources=_deduplicate_sources(sources),
    )


def verify_text_citations(
    query: str,
    hits: Sequence[RetrievalHit],
    *,
    conclusion: str | None = None,
    max_citations: int = 5,
    policy_scope: bool = False,
) -> CitationCheck:
    """核查原文候选是否与问题有可解释的词法交集；永不放行自动计分。"""
    if not query.strip() or not hits:
        return _text_refusal("没有检索到可核查的规则原文。")
    if not _query_is_in_scope(query, require_intent=not policy_scope):
        return _text_refusal("问题不属于当前综测/推免细则的可核查范围。")
    # 延迟导入，避免 router 与 citation/query 的模块初始化环。
    from .query import rewrite_retrieval_query

    query_tokens = _meaningful_tokens(rewrite_retrieval_query(query))
    query_topics = _policy_topics(query)
    if not query_tokens:
        return _text_refusal("问题缺少可用于核查引用的有效规则术语。")

    accepted: list[RetrievalHit] = []
    for hit in hits:
        if not has_precise_locator(hit.clause.source):
            continue
        clause_tokens = _meaningful_tokens(hit.clause.text)
        overlap = query_tokens & clause_tokens
        channels = set(hit.component_ranks)
        semantic_supported = bool(overlap) or bool(query_topics & _policy_topics(hit.clause.text))
        dual_supported = {"bm25", "vector"}.issubset(channels) and semantic_supported
        lexical_supported = hit.channel in {None, "bm25"} and len(overlap) >= 2
        if not (dual_supported or lexical_supported):
            continue
        accepted.append(hit)
        if len(accepted) >= max_citations:
            break

    if not accepted:
        return _text_refusal("召回片段与问题之间缺少可复核的双路/词法支撑。")

    if conclusion:
        available_numbers = {
            match.group(0)
            for hit in accepted
            for match in _NUMBER_RE.finditer(hit.clause.text)
        }
        unsupported_numbers = [
            number
            for number in dict.fromkeys(_NUMBER_RE.findall(conclusion))
            if number not in available_numbers
        ]
        if unsupported_numbers:
            return CitationCheck(
                supported=False,
                disposition="refuse",
                reason="答复包含引用片段中不存在的数字，已阻断。",
                unsupported_items=[f"number:{number}" for number in unsupported_numbers],
            )

    return CitationCheck(
        supported=True,
        disposition="manual_review",
        reason="检索到可定位且与问题相关的原文候选；文本通道只供人工确认，不自动计分。",
        citation_ids=[hit.clause.id for hit in accepted],
        sources=_deduplicate_sources([hit.clause.source for hit in accepted]),
    )


def _meaningful_tokens(text: str) -> set[str]:
    return {
        token
        for token in tokenize_for_search(text)
        if len(token) >= 2 and token not in _GENERIC_QUERY_TOKENS
    }


def _query_is_in_scope(text: str, *, require_intent: bool) -> bool:
    folded = text.casefold()
    # “电子期刊”是常见图书馆服务说法，不能仅凭“期刊”判定为论文政策问题。
    if "电子期刊" in folded and not any(
        marker in folded for marker in ("论文", "科研", "成果", "发表", "作者")
    ):
        folded = folded.replace("电子期刊", "")
    return (
        not any(marker in folded for marker in _OUT_OF_SCOPE_MARKERS)
        and any(anchor in folded for anchor in _POLICY_QUERY_ANCHORS)
        and (not require_intent or any(intent in folded for intent in _POLICY_INTENT_ANCHORS))
    )


def _policy_topics(text: str) -> set[int]:
    folded = text.casefold()
    return {
        index
        for index, group in enumerate(_POLICY_TOPIC_GROUPS)
        if any(term in folded for term in group)
    }


def _text_refusal(reason: str) -> CitationCheck:
    return CitationCheck(
        supported=False,
        disposition="refuse",
        reason=reason + " 系统不做无依据回答。",
    )


def _deduplicate_sources(sources: Sequence[SourceRef]) -> list[SourceRef]:
    unique: dict[str, SourceRef] = {}
    for source in sources:
        key = source.model_dump_json()
        unique.setdefault(key, source)
    return list(unique.values())


__all__ = [
    "CitationCheck",
    "CitationDisposition",
    "has_precise_locator",
    "verify_score_breakdown",
    "verify_text_citations",
]

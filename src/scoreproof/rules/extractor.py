"""规则抽取器骨架（P1 先用"程序抽取 + 人工校对一遍"，P2 再接 LLM）。

纪律：
1. LLM **只**输出结构化草稿（等级 -> 分值 + 出处），不参与任何计算；
2. 所有草稿必须带 ``source``，没有出处的草稿一律丢弃（不可溯源 = 不可用）；
3. 抽完必须人工校对一遍再入规则库（项目总结第 6 节易翻车点 3/4）。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config import get_settings
from ..errors import DataSourceError, UnsupportedModality
from ..normalize import normalize_academic_year, normalize_level
from ..schema import ConstraintSpec, Rule, SourceRef

# 给 LLM 的抽取契约（P2 接真实调用时直接复用）
RULE_DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "description": "加分类别，如 学科竞赛"},
        "level": {"type": "string", "description": "等级/名次原文，如 省级二等奖"},
        "score": {"type": "number", "description": "该等级分值，必须来自原文数字"},
        "synonyms": {"type": "array", "items": {"type": "string"}},
        "cap": {"type": ["number", "null"], "description": "该项封顶，原文没有则为 null"},
        "team_factor": {"type": ["number", "null"], "description": "团队折算系数"},
        "clause": {"type": "string", "description": "条款编号，如 第三章第7条"},
        "evidence_quote": {"type": "string", "description": "支撑该规则的原文片段（必须逐字摘录）"},
    },
    "required": ["category", "level", "score", "evidence_quote"],
}


@dataclass
class RuleDraft:
    """LLM/程序抽出的规则草稿（尚未校对）。"""

    category: str
    level: str
    score: float
    evidence_quote: str
    synonyms: list[str] = field(default_factory=list)
    cap: float | None = None
    team_factor: float | None = None
    clause: str | None = None
    confidence: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_rule(
        self,
        *,
        academic_year: str,
        doc: str,
        college: str | None = None,
        page: int | None = None,
        table: str | None = None,
        dedup_group: str | None = None,
        priority: int = 0,
    ) -> Rule:
        """草稿 -> 正式规则。**这是唯一的转换入口，便于统一审计。**"""
        canonical = normalize_level(self.level).canonical
        return Rule(
            academic_year=normalize_academic_year(academic_year),
            college=college,
            category=self.category,
            level=canonical,
            score=float(self.score),
            synonyms=sorted({s for s in [self.level, *self.synonyms] if s and s != canonical}),
            constraints=ConstraintSpec(
                dedup_group=dedup_group or self.category,
                cap=self.cap,
                team_factor=self.team_factor if self.team_factor is not None else 1.0,
            ),
            source=SourceRef(doc=doc, page=page, table=table, clause=self.clause,
                             text=self.evidence_quote[:200]),
            priority=priority,
            raw_text=self.evidence_quote[:500],
        )


class Extractor(Protocol):
    """抽取器协议：程序抽取 / LLM 抽取都实现它。"""

    def extract(self, text: str, **kwargs) -> list[RuleDraft]: ...


class HeuristicExtractor:
    """无模型抽取：从"等级 ... N分"式文本里直接抓规则（P1 主力）。

    上限明确：只处理行内一一对应的写法，复杂表格交给表格解析 + 人工校对。
    """

    def __init__(self, *, category: str = "未分类", require_score: bool = True) -> None:
        import re

        self.category = category
        self.require_score = require_score
        self._re = re.compile(
            r"(?P<level>[\u4e00-\u9fff]{0,6}(?:一等|二等|三等|特等|优秀|入围|参与)奖"
            r"|[\u4e00-\u9fff]{0,4}(?:第[一二三四1-4]名|冠军|亚军|季军))"
            r"[^\d]{0,8}(?P<score>\d+(?:\.\d+)?)\s*分?"
        )

    def extract(self, text: str, **kwargs) -> list[RuleDraft]:
        drafts: list[RuleDraft] = []
        for line in (text or "").splitlines():
            for m in self._re.finditer(line):
                drafts.append(
                    RuleDraft(
                        category=kwargs.get("category", self.category),
                        level=m.group("level"),
                        score=float(m.group("score")),
                        evidence_quote=line.strip(),
                        confidence=0.7,
                        meta={"extractor": "heuristic"},
                    )
                )
        return drafts


class LLMExtractor:
    """LLM 抽取（只在配置了 API Key 时可用；默认走 DeepSeek 文本模型）。

    未实现真实调用时保持显式失败 —— 宁可报"未接入"，也不返回编造数据。
    """

    def __init__(self, *, model: str | None = None, client: Any = None) -> None:
        settings = get_settings()
        self.model = model or settings.llm_model
        self.base_url = settings.llm_base_url
        self.client = client

    def available(self) -> bool:
        if self.client is not None:
            return True
        if not get_settings().llm_configured:
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def build_prompt(self, text: str, **kwargs) -> list[dict[str, str]]:
        """抽取提示词：明确"只抽原文出现的分值、必须给原文片段"。"""
        system = (
            "你是综测加分细则的结构化抽取器。只输出 JSON 数组，每个元素符合给定 schema；"
            "score 必须是原文出现的数字，禁止推算、禁止补全；evidence_quote 必须逐字摘录原文。"
            "找不到分值的条目不要输出。"
        )
        payload = {
            "schema": RULE_DRAFT_SCHEMA,
            "defaults": {k: v for k, v in kwargs.items() if v is not None},
            "text": text,
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def extract(self, text: str, **kwargs) -> list[RuleDraft]:  # pragma: no cover - P2
        if not self.available():
            raise DataSourceError(
                "LLM 抽取不可用：缺少 DEEPSEEK_API_KEY 或未安装 openai",
                detail={"hint": "复制 .env.example 为 .env 并填入 Key，或 uv sync --extra llm"},
            )
        raise UnsupportedModality(
            "LLMExtractor 尚未接入真实调用（P2 任务）",
            detail={"model": self.model, "todo": "实现 responses.create / chat.completions 解析 + 校验"},
        )


def drafts_to_rules(
    drafts: Iterable[RuleDraft],
    *,
    academic_year: str,
    doc: str,
    college: str | None = None,
    page: int | None = None,
    table: str | None = None,
) -> list[Rule]:
    """批量转换，并丢弃没有出处的草稿（不可溯源 = 不可用）。"""
    out: list[Rule] = []
    for d in drafts:
        if not (d.evidence_quote or "").strip():
            continue
        out.append(
            d.to_rule(
                academic_year=academic_year, doc=doc, college=college, page=page, table=table
            )
        )
    return out


__all__ = [
    "RULE_DRAFT_SCHEMA",
    "Extractor",
    "HeuristicExtractor",
    "LLMExtractor",
    "RuleDraft",
    "drafts_to_rules",
]

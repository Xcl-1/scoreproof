"""规则抽取器骨架（P1 先用"程序抽取 + 人工校对一遍"，P2 再接 LLM）。

纪律：
1. LLM **只**输出结构化草稿（等级 -> 分值 + 出处），不参与任何计算；
2. 所有草稿必须带 ``source``，没有出处的草稿一律丢弃（不可溯源 = 不可用）；
3. 抽完必须人工校对一遍再入规则库（项目总结第 6 节易翻车点 3/4）。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from ..config import get_settings
from ..errors import DataSourceError, SchemaValidationError
from ..normalize import normalize_academic_year, normalize_level
from ..observability import CostLedger, classify_model_tier, pricing_from_settings
from ..schema import ConstraintSpec, Rule, SourceRef
from .gateway import ExtractionGateway, GatewayContext, GatewayReport, RuleDraftInput, chunk_hash

# 给 LLM 的抽取契约直接由严格 Pydantic 模型生成，避免提示 schema 与网关漂移。
RULE_DRAFT_SCHEMA: dict[str, Any] = RuleDraftInput.model_json_schema()
RULE_BATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"rules": {"type": "array", "items": RULE_DRAFT_SCHEMA}},
    "required": ["rules"],
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
    rank: str | None = None
    item_name: str | None = None
    effective_date: str | None = None
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
        char_start: int | None = None,
        char_end: int | None = None,
    ) -> Rule:
        """草稿 -> 正式规则。**这是唯一的转换入口，便于统一审计。**"""
        canonical = normalize_level(self.level).canonical
        return Rule(
            academic_year=normalize_academic_year(academic_year),
            college=college,
            category=self.category,
            level=canonical,
            rank=self.rank,
            item_name=self.item_name,
            score=float(self.score),
            synonyms=sorted({s for s in [self.level, *self.synonyms] if s and s != canonical}),
            constraints=ConstraintSpec(
                dedup_group=dedup_group or self.category,
                cap=self.cap,
                team_factor=self.team_factor if self.team_factor is not None else 1.0,
            ),
            source=SourceRef(
                doc=doc,
                page=page,
                table=table,
                clause=self.clause,
                text=self.evidence_quote[:200],
                char_start=char_start,
                char_end=char_end,
            ),
            priority=priority,
            raw_text=self.evidence_quote[:500],
        )


class Extractor(Protocol):
    """抽取器协议：程序抽取 / LLM 抽取都实现它。"""

    def extract(self, text: str, **kwargs) -> list[RuleDraft]: ...


class ExtractionCache(Protocol):
    """抽取缓存最小契约，RuleStore 与测试替身均可实现。"""

    def get_extraction_cache(self, *, chunk_hash: str, model: str, variant: str) -> list[dict] | None: ...

    def set_extraction_cache(
        self, *, chunk_hash: str, model: str, variant: str, payloads: list[dict]
    ) -> None: ...


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

    使用 LangChain 的 OpenAI 兼容接入层调用 DeepSeek。返回值仍是未经信任的
    草稿；生产路径应调用 ``extract_validated``，由验证网关决定是否可发布。
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        client: Any = None,
        cache: ExtractionCache | None = None,
        cost_ledger: CostLedger | None = None,
        material_id: str | None = None,
        batch_id: str | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.llm_model
        self.base_url = settings.llm_base_url
        self.client = client
        self.cache = cache
        self.cost_ledger = cost_ledger
        self.material_id = material_id
        self.batch_id = batch_id

    def available(self) -> bool:
        if self.client is not None:
            return True
        if not get_settings().llm_configured:
            return False
        try:
            import langchain_openai  # noqa: F401
        except ImportError:
            return False
        return True

    def build_prompt(
        self, text: str, *, variant: str = "direct", **kwargs
    ) -> list[dict[str, str]]:
        """抽取提示词：明确"只抽原文出现的分值、必须给原文片段"。"""
        if variant == "clause_first":
            method = "先逐条识别原文中的规则条款，再从每条已识别条款抽取字段；不要合并不同条款。"
        elif variant == "direct":
            method = "直接从原文抽取符合 schema 的规则数组。"
        else:
            raise ValueError(f"未知抽取提示变体：{variant}")
        system = (
            "你是综测加分细则的结构化抽取器。只输出 JSON 对象 {\"rules\": [...]}，"
            "rules 中每个元素符合给定 schema；"
            "score 必须是原文出现的数字，禁止推算、禁止补全；evidence_quote 必须逐字摘录原文。"
            f"找不到分值的条目不要输出。{method}"
        )
        payload = {
            "schema": RULE_BATCH_SCHEMA,
            "example": {"rules": []},
            "defaults": {k: v for k, v in kwargs.items() if v is not None},
            "text": text,
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def _client_for(self, *, temperature: float) -> Any:
        if self.client is not None:
            return self.client
        settings = get_settings()
        if not settings.llm_api_key:
            raise DataSourceError("LLM 抽取不可用：缺少 DEEPSEEK_API_KEY")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - 取决于可选依赖
            raise DataSourceError(
                "LLM 抽取不可用：未安装 langchain-openai",
                detail={"hint": "uv sync --extra llm"},
            ) from exc
        return ChatOpenAI(
            model=self.model,
            api_key=settings.llm_api_key,
            base_url=self.base_url,
            temperature=temperature,
            max_tokens=4096,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    @staticmethod
    def _response_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            return "".join(parts)
        raise SchemaValidationError("LLM 返回内容不是文本", detail={"type": type(content).__name__})

    @staticmethod
    def _decode_payloads(raw: str) -> list[dict[str, Any]]:
        text = raw.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(
                "LLM 返回的内容不是合法 JSON",
                detail={"line": exc.lineno, "column": exc.colno},
            ) from exc
        if isinstance(payload, dict) and set(payload) == {"rules"}:
            payload = payload["rules"]
        if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
            raise SchemaValidationError("LLM 返回必须是 JSON 对象数组")
        return [dict(item) for item in payload]

    def extract_payloads(
        self,
        text: str,
        *,
        variant: str = "direct",
        temperature: float = 0.0,
        **kwargs,
    ) -> list[dict[str, Any]]:
        if not self.available():
            raise DataSourceError(
                "LLM 抽取不可用：缺少 DEEPSEEK_API_KEY 或未安装 langchain-openai",
                detail={"hint": "复制 .env.example 为 .env 并填入 Key，或 uv sync --extra llm"},
            )
        digest = chunk_hash(text)
        messages = self.build_prompt(text, variant=variant, **kwargs)
        prompt_hash = hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        cache_variant = f"{variant}:t={temperature:g}:p={prompt_hash}"
        if self.cache is not None:
            cached = self.cache.get_extraction_cache(
                chunk_hash=digest, model=self.model, variant=cache_variant
            )
            if cached is not None:
                if self.cost_ledger is not None:
                    self.cost_ledger.record_cache_hit(
                        provider="deepseek",
                        model=self.model,
                        model_tier=classify_model_tier(self.model),
                        purpose=(
                            "rule_extraction_secondary"
                            if variant != "direct"
                            else "rule_extraction_primary"
                        ),
                        material_id=self.material_id or f"chunk:{digest}",
                        batch_id=self.batch_id,
                    )
                return cached
        purpose = (
            "rule_extraction_secondary" if variant != "direct" else "rule_extraction_primary"
        )
        try:
            response = self._client_for(temperature=temperature).invoke(messages)
        except Exception as exc:
            if self.cost_ledger is not None:
                self.cost_ledger.record_failure(
                    provider="deepseek",
                    model=self.model,
                    model_tier=classify_model_tier(self.model),
                    purpose=purpose,
                    error=exc,
                    material_id=self.material_id or f"chunk:{digest}",
                    batch_id=self.batch_id,
                )
            raise
        if self.cost_ledger is not None:
            self.cost_ledger.record_response(
                response,
                provider="deepseek",
                model=self.model,
                model_tier=classify_model_tier(self.model),
                purpose=purpose,
                material_id=self.material_id or f"chunk:{digest}",
                batch_id=self.batch_id,
                pricing=pricing_from_settings(get_settings()),
            )
        payloads = self._decode_payloads(self._response_text(response))
        if self.cache is not None:
            self.cache.set_extraction_cache(
                chunk_hash=digest, model=self.model, variant=cache_variant, payloads=payloads
            )
        return payloads

    def extract(self, text: str, **kwargs) -> list[RuleDraft]:
        """兼容旧协议的严格解析入口；返回值仍不可绕过网关直接发布。"""
        payloads = self.extract_payloads(text, **kwargs)
        try:
            strict = [RuleDraftInput.model_validate(item) for item in payloads]
        except Exception as exc:
            raise SchemaValidationError("LLM 草稿未通过严格 schema 校验") from exc
        return [
            RuleDraft(
                category=item.category,
                level=item.level,
                score=item.score,
                evidence_quote=item.evidence_quote,
                synonyms=item.synonyms,
                cap=item.cap,
                team_factor=item.team_factor,
                clause=item.clause,
                rank=item.rank,
                item_name=item.item_name,
                effective_date=item.effective_date,
                meta={"extractor": "llm", "model": self.model},
            )
            for item in strict
        ]

    def extract_validated(
        self,
        text: str,
        *,
        context: GatewayContext,
        gateway: ExtractionGateway | None = None,
        risk_level: Literal["normal", "high"] = "normal",
        existing_rules: Iterable[Rule] = (),
        **kwargs,
    ) -> GatewayReport:
        """执行真实抽取并立即过网关；高风险块触发第二种提示交叉验证。"""
        checker = gateway or ExtractionGateway()
        current_rules = list(existing_rules)
        primary = self.extract_payloads(text, variant="direct", temperature=0.0, **kwargs)
        preliminary = checker.validate_batch(
            primary,
            source_text=text,
            context=context,
            existing_rules=current_rules,
        )
        secondary = None
        low_confidence = any(
            issue.layer in {3, 4} for item in preliminary.items for issue in item.issues
        )
        if risk_level == "high" or low_confidence:
            secondary = self.extract_payloads(
                text, variant="clause_first", temperature=0.3, **kwargs
            )
        if secondary is None:
            return preliminary
        return checker.validate_batch(
            primary,
            source_text=text,
            context=context,
            secondary_payloads=secondary,
            existing_rules=current_rules,
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
    "RULE_BATCH_SCHEMA",
    "Extractor",
    "ExtractionCache",
    "HeuristicExtractor",
    "LLMExtractor",
    "RuleDraft",
    "drafts_to_rules",
]

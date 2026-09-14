"""五个 LangChain 工具及其确定性运行时。"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal, Protocol

from langchain_core.tools import BaseTool, tool

from ..calc.engine import compute_claims
from ..normalize import normalize_level
from ..retrieval.router import RetrievalHit
from ..schema import Ruleset
from .models import (
    AskClarificationInput,
    CalcClaimInput,
    CalcScoreInput,
    CheckEvidenceInput,
    FieldDiff,
    LookupRuleInput,
    SearchClausesInput,
)

ClauseSearcher = Callable[[str, dict[str, Any], int], list[RetrievalHit]]


class AuditSink(Protocol):
    def record_tool_call(
        self,
        *,
        call_id: str,
        session_id: str,
        tool_name: str,
        input_payload: dict,
        result_summary: str,
        duration_ms: float,
        model_version: str,
        status: Literal["ok", "error", "degraded"],
        error: str | None = None,
        created_at: datetime | None = None,
    ) -> None: ...


def ruleset_version(ruleset: Ruleset) -> str:
    """从会影响核算的规则内容生成稳定版本，排除加载时间等非业务字段。"""
    rows = []
    for rule in sorted(ruleset.rules, key=lambda item: item.id):
        payload = rule.model_dump(mode="json", exclude={"created_at"})
        rows.append(payload)
    import hashlib

    raw = json.dumps(
        {"declared_version": ruleset.version, "rules": rows},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"rv_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


class ToolRuntime:
    """一次编排请求的工具依赖与审计上下文。"""

    def __init__(
        self,
        ruleset: Ruleset,
        *,
        session_id: str,
        model_version: str,
        audit_sink: AuditSink | None = None,
        clause_searcher: ClauseSearcher | None = None,
        student_aliases: dict[str, str] | None = None,
    ) -> None:
        self.ruleset = ruleset
        self.session_id = session_id
        self.model_version = model_version
        self.audit_sink = audit_sink
        self.clause_searcher = clause_searcher
        self.student_aliases = dict(student_aliases or {})
        self.version = ruleset_version(ruleset)

    def resolve_student_id(self, student_id: str) -> str:
        if not self.student_aliases:
            return student_id
        try:
            return self.student_aliases[student_id]
        except KeyError as exc:
            raise ValueError("student_id 必须使用编排层提供的匿名引用") from exc

    def audited(self, name: str, args: dict[str, Any], operation: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        status: Literal["ok", "error", "degraded"] = "ok"
        error: str | None = None
        result: Any = None
        try:
            result = operation()
            return result
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if self.audit_sink is not None:
                elapsed = (time.perf_counter() - started) * 1000
                summary = _result_summary(result, error=error)
                self.audit_sink.record_tool_call(
                    call_id=f"tc_{uuid.uuid4().hex}",
                    session_id=self.session_id,
                    tool_name=name,
                    input_payload=args,
                    result_summary=summary,
                    duration_ms=round(elapsed, 3),
                    model_version=self.model_version,
                    status=status,
                    error=error,
                )

    def record_event(self, name: str, payload: dict[str, Any], *, summary: str) -> None:
        """把编排降级等非业务工具事件写入同一审计时间线。"""
        if self.audit_sink is None:
            return
        self.audit_sink.record_tool_call(
            call_id=f"tc_{uuid.uuid4().hex}",
            session_id=self.session_id,
            tool_name=name,
            input_payload=payload,
            result_summary=summary[:1000],
            duration_ms=0.0,
            model_version=self.model_version,
            status="degraded",
            error=None,
        )


def build_tools(runtime: ToolRuntime) -> list[BaseTool]:
    """为单次请求创建五个带 Pydantic schema 的 `@tool`。"""

    @tool("lookup_rule", args_schema=LookupRuleInput)
    def lookup_rule(
        academic_year: str,
        category: str,
        level: str,
        college: str | None = None,
        rank: str | None = None,
        item_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """按学年、学院、类别、等级、名次和项目名精确查询可溯源规则。"""
        args = {
            "academic_year": academic_year,
            "college": college,
            "category": category,
            "level": level,
            "rank": rank,
            "item_name": item_name,
        }

        def operation() -> list[dict[str, Any]]:
            canonical = normalize_level(level).canonical
            found = []
            for rule in runtime.ruleset.filter(
                academic_year=academic_year, college=college, category=category
            ).rules:
                keys = [normalize_level(value).canonical for value in rule.match_keys]
                if canonical not in keys:
                    continue
                if rank is not None and rule.rank != rank:
                    continue
                if item_name is not None and rule.item_name != item_name:
                    continue
                found.append(rule.model_dump(mode="json"))
            return found

        return runtime.audited("lookup_rule", args, operation)

    @tool("search_clauses", args_schema=SearchClausesInput)
    def search_clauses(
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """从活动 BM25+向量索引检索带页码与坐标的原文条款。"""
        clean_filters = dict(filters or {})
        args = {"query": query, "filters": clean_filters, "top_k": top_k}

        def operation() -> list[dict[str, Any]]:
            if runtime.clause_searcher is None:
                return []
            hits = runtime.clause_searcher(query, clean_filters, top_k)
            return [
                {
                    "id": hit.clause.id,
                    "text": hit.clause.text,
                    "source": hit.clause.source.model_dump(mode="json"),
                    "score": hit.score,
                    "rank": hit.rank,
                    "channel": hit.channel,
                }
                for hit in hits
            ]

        return runtime.audited("search_clauses", args, operation)

    @tool("calc_score", args_schema=CalcScoreInput)
    def calc_score(
        claims: list[dict[str, Any]],
        ruleset_version: str,
        student_id: str,
    ) -> dict[str, Any]:
        """只用结构化申报与锁定规则版本运行纯 Python 确定性核算。"""
        args = {
            "claims": claims,
            "ruleset_version": ruleset_version,
            "student_id": student_id,
        }

        def operation() -> dict[str, Any]:
            if ruleset_version != runtime.version:
                raise ValueError(
                    f"规则版本不匹配：会话锁定 {runtime.version}，收到 {ruleset_version}"
                )
            typed = [CalcClaimInput.model_validate(item) for item in claims]
            resolved_student_id = runtime.resolve_student_id(student_id)
            claim_models = [item.to_claim(student_id=resolved_student_id) for item in typed]
            year = next((item.academic_year for item in claim_models if item.academic_year), None)
            college_value = next((item.college for item in claim_models if item.college), None)
            return compute_claims(
                claim_models,
                runtime.ruleset,
                academic_year=year,
                college=college_value,
            ).model_dump(mode="json")

        return runtime.audited("calc_score", args, operation)

    @tool("check_evidence", args_schema=CheckEvidenceInput)
    def check_evidence(
        claim: dict[str, Any], evidence_fields: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """逐字段比较申报与证据抽取值，返回缺失和不一致项。"""
        args = {"claim": claim, "evidence_fields": evidence_fields}

        def operation() -> list[dict[str, Any]]:
            fields = sorted(set(claim) | set(evidence_fields))
            diffs: list[FieldDiff] = []
            for field in fields:
                left = claim.get(field)
                right = evidence_fields.get(field)
                left_norm = _normalise_field(left)
                right_norm = _normalise_field(right)
                if field not in claim:
                    reason = "申报字段缺失"
                elif field not in evidence_fields:
                    reason = "证据字段缺失"
                elif left_norm == right_norm:
                    reason = "一致"
                else:
                    reason = "字段值不一致"
                diffs.append(
                    FieldDiff(
                        field=field,
                        claim_value=left,
                        evidence_value=right,
                        matches=reason == "一致",
                        reason=reason,
                    )
                )
            return [item.model_dump(mode="json") for item in diffs]

        return runtime.audited("check_evidence", args, operation)

    @tool("ask_clarification", args_schema=AskClarificationInput)
    def ask_clarification(missing_fields: list[str], question: str) -> dict[str, Any]:
        """在必要字段缺失或工具空结果时请求用户补充，不产生任何分值。"""
        args = {"missing_fields": missing_fields, "question": question}
        return runtime.audited(
            "ask_clarification",
            args,
            lambda: {"status": "needs_clarification", **args},
        )

    return [lookup_rule, search_clauses, calc_score, check_evidence, ask_clarification]


def _normalise_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return "".join(value.split()).casefold()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _result_summary(result: Any, *, error: str | None) -> str:
    if error:
        return error[:1000]
    if isinstance(result, list):
        prefix = f"list[{len(result)}] "
    elif isinstance(result, dict):
        prefix = f"dict[{len(result)}] "
    else:
        prefix = f"{type(result).__name__} "
    return (prefix + json.dumps(result, ensure_ascii=False, default=str))[:1000]


__all__ = ["ClauseSearcher", "ToolRuntime", "build_tools", "ruleset_version"]

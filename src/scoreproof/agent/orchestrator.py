"""LangChain 工具绑定 + 显式状态机 + 编排层数字护栏。"""

from __future__ import annotations

import json
import re
import threading
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from ..normalize import normalize_level
from ..schema import Claim, ScoreBreakdown
from .models import (
    CalcClaimInput,
    NumberValidation,
    OrchestrationRequest,
    OrchestrationResult,
)
from .tools import AuditSink, ClauseSearcher, ToolRuntime, build_tools

_NUMBER_RE = re.compile(r"(?<![\d.])[-+]?\d+(?:\.\d+)?(?![\d.])")
_MODEL_STUDENT_REF = "CURRENT_STUDENT"


class BoundModel(Protocol):
    def invoke(self, messages: list[Any]) -> Any: ...


class ToolCallingModel(Protocol):
    def bind_tools(self, tools: list[BaseTool]) -> BoundModel: ...

    def invoke(self, messages: list[Any]) -> Any: ...


class SessionVersionConflict(RuntimeError):
    pass


class SessionRuleLock:
    """进程内轻量会话锁；同一 session 只允许观察一个规则版本。"""

    def __init__(self) -> None:
        self._versions: dict[str, str] = {}
        self._lock = threading.Lock()

    def acquire(self, session_id: str, ruleset_version: str) -> str:
        with self._lock:
            existing = self._versions.setdefault(session_id, ruleset_version)
        if existing != ruleset_version:
            raise SessionVersionConflict("当前会话的规则版本已变化，请新建会话后重试。")
        return existing

    def get(self, session_id: str) -> str | None:
        with self._lock:
            return self._versions.get(session_id)


class ScoreProofOrchestrator:
    """显式状态机：路由 → 工具调用 → 数字校验 → 最终分支。"""

    def __init__(
        self,
        ruleset,
        *,
        model: ToolCallingModel | None = None,
        model_version: str = "rules-only",
        audit_sink: AuditSink | None = None,
        clause_searcher: ClauseSearcher | None = None,
        sessions: SessionRuleLock | None = None,
        max_tool_rounds: int = 4,
    ) -> None:
        if not 1 <= max_tool_rounds <= 4:
            raise ValueError("max_tool_rounds 必须在 1 到 4 之间")
        self.ruleset = ruleset
        self.model = model
        self.model_version = model_version
        self.audit_sink = audit_sink
        self.clause_searcher = clause_searcher
        self.sessions = sessions or SessionRuleLock()
        self.max_tool_rounds = max_tool_rounds

    def run(self, request: OrchestrationRequest) -> OrchestrationResult:
        trace = ["route"]
        runtime = ToolRuntime(
            self.ruleset,
            session_id=request.session_id,
            model_version=self.model_version,
            audit_sink=self.audit_sink,
            clause_searcher=self.clause_searcher,
            student_aliases=(
                {_MODEL_STUDENT_REF: request.claims[0].student_id}
                if self.model is not None and request.claims
                else None
            ),
        )
        try:
            self.sessions.acquire(request.session_id, runtime.version)
        except SessionVersionConflict as exc:
            return self._result(
                request,
                runtime,
                outcome="refusal",
                answer=str(exc),
                trace=[*trace, "ruleset_version_conflict", "refusal"],
            )

        tools = build_tools(runtime)
        by_name = {item.name: item for item in tools}
        ledger: ScoreBreakdown | None = None
        calls: list[str] = []
        degraded = self.model is None

        if self.model is not None:
            try:
                bound = self.model.bind_tools(tools)
                messages: list[Any] = self._route_messages(request, runtime.version)
                for round_index in range(1, self.max_tool_rounds + 1):
                    response = bound.invoke(messages)
                    model_calls = list(getattr(response, "tool_calls", None) or [])
                    if not model_calls:
                        break
                    trace.append(f"tool_round_{round_index}")
                    messages.append(response)
                    for call_index, call in enumerate(model_calls):
                        name = str(call.get("name", ""))
                        args = dict(call.get("args") or {})
                        call_id = str(call.get("id") or f"call_{round_index}_{call_index}")
                        selected = by_name.get(name)
                        if selected is None:
                            raise RuntimeError(f"模型请求了未注册工具：{name}")
                        result = selected.invoke(args)
                        calls.append(name)
                        if name == "ask_clarification":
                            trace.append("clarification")
                            return self._result(
                                request,
                                runtime,
                                outcome="clarification",
                                answer=str(result["question"]),
                                trace=trace,
                                calls=calls,
                            )
                        if _is_empty_tool_result(name, result):
                            return self._clarify_empty(
                                request, runtime, by_name, calls, trace, source_tool=name
                            )
                        messages.append(
                            ToolMessage(
                                content=json.dumps(
                                    _redact_for_model(result), ensure_ascii=False, default=str
                                ),
                                tool_call_id=call_id,
                            )
                        )
                        if name == "calc_score":
                            ledger = ScoreBreakdown.model_validate(result)
                    if ledger is not None:
                        break
            except Exception as exc:
                degraded = True
                runtime.record_event(
                    "orchestration_fallback",
                    {"reason_type": type(exc).__name__},
                    summary=f"模型工具编排失败，转规则路由：{exc}",
                )
                trace.append("model_fallback")

        if ledger is None:
            fallback = self._fallback(request, runtime, by_name, calls, trace)
            if isinstance(fallback, OrchestrationResult):
                fallback.degraded = degraded or self.model is not None
                return fallback
            ledger = fallback
            degraded = True

        trace.append("number_validation")
        answer, validation, blocked = self._answer_with_guard(ledger)
        trace.append("answer")
        return self._result(
            request,
            runtime,
            outcome="answer",
            answer=answer,
            trace=trace,
            calls=calls,
            ledger=ledger,
            degraded=degraded,
            validation=validation,
            blocked=blocked,
        )

    def _fallback(
        self,
        request: OrchestrationRequest,
        runtime: ToolRuntime,
        tools: dict[str, BaseTool],
        calls: list[str],
        trace: list[str],
    ) -> ScoreBreakdown | OrchestrationResult:
        trace.append("rules_fallback")
        runtime.record_event(
            "orchestration_fallback",
            {"has_model": self.model is not None},
            summary="直接规则路由执行 lookup_rule 与 calc_score",
        )
        claims = request.claims
        if not claims:
            parsed = _claim_from_query(request, self.ruleset)
            if parsed is None:
                return self._clarify(
                    request,
                    runtime,
                    tools,
                    calls,
                    trace,
                    ["academic_year", "category", "level"],
                    "请补充待核算项目的学年、类别和完整等级。",
                )
            claims = [parsed]
            trace.append("regex_keyword_route")
        for claim in claims:
            year = claim.academic_year or request.academic_year
            level = claim.level or claim.raw_text.strip()
            missing = [
                name
                for name, value in (("academic_year", year), ("category", claim.category), ("level", level))
                if not value
            ]
            if missing:
                return self._clarify(
                    request,
                    runtime,
                    tools,
                    calls,
                    trace,
                    missing,
                    "请补充申报项目的学年、类别和获奖等级。",
                )
            lookup_args = {
                "academic_year": year,
                "college": claim.college or request.college,
                "category": claim.category,
                "level": level,
                "rank": claim.extra.get("rank"),
                "item_name": claim.extra.get("item_name"),
            }
            found = tools["lookup_rule"].invoke(lookup_args)
            calls.append("lookup_rule")
            if not found:
                return self._clarify_empty(
                    request, runtime, tools, calls, trace, source_tool="lookup_rule"
                )

        calc_claims = [
            CalcClaimInput.from_claim(_claim_with_defaults(item, request)).model_dump(mode="python")
            for item in claims
        ]
        payload = {
            "claims": calc_claims,
            "ruleset_version": runtime.version,
            "student_id": (
                _MODEL_STUDENT_REF if runtime.student_aliases else claims[0].student_id
            ),
        }
        result = tools["calc_score"].invoke(payload)
        calls.append("calc_score")
        return ScoreBreakdown.model_validate(result)

    def _clarify_empty(
        self,
        request: OrchestrationRequest,
        runtime: ToolRuntime,
        tools: dict[str, BaseTool],
        calls: list[str],
        trace: list[str],
        *,
        source_tool: str,
    ) -> OrchestrationResult:
        return self._clarify(
            request,
            runtime,
            tools,
            calls,
            trace,
            ["rule_match"],
            "没有查到对应规则，请核对学年、学院、类别和等级，或补充材料。",
            source_tool=source_tool,
        )

    def _clarify(
        self,
        request: OrchestrationRequest,
        runtime: ToolRuntime,
        tools: dict[str, BaseTool],
        calls: list[str],
        trace: list[str],
        missing_fields: list[str],
        question: str,
        *,
        source_tool: str | None = None,
    ) -> OrchestrationResult:
        trace.extend(["empty_result_branch" if source_tool else "missing_field_branch", "clarification"])
        payload = tools["ask_clarification"].invoke(
            {"missing_fields": missing_fields, "question": question}
        )
        calls.append("ask_clarification")
        return self._result(
            request,
            runtime,
            outcome="clarification",
            answer=str(payload["question"]),
            trace=trace,
            calls=calls,
        )

    def _answer_with_guard(
        self, ledger: ScoreBreakdown
    ) -> tuple[str, NumberValidation, int]:
        blocked = 0
        if self.model is not None:
            prompt = [
                SystemMessage(
                    content=(
                        "根据给定账本生成简短中文答复。只能复述账本中已有数字，必须说明总分和出处；"
                        "禁止估算或添加新数字。"
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        _redact_for_model(ledger.model_dump(mode="json")), ensure_ascii=False
                    )
                ),
            ]
            for _attempt in range(2):
                try:
                    response = self.model.invoke(prompt)
                    candidate = _response_text(response).strip()
                    validation = validate_answer_numbers(candidate, ledger)
                    if candidate and validation.valid:
                        return candidate, validation, blocked
                    blocked += 1
                    prompt.append(
                        HumanMessage(
                            content=(
                                "上一版含账本外数字，已拦截。删除这些数字后重写："
                                + "、".join(validation.unsupported)
                            )
                        )
                    )
                except Exception:
                    blocked += 1
                    break
        answer = deterministic_answer(ledger)
        return answer, validate_answer_numbers(answer, ledger), blocked

    def _route_messages(self, request: OrchestrationRequest, version: str) -> list[Any]:
        payload = {
            "query": _redact_query(request),
            "academic_year": request.academic_year,
            "college": request.college,
            "ruleset_version": version,
            "claims": [
                CalcClaimInput.from_claim(_claim_with_defaults(item, request)).model_dump(mode="json")
                for item in request.claims
                if item.level or item.raw_text.strip()
            ],
            "student_id": _MODEL_STUDENT_REF if request.claims else None,
        }
        return [
            SystemMessage(
                content=(
                    "你是综测核算工具路由器。必须调用工具，禁止自行给分。"
                    "有结构化申报时先 lookup_rule，再使用 calc_score；工具空结果不得猜测。"
                )
            ),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]

    def _result(
        self,
        request: OrchestrationRequest,
        runtime: ToolRuntime,
        *,
        outcome: str,
        answer: str,
        trace: list[str],
        calls: list[str] | None = None,
        ledger: ScoreBreakdown | None = None,
        degraded: bool = False,
        validation: NumberValidation | None = None,
        blocked: int = 0,
    ) -> OrchestrationResult:
        return OrchestrationResult(
            outcome=outcome,
            answer=answer,
            session_id=request.session_id,
            ruleset_version=runtime.version,
            ledger=ledger,
            degraded=degraded,
            model_version=self.model_version,
            state_trace=trace,
            tool_calls=list(calls or []),
            number_validation=validation,
            blocked_answer_count=blocked,
        )


def validate_answer_numbers(answer: str, ledger: ScoreBreakdown) -> NumberValidation:
    seen = _extract_numbers(answer)
    allowed = set(_extract_numbers(json.dumps(ledger.model_dump(mode="json"), ensure_ascii=False)))
    allowed_keys = {_decimal_key(item) for item in allowed}
    unsupported = [item for item in seen if _decimal_key(item) not in allowed_keys]
    return NumberValidation(valid=not unsupported, seen=seen, unsupported=unsupported)


def deterministic_answer(ledger: ScoreBreakdown) -> str:
    counted = [item for item in ledger.matches if item.counted and item.source is not None]
    answer = f"核算结果为 {ledger.total:g} 分。"
    if counted:
        source = counted[0].source
        assert source is not None
        bits = [source.doc]
        if source.page is not None:
            bits.append(f"第 {source.page} 页")
        if source.clause:
            bits.append(source.clause)
        answer += "依据：" + "，".join(bits) + "。"
    if ledger.unmatched_claims:
        answer += "存在未命中规则的申报，请人工复核。"
    return answer


def make_deepseek_model(*, model: str, api_key: str, base_url: str) -> ToolCallingModel:
    """延迟导入模型适配，未安装 llm extra 时规则降级仍可工作。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_tokens=1200,
        max_retries=0,
        timeout=45,
    )


def _claim_with_defaults(claim: Claim, request: OrchestrationRequest) -> Claim:
    return claim.model_copy(
        update={
            "academic_year": claim.academic_year or request.academic_year,
            "college": claim.college or request.college,
        }
    )


def _claim_from_query(request: OrchestrationRequest, ruleset) -> Claim | None:
    """模型不可用时只做可解释的正则/关键词解析，找不到完整键就澄清。"""
    year_match = re.search(r"20\d{2}(?:-20\d{2})?", request.query)
    year = request.academic_year or (year_match.group(0) if year_match else None)
    candidates = ruleset.filter(
        academic_year=year,
        college=request.college,
    ).rules
    category = next(
        (
            value
            for value in sorted({item.category for item in candidates}, key=len, reverse=True)
            if value in request.query
        ),
        None,
    )
    if category is None:
        return None
    category_rules = [item for item in candidates if item.category == category]
    aliases = sorted(
        ((alias, rule.level) for rule in category_rules for alias in rule.match_keys),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    level = next(
        (
            canonical
            for alias, canonical in aliases
            if normalize_level(alias).canonical in normalize_level(request.query).canonical
        ),
        None,
    )
    if not year or level is None:
        return None
    return Claim(
        student_id=f"session:{request.session_id}",
        academic_year=year,
        college=request.college,
        category=category,
        level=level,
        raw_text=request.query,
    )


def _is_empty_tool_result(name: str, result: Any) -> bool:
    return name != "ask_clarification" and result in (None, [], {})


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return ""


def _redact_query(request: OrchestrationRequest) -> str:
    text = request.query
    for claim in request.claims:
        for sensitive in (claim.student_id, claim.student_name):
            if sensitive:
                text = text.replace(sensitive, _MODEL_STUDENT_REF)
    return text


def _redact_for_model(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (_MODEL_STUDENT_REF if key == "student_id" else _redact_for_model(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_for_model(item) for item in value]
    return value


def _extract_numbers(text: str) -> list[str]:
    return list(dict.fromkeys(match.group(0) for match in _NUMBER_RE.finditer(text)))


def _decimal_key(value: str) -> Decimal | str:
    try:
        return Decimal(value).normalize()
    except InvalidOperation:
        return value


__all__ = [
    "ScoreProofOrchestrator",
    "SessionRuleLock",
    "SessionVersionConflict",
    "deterministic_answer",
    "make_deepseek_model",
    "validate_answer_numbers",
]

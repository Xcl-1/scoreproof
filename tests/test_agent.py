"""阶段 4.4：五工具、状态机、数字护栏、降级和审计。"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError
from typer.testing import CliRunner

from scoreproof.agent import (
    OrchestrationRequest,
    ScoreProofOrchestrator,
    SessionRuleLock,
    ToolRuntime,
    build_tools,
    ruleset_version,
    validate_answer_numbers,
)
from scoreproof.agent.models import CalcClaimInput
from scoreproof.cli import app
from scoreproof.retrieval.router import Clause, RetrievalHit
from scoreproof.rules.store import RuleStore
from scoreproof.schema import Ruleset, SourceRef

from .conftest import make_claim, make_rule


class ScriptedModel:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.bound_names: list[str] = []
        self.received_messages: list[list[Any]] = []

    def bind_tools(self, tools) -> ScriptedModel:
        self.bound_names = [item.name for item in tools]
        return self

    def invoke(self, messages):
        self.received_messages.append(list(messages))
        if not self.responses:
            raise AssertionError("脚本模型响应已耗尽")
        return self.responses.pop(0)


@pytest.fixture
def one_rule() -> Ruleset:
    return Ruleset(
        rules=[make_rule("省级二等奖", 8, team_factor=0.5, rule_id="r_agent")]
    )


def _request(*, session_id: str = "session-agent") -> OrchestrationRequest:
    return OrchestrationRequest(
        query="我获得省级二等奖，能加多少分？",
        session_id=session_id,
        claims=[make_claim(level="省级二等奖", claim_id="c_agent")],
        academic_year="2025-2026",
    )


class TestTools:
    def test_five_tools_have_pydantic_schemas_and_run_independently(
        self, one_rule: Ruleset, tmp_path
    ) -> None:
        store = RuleStore(tmp_path / "rules.sqlite")
        runtime = ToolRuntime(
            one_rule,
            session_id="s-tools",
            model_version="mock-v1",
            audit_sink=store,
            clause_searcher=lambda query, filters, top_k: [
                RetrievalHit(
                    clause=Clause(
                        id="clause-1",
                        text="省级二等奖计八分",
                        source=SourceRef(doc="细则.pdf", page=4, text="省级二等奖计八分"),
                    ),
                    score=1.0,
                    rank=1,
                    channel="rrf",
                )
            ],
        )
        tools = {item.name: item for item in build_tools(runtime)}
        assert set(tools) == {
            "lookup_rule",
            "search_clauses",
            "calc_score",
            "check_evidence",
            "ask_clarification",
        }
        assert all(item.args_schema is not None for item in tools.values())

        found = tools["lookup_rule"].invoke(
            {
                "academic_year": "2025-2026",
                "college": None,
                "category": "学科竞赛",
                "level": "省级二等奖",
                "rank": None,
                "item_name": None,
            }
        )
        assert found[0]["score"] == 8
        hits = tools["search_clauses"].invoke(
            {"query": "省级二等奖", "filters": {}, "top_k": 1}
        )
        assert hits[0]["source"]["page"] == 4
        diffs = tools["check_evidence"].invoke(
            {"claim": {"level": "省级二等奖"}, "evidence_fields": {"level": "省级一等奖"}}
        )
        assert diffs[0]["matches"] is False
        question = tools["ask_clarification"].invoke(
            {"missing_fields": ["level"], "question": "请补充等级"}
        )
        assert question["status"] == "needs_clarification"
        calc = tools["calc_score"].invoke(
            {
                "claims": [
                    {
                        "claim_id": "c1",
                        "academic_year": "2025-2026",
                        "college": None,
                        "category": "学科竞赛",
                        "level": "省级二等奖",
                        "team": False,
                        "catalog_listed": True,
                    }
                ],
                "ruleset_version": runtime.version,
                "student_id": "2023001",
            }
        )
        assert calc["total"] == 8
        audits = store.list_tool_calls(session_id="s-tools")
        store.close()
        assert len(audits) == 5
        assert {row["tool_name"] for row in audits} == set(tools)
        assert all(row["model_version"] == "mock-v1" for row in audits)
        assert all(row["duration_ms"] >= 0 for row in audits)

    def test_calc_input_rejects_free_form_and_coercion(self) -> None:
        with pytest.raises(ValidationError):
            CalcClaimInput.model_validate("省级二等奖加几分")
        with pytest.raises(ValidationError):
            CalcClaimInput.model_validate(
                {
                    "claim_id": "c1",
                    "category": "学科竞赛",
                    "level": "省级二等奖",
                    "team": "false",
                }
            )

    def test_ruleset_version_is_stable_and_content_addressed(self, one_rule: Ruleset) -> None:
        assert ruleset_version(one_rule) == ruleset_version(one_rule.model_copy(deep=True))
        changed = one_rule.model_copy(deep=True)
        changed.rules[0].score = 9
        assert ruleset_version(one_rule) != ruleset_version(changed)


class TestOrchestrator:
    def test_bind_tools_and_full_state_machine(self, one_rule: Ruleset, tmp_path) -> None:
        version = ruleset_version(one_rule)
        claim_payload = CalcClaimInput.from_claim(_request().claims[0]).model_dump(mode="python")
        model = ScriptedModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "lookup_rule",
                            "args": {
                                "academic_year": "2025-2026",
                                "college": None,
                                "category": "学科竞赛",
                                "level": "省级二等奖",
                                "rank": None,
                                "item_name": None,
                            },
                            "id": "call-lookup",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calc_score",
                            "args": {
                                "claims": [claim_payload],
                                "ruleset_version": version,
                                "student_id": "CURRENT_STUDENT",
                            },
                            "id": "call-calc",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="核算结果为 8 分，依据合成细则.pdf 第 4 页。"),
            ]
        )
        store = RuleStore(tmp_path / "audit.sqlite")
        result = ScoreProofOrchestrator(
            one_rule, model=model, model_version="mock-tools", audit_sink=store
        ).run(_request())
        audits = store.list_tool_calls(session_id="session-agent")
        store.close()
        assert model.bound_names == [
            "lookup_rule",
            "search_clauses",
            "calc_score",
            "check_evidence",
            "ask_clarification",
        ]
        assert result.outcome == "answer" and result.ledger is not None
        assert result.ledger.total == 8 and result.ledger.student_id == "2023001"
        assert result.number_validation.valid
        assert result.degraded is False
        assert result.tool_calls == ["lookup_rule", "calc_score"]
        assert "number_validation" in result.state_trace
        assert [row["tool_name"] for row in audits] == ["lookup_rule", "calc_score"]

    def test_empty_tool_result_forces_clarification(self, one_rule: Ruleset, tmp_path) -> None:
        model = ScriptedModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "lookup_rule",
                            "args": {
                                "academic_year": "2025-2026",
                                "college": None,
                                "category": "其它",
                                "level": "不存在",
                            },
                            "id": "empty",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )
        store = RuleStore(tmp_path / "empty.sqlite")
        result = ScoreProofOrchestrator(one_rule, model=model, audit_sink=store).run(_request())
        names = [row["tool_name"] for row in store.list_tool_calls()]
        store.close()
        assert result.outcome == "clarification" and result.ledger is None
        assert "empty_result_branch" in result.state_trace
        assert names == ["lookup_rule", "ask_clarification"]

    def test_broken_model_falls_back_to_lookup_and_calc(self, one_rule: Ruleset, tmp_path) -> None:
        class BrokenModel:
            def bind_tools(self, tools):
                raise TimeoutError("provider timeout")

            def invoke(self, messages):
                raise AssertionError("不应生成答案")

        store = RuleStore(tmp_path / "fallback.sqlite")
        result = ScoreProofOrchestrator(
            one_rule, model=BrokenModel(), model_version="broken-v1", audit_sink=store
        ).run(_request())
        names = [row["tool_name"] for row in store.list_tool_calls()]
        store.close()
        assert result.outcome == "answer" and result.ledger.total == 8
        assert result.degraded is True and result.number_validation.valid
        assert "lookup_rule" in names and "calc_score" in names
        assert "orchestration_fallback" in names

    def test_rules_fallback_parses_explicit_query_without_claims(self, one_rule: Ruleset) -> None:
        request = OrchestrationRequest(
            query="2025-2026学年学科竞赛省级二等奖能加多少分？",
            session_id="query-only",
        )
        result = ScoreProofOrchestrator(one_rule).run(request)
        assert result.outcome == "answer" and result.ledger.total == 8
        assert "regex_keyword_route" in result.state_trace
        assert result.tool_calls == ["lookup_rule", "calc_score"]

    def test_two_hallucinated_answers_are_blocked_then_deterministic(self, one_rule: Ruleset) -> None:
        version = ruleset_version(one_rule)
        claim_payload = CalcClaimInput.from_claim(_request().claims[0]).model_dump(mode="python")
        model = ScriptedModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calc_score",
                            "args": {
                                "claims": [claim_payload],
                                "ruleset_version": version,
                                "student_id": "CURRENT_STUDENT",
                            },
                            "id": "calc",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="可以加 99 分。"),
                AIMessage(content="修正后可以加 98 分。"),
            ]
        )
        result = ScoreProofOrchestrator(one_rule, model=model).run(_request())
        assert result.blocked_answer_count == 2
        assert "99" not in result.answer and "98" not in result.answer
        assert result.number_validation.valid and result.ledger.total == 8

    def test_model_messages_redact_student_identity(self, one_rule: Ruleset) -> None:
        model = ScriptedModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ask_clarification",
                            "args": {
                                "missing_fields": ["evidence"],
                                "question": "请补充证据",
                            },
                            "id": "clarify",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )
        claim = make_claim(level="省级二等奖", student_id="20239999")
        claim.student_name = "真实姓名测试"
        result = ScoreProofOrchestrator(one_rule, model=model).run(
            OrchestrationRequest(
                query="20239999真实姓名测试想核算省级二等奖",
                claims=[claim],
            )
        )
        sent = "\n".join(
            str(getattr(message, "content", ""))
            for batch in model.received_messages
            for message in batch
        )
        assert result.outcome == "clarification"
        assert "20239999" not in sent and "真实姓名测试" not in sent
        assert "CURRENT_STUDENT" in sent

    def test_one_hundred_fabricated_numbers_are_all_rejected(self, one_rule: Ruleset) -> None:
        result = ScoreProofOrchestrator(one_rule).run(_request())
        assert result.ledger is not None
        validations = [
            validate_answer_numbers(f"核算结果是 {number} 分", result.ledger)
            for number in range(1000, 1100)
        ]
        assert len(validations) == 100
        assert all(not item.valid for item in validations)

    def test_session_locks_ruleset_version(self, one_rule: Ruleset) -> None:
        sessions = SessionRuleLock()
        first = ScoreProofOrchestrator(one_rule, sessions=sessions).run(_request(session_id="same"))
        changed = one_rule.model_copy(deep=True)
        changed.rules[0].score = 9
        second = ScoreProofOrchestrator(changed, sessions=sessions).run(
            _request(session_id="same")
        )
        assert first.outcome == "answer"
        assert second.outcome == "refusal"
        assert "ruleset_version_conflict" in second.state_trace


def test_audit_payload_is_valid_json(one_rule: Ruleset, tmp_path) -> None:
    store = RuleStore(tmp_path / "json.sqlite")
    ScoreProofOrchestrator(one_rule, audit_sink=store).run(_request(session_id="json"))
    rows = store.list_tool_calls(session_id="json")
    store.close()
    assert rows
    assert all(isinstance(json.loads(row["input_json"]), dict) for row in rows)


def test_ask_score_cli_runs_end_to_end_without_model(one_rule: Ruleset, tmp_path) -> None:
    db = tmp_path / "cli.sqlite"
    with RuleStore(db) as store:
        store.upsert_rules(one_rule.rules)
    result = CliRunner().invoke(
        app,
        [
            "ask-score",
            "我获得省级二等奖，能加多少分？",
            "--student-id",
            "2023001",
            "--year",
            "2025-2026",
            "--category",
            "学科竞赛",
            "--level",
            "省级二等奖",
            "--no-model",
            "--db",
            str(db),
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"outcome": "answer"' in result.output
    assert '"total": 8.0' in result.output
    with RuleStore(db) as store:
        assert [row["tool_name"] for row in store.list_tool_calls()][-2:] == [
            "lookup_rule",
            "calc_score",
        ]

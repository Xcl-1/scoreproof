"""服务层测试：健康检查、核算、SSE、解释、证据查重。

状态是模块级单例，测试里用 monkeypatch 注入内存规则库，避免污染真实 data/。
"""

from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from scoreproof.api.app import app, get_state  # noqa: E402
from scoreproof.indexing import DocumentChunk, HybridIndexManifestStore  # noqa: E402
from scoreproof.review import fact_hash  # noqa: E402
from scoreproof.rules.gateway import ExtractionGateway  # noqa: E402
from scoreproof.schema import (  # noqa: E402
    Evidence,
    Ruleset,
    SourceRef,  # noqa: E402
)

from .conftest import make_rule  # noqa: E402


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    ruleset = Ruleset(
        rules=[
            make_rule("省级一等奖", 10, group="学科竞赛", cap=20, rule_id="r1"),
            make_rule("省级二等奖", 8, group="学科竞赛", cap=20, team_factor=0.5, rule_id="r2"),
            make_rule("一等奖", 6, category="文体活动", group="文体活动", rule_id="r3"),
        ]
    )
    state = get_state()
    monkeypatch.setattr(state, "ruleset", ruleset)
    monkeypatch.setattr(state, "evidence", {})
    monkeypatch.setattr(
        state,
        "settings",
        replace(
            state.settings,
            db_path=tmp_path / "rules.sqlite",
            cost_db_path=tmp_path / "rules.sqlite",
            index_db_path=tmp_path / "index.sqlite",
            vector_dir=tmp_path / "chroma",
        ),
    )
    monkeypatch.setattr(state, "_agent_sessions", None)
    with TestClient(app) as c:
        # startup 钩子会尝试从磁盘加载，这里再注入一次确保用的是内存规则
        get_state().ruleset = ruleset
        yield c


CLAIM_OK = {
    "student_id": "2023001",
    "academic_year": "2025-2026",
    "category": "学科竞赛",
    "raw_text": "省二等奖",
    "level": "省级二等奖",
}


class TestBase:
    def test_health(self, client: TestClient) -> None:
        res = client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ok" and body["rules_loaded"] == 3

    def test_config_hides_secrets(self, client: TestClient) -> None:
        res = client.get("/api/config")
        assert res.status_code == 200
        text = json.dumps(res.json(), ensure_ascii=False)
        assert "api_key" not in text.lower()
        assert "llm_configured" in res.json()

    def test_cost_summary_is_real_read_only_endpoint(self, client: TestClient) -> None:
        res = client.get("/api/costs/summary")
        assert res.status_code == 200
        body = res.json()
        assert body["source_database"] == "rules.sqlite"
        assert body["external_calls"] == 0
        assert "by_model" in body

    def test_review_api_state_machine_and_error_codes(self, client: TestClient) -> None:
        create_payload = {
            "task_type": "manual_request",
            "reason_codes": ["manual_request"],
            "priority": "high",
            "evidence_id": "evidence-review-api",
            "field_name": "级别",
            "current_normalized_value": {"value": "省级"},
            "audit_metadata": {
                "source_component": "manual",
                "input_fact_hash": fact_hash({"api": "review", "id": 1}),
                "trigger_version": "api-test-v1",
                "sensitive_fields_redacted": True,
            },
        }
        created = client.post("/api/reviews", json=create_payload)
        assert created.status_code == 200
        task = created.json()
        task_id = task["review_task_id"]

        duplicate = client.post("/api/reviews", json=create_payload)
        assert duplicate.status_code == 200
        assert duplicate.json()["review_task_id"] == task_id
        assert client.get("/api/reviews", params={"status": "pending"}).json()["count"] == 1
        assert client.get(f"/api/reviews/{task_id}").status_code == 200
        assert client.get("/api/reviews/rvw_missing").status_code == 404

        started = client.post(
            f"/api/reviews/{task_id}/start",
            json={"expected_revision": 1, "operator_token": "reviewer:api"},
        )
        assert started.status_code == 200
        stale = client.post(
            f"/api/reviews/{task_id}/resolve",
            json={
                "expected_revision": 1,
                "resolver_token": "reviewer:api",
                "resolution": "confirmed",
                "apply_to_evidence": False,
            },
        )
        assert stale.status_code == 409
        invalid = client.post(
            f"/api/reviews/{task_id}/resolve",
            json={
                "expected_revision": 2,
                "resolver_token": "reviewer:api",
                "resolution": "corrected",
                "corrected_value": {"value": "不存在级别"},
                "apply_to_evidence": False,
            },
        )
        assert invalid.status_code == 422
        resolved = client.post(
            f"/api/reviews/{task_id}/resolve",
            json={
                "expected_revision": 2,
                "resolver_token": "reviewer:api",
                "resolution": "corrected",
                "corrected_value": {"value": "省部级"},
                "apply_to_evidence": False,
            },
        )
        assert resolved.status_code == 200
        assert resolved.json()["manual_corrected_value"] == {"value": "省级"}
        illegal = client.post(
            f"/api/reviews/{task_id}/start",
            json={"expected_revision": 3, "operator_token": "reviewer:api"},
        )
        assert illegal.status_code == 409

    def test_user_trial_api_template_stays_blocked(self, client: TestClient) -> None:
        template = client.get("/api/eval/user-trial/template")
        assert template.status_code == 200
        assert template.json()["sessions"] == []
        evaluated = client.post("/api/eval/user-trial", json=template.json())
        assert evaluated.status_code == 200
        body = evaluated.json()
        assert body["sample_size"] == 0
        assert body["formal_gate_eligible"] is False

    def test_rule_extraction_eval_api_keeps_synthetic_sample_as_smoke(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        api_module = importlib.import_module("scoreproof.api.app")

        class FakeExtractor:
            model = "fake-rule-model"

            def __init__(self, **_: object) -> None:
                pass

            def available(self) -> bool:
                return False

            def extract_validated(self, text: str, **kwargs: object):
                return ExtractionGateway().validate_batch(
                    [
                        {
                            "category": "学科竞赛",
                            "level": "省级一等奖",
                            "score": 3.0,
                            "evidence_quote": text,
                        }
                    ],
                    source_text=text,
                    context=kwargs["context"],
                )

        monkeypatch.setattr(api_module, "LLMExtractor", FakeExtractor)
        payload = {
            "dataset_version": "api-smoke",
            "cases": [
                {
                    "case_id": "api-1",
                    "source_id": "synthetic-rule-1",
                    "source_text": "省级一等奖计3分。",
                    "academic_year": "2025-2026",
                    "category_hint": "学科竞赛",
                    "real_source": False,
                    "synthetic": True,
                    "expected_rules": [
                        {
                            "category": "学科竞赛",
                            "level": "省级一等奖",
                            "score": 3.0,
                            "evidence_quote": "省级一等奖计3分。",
                        }
                    ],
                }
            ],
        }
        response = client.post("/api/eval/rule-extraction", json=payload)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["exact_rule_accuracy"]["value"] == 1.0
        assert body["smoke_test_only"] is True
        assert body["formal_gate_eligible"] is False

    def test_index_page(self, client: TestClient) -> None:
        assert "scoreproof" in client.get("/").text

    def test_openapi(self, client: TestClient) -> None:
        assert client.get("/openapi.json").status_code == 200

    def test_release_readiness_never_promotes_smoke_reports(self, client: TestClient) -> None:
        response = client.get("/api/release-readiness", params={"candidate_version": "test-candidate"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["candidate_version"] == "test-candidate"
        assert payload["ready"] is False
        assert "certificate_fields" in payload["blocking_gate_ids"]
        assert "evidence_dedup" in payload["blocking_gate_ids"]


class TestCalc:
    def test_calc_returns_traceable_breakdown(self, client: TestClient) -> None:
        res = client.post("/api/calc", json={"claims": [CLAIM_OK], "academic_year": "2025-2026"})
        assert res.status_code == 200
        bd = res.json()["breakdown"]
        assert bd["total"] == 8
        counted = [m for m in bd["matches"] if m["counted"]]
        assert counted and counted[0]["source"]["doc"] == "合成细则.pdf"

    def test_same_group_take_max(self, client: TestClient) -> None:
        payload = {
            "claims": [
                CLAIM_OK,
                {**CLAIM_OK, "raw_text": "省一等奖", "level": "省级一等奖"},
            ],
            "academic_year": "2025-2026",
        }
        res = client.post("/api/calc", json=payload)
        assert res.json()["breakdown"]["total"] == 10

    def test_team_factor(self, client: TestClient) -> None:
        payload = {"claims": [{**CLAIM_OK, "team": True}], "academic_year": "2025-2026"}
        assert client.post("/api/calc", json=payload).json()["breakdown"]["total"] == 4

    def test_unmatched_is_refused_not_guessed(self, client: TestClient) -> None:
        payload = {
            "claims": [{**CLAIM_OK, "raw_text": "宿舍卫生优秀", "level": None, "category": "宿舍管理"}],
            "academic_year": "2025-2026",
        }
        bd = client.post("/api/calc", json=payload).json()["breakdown"]
        assert bd["total"] == 0 and bd["unmatched_claims"]

    def test_empty_claims_rejected(self, client: TestClient) -> None:
        assert client.post("/api/calc", json={"claims": []}).status_code == 422

    def test_agent_endpoint_real_rules_fallback(self, client: TestClient) -> None:
        response = client.post(
            "/api/agent",
            json={
                "query": "我获得省级二等奖，能加多少分？",
                "use_model": False,
                "academic_year": "2025-2026",
                "claims": [CLAIM_OK],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["outcome"] == "answer"
        assert body["ledger"]["total"] == 8
        assert body["number_validation"]["valid"] is True
        assert body["tool_calls"] == ["lookup_rule", "calc_score"]

    def test_agent_endpoint_model_factory_failure_falls_back(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = get_state()
        monkeypatch.setattr(state, "settings", replace(state.settings, llm_api_key="fake"))

        def broken_factory(**kwargs):
            raise ImportError("langchain-openai unavailable")

        monkeypatch.setattr("scoreproof.agent.orchestrator.make_deepseek_model", broken_factory)
        response = client.post(
            "/api/agent",
            json={
                "query": "省级二等奖能加多少分？",
                "academic_year": "2025-2026",
                "claims": [CLAIM_OK],
            },
        )
        assert response.status_code == 200
        assert response.json()["ledger"]["total"] == 8
        assert response.json()["degraded"] is True

    def test_sse_stream(self, client: TestClient) -> None:
        with client.stream(
            "POST", "/api/calc/stream", json={"claims": [CLAIM_OK], "academic_year": "2025-2026"}
        ) as res:
            assert res.status_code == 200
            assert res.headers["content-type"].startswith("text/event-stream")
            body = "".join(res.iter_text())
        assert "event: start" in body
        assert "event: group" in body
        assert "event: done" in body
        assert '"total": 8' in body.replace(" ", " ")


class TestBatchCalc:
    def test_persistent_batch_idempotency_listing_and_exports(self, client: TestClient) -> None:
        payload = {
            "claims": [
                {**CLAIM_OK, "student_name": "张同学"},
                {
                    **CLAIM_OK,
                    "student_id": "2023002",
                    "student_name": "=FORMULA()",
                    "raw_text": "省一等奖",
                    "level": "省级一等奖",
                },
            ],
            "academic_year": "2025-2026",
        }
        headers = {"Idempotency-Key": "api-batch-test-001"}
        created = client.post("/api/batches", json=payload, headers=headers)
        assert created.status_code == 201, created.text
        body = created.json()
        task = body["task"]
        batch_id = task["batch_id"]
        assert body["created"] is True
        assert task["status"] == "succeeded"
        assert task["claim_count"] == 2 and task["student_count"] == 2
        assert task["result_hash"]
        assert body["artifact"]["results"]["2023001"]["total"] == 8
        assert body["artifact"]["results"]["2023002"]["total"] == 10
        assert [event["event_type"] for event in body["events"]] == [
            "created",
            "started",
            "succeeded",
        ]

        repeated = client.post("/api/batches", json=payload, headers=headers)
        assert repeated.status_code == 200
        assert repeated.json()["created"] is False
        assert repeated.json()["task"]["batch_id"] == batch_id

        conflicting = client.post(
            "/api/batches",
            json={**payload, "college": "不同学院"},
            headers=headers,
        )
        assert conflicting.status_code == 409
        assert conflicting.json()["detail"]["code"] == "version_conflict"

        listed = client.get("/api/batches")
        assert listed.status_code == 200
        assert listed.json()["count"] == 1
        assert "artifact" not in listed.json()["tasks"][0]

        detail = client.get(f"/api/batches/{batch_id}")
        assert detail.status_code == 200
        assert detail.json()["task"]["rule_content_hash"] == task["rule_content_hash"]
        assert client.get("/api/batches/bat_missing").status_code == 404

        exported_json = client.get(f"/api/batches/{batch_id}/export?format=json")
        assert exported_json.status_code == 200
        assert exported_json.json()["batch_id"] == batch_id
        assert "attachment" in exported_json.headers["content-disposition"]

        exported_csv = client.get(f"/api/batches/{batch_id}/export?format=csv")
        assert exported_csv.status_code == 200
        decoded = exported_csv.content.decode("utf-8-sig")
        assert "rule_version_id" in decoded
        assert "'=FORMULA()" in decoded

    def test_batch_rejects_bad_idempotency_key(self, client: TestClient) -> None:
        response = client.post(
            "/api/batches",
            json={"claims": [CLAIM_OK]},
            headers={"Idempotency-Key": "bad"},
        )
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "schema_validation_error"
        duplicate_ids = client.post(
            "/api/batches",
            json={
                "claims": [
                    {**CLAIM_OK, "id": "same-claim"},
                    {**CLAIM_OK, "id": "same-claim"},
                ]
            },
        )
        assert duplicate_ids.status_code == 422


class TestRetrievalEndpoints:
    def test_explain(self, client: TestClient) -> None:
        res = client.post("/api/explain", json={"claim": CLAIM_OK})
        assert res.status_code == 200
        body = res.json()
        assert body["channel"] == "structured" and body["matched"] is True
        assert body["source"]["page"] == 4
        assert body["claim"]["canonical_level"] == "省级二等奖"

    def test_refusal_check_positive(self, client: TestClient) -> None:
        payload = {"claim": {**CLAIM_OK, "raw_text": "无关内容", "level": None, "category": "其它"}}
        body = client.post("/api/refusal-check", json=payload).json()
        assert body["refused"] is True and body["channel"] == "none"

    def test_refusal_check_negative(self, client: TestClient) -> None:
        body = client.post("/api/refusal-check", json={"claim": CLAIM_OK}).json()
        assert body["refused"] is False

    def test_hybrid_search(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = get_state().settings
        chunks = [
            DocumentChunk(
                logical_key=f"page:{index}",
                text=text,
                kind="page",
                source=SourceRef(doc="rules.pdf", page=index, text=text),
                metadata={"academic_year": "2025-2026", "college": "计算机学院"},
            )
            for index, text in enumerate(
                [
                    "学科竞赛国家级一等奖计十五分",
                    "志愿服务每满二十小时计一分",
                    "文体活动校级一等奖计三分",
                    "知识产权第一专利人可申请创新加分",
                ],
                start=1,
            )
        ]
        with HybridIndexManifestStore(
            settings.index_db_path,
            vector_dir=settings.vector_dir,
            embedding_model="scoreproof-hash-v1",
        ) as index:
            index.sync_document(
                doc_id="rules",
                source_path="rules.pdf",
                document_bytes=b"rules-v1",
                chunks=chunks,
                embedding_model="scoreproof-hash-v1",
            )
        response = client.post(
            "/api/search",
            json={
                "query": "志愿服务",
                "top_k": 2,
                "academic_year": "2025-2026",
                "college": "计算机学院",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 2
        assert body["hits"][0]["id"] == "rules:page:2"
        assert body["hits"][0]["channel"] == "rrf"

        citation = client.post(
            "/api/citation-check",
            json={"query": "志愿服务如何换算成绩", "top_k": 5},
        )
        assert citation.status_code == 200
        assert citation.json()["disposition"] == "manual_review"
        refused = client.post(
            "/api/citation-check",
            json={"query": "宿舍空调坏了找谁维修", "top_k": 5},
        )
        assert refused.status_code == 200
        assert refused.json()["disposition"] == "refuse"

        class FakeReranker:
            model_version = "fake"

            def __init__(self, **kwargs) -> None:
                pass

            def score(self, query: str, documents: list[str]) -> list[float]:
                return [float(-index) for index in range(len(documents))]

        api_module = importlib.import_module("scoreproof.api.app")
        monkeypatch.setattr(api_module, "FastEmbedReranker", FakeReranker)
        reranked = client.post(
            "/api/search",
            json={
                "query": "志愿服务",
                "top_k": 2,
                "academic_year": "2025-2026",
                "college": "计算机学院",
                "rerank": True,
            },
        )
        assert reranked.status_code == 200
        assert reranked.json()["hits"][0]["channel"] == "rerank"
        assert reranked.json()["hits"][0]["rerank_score"] is not None
        assert get_state().reranker_provider() is get_state().reranker_provider()


class TestEvidence:
    def test_extract_certificate_upload_entry(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        state = get_state()
        monkeypatch.setattr(state, "settings", replace(state.settings, data_dir=tmp_path / "data"))
        evidence = Evidence(
            id="ev_extract",
            type="image",
            ocr_text="学生001 国家级一等奖",
            fields={"姓名": "学生001", "级别": "国家级", "奖项/名次": "一等奖"},
            field_confidence={"姓名": 0.95, "级别": 0.9, "奖项/名次": 0.9},
            extractor="ocr+llm",
        )

        class Result:
            def __init__(self) -> None:
                self.evidence = evidence

            def model_dump(self, **_: object) -> dict:
                return {
                    "source": "uploaded",
                    "extraction": {
                        "fields": {"姓名": {"normalized_value": "学生001"}},
                        "vlm": {"requested": False, "called": False},
                    },
                    "evidence": evidence.model_dump(mode="json"),
                }

        def fake_extract(path: Path, **_: object) -> Result:
            assert path.exists() and path.name.startswith("certificate-")
            return Result()

        api_module = importlib.import_module("scoreproof.api.app")
        monkeypatch.setattr(api_module, "run_certificate_extraction", fake_extract)
        response = client.post(
            "/api/evidence/extract-certificate",
            files={"file": ("award.png", b"real-file-bytes", "image/png")},
        )
        assert response.status_code == 200, response.text
        assert response.json()["evidence"]["id"] == "ev_extract"
        assert "ev_extract" in get_state().evidence

    def test_upsert_and_duplicate_detection(self, client: TestClient) -> None:
        base = {
            "type": "image",
            "path": "data/raw/cert_a.jpg",
            "ocr_text": "省级二等奖 张三 2025",
            "fields": {"赛事": "数学建模", "等级": "省级二等奖", "姓名": "张三"},
            "field_confidence": {"等级": 0.95, "姓名": 0.6},
            "phash": "deadbeef",
            "extractor": "ocr+llm",
        }
        first = client.post("/api/evidence", json={"evidence": base})
        assert first.status_code == 200
        assert first.json()["requires_review"] is True  # 姓名置信度低 + 未人工校对

        # 同一张图（同 pHash）+ 同字段指纹 -> 应被识别为重复申报
        clone = {**base, "id": "ev_same", "path": "data/raw/cert_a_copy.jpg"}
        client.post("/api/evidence", json={"evidence": clone})

        dupes = client.get("/api/evidence/duplicates").json()
        assert dupes["total_evidence"] == 2
        assert any(len(v) > 1 for v in dupes["by_phash"].values())
        assert any(len(v) > 1 for v in dupes["by_fingerprint"].values())

    def test_bad_confidence_rejected(self, client: TestClient) -> None:
        payload = {
            "evidence": {
                "type": "image",
                "fields": {"等级": "省级二等奖"},
                "field_confidence": {"等级": 1.5},  # 超出 [0,1]
            }
        }
        assert client.post("/api/evidence", json=payload).status_code == 422

    def test_fingerprint_differs_for_different_fields(self) -> None:
        a = Evidence(type="image", fields={"赛事": "数学建模", "等级": "省级二等奖"})
        b = Evidence(type="image", fields={"赛事": "数学建模", "等级": "省级一等奖"})
        assert a.fingerprint() != b.fingerprint()

    def test_fingerprint_falls_back_to_ocr_text(self) -> None:
        a = Evidence(type="image", ocr_text="省级二等奖 张三")
        b = Evidence(type="image", ocr_text="省级二等奖 张三 ")
        assert a.fingerprint() == b.fingerprint()

    def test_pair_compare_and_claim_consistency_endpoints(self, client: TestClient) -> None:
        fields = {
            "姓名": "张三",
            "赛事名称": "数学建模竞赛",
            "级别": "省级",
            "奖项/名次": "二等奖",
            "获奖日期": "2025-10-02",
            "颁发单位": "省竞赛组委会",
            "团队属性": "个人",
            "申报类别": "学科竞赛",
        }
        for evidence_id, image_hash in (("ev_left", "0000000000000000"), ("ev_right", "0000000000000001")):
            response = client.post(
                "/api/evidence",
                json={
                    "evidence": {
                        "id": evidence_id,
                        "type": "image",
                        "fields": fields,
                        "phash": image_hash,
                    }
                },
            )
            assert response.status_code == 200

        compared = client.post(
            "/api/evidence/compare",
            json={"left_id": "ev_left", "right_id": "ev_right"},
        )
        assert compared.status_code == 200, compared.text
        assert compared.json()["status"] == "确定重复"
        grouped = client.get("/api/evidence/duplicates").json()
        assert grouped["total_pairs"] == 1
        assert grouped["flagged_pairs"][0]["left_id"] == "ev_left"

        checked = client.post(
            "/api/evidence/check-claim",
            json={
                "claim": {
                    "id": "claim-1",
                    "student_id": "2025001",
                    "student_name": "张三",
                    "academic_year": "2025-2026",
                    "category": "学科竞赛",
                    "raw_text": "省级二等奖",
                    "team": False,
                    "extra": {"event_name": "数学建模竞赛"},
                },
                "evidence_id": "ev_left",
                "policy": {
                    "allowed_issuers": ["省竞赛组委会"],
                    "catalog_events": ["数学建模竞赛"],
                },
            },
        )
        assert checked.status_code == 200, checked.text
        assert checked.json()["status"] == "通过"

    def test_pair_compare_rejects_missing_or_same_id(self, client: TestClient) -> None:
        client.post(
            "/api/evidence",
            json={"evidence": {"id": "ev_one", "type": "image"}},
        )
        client.post(
            "/api/evidence",
            json={"evidence": {"id": "ev_two", "type": "image"}},
        )
        assert client.get("/api/evidence/duplicates").json()["by_fingerprint"] == {}
        same = client.post(
            "/api/evidence/compare",
            json={"left_id": "ev_one", "right_id": "ev_one"},
        )
        assert same.status_code == 422
        missing = client.post(
            "/api/evidence/compare",
            json={"left_id": "ev_one", "right_id": "missing"},
        )
        assert missing.status_code == 404

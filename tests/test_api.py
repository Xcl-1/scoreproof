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

    def test_index_page(self, client: TestClient) -> None:
        assert "scoreproof" in client.get("/").text

    def test_openapi(self, client: TestClient) -> None:
        assert client.get("/openapi.json").status_code == 200


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

        monkeypatch.setattr(
            "scoreproof.agent.orchestrator.make_deepseek_model", broken_factory
        )
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

    def test_hybrid_search(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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

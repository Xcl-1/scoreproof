"""服务层测试：健康检查、核算、SSE、解释、证据查重。

状态是模块级单例，测试里用 monkeypatch 注入内存规则库，避免污染真实 data/。
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from scoreproof.api.app import app, get_state  # noqa: E402
from scoreproof.schema import Evidence, Ruleset  # noqa: E402

from .conftest import make_rule  # noqa: E402


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
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


class TestEvidence:
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

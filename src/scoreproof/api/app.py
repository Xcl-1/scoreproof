"""服务层：FastAPI + SSE 流式 + 引用面板所需的数据。

P3 目标（项目总结第 8 节）：同学自助查询 + 班委批量核算。
本文件只做"薄适配"：解析请求 -> 调 calc/retrieval -> 返回带出处的结构，
所有业务逻辑都在 calc / retrieval / rules 里，便于单元测试。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..calc.engine import EngineConfig, compute_claims
from ..config import get_settings
from ..errors import ScoreProofError
from ..ingest.excel_loader import load_rules
from ..retrieval.router import Router
from ..rules.store import RuleStore
from ..schema import Claim, Evidence, Ruleset

ClaimStatusLiteral = Literal["待核对", "已核对", "已驳回", "低置信", "未命中规则"]


# ======================================================================
# 请求 / 响应模型
# ======================================================================


class ClaimIn(BaseModel):
    student_id: str
    student_name: str | None = None
    academic_year: str | None = None
    college: str | None = None
    category: str = "未分类"
    raw_text: str = ""
    level: str | None = None
    team: bool = False
    catalog_listed: bool = True
    evidence_ids: list[str] = Field(default_factory=list)

    def to_claim(self) -> Claim:
        return Claim(**self.model_dump())


class CalcRequest(BaseModel):
    claims: list[ClaimIn] = Field(min_length=1)
    academic_year: str | None = None
    college: str | None = None
    strict_year: bool = True
    fuzzy_fallback: bool = True

    def config(self) -> EngineConfig:
        return EngineConfig(strict_year=self.strict_year, fuzzy_fallback=self.fuzzy_fallback)


class ExplainRequest(BaseModel):
    claim: ClaimIn
    top_k: int = Field(default=5, ge=1, le=20)


class EvidenceIn(BaseModel):
    evidence: Evidence


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    rules_loaded: int
    llm_configured: bool


# ======================================================================
# 状态：规则库由入口一次性加载（P1 简化，P4 换增量索引）
# ======================================================================


class AppState:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.ruleset: Ruleset = Ruleset()
        self.evidence: dict[str, Evidence] = {}
        self._store: RuleStore | None = None

    # ---------- 规则库 ----------

    @property
    def store(self) -> RuleStore:
        if self._store is None:
            self._store = RuleStore(self.settings.db_path)
        return self._store

    def load_ruleset(self) -> Ruleset:
        """优先从 SQLite 读；没有则尝试 data/rules/rules.json。"""
        try:
            rules = self.store.list_rules()
            if rules:
                self.ruleset = Ruleset(rules=rules, meta={"origin": str(self.settings.db_path)})
                return self.ruleset
        except Exception:  # pragma: no cover - 首次运行数据库为空
            pass
        json_path = self.settings.rules_dir / "rules.json"
        if json_path.exists():
            self.ruleset = Ruleset.from_json(json_path)
        return self.ruleset

    def ensure_rules(self) -> Ruleset:
        if not self.ruleset.rules:
            self.load_ruleset()
        return self.ruleset


state = AppState()


def get_state() -> AppState:
    return state


# ======================================================================
# 应用
# ======================================================================


def create_app() -> FastAPI:
    app = FastAPI(
        title="scoreproof · 综测加分核算系统",
        version=__version__,
        description=(
            "异构文档 -> 结构化规则库 -> 确定性计算。"
            "LLM 只做抽取与条款定位，所有数值计算由 Python 代码完成，每条分值可回溯原文。"
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 本地工具；上公网前务必收敛
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _startup() -> None:  # pragma: no cover - 启动钩子
        st = get_state()
        st.settings.ensure_dirs()
        st.load_ruleset()

    # ---------------- 基础 ----------------

    @app.get("/health", response_model=HealthResponse, tags=["base"])
    def health() -> HealthResponse:
        st = get_state()
        return HealthResponse(
            status="ok",
            version=__version__,
            rules_loaded=len(st.ensure_rules()),
            llm_configured=st.settings.llm_configured,
        )

    @app.get("/api/config", tags=["base"])
    def read_config() -> dict:
        return get_state().settings.safe_repr()

    # ---------------- 规则库 ----------------

    @app.get("/api/rules", tags=["rules"])
    def list_rules(
        academic_year: str | None = None,
        college: str | None = None,
        category: str | None = None,
    ) -> dict:
        st = get_state()
        rules = st.store.list_rules(academic_year=academic_year, college=college, category=category)
        return {"count": len(rules), "rules": [r.model_dump(mode="json") for r in rules]}

    @app.post("/api/rules/import", tags=["rules"])
    async def import_rules(
        file: UploadFile = File(...),
        academic_year: str | None = None,
        college: str | None = None,
    ) -> dict:
        """上传规则表（Excel）-> 解析 -> 入库。第一版仍建议人工校对一遍。"""
        st = get_state()
        tmp = _tmp_path(file.filename or "rules.xlsx")
        tmp.write_bytes(await file.read())
        year = academic_year or _guess_year(tmp)
        if not year:
            raise HTTPException(
                status_code=400,
                detail={"code": "schema_validation_error",
                        "message": "无法确定学年，请在查询参数里显式传 academic_year=2025-2026",
                        "detail": {"filename": tmp.name}},
            )
        try:
            rules = load_rules(tmp, academic_year=year, college=college)
        except ScoreProofError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
        n = st.store.upsert_rules(rules)
        st.load_ruleset()
        return {"imported": n, "total": len(st.ruleset), "academic_year": year}

    # ---------------- 核算 ----------------

    @app.post("/api/calc", tags=["calc"])
    def calc(req: CalcRequest) -> dict:
        st = get_state()
        ruleset = st.ensure_rules()
        claims = [c.to_claim() for c in req.claims]
        year = req.academic_year or next((c.academic_year for c in claims if c.academic_year), None)
        breakdown = compute_claims(
            claims, ruleset, academic_year=year, college=req.college, config=req.config()
        )
        return {"breakdown": breakdown.to_dict(), "claims": [c.model_dump(mode="json") for c in claims]}

    @app.post("/api/calc/stream", tags=["calc"])
    async def calc_stream(req: CalcRequest) -> StreamingResponse:
        """SSE：逐组推送账目，前端可边算边渲染引用面板。"""
        st = get_state()
        ruleset = st.ensure_rules()
        claims = [c.to_claim() for c in req.claims]
        year = req.academic_year or next((c.academic_year for c in claims if c.academic_year), None)
        breakdown = compute_claims(
            claims, ruleset, academic_year=year, college=req.college, config=req.config()
        )

        async def events() -> AsyncIterator[str]:
            yield _sse("start", {"student_id": breakdown.student_id, "student_count": len(claims)})
            for group in breakdown.groups:
                yield _sse("group", group.model_dump(mode="json"))
                await asyncio.sleep(0)  # 让出事件循环，保证真的流式
            for note in breakdown.notes:
                yield _sse("note", {"text": note})
            yield _sse("done", {"total": breakdown.total, "unmatched": breakdown.unmatched_claims})
            yield "event: close\ndata: {}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---------------- 检索 / 解释 ----------------

    @app.post("/api/explain", tags=["retrieval"])
    def explain(req: ExplainRequest) -> dict:
        st = get_state()
        router = Router(st.ensure_rules(), academic_year=req.claim.academic_year,
                        college=req.claim.college)
        return router.explain(req.claim.to_claim(), top_k=req.top_k)

    @app.post("/api/refusal-check", tags=["retrieval"])
    def refusal_check(req: ExplainRequest) -> dict:
        """只关心"该不该拒答"的场景（简历指标：未找到规则时正确拒答率 100%）。"""
        st = get_state()
        router = Router(st.ensure_rules(), academic_year=req.claim.academic_year,
                        college=req.claim.college)
        res = router.route(req.claim.to_claim())
        return {"refused": res.refused, "channel": res.channel, "reason": res.reason}

    # ---------------- 证据 / 多模态（P2 入口） ----------------

    @app.post("/api/evidence", tags=["evidence"])
    def upsert_evidence(payload: EvidenceIn) -> dict:
        st = get_state()
        st.evidence[payload.evidence.id] = payload.evidence
        return {
            "id": payload.evidence.id,
            "requires_review": payload.evidence.requires_review(),
            "fingerprint": payload.evidence.fingerprint(),
        }

    @app.get("/api/evidence/duplicates", tags=["evidence"])
    def duplicates() -> dict:
        """重复申报检测（pHash + 字段指纹）。"""
        st = get_state()
        by_phash: dict[str, list[str]] = {}
        by_fp: dict[str, list[str]] = {}
        for ev in st.evidence.values():
            if ev.phash:
                by_phash.setdefault(ev.phash, []).append(ev.id)
            by_fp.setdefault(ev.fingerprint(), []).append(ev.id)
        dupes = {
            "by_phash": {k: v for k, v in by_phash.items() if len(v) > 1},
            "by_fingerprint": {k: v for k, v in by_fp.items() if len(v) > 1},
        }
        return {"total_evidence": len(st.evidence), **dupes}

    # ---------------- 调试页 ----------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return _INDEX_HTML

    return app


# ======================================================================
# 辅助
# ======================================================================


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _tmp_path(filename: str) -> Path:
    settings = get_settings()
    settings.ensure_dirs()
    tmp_dir = settings.data_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / Path(filename).name


def _guess_year(path: Path) -> str:
    """从文件名猜学年，如 ``2025综测细则.xlsx`` -> ``2025-2026``。"""
    import re

    m = re.search(r"(20\d{2})", path.name)
    return f"{m.group(1)}-{int(m.group(1)) + 1}" if m else ""


_INDEX_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>scoreproof</title>
<style>
 body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:2rem;max-width:900px}
 code,pre{background:#f5f5f5;padding:.15rem .35rem;border-radius:4px}
 li{margin:.35rem 0}
</style></head><body>
<h1>scoreproof · 综测加分核算系统</h1>
<p>确定性计算 + 双通道检索，每条分值可回溯原文。前端在 <code>web/</code>（P3 任务）。</p>
<ul>
 <li><a href="/docs">/docs</a> — OpenAPI 交互文档</li>
 <li><a href="/health">/health</a> — 健康检查与规则数</li>
 <li><code>POST /api/calc</code> — 提交申报条目 -> 返回可回溯账目</li>
 <li><code>POST /api/explain</code> — 单条申报的通道命中与原文引用</li>
 <li><code>POST /api/refusal-check</code> — 未找到规则时是否正确拒答</li>
</ul>
</body></html>"""


app = create_app()

__all__ = ["AppState", "app", "create_app", "get_state"]

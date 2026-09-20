"""服务层：FastAPI + SSE 流式 + 引用面板所需的数据。

V3.0 的 P0 仅保留最小演示 API；完整 Web 三端延后至 P2。
本文件只做"薄适配"：解析请求 -> 调 calc/retrieval -> 返回带出处的结构，
所有业务逻辑都在 calc / retrieval / rules 里，便于单元测试。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..calc.engine import EngineConfig, compute_claims
from ..config import PROJECT_ROOT, get_settings
from ..errors import ScoreProofError
from ..eval.readiness import build_release_readiness
from ..eval.user_trial import UserTrialDataset, evaluate_user_trials, user_trial_template
from ..evidence.certificate import extract_certificate as run_certificate_extraction
from ..evidence.consistency import ConsistencyPolicy, compare_claim_evidence
from ..evidence.dedup import DuplicateThresholds, compare_evidence, fact_fingerprint
from ..indexing import EmbeddingFunction, HybridIndexManifestStore, make_embedding_provider
from ..ingest.excel_loader import load_rules
from ..observability import CostLedger
from ..retrieval.citation import verify_text_citations
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.query import rewrite_retrieval_query
from ..retrieval.rerank import FastEmbedReranker, RerankingRetriever
from ..retrieval.router import Retriever, Router
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


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    academic_year: str | None = None
    college: str | None = None
    doc_id: str | None = None
    rerank: bool = False


class AgentRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str | None = None
    claims: list[ClaimIn] = Field(default_factory=list)
    academic_year: str | None = None
    college: str | None = None
    use_model: bool = True


class EvidenceIn(BaseModel):
    evidence: Evidence


class EvidenceCompareRequest(BaseModel):
    left_id: str
    right_id: str
    thresholds: DuplicateThresholds = Field(default_factory=DuplicateThresholds)


class ClaimEvidenceCheckRequest(BaseModel):
    claim: Claim
    evidence_id: str
    policy: ConsistencyPolicy = Field(default_factory=ConsistencyPolicy)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    rules_loaded: int
    llm_configured: bool


# ======================================================================
# 状态：规则库由入口一次性加载（阶段 4 改为增量索引）
# ======================================================================


class AppState:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.ruleset: Ruleset = Ruleset()
        self.evidence: dict[str, Evidence] = {}
        self._store: RuleStore | None = None
        self._embedding_cache: tuple[tuple[str, str | None, str], EmbeddingFunction, str] | None = (
            None
        )
        self._reranker_cache: tuple[tuple[str, str], FastEmbedReranker] | None = None
        self._agent_sessions: Any = None

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

    def embedding_provider(self) -> tuple[EmbeddingFunction, str]:
        """按配置复用模型实例，避免每次 API 查询重新加载 ONNX 权重。"""
        key = (
            self.settings.embedding_backend,
            self.settings.embedding_model,
            str(self.settings.model_cache_dir),
        )
        if self._embedding_cache is None or self._embedding_cache[0] != key:
            embeddings, version = make_embedding_provider(
                backend=self.settings.embedding_backend,
                model_name=self.settings.embedding_model,
                cache_dir=self.settings.model_cache_dir,
            )
            self._embedding_cache = (key, embeddings, version)
        return self._embedding_cache[1], self._embedding_cache[2]

    def reranker_provider(self) -> FastEmbedReranker:
        key = (self.settings.reranker_model, str(self.settings.model_cache_dir))
        if self._reranker_cache is None or self._reranker_cache[0] != key:
            self._reranker_cache = (
                key,
                FastEmbedReranker(
                    model_name=self.settings.reranker_model,
                    cache_dir=self.settings.model_cache_dir,
                ),
            )
        return self._reranker_cache[1]

    def agent_sessions(self) -> Any:
        if self._agent_sessions is None:
            from ..agent import SessionRuleLock

            self._agent_sessions = SessionRuleLock()
        return self._agent_sessions


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

    @app.get("/api/release-readiness", tags=["base"])
    def release_readiness(candidate_version: str = __version__) -> dict:
        """只读汇总固定评测报告；缺失正式数据时明确返回阻塞。"""
        return build_release_readiness(
            PROJECT_ROOT / "reports",
            candidate_version=candidate_version,
        ).model_dump(mode="json")

    @app.get("/api/costs/summary", tags=["base"])
    def cost_summary() -> dict:
        """只读汇总模型 token/成本；不返回提示词、回复或用户原始标识。"""
        settings = get_state().settings
        with CostLedger(settings.cost_db_path) as ledger:
            return ledger.report().model_dump(mode="json")

    @app.get("/api/eval/user-trial/template", tags=["evaluation"])
    def user_trial_dataset_template() -> dict:
        """返回显式不可通过正式门禁的空试用模板。"""
        return user_trial_template().model_dump(mode="json")

    @app.post("/api/eval/user-trial", tags=["evaluation"])
    def evaluate_user_trial_api(dataset: UserTrialDataset) -> dict:
        """评测已授权试用元数据；接口不接收姓名、学号、材料或自由文本。"""
        return evaluate_user_trials(dataset).model_dump(mode="json")

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

    @app.post("/api/search", tags=["retrieval"])
    def search(req: SearchRequest) -> dict:
        """查询活动混合索引；返回 RRF 排名及两路原始名次。"""
        st = get_state()
        settings = st.settings
        embeddings, model_version = st.embedding_provider()
        with HybridIndexManifestStore(
            settings.index_db_path,
            vector_dir=settings.vector_dir,
            embeddings=embeddings,
            embedding_model=model_version,
        ) as index:
            retriever: Retriever = HybridRetriever(
                index,
                academic_year=req.academic_year,
                college=req.college,
                doc_id=req.doc_id,
                query_rewriter=rewrite_retrieval_query,
            )
            if req.rerank:
                retriever = RerankingRetriever(
                    retriever,
                    st.reranker_provider(),
                    candidate_k=settings.rerank_candidate_k,
                    base_rank_weight=settings.rerank_base_weight,
                    rerank_rank_weight=settings.rerank_model_weight,
                )
            hits = retriever.search(req.query, top_k=req.top_k)
        return {
            "count": len(hits),
            "hits": [
                {
                    "id": hit.clause.id,
                    "text": hit.clause.text,
                    "source": hit.clause.source.model_dump(mode="json"),
                    "score": hit.score,
                    "rank": hit.rank,
                    "channel": hit.channel,
                    "component_ranks": hit.component_ranks,
                    "rerank_score": hit.rerank_score,
                }
                for hit in hits
            ],
        }

    @app.post("/api/citation-check", tags=["retrieval"])
    def citation_check(req: SearchRequest) -> dict:
        """核查真实索引出处是否足以支撑候选答复；文本命中不自动给分。"""
        st = get_state()
        settings = st.settings
        embeddings, model_version = st.embedding_provider()
        with HybridIndexManifestStore(
            settings.index_db_path,
            vector_dir=settings.vector_dir,
            embeddings=embeddings,
            embedding_model=model_version,
        ) as index:
            retriever = HybridRetriever(
                index,
                academic_year=req.academic_year,
                college=req.college,
                doc_id=req.doc_id,
                query_rewriter=rewrite_retrieval_query,
            )
            hits = retriever.search(req.query, top_k=req.top_k)
        return verify_text_citations(req.query, hits).model_dump(mode="json")

    # ---------------- 工具编排 ----------------

    @app.post("/api/agent", tags=["agent"])
    def agent(req: AgentRequest) -> dict:
        """LangChain 工具路由；模型不可用时自动回退到确定性查表与核算。"""
        from ..agent import OrchestrationRequest, ScoreProofOrchestrator
        from ..agent.orchestrator import make_deepseek_model

        st = get_state()
        settings = st.settings
        model = None
        if req.use_model and settings.llm_configured:
            assert settings.llm_api_key is not None
            try:
                model = make_deepseek_model(
                    model=settings.llm_model,
                    api_key=settings.llm_api_key,
                    base_url=settings.llm_base_url,
                )
            except Exception:
                model = None

        def clause_searcher(query: str, filters: dict[str, Any], top_k: int) -> list:
            embeddings, model_version = st.embedding_provider()
            with HybridIndexManifestStore(
                settings.index_db_path,
                vector_dir=settings.vector_dir,
                embeddings=embeddings,
                embedding_model=model_version,
            ) as index:
                return HybridRetriever(
                    index,
                    academic_year=filters.get("academic_year"),
                    college=filters.get("college"),
                    doc_id=filters.get("doc_id"),
                    query_rewriter=rewrite_retrieval_query,
                ).search(query, top_k=top_k)

        audit_store = RuleStore(settings.db_path)
        cost_ledger = CostLedger(
            settings.cost_db_path,
            subject_salt=settings.cost_id_salt,
        )
        try:
            request = OrchestrationRequest(
                query=req.query,
                **({"session_id": req.session_id} if req.session_id else {}),
                claims=[item.to_claim() for item in req.claims],
                academic_year=req.academic_year,
                college=req.college,
            )
            result = ScoreProofOrchestrator(
                st.ensure_rules(),
                model=model,
                model_version=settings.llm_model if model is not None else "rules-only",
                audit_sink=audit_store,
                clause_searcher=clause_searcher,
                sessions=st.agent_sessions(),
                cost_ledger=cost_ledger,
            ).run(request)
            return result.model_dump(mode="json")
        finally:
            cost_ledger.close()
            audit_store.close()

    # ---------------- 证据 / 多模态（阶段 6 最小入口） ----------------

    @app.post("/api/evidence/extract-certificate", tags=["evidence"])
    async def extract_certificate_api(
        file: UploadFile = File(...),
        preprocess_image: bool = True,
        confidence_threshold: float = 0.8,
        vlm_provider: str | None = None,
    ) -> dict:
        """上传真实图片并执行 RapidOCR + DeepSeek 文本抽取主链路。"""
        if not 0 <= confidence_threshold <= 1:
            raise HTTPException(status_code=422, detail="confidence_threshold 必须位于 [0,1]")
        suffix = Path(file.filename or "certificate.png").suffix.lower()
        if suffix not in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}:
            raise HTTPException(status_code=400, detail="不支持的图片格式")
        content = await file.read()
        if not content or len(content) > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="图片必须非空且不超过 20 MiB")
        settings = get_state().settings
        settings.ensure_dirs()
        target = settings.data_dir / "tmp" / f"certificate-{uuid.uuid4().hex}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        try:
            with CostLedger(settings.cost_db_path) as ledger:
                result = run_certificate_extraction(
                    target,
                    run_preprocess=preprocess_image,
                    confidence_threshold=confidence_threshold,
                    provider=vlm_provider,
                    cost_ledger=ledger,
                    batch_id=f"certificate-api:{uuid.uuid4().hex}",
                )
        except ScoreProofError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
        st = get_state()
        st.evidence[result.evidence.id] = result.evidence
        return result.model_dump(mode="json", by_alias=True)

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
        """兼容旧分组，并返回 SHA-256 + pHash 距离 + 字段事实的联合查重结果。"""
        st = get_state()
        by_phash: dict[str, list[str]] = {}
        by_fp: dict[str, list[str]] = {}
        for ev in st.evidence.values():
            if ev.phash:
                by_phash.setdefault(ev.phash, []).append(ev.id)
            fingerprint = fact_fingerprint(ev)
            if fingerprint is not None:
                by_fp.setdefault(fingerprint, []).append(ev.id)
        dupes = {
            "by_phash": {k: v for k, v in by_phash.items() if len(v) > 1},
            "by_fingerprint": {k: v for k, v in by_fp.items() if len(v) > 1},
        }
        decisions = [
            compare_evidence(left, right)
            for left, right in combinations(st.evidence.values(), 2)
        ]
        return {
            "total_evidence": len(st.evidence),
            "total_pairs": len(decisions),
            "flagged_pairs": [
                item.model_dump(mode="json") for item in decisions if item.flagged
            ],
            **dupes,
        }

    @app.post("/api/evidence/compare", tags=["evidence"])
    def compare_evidence_api(payload: EvidenceCompareRequest) -> dict:
        """按 ID 对两份已保存证据执行可解释联合查重。"""
        st = get_state()
        if payload.left_id == payload.right_id:
            raise HTTPException(status_code=422, detail="left_id 与 right_id 必须不同")
        left = st.evidence.get(payload.left_id)
        right = st.evidence.get(payload.right_id)
        if left is None or right is None:
            missing = [
                item
                for item, evidence in ((payload.left_id, left), (payload.right_id, right))
                if evidence is None
            ]
            raise HTTPException(status_code=404, detail={"missing_evidence_ids": missing})
        return compare_evidence(left, right, thresholds=payload.thresholds).model_dump(mode="json")

    @app.post("/api/evidence/check-claim", tags=["evidence"])
    def check_claim_evidence_api(payload: ClaimEvidenceCheckRequest) -> dict:
        """确定性核对申报与证据；缺字段明确进入人工复核。"""
        evidence = get_state().evidence.get(payload.evidence_id)
        if evidence is None:
            raise HTTPException(status_code=404, detail="evidence_id 不存在")
        return compare_claim_evidence(
            payload.claim,
            evidence,
            policy=payload.policy,
        ).model_dump(mode="json")

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
<p>确定性计算 + 双通道检索，每条分值可回溯原文。完整前端按 V3.0 延后至 P2。</p>
<ul>
 <li><a href="/docs">/docs</a> — OpenAPI 交互文档</li>
 <li><a href="/health">/health</a> — 健康检查与规则数</li>
 <li><code>POST /api/calc</code> — 提交申报条目 -> 返回可回溯账目</li>
 <li><code>POST /api/explain</code> — 单条申报的通道命中与原文引用</li>
 <li><code>POST /api/citation-check</code> — 核查真实索引引用并给出人工确认/拒答分支</li>
 <li><code>POST /api/refusal-check</code> — 未找到规则时是否正确拒答</li>
 <li><code>POST /api/agent</code> — 工具编排、结构化核算与数字/引用门禁</li>
 <li><code>GET/POST /api/eval/user-trial</code> — 无 PII 的试用模板与正式门禁评测</li>
</ul>
</body></html>"""


app = create_app()

__all__ = ["AppState", "app", "create_app", "get_state"]

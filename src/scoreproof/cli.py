"""命令行入口：``scoreproof <command>``。

设计目标：不用起服务也能完成 P1 全流程（解析 -> 入库 -> 核算 -> 回测）。
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .calc.engine import EngineConfig, compute_all
from .config import PROJECT_ROOT, get_settings
from .eval.backtest import (
    item_reference_template,
    load_ground_truth,
    load_item_expectations,
    run_backtest,
)
from .eval.certificate import evaluate_certificate_fields, load_jsonl
from .eval.citation import evaluate_citation_refusal, load_refusal_cases
from .eval.dedup import evaluate_dedup_pairs, load_dedup_dataset
from .eval.gateway import GatewayNegativeCase, evaluate_gateway_negatives
from .eval.readiness import build_release_readiness, run_quality_gates
from .eval.retrieval import (
    build_ablation_report,
    evaluate_retriever,
    load_retrieval_cases,
)
from .evidence.certificate import extract_certificate
from .evidence.consistency import ConsistencyPolicy, compare_claim_evidence
from .evidence.dedup import DuplicateThresholds, compare_evidence
from .indexing import (
    DocumentChunk,
    EmbeddingFunction,
    FastEmbedEmbeddings,
    HybridIndexManifestStore,
    IndexManifestStore,
    make_embedding_provider,
)
from .ingest.excel_loader import load_claims, load_rules
from .ingest.image_loader import check_quality, phash, preprocess, run_ocr
from .ingest.pdf_loader import load_pdf
from .observability import CostLedger
from .retrieval.hybrid import BM25Retriever, HybridRetriever
from .retrieval.query import rewrite_retrieval_query
from .retrieval.rerank import FastEmbedReranker, RerankingRetriever
from .retrieval.router import Retriever, Router, clauses_from_pdf_pages
from .rules.extractor import LLMExtractor
from .rules.gateway import GatewayContext
from .rules.store import RuleStore
from .schema import Claim, Evidence, Ruleset, SourceRef

app = typer.Typer(
    help="综测加分核算系统：Excel/PDF 解析 -> 规则库 -> 确定性核算 -> 回测",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _embedding_provider(
    *, backend: str, model_name: str | None, cache_dir: Path
) -> tuple[EmbeddingFunction, str]:
    try:
        return make_embedding_provider(backend=backend, model_name=model_name, cache_dir=cache_dir)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _pdf_index_chunks(
    *,
    pdf: Path,
    pages: Sequence[Any],
    chunk_mode: str,
    academic_year: str | None,
    college: str | None,
) -> list[DocumentChunk]:
    if chunk_mode not in {"page", "block"}:
        raise typer.BadParameter("chunk-mode 只支持 page 或 block")
    common = {"academic_year": academic_year, "college": college}
    if chunk_mode == "page":
        return [
            DocumentChunk(
                logical_key=f"page:{page.page}",
                text=page.text,
                kind="page",
                source=SourceRef(doc=pdf.name, page=page.page, text=page.text[:200]),
                metadata={
                    **common,
                    "block_count": len(page.blocks),
                    "is_probably_scanned": page.is_probably_scanned,
                },
            )
            for page in pages
        ]
    chunks: list[DocumentChunk] = []
    for page in pages:
        raw_texts = [str(block.get("text", "")).strip() for block in page.blocks]
        index_texts = _table_context_texts(raw_texts)
        for block_index, block in enumerate(page.blocks):
            source_text = raw_texts[block_index]
            if not source_text:
                continue
            text = index_texts[block_index]
            bbox = block.get("bbox")
            chunks.append(
                DocumentChunk(
                    logical_key=f"page:{page.page}:block:{block_index}",
                    text=text,
                    kind="paragraph",
                    source=SourceRef(
                        doc=pdf.name,
                        page=page.page,
                        text=source_text[:200],
                        bbox=tuple(bbox) if bbox else None,
                    ),
                    metadata={
                        **common,
                        "block_index": block_index,
                        **({"original_text": source_text} if text != source_text else {}),
                    },
                )
            )
    return chunks


def _table_context_texts(texts: Sequence[str]) -> list[str]:
    """把 PDF 表格中独立的级别单元格前缀到后续获奖行，保留原 block ID。"""
    level_headers = {"国际级", "国家级", "省部级", "校级"}
    award_markers = ("特等奖", "一等奖", "二等奖", "三等奖", "单项奖")
    level: str | None = None
    enriched: list[str] = []
    for text in texts:
        compact = "".join(text.split())
        if compact in level_headers:
            level = compact
            enriched.append(text)
        elif level and text and any(marker in compact for marker in award_markers):
            enriched.append(f"{level}\n{text}")
        else:
            enriched.append(text)
    return enriched


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"scoreproof {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(
        None, "--version", "-V", callback=_version_callback, is_eager=True, help="显示版本"
    ),
) -> None:
    """全局选项。"""


# ======================================================================
# 规则库
# ======================================================================


@app.command("import-rules")
def import_rules(
    excel: Path = typer.Argument(..., exists=True, help="规则表（等级×类别 加分矩阵）"),
    academic_year: str = typer.Option(..., "--year", "-y", help="学年，如 2025-2026"),
    college: str | None = typer.Option(None, "--college", help="学院；不填为校级通用"),
    sheet: str = typer.Option("0", "--sheet", help="工作表名或索引"),
    db: Path | None = typer.Option(None, "--db", help="SQLite 路径，默认取配置"),
    export: Path | None = typer.Option(None, "--export", help="同时导出 JSON 供人工校对"),
) -> None:
    """把规则表导入 SQLite（建议随后人工校对一遍）。"""
    settings = get_settings()
    sheet_arg: str | int = int(sheet) if sheet.isdigit() else sheet
    rules = load_rules(excel, sheet=sheet_arg, academic_year=academic_year, college=college)
    if not rules:
        console.print("[yellow]没有解析出任何规则：检查表头是否包含 类别/等级/分值[/yellow]")
        raise typer.Exit(code=1)
    store = RuleStore(db or settings.db_path)
    n = store.upsert_rules(rules)
    console.print(f"[green]已导入 {n} 条规则[/green]（库内共 {store.count()} 条）-> {store.path}")
    if export:
        Ruleset(rules=store.list_rules()).to_json(export)
        console.print(f"已导出 JSON：{export}")
    store.close()


@app.command("list-rules")
def show_rules(
    academic_year: str | None = typer.Option(None, "--year", "-y"),
    college: str | None = typer.Option(None, "--college"),
    category: str | None = typer.Option(None, "--category"),
    db: Path | None = typer.Option(None, "--db"),
) -> None:
    """查看规则库。"""
    settings = get_settings()
    store = RuleStore(db or settings.db_path)
    rules = store.list_rules(academic_year=academic_year, college=college, category=category)
    store.close()
    table = Table(title=f"规则库（{len(rules)} 条）")
    for col in ("学年", "学院", "类别", "等级", "分值", "互斥组", "封顶", "团队系数", "出处"):
        table.add_column(col)
    for r in rules:
        table.add_row(
            r.academic_year,
            r.college or "全校",
            r.category,
            r.level,
            f"{r.score:g}",
            r.constraints.dedup_group,
            "不设" if r.constraints.cap is None else f"{r.constraints.cap:g}",
            f"{r.constraints.team_factor:g}",
            r.source.short(),
        )
    console.print(table)


@app.command("ask-score")
def ask_score(
    query: str = typer.Argument(..., help="自然语言核算问题"),
    student_id: str = typer.Option(..., "--student-id", help="学号"),
    academic_year: str = typer.Option(..., "--year", "-y", help="学年"),
    category: str = typer.Option(..., "--category", help="申报类别"),
    level: str = typer.Option(..., "--level", help="规范化等级/名次"),
    college: str | None = typer.Option(None, "--college", help="学院"),
    team: bool = typer.Option(False, "--team/--individual", help="团队或个人项目"),
    use_model: bool = typer.Option(True, "--model/--no-model", help="启用模型工具路由"),
    session_id: str | None = typer.Option(
        None, "--session-id", help="审计会话 ID；规则版本锁在同一服务进程内复用"
    ),
    db: Path | None = typer.Option(None, "--db", help="规则库与工具审计 SQLite"),
) -> None:
    """通过工具编排查规则并核算；模型失败时自动降级，绝不让模型心算。"""
    from .agent import OrchestrationRequest, ScoreProofOrchestrator
    from .agent.orchestrator import make_deepseek_model

    settings = get_settings()
    database = db or settings.db_path
    store = RuleStore(database)
    cost_ledger = CostLedger(database, subject_salt=settings.cost_id_salt)
    try:
        ruleset = store.load_ruleset(academic_year=academic_year, college=college)
        if not ruleset.rules:
            console.print("[yellow]规则库没有对应学年/学院的规则，无法核算。[/yellow]")
            raise typer.Exit(code=2)
        model = None
        if use_model and settings.llm_configured:
            assert settings.llm_api_key is not None
            try:
                model = make_deepseek_model(
                    model=settings.llm_model,
                    api_key=settings.llm_api_key,
                    base_url=settings.llm_base_url,
                )
            except Exception:
                model = None
        claim = Claim(
            student_id=student_id,
            academic_year=academic_year,
            college=college,
            category=category,
            raw_text=level,
            level=level,
            team=team,
        )
        request_data: dict[str, Any] = {
            "query": query,
            "claims": [claim],
            "academic_year": academic_year,
            "college": college,
        }
        if session_id:
            request_data["session_id"] = session_id
        result = ScoreProofOrchestrator(
            ruleset,
            model=model,
            model_version=settings.llm_model if model is not None else "rules-only",
            audit_sink=store,
            cost_ledger=cost_ledger,
        ).run(OrchestrationRequest(**request_data))
        console.print_json(result.model_dump_json())
        if result.outcome != "answer":
            raise typer.Exit(code=2)
    finally:
        cost_ledger.close()
        store.close()


# ======================================================================
# 解析
# ======================================================================


@app.command("parse-pdf")
def parse_pdf(
    pdf: Path = typer.Argument(..., exists=True, help="细则 PDF"),
    with_tables: bool = typer.Option(True, "--tables/--no-tables", help="是否抽表格"),
    out: Path | None = typer.Option(None, "--out", help="导出 JSON"),
) -> None:
    """抽取 PDF 文本与表格，检查扫描件风险。"""
    pages = load_pdf(pdf, with_tables=with_tables)
    scanned = [p.page for p in pages if p.is_probably_scanned]
    console.print(f"共 {len(pages)} 页；文本量 {sum(p.char_count for p in pages)} 字符")
    if scanned:
        console.print(
            f"[yellow]疑似扫描件页（需 OCR + 人工校对）：{scanned}[/yellow]"
        )
    if out:
        payload = [
            {"page": p.page, "text": p.text, "tables": p.tables, "meta": p.meta} for p in pages
        ]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"已写出：{out}")
    else:
        first = next((p for p in pages if p.char_count), None)
        if first:
            console.print(first.text[:800])


@app.command("sync-pdf-manifest")
def sync_pdf_manifest(
    pdf: Path = typer.Argument(..., exists=True, dir_okay=False, help="待同步的规则 PDF"),
    doc_id: str | None = typer.Option(None, "--doc-id", help="稳定文档 ID；默认使用文件名"),
    embedding_model: str = typer.Option(
        "unembedded-v1",
        "--embedding-model",
        help="索引模型版本；阶段 4.2 接入向量前使用 unembedded-v1",
    ),
    db: Path | None = typer.Option(None, "--db", help="manifest 所在 SQLite"),
) -> None:
    """按页生成逻辑块，计算双级 Hash，并原子发布增量 manifest。"""
    settings = get_settings()
    pages = load_pdf(pdf)
    scanned_pages = [page.page for page in pages if not page.text.strip()]
    if scanned_pages:
        console.print(
            f"[yellow]以下页面没有可索引文本，必须先 OCR，未发布 manifest：{scanned_pages}[/yellow]"
        )
        raise typer.Exit(code=2)
    chunks = [
        DocumentChunk(
            logical_key=f"page:{page.page}",
            text=page.text,
            kind="page",
            source=SourceRef(doc=pdf.name, page=page.page, text=page.text[:200]),
            metadata={
                "block_count": len(page.blocks),
                "is_probably_scanned": page.is_probably_scanned,
            },
        )
        for page in pages
    ]
    if not chunks:
        console.print("[yellow]PDF 没有可索引文本；疑似扫描件需先 OCR[/yellow]")
        raise typer.Exit(code=2)
    with IndexManifestStore(db or settings.db_path) as store:
        result = store.sync_document(
            doc_id=doc_id or pdf.name,
            source_path=pdf,
            document_bytes=pdf.read_bytes(),
            chunks=chunks,
            embedding_model=embedding_model,
        )
    payload = result.model_dump(mode="json")
    payload["rebuild_keys"] = result.diff.rebuild_keys
    console.print_json(json.dumps(payload, ensure_ascii=False))


@app.command("sync-pdf-hybrid")
def sync_pdf_hybrid(
    pdf: Path = typer.Argument(..., exists=True, dir_okay=False, help="待同步的规则 PDF"),
    doc_id: str | None = typer.Option(None, "--doc-id", help="稳定文档 ID；默认使用文件名"),
    academic_year: str | None = typer.Option(None, "--year", "-y"),
    college: str | None = typer.Option(None, "--college"),
    chunk_mode: str = typer.Option("page", "--chunk-mode", help="page 或 block"),
    embedding_backend: str = typer.Option("hash", "--embedding-backend", help="hash 或 fastembed"),
    embedding_model: str | None = typer.Option(None, "--embedding-model"),
    model_cache: Path | None = typer.Option(None, "--model-cache"),
    db: Path | None = typer.Option(None, "--db", help="检索 Manifest SQLite"),
    vector_dir: Path | None = typer.Option(None, "--vector-dir", help="Chroma 持久化目录"),
) -> None:
    """把 PDF 作为同一 Manifest 批次发布到 BM25 与 Chroma。"""
    settings = get_settings()
    pages = load_pdf(pdf)
    scanned_pages = [page.page for page in pages if not page.text.strip()]
    if scanned_pages:
        console.print(
            f"[yellow]以下页面没有可索引文本，必须先 OCR，未发布索引：{scanned_pages}[/yellow]"
        )
        raise typer.Exit(code=2)
    chunks = _pdf_index_chunks(
        pdf=pdf,
        pages=pages,
        chunk_mode=chunk_mode,
        academic_year=academic_year,
        college=college,
    )
    if not chunks:
        console.print("[yellow]PDF 没有可索引文本；疑似扫描件需先 OCR[/yellow]")
        raise typer.Exit(code=2)
    embeddings, model_version = _embedding_provider(
        backend=embedding_backend,
        model_name=embedding_model,
        cache_dir=model_cache or settings.model_cache_dir,
    )
    with HybridIndexManifestStore(
        db or settings.index_db_path,
        vector_dir=vector_dir or settings.vector_dir,
        embeddings=embeddings,
        embedding_model=model_version,
    ) as store:
        result = store.sync_document(
            doc_id=doc_id or pdf.name,
            source_path=pdf,
            document_bytes=pdf.read_bytes(),
            chunks=chunks,
            embedding_model=model_version,
        )
        batch = store.search_batch(result.manifest_id) if result.manifest_id else None
    payload = result.model_dump(mode="json")
    payload["rebuild_keys"] = result.diff.rebuild_keys
    payload["search_batch"] = batch.model_dump(mode="json") if batch else None
    console.print_json(json.dumps(payload, ensure_ascii=False, default=str))


@app.command("search-index")
def search_index(
    query: str = typer.Argument(..., help="待检索的问题或条款关键词"),
    top_k: int = typer.Option(5, "--top-k", min=1, max=20),
    academic_year: str | None = typer.Option(None, "--year", "-y"),
    college: str | None = typer.Option(None, "--college"),
    doc_id: str | None = typer.Option(None, "--doc-id"),
    embedding_backend: str = typer.Option("hash", "--embedding-backend"),
    embedding_model: str | None = typer.Option(None, "--embedding-model"),
    model_cache: Path | None = typer.Option(None, "--model-cache"),
    rerank: bool = typer.Option(False, "--rerank", help="用 CrossEncoder 精排并与 RRF 排名融合"),
    reranker_model: str = typer.Option("BAAI/bge-reranker-base", "--reranker-model"),
    candidate_k: int = typer.Option(20, "--candidate-k", min=1),
    base_rank_weight: float = typer.Option(4.0, "--base-rank-weight", min=0.001),
    rerank_rank_weight: float = typer.Option(1.0, "--rerank-rank-weight", min=0.001),
    db: Path | None = typer.Option(None, "--db"),
    vector_dir: Path | None = typer.Option(None, "--vector-dir"),
) -> None:
    """查询活动 Manifest：BM25 与向量并行召回后用 RRF 融合。"""
    settings = get_settings()
    embeddings, model_version = _embedding_provider(
        backend=embedding_backend,
        model_name=embedding_model,
        cache_dir=model_cache or settings.model_cache_dir,
    )
    with HybridIndexManifestStore(
        db or settings.index_db_path,
        vector_dir=vector_dir or settings.vector_dir,
        embeddings=embeddings,
        embedding_model=model_version,
    ) as store:
        retriever: Retriever = HybridRetriever(
            store,
            academic_year=academic_year,
            college=college,
            doc_id=doc_id,
            query_rewriter=rewrite_retrieval_query,
        )
        if rerank:
            retriever = RerankingRetriever(
                retriever,
                FastEmbedReranker(
                    model_name=reranker_model,
                    cache_dir=model_cache or settings.model_cache_dir,
                ),
                candidate_k=candidate_k,
                base_rank_weight=base_rank_weight,
                rerank_rank_weight=rerank_rank_weight,
            )
        hits = retriever.search(query, top_k=top_k)
    payload = [
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
    ]
    console.print_json(json.dumps({"count": len(payload), "hits": payload}, ensure_ascii=False))


@app.command("eval-retrieval")
def eval_retrieval(
    dataset: Path = typer.Argument(..., exists=True, dir_okay=False),
    db: Path | None = typer.Option(None, "--db"),
    vector_dir: Path | None = typer.Option(None, "--vector-dir"),
    embedding_model: str = typer.Option("BAAI/bge-small-zh-v1.5", "--embedding-model"),
    reranker_model: str = typer.Option("BAAI/bge-reranker-base", "--reranker-model"),
    model_cache: Path | None = typer.Option(None, "--model-cache"),
    candidate_k: int = typer.Option(20, "--candidate-k", min=1),
    base_rank_weight: float = typer.Option(4.0, "--base-rank-weight", min=0.001),
    rerank_rank_weight: float = typer.Option(1.0, "--rerank-rank-weight", min=0.001),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    """在冻结集上复跑 A=BM25、B=向量+RRF、C=Rerank 三档消融。"""
    settings = get_settings()
    version, kind, cases = load_retrieval_cases(dataset)
    embeddings = FastEmbedEmbeddings(
        model_name=embedding_model,
        cache_dir=model_cache or settings.model_cache_dir,
    )
    with HybridIndexManifestStore(
        db or settings.index_db_path,
        vector_dir=vector_dir or settings.vector_dir,
        embeddings=embeddings,
        embedding_model=embeddings.model_version,
    ) as store:
        hybrid = HybridRetriever(store, query_rewriter=rewrite_retrieval_query)
        baseline = BM25Retriever(hybrid)
        reranker = FastEmbedReranker(
            model_name=reranker_model,
            cache_dir=model_cache or settings.model_cache_dir,
        )
        reranked = RerankingRetriever(
            hybrid,
            reranker,
            candidate_k=candidate_k,
            base_rank_weight=base_rank_weight,
            rerank_rank_weight=rerank_rank_weight,
        )
        reports = [
            evaluate_retriever(baseline, cases, variant="A"),
            evaluate_retriever(
                hybrid,
                cases,
                variant="B",
                embedding_count=lambda: embeddings.query_count,
            ),
            evaluate_retriever(
                reranked,
                cases,
                variant="C",
                embedding_count=lambda: embeddings.query_count,
                rerank_pair_count=lambda: reranked.last_pair_count,
            ),
        ]
        active_manifest_ids = [batch.manifest_id for batch in store.active_batches()]
    report = build_ablation_report(
        dataset_version=version,
        dataset_kind=kind,
        variants=reports,
        notes=[
            f"Embedding={embeddings.model_version}",
            f"Reranker={reranker.model_version}",
            (
                f"Rerank rank fusion: base={base_rank_weight:g}, "
                f"cross_encoder={rerank_rank_weight:g}, candidate_k={candidate_k}"
            ),
            "Query rewrite=scoreproof-default-v1 (BM25 only)",
            f"Active manifests={','.join(active_manifest_ids)}",
            "冻结集为基于真实公开细则人工整理的查询，不是生产用户日志。",
        ],
    )
    payload = report.model_dump_json(indent=2)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        console.print(f"已写出：{out}")
    console.print_json(payload)


@app.command("eval-citation-refusal")
def eval_citation_refusal(
    positive_dataset: Path = typer.Argument(..., exists=True, dir_okay=False),
    negative_dataset: Path = typer.Argument(..., exists=True, dir_okay=False),
    db: Path | None = typer.Option(None, "--db"),
    vector_dir: Path | None = typer.Option(None, "--vector-dir"),
    embedding_model: str = typer.Option("BAAI/bge-small-zh-v1.5", "--embedding-model"),
    model_cache: Path | None = typer.Option(None, "--model-cache"),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    """成对评测引用定位、应拒答正确率与可回答问题误拒率。"""
    settings = get_settings()
    positive_version, positive_kind, positive_cases = load_retrieval_cases(positive_dataset)
    negative_version, negative_kind, negative_cases = load_refusal_cases(negative_dataset)
    embeddings = FastEmbedEmbeddings(
        model_name=embedding_model,
        cache_dir=model_cache or settings.model_cache_dir,
    )
    with HybridIndexManifestStore(
        db or settings.index_db_path,
        vector_dir=vector_dir or settings.vector_dir,
        embeddings=embeddings,
        embedding_model=embeddings.model_version,
    ) as store:
        retriever = HybridRetriever(store, query_rewriter=rewrite_retrieval_query)
        active_manifest_ids = [batch.manifest_id for batch in store.active_batches()]
        report = evaluate_citation_refusal(
            retriever,
            positive_cases,
            negative_cases,
            positive_dataset_version=positive_version,
            positive_dataset_kind=positive_kind,
            negative_dataset_version=negative_version,
            negative_dataset_kind=negative_kind,
            embedding_count=lambda: embeddings.query_count,
            notes=[
                f"Embedding={embeddings.model_version}",
                f"Active manifests={','.join(active_manifest_ids)}",
                "正样本来自真实公开细则人工冻结问法；负样本为人工构造的域外边界，不是生产日志。",
                "文本引用命中只进入人工确认，不自动计分。",
            ],
        )
    payload = report.model_dump_json(indent=2)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        console.print(f"已写出：{out}")
    console.print_json(payload)
    if not report.passed:
        raise typer.Exit(code=2)


@app.command("rollback-index-manifest")
def rollback_index_manifest(
    doc_id: str = typer.Argument(..., help="稳定文档 ID"),
    manifest_id: str | None = typer.Option(None, "--manifest-id", help="默认回滚到上一版本"),
    db: Path | None = typer.Option(None, "--db"),
) -> None:
    """把活动索引原子切回指定或上一份完整 manifest。"""
    settings = get_settings()
    with IndexManifestStore(db or settings.db_path) as store:
        result = store.rollback(doc_id, manifest_id=manifest_id)
    console.print_json(result.model_dump_json())


@app.command("delete-index-document")
def delete_index_document(
    doc_id: str = typer.Argument(..., help="稳定文档 ID"),
    db: Path | None = typer.Option(None, "--db"),
) -> None:
    """从活动索引删除文档，同时保留可审计历史快照。"""
    settings = get_settings()
    with IndexManifestStore(db or settings.db_path) as store:
        result = store.delete_document(doc_id)
    console.print_json(result.model_dump_json())


@app.command("parse-claims")
def parse_claims(
    excel: Path = typer.Argument(..., exists=True, help="综测表 Excel"),
    sheet: str = typer.Option("0", "--sheet"),
    academic_year: str | None = typer.Option(None, "--year", "-y"),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    """解析综测表 -> 申报条目（含合并单元格 fill-down）。"""
    sheet_arg: str | int = int(sheet) if sheet.isdigit() else sheet
    claims = load_claims(excel, sheet=sheet_arg, academic_year=academic_year)
    console.print(f"解析出 {len(claims)} 条申报，涉及 {len({c.student_id for c in claims})} 名学生")
    preview = Table(title="前 10 条")
    for col in ("学号", "姓名", "类别", "原文", "归一化等级"):
        preview.add_column(col)
    for c in claims[:10]:
        preview.add_row(c.student_id, c.student_name or "", c.category, c.raw_text,
                        c.level or "[red]未识别[/red]")
    console.print(preview)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps([c.model_dump(mode="json") for c in claims], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        console.print(f"已写出：{out}")


@app.command("parse-image")
def parse_image(
    image: Path = typer.Argument(..., exists=True, dir_okay=False, help="奖状/证书图片"),
    run_preprocess: bool = typer.Option(False, "--preprocess/--raw", help="OCR 前是否预处理"),
    with_ocr: bool = typer.Option(True, "--ocr/--no-ocr", help="是否运行 RapidOCR"),
    processed_dir: Path | None = typer.Option(None, "--processed-dir", help="预处理图片输出目录"),
    out: Path | None = typer.Option(None, "--out", help="导出 JSON"),
) -> None:
    """检查图片质量、计算 pHash，并可执行本地 OCR。"""
    quality = check_quality(image)
    target = preprocess(image, out_dir=processed_dir) if run_preprocess else image
    ocr_result = run_ocr(target) if with_ocr else None
    payload = {
        "source": str(image),
        "processed": str(target) if run_preprocess else None,
        "quality": {
            "width": quality.width,
            "height": quality.height,
            "sharpness": quality.sharpness,
            "is_blurry": quality.is_blurry,
            "needs_retake": quality.needs_retake,
            "rotation_applied": quality.rotation_applied,
            "notes": quality.notes,
        },
        "phash": phash(image),
        "ocr": None
        if ocr_result is None
        else {
            "engine": ocr_result.engine,
            "elapsed_seconds": ocr_result.elapsed_seconds,
            "mean_confidence": ocr_result.mean_confidence,
            "text": ocr_result.text,
            "lines": [
                {"text": line.text, "bbox": line.bbox, "confidence": line.confidence}
                for line in ocr_result.lines
            ],
        },
    }
    console.print_json(json.dumps(payload, ensure_ascii=False))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"已写出：{out}")


@app.command("extract-certificate")
def extract_certificate_command(
    image: Path = typer.Argument(..., exists=True, dir_okay=False, help="奖状/证书图片"),
    run_preprocess: bool = typer.Option(True, "--preprocess/--raw", help="是否预处理后再 OCR"),
    processed_dir: Path | None = typer.Option(None, "--processed-dir", help="预处理图片输出目录"),
    confidence_threshold: float = typer.Option(
        0.8, "--confidence-threshold", min=0.0, max=1.0
    ),
    vlm_provider: str | None = typer.Option(
        None, "--vlm-provider", help="仅覆盖触发判断：qwen-vl-plus 或 glm-4v"
    ),
    out: Path | None = typer.Option(None, "--out", help="导出完整 JSON（含 Evidence）"),
    cost_db: Path | None = typer.Option(None, "--cost-db", help="模型 token/成本账本 SQLite"),
) -> None:
    """真实图片 -> RapidOCR -> DeepSeek 文本结构化 -> 校验/置信度/复核状态。"""
    settings = get_settings()
    with CostLedger(cost_db or settings.cost_db_path) as ledger:
        result = extract_certificate(
            image,
            run_preprocess=run_preprocess,
            processed_dir=processed_dir,
            confidence_threshold=confidence_threshold,
            provider=vlm_provider,
            cost_ledger=ledger,
            batch_id=f"certificate:{_file_sha256(image)[:16]}",
        )
    encoded = result.model_dump_json(indent=2, by_alias=True)
    console.print_json(encoded)
    if result.extraction.vlm.requested and not result.extraction.vlm.called:
        console.print("[yellow]VLM 未调用；低置信字段已明确转入人工复核。[/yellow]")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(encoded, encoding="utf-8")
        console.print(f"已写出：{out}")


@app.command("eval-certificate-fields")
def eval_certificate_fields_command(
    labels: Path = typer.Argument(..., exists=True, dir_okay=False, help="字段标签 JSONL"),
    predictions: Path | None = typer.Option(
        None,
        "--predictions",
        exists=True,
        dir_okay=False,
        help="已有预测 JSONL；省略时从标签中的 image_path 实际抽取",
    ),
    predictions_out: Path | None = typer.Option(
        None, "--predictions-out", help="保存本次实际抽取的逐图预测 JSONL"
    ),
    run_preprocess: bool = typer.Option(False, "--preprocess/--raw", help="实际抽取时是否预处理"),
    dataset_version: str = typer.Option("certificate-fields-v1", "--dataset-version"),
    out: Path | None = typer.Option(None, "--out", help="评测报告 JSON"),
    cost_db: Path | None = typer.Option(None, "--cost-db", help="实际抽取时的模型成本账本"),
) -> None:
    """评测字段 micro-F1/逐字段 F1/整证正确率/VLM 触发率与样本量。"""
    label_rows = load_jsonl(labels)
    if predictions is not None:
        prediction_rows = load_jsonl(predictions)
    else:
        prediction_rows = []
        settings = get_settings()
        with CostLedger(cost_db or settings.cost_db_path) as ledger:
            for label in label_rows:
                image_path = label.get("image_path")
                if not isinstance(image_path, str) or not image_path.strip():
                    raise typer.BadParameter("省略 --predictions 时，每条标签必须包含 image_path")
                image = Path(image_path)
                result = extract_certificate(
                    image,
                    run_preprocess=run_preprocess,
                    cost_ledger=ledger,
                    batch_id=f"certificate-eval:{_file_sha256(image)[:16]}",
                )
                payload = result.model_dump(mode="json", by_alias=True)
                payload["evidence_id"] = str(label.get("evidence_id") or "")
                prediction_rows.append(payload)
        if predictions_out:
            predictions_out.parent.mkdir(parents=True, exist_ok=True)
            predictions_out.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in prediction_rows) + "\n",
                encoding="utf-8",
            )
    report = evaluate_certificate_fields(
        label_rows,
        prediction_rows,
        dataset_version=dataset_version,
    )
    encoded = report.model_dump_json(indent=2)
    console.print_json(encoded)
    if report.smoke_test_only:
        console.print("[yellow]仅烟雾测试：不得作为 n≥30 的正式字段 F1 验收。[/yellow]")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(encoded, encoding="utf-8")
        console.print(f"已写出：{out}")


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"JSON 顶层必须是对象：{path}")
    return payload


def _load_evidence_source(path: Path, fields_path: Path | None = None) -> Evidence:
    if path.suffix.lower() == ".json":
        payload = _json_object(path)
        nested = payload.get("evidence", payload)
        if not isinstance(nested, dict):
            raise typer.BadParameter(f"evidence 必须是对象：{path}")
        return Evidence.model_validate(nested)
    fields = _json_object(fields_path) if fields_path is not None else {}
    if "fields" in fields and isinstance(fields["fields"], dict):
        fields = fields["fields"]
    return Evidence(type="image", path=str(path), fields=fields, phash=phash(path))


@app.command("compare-evidence")
def compare_evidence_command(
    left: Path = typer.Argument(..., exists=True, dir_okay=False, help="左侧图片或 Evidence JSON"),
    right: Path = typer.Argument(..., exists=True, dir_okay=False, help="右侧图片或 Evidence JSON"),
    left_fields: Path | None = typer.Option(
        None, "--left-fields", exists=True, dir_okay=False, help="左图结构化字段 JSON"
    ),
    right_fields: Path | None = typer.Option(
        None, "--right-fields", exists=True, dir_okay=False, help="右图结构化字段 JSON"
    ),
    phash_definite_max: int = typer.Option(2, "--phash-definite-max", min=0, max=64),
    phash_suspected_max: int = typer.Option(10, "--phash-suspected-max", min=0, max=64),
    out: Path | None = typer.Option(None, "--out", help="导出查重决策 JSON"),
) -> None:
    """真实文件/Evidence JSON -> SHA-256 + pHash + 字段事实联合查重。"""
    limits = DuplicateThresholds(
        phash_definite_max=phash_definite_max,
        phash_suspected_max=phash_suspected_max,
    )
    decision = compare_evidence(
        _load_evidence_source(left, left_fields),
        _load_evidence_source(right, right_fields),
        thresholds=limits,
    )
    encoded = decision.model_dump_json(indent=2)
    console.print_json(encoded)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(encoded, encoding="utf-8")
        console.print(f"已写出：{out}")


@app.command("check-evidence-consistency")
def check_evidence_consistency_command(
    claim_json: Path = typer.Argument(..., exists=True, dir_okay=False, help="Claim JSON"),
    evidence_json: Path = typer.Argument(..., exists=True, dir_okay=False, help="Evidence JSON"),
    policy_json: Path | None = typer.Option(
        None, "--policy", exists=True, dir_okay=False, help="别名/目录/颁发单位策略 JSON"
    ),
    out: Path | None = typer.Option(None, "--out", help="导出逐字段一致性报告"),
) -> None:
    """确定性比对申报与证据；信息不足不会当作一致。"""
    claim_payload = _json_object(claim_json)
    evidence_payload = _json_object(evidence_json)
    claim = Claim.model_validate(claim_payload.get("claim", claim_payload))
    evidence = Evidence.model_validate(evidence_payload.get("evidence", evidence_payload))
    policy = ConsistencyPolicy.model_validate(_json_object(policy_json)) if policy_json else None
    report = compare_claim_evidence(claim, evidence, policy=policy)
    encoded = report.model_dump_json(indent=2)
    console.print_json(encoded)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(encoded, encoding="utf-8")
        console.print(f"已写出：{out}")


@app.command("eval-evidence-dedup")
def eval_evidence_dedup_command(
    dataset: Path = typer.Argument(..., exists=True, dir_okay=False, help="成对查重评测 JSON"),
    phash_definite_max: int = typer.Option(2, "--phash-definite-max", min=0, max=64),
    phash_suspected_max: int = typer.Option(10, "--phash-suspected-max", min=0, max=64),
    out: Path | None = typer.Option(None, "--out", help="导出评测报告 JSON"),
) -> None:
    """输出 Recall/Precision/F1、混淆矩阵和 n≥50 正式门禁。"""
    version, independent, cases = load_dedup_dataset(dataset)
    report = evaluate_dedup_pairs(
        cases,
        dataset_version=version,
        independent_real_pairs=independent,
        thresholds=DuplicateThresholds(
            phash_definite_max=phash_definite_max,
            phash_suspected_max=phash_suspected_max,
        ),
    )
    encoded = report.model_dump_json(indent=2)
    console.print_json(encoded)
    if report.smoke_test_only:
        console.print("[yellow]仅烟雾测试：不得作为 n≥50 对的正式查重验收。[/yellow]")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(encoded, encoding="utf-8")
        console.print(f"已写出：{out}")


@app.command("quality-gates")
def quality_gates_command(
    out: Path = typer.Option(
        PROJECT_ROOT / "reports" / "quality-gates-v1.json",
        "--out",
        help="导出真实质量检查报告",
    ),
) -> None:
    """固定执行 pytest、Ruff、mypy、uv lock 与 git diff 检查。"""
    report = run_quality_gates(PROJECT_ROOT)
    encoded = report.model_dump_json(indent=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(encoded, encoding="utf-8")
    console.print_json(encoded)
    console.print(f"已写出：{out}")
    if not report.all_checks_passed:
        raise typer.Exit(code=2)
    if not report.source_tree_clean:
        console.print("[yellow]检查通过，但工作树不干净；候选版本尚未冻结。[/yellow]")


@app.command("release-readiness")
def release_readiness_command(
    candidate_version: str = typer.Option(
        ..., "--candidate-version", help="待冻结的 Git 提交或版本标识"
    ),
    report_dir: Path = typer.Option(PROJECT_ROOT / "reports", "--report-dir"),
    out: Path = typer.Option(
        PROJECT_ROOT / "reports" / "release-readiness-v1.json",
        "--out",
    ),
    enforce: bool = typer.Option(True, "--enforce/--no-enforce", help="阻塞时以退出码 2 结束"),
) -> None:
    """汇总固定评测产物，烟雾结果不能使正式 RC 门禁通过。"""
    report = build_release_readiness(report_dir, candidate_version=candidate_version)
    encoded = report.model_dump_json(indent=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(encoded, encoding="utf-8")
    console.print_json(encoded)
    console.print(f"已写出：{out}")
    if enforce and not report.ready:
        console.print(f"[red]RC 阻塞：{', '.join(report.blocking_gate_ids)}[/red]")
        raise typer.Exit(code=2)


@app.command("cost-report")
def cost_report_command(
    db: Path | None = typer.Option(None, "--db", help="cost_events 所在 SQLite"),
    out: Path | None = typer.Option(None, "--out", help="保存成本汇总 JSON"),
) -> None:
    """汇总真实模型 token、缓存、模型分级和显式价格成本。"""
    settings = get_settings()
    with CostLedger(db or settings.cost_db_path) as ledger:
        report = ledger.report()
    encoded = report.model_dump_json(indent=2)
    console.print_json(encoded)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(encoded, encoding="utf-8")
        console.print(f"已写出：{out}")


# ======================================================================
# 核算
# ======================================================================


def _load_ruleset(db: Path | None) -> Ruleset:
    settings = get_settings()
    store = RuleStore(db or settings.db_path)
    rules = store.list_rules()
    store.close()
    if not rules:
        console.print("[yellow]规则库是空的：先跑 import-rules[/yellow]")
        raise typer.Exit(code=1)
    return Ruleset(rules=rules)


@app.command("calc")
def calc(
    excel: Path = typer.Argument(..., exists=True, help="综测表 Excel"),
    sheet: str = typer.Option("0", "--sheet"),
    academic_year: str | None = typer.Option(None, "--year", "-y"),
    college: str | None = typer.Option(None, "--college"),
    student: str | None = typer.Option(None, "--student", help="只算某个学号"),
    db: Path | None = typer.Option(None, "--db"),
    out: Path | None = typer.Option(None, "--out", help="导出 JSON 账目"),
    show_matches: bool = typer.Option(True, "--show/--quiet"),
) -> None:
    """批量核算综测表，输出可回溯账目。"""
    sheet_arg: str | int = int(sheet) if sheet.isdigit() else sheet
    claims = load_claims(excel, sheet=sheet_arg, academic_year=academic_year)
    ruleset = _load_ruleset(db)
    results = compute_all(claims, ruleset, academic_year=academic_year, college=college,
                          config=EngineConfig())
    if student:
        results = {k: v for k, v in results.items() if k == student}
        if not results:
            console.print(f"[red]没有找到学号 {student}[/red]")
            raise typer.Exit(code=1)

    table = Table(title=f"核算结果（{len(results)} 人，规则 {len(ruleset)} 条）")
    for col in ("学号", "姓名", "总分", "计入条目", "未命中", "待复核"):
        table.add_column(col)
    name_of = {c.student_id: c.student_name for c in claims}
    for sid, bd in results.items():
        counted = sum(1 for m in bd.matches if m.counted and m.rule_id)
        table.add_row(
            sid,
            name_of.get(sid) or "",
            f"[bold]{bd.total:g}[/bold]",
            str(counted),
            str(len(bd.unmatched_claims)),
            str(len(bd.review_claims)),
        )
    console.print(table)

    if show_matches:
        for sid, bd in results.items():
            if not bd.matches:
                continue
            detail = Table(title=f"{sid} 明细")
            for col in ("申报", "等级", "分值", "是否计入", "通道", "出处", "说明"):
                detail.add_column(col)
            for m in bd.matches:
                detail.add_row(
                    m.claim_id,
                    (m.matched_key or "-"),
                    f"{m.score:g}",
                    "[green]是[/green]" if m.counted else "[red]否[/red]",
                    m.channel,
                    m.source.short() if m.source else "-",
                    m.reason or "",
                )
            console.print(detail)

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({sid: bd.to_dict() for sid, bd in results.items()},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        console.print(f"已写出：{out}")


@app.command("explain")
def explain(
    text: str = typer.Argument(..., help="申报原文，如 省二等奖"),
    category: str = typer.Option("学科竞赛", "--category"),
    academic_year: str | None = typer.Option(None, "--year", "-y"),
    college: str | None = typer.Option(None, "--college"),
    db: Path | None = typer.Option(None, "--db"),
) -> None:
    """解释单条申报走哪个通道、命中哪条规则、引用哪段原文。"""
    ruleset = _load_ruleset(db)
    router = Router(ruleset, academic_year=academic_year, college=college)
    claim = Claim(student_id="demo", raw_text=text, category=category,
                  academic_year=academic_year, college=college)
    console.print_json(json.dumps(router.explain(claim), ensure_ascii=False))


@app.command("route-pdf")
def route_pdf(
    pdf: Path = typer.Argument(..., exists=True, help="细则 PDF（兜底通道语料）"),
    text: str = typer.Option(..., "--text", help="申报原文"),
    db: Path | None = typer.Option(None, "--db"),
    top_k: int = typer.Option(5, "--top-k"),
) -> None:
    """用 PDF 原文做兜底检索（BM25），演示未命中时如何溯源与拒答。"""
    ruleset = _load_ruleset(db)
    clauses = clauses_from_pdf_pages(load_pdf(pdf))
    router = Router(ruleset, clauses=clauses)
    claim = Claim(student_id="demo", raw_text=text, category="未分类")
    console.print_json(json.dumps(router.explain(claim, top_k=top_k), ensure_ascii=False))


# ======================================================================
# 可信抽取与回测
# ======================================================================


@app.command("extract-rules-llm")
def extract_rules_llm(
    source: Path = typer.Argument(..., exists=True, help="待抽取的 UTF-8 规则文本块"),
    academic_year: str = typer.Option(..., "--year", "-y", help="规则适用学年"),
    college: str | None = typer.Option(None, "--college"),
    page: int | None = typer.Option(None, "--page", min=1),
    allowed_level: list[str] | None = typer.Option(
        None,
        "--allowed-level",
        help="本文档允许的自定义等级/身份，可重复传入",
    ),
    double_check: bool = typer.Option(
        False, "--double-check", help="使用第二种提示独立抽取并交叉验证"
    ),
    publish: bool = typer.Option(
        False, "--publish", help="人工确认后发布；未通过网关的批次仍会被强制阻止"
    ),
    report_out: Path | None = typer.Option(None, "--report-out", help="保存完整网关报告 JSON"),
    db: Path | None = typer.Option(None, "--db"),
) -> None:
    """调用文本模型抽取规则，并经过五道验证及独立发布冲突门禁。"""
    settings = get_settings()
    text = source.read_text(encoding="utf-8")
    context = GatewayContext(
        academic_year=academic_year,
        college=college,
        doc=source.name,
        page=page,
        allowed_levels=frozenset(allowed_level or []),
    )
    database = db or settings.db_path
    with RuleStore(database) as store, CostLedger(database) as ledger:
        # 先按旧构造契约创建，保持第三方/测试替身兼容；正式实现再注入账本上下文。
        extractor = LLMExtractor(cache=store)
        if isinstance(extractor, LLMExtractor):
            extractor.cost_ledger = ledger
            extractor.material_id = f"document:{_file_sha256(source)}"
            extractor.batch_id = f"rule-extraction:{_file_sha256(source)[:16]}"
        report = extractor.extract_validated(
            text,
            context=context,
            risk_level="high" if double_check else "normal",
            existing_rules=store.list_rules(enabled_only=False),
        )
        if publish:
            count = store.publish_extraction_report(report, model=extractor.model)
            console.print(f"[green]已发布 {count} 条通过网关的规则[/green]")
        else:
            store.record_extraction_report(report, model=extractor.model)
    console.print_json(json.dumps(report.metrics(), ensure_ascii=False))
    if report_out:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"已写出：{report_out}")
    if not report.publishable:
        raise typer.Exit(code=2)


@app.command("eval-extraction-gateway")
def eval_extraction_gateway(
    dataset: Path = typer.Argument(..., exists=True, help="网关负例集 JSON"),
    academic_year: str = typer.Option(..., "--year", "-y"),
    college: str | None = typer.Option(None, "--college"),
    out: Path | None = typer.Option(None, "--out", help="保存评测报告 JSON"),
) -> None:
    """复跑分层构造负例，报告五道校验与发布门禁的 Wilson 区间。"""
    raw = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list):
        raise typer.BadParameter("数据集必须是包含 dataset_version 与 cases 数组的 JSON 对象")
    cases = [GatewayNegativeCase.model_validate(item) for item in raw["cases"]]
    report = evaluate_gateway_negatives(
        cases,
        context=GatewayContext(
            academic_year=academic_year,
            college=college,
            doc=dataset.name,
        ),
        dataset_version=str(raw.get("dataset_version") or "unversioned"),
    )
    encoded = report.model_dump_json(indent=2)
    console.print_json(encoded)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(encoded, encoding="utf-8")
        console.print(f"已写出：{out}")
    target_failed = any(
        result.detection_rate < 0.99 for result in report.target_results.values()
    )
    if report.detection_rate < 0.99 or target_failed:
        raise typer.Exit(code=2)


def _sheet_arg(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@app.command("export-backtest-template")
def export_backtest_template(
    excel: Path = typer.Argument(..., exists=True, help="往年综测表（申报明细）"),
    out: Path = typer.Option(..., "--out", help="逐项历史参照/裁决模板 .xlsx/.csv"),
    sheet: str = typer.Option("0", "--sheet"),
    academic_year: str | None = typer.Option(None, "--year", "-y"),
) -> None:
    """从真实申报明细生成稳定的逐项回测标注模板。"""
    claims = load_claims(excel, sheet=_sheet_arg(sheet), academic_year=academic_year)
    out.parent.mkdir(parents=True, exist_ok=True)
    template = item_reference_template(claims)
    if out.suffix.lower() == ".csv":
        template.to_csv(out, index=False, encoding="utf-8-sig")
    else:
        template.to_excel(out, index=False, sheet_name="逐项参照")
    console.print(f"已生成 {len(claims)} 条逐项参照模板：[bold]{out}[/bold]")
    console.print("请填写“历史/裁决得分”；只有经业务确认的行才将“已裁决”设为“是”。")


@app.command("backtest")
def backtest(
    excel: Path = typer.Argument(..., exists=True, help="往年综测表（申报明细）"),
    truth: Path = typer.Option(..., "--truth", exists=True, help="往年汇总表（学号/总分）"),
    item_reference: Path | None = typer.Option(
        None, "--item-reference", exists=True, help="逐项历史参照/业务裁决表"
    ),
    sheet: str = typer.Option("0", "--sheet"),
    truth_sheet: str = typer.Option("0", "--truth-sheet"),
    item_sheet: str = typer.Option("0", "--item-sheet"),
    academic_year: str | None = typer.Option(None, "--year", "-y"),
    college: str | None = typer.Option(None, "--college"),
    mode: str = typer.Option(
        "historical-reference", "--mode", help="historical-reference 或 adjudicated-truth"
    ),
    required_students: int | None = typer.Option(
        None, "--required-students", min=1, help="启用样本量与数据完整性门禁；正式验收填 52"
    ),
    student_col: str = typer.Option("学号", "--student-col"),
    total_col: str = typer.Option("总分", "--total-col"),
    item_col: str = typer.Option("申报项标识", "--item-col"),
    item_score_col: str = typer.Option("历史/裁决得分", "--item-score-col"),
    out: Path | None = typer.Option(None, "--out", help="导出完整 JSON 报告"),
    diff_out: Path | None = typer.Option(None, "--diff-out", help="导出完整差异 CSV"),
    db: Path | None = typer.Option(None, "--db"),
) -> None:
    """执行逐人、逐项回测；无业务裁决时只报告与历史人工结果的一致性。"""
    started = time.perf_counter()
    normalized_mode = mode.strip().lower().replace("-", "_")
    if normalized_mode not in {"historical_reference", "adjudicated_truth"}:
        raise typer.BadParameter("--mode 只能是 historical-reference 或 adjudicated-truth")
    claims = load_claims(excel, sheet=_sheet_arg(sheet), academic_year=academic_year)
    ruleset = _load_ruleset(db)
    ground_truth = load_ground_truth(
        truth,
        sheet=_sheet_arg(truth_sheet),
        student_col=student_col,
        total_col=total_col,
    )
    item_expectations = (
        load_item_expectations(
            item_reference,
            sheet=_sheet_arg(item_sheet),
            student_col=student_col,
            item_col=item_col,
            score_col=item_score_col,
        )
        if item_reference
        else None
    )
    report = run_backtest(
        claims,
        ruleset,
        ground_truth,
        item_expectations=item_expectations,
        mode=normalized_mode,  # type: ignore[arg-type]
        required_students=required_students,
        academic_year=academic_year,
        college=college,
    )
    report.meta["input_sha256"] = {
        "claims": _file_sha256(excel),
        "totals": _file_sha256(truth),
        **({"items": _file_sha256(item_reference)} if item_reference else {}),
    }
    report.meta["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    summary = report.summary()
    console.print_json(json.dumps(summary, ensure_ascii=False))
    console.print(
        f"{report.person_metric_name} [bold]{report.accuracy:.2%}[/bold]"
        f"（{report.exact}/{report.total_students}），MAE {report.mean_absolute_error}，"
        f"最大误差 {report.max_absolute_error}"
    )
    if report.item_agreement is not None:
        console.print(
            f"{report.item_metric_name} [bold]{report.item_agreement:.2%}[/bold]"
            f"（{report.matched_items}/{report.total_items}）"
        )
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        console.print(f"已写出完整报告：{out}")
    if diff_out:
        diff_out.parent.mkdir(parents=True, exist_ok=True)
        report.diffs_frame().to_csv(diff_out, index=False, encoding="utf-8-sig")
        console.print(f"已写出完整差异：{diff_out}")
    if required_students is not None and not report.gate_passed:
        raise typer.Exit(code=2)


# ======================================================================
# 服务
# ======================================================================


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """启动 FastAPI 服务（/docs 可交互调试）。"""
    import uvicorn

    uvicorn.run("scoreproof.api.app:app", host=host, port=port, reload=reload)


@app.command("doctor")
def doctor() -> None:
    """环境自检：依赖、配置、目录、规则库。"""
    settings = get_settings()
    table = Table(title="环境自检")
    for col in ("项目", "状态", "说明"):
        table.add_column(col)

    def add(name: str, ok: bool, note: str) -> None:
        table.add_row(name, "[green]OK[/green]" if ok else "[yellow]缺失[/yellow]", note)

    add("数据目录", settings.data_dir.exists(), str(settings.data_dir))
    add("规则库", settings.db_path.exists(), str(settings.db_path))
    add("混合索引库", settings.index_db_path.exists(), str(settings.index_db_path))
    add("向量目录", settings.vector_dir.exists(), str(settings.vector_dir))
    add("LLM", settings.llm_configured, f"{settings.llm_model} @ {settings.llm_base_url}")
    for mod, note, extra in (
        ("pandas", "Excel 解析", "核心"),
        ("openpyxl", "xlsx 引擎", "核心"),
        ("pymupdf", "PDF 文本", "核心"),
        ("pdfplumber", "PDF 表格", "可选"),
        ("docx", "Word 解析", "核心"),
        ("fastapi", "服务层", "核心"),
        ("rank_bm25", "BM25 召回", "retrieval"),
        ("chromadb", "向量持久化", "retrieval"),
        ("langchain_chroma", "LangChain 向量检索", "retrieval"),
        ("langchain_openai", "LLM 抽取接入", "llm"),
        ("rapidocr_onnxruntime", "OCR", "P2"),
        ("imagehash", "查重", "P2"),
        ("cv2", "图像预处理", "P2"),
    ):
        try:
            __import__(mod)
            add(f"依赖 {mod}", True, note)
        except ImportError:
            add(f"依赖 {mod}", False, f"{note}（{'必需' if extra == '核心' else extra}）")
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()

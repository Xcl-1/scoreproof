"""命令行入口：``scoreproof <command>``。

设计目标：不用起服务也能完成 P1 全流程（解析 -> 入库 -> 核算 -> 回测）。
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .calc.engine import EngineConfig, compute_all
from .config import get_settings
from .eval.backtest import load_ground_truth, run_backtest
from .eval.gateway import GatewayNegativeCase, evaluate_gateway_negatives
from .ingest.excel_loader import load_claims, load_rules
from .ingest.image_loader import check_quality, phash, preprocess, run_ocr
from .ingest.pdf_loader import load_pdf
from .retrieval.router import Router, clauses_from_pdf_pages
from .rules.extractor import LLMExtractor
from .rules.gateway import GatewayContext
from .rules.store import RuleStore
from .schema import Claim, Ruleset

app = typer.Typer(
    help="综测加分核算系统：Excel/PDF 解析 -> 规则库 -> 确定性核算 -> 回测",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


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
    )
    with RuleStore(db or settings.db_path) as store:
        extractor = LLMExtractor(cache=store)
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


@app.command("backtest")
def backtest(
    excel: Path = typer.Argument(..., exists=True, help="往年综测表（申报明细）"),
    truth: Path = typer.Option(..., "--truth", exists=True, help="往年汇总表（学号/总分）"),
    sheet: str = typer.Option("0", "--sheet"),
    truth_sheet: str = typer.Option("0", "--truth-sheet"),
    academic_year: str | None = typer.Option(None, "--year", "-y"),
    college: str | None = typer.Option(None, "--college"),
    student_col: str = typer.Option("学号", "--student-col"),
    total_col: str = typer.Option("总分", "--total-col"),
    out: Path | None = typer.Option(None, "--out", help="导出逐人比对 CSV/JSON"),
    db: Path | None = typer.Option(None, "--db"),
) -> None:
    """用往年综测表回测计算准确率（**没测出来就不写数字**）。"""
    sheet_arg: str | int = int(sheet) if sheet.isdigit() else sheet
    truth_arg: str | int = int(truth_sheet) if truth_sheet.isdigit() else truth_sheet
    claims = load_claims(excel, sheet=sheet_arg, academic_year=academic_year)
    ruleset = _load_ruleset(db)
    ground_truth = load_ground_truth(truth, sheet=truth_arg, student_col=student_col,
                                     total_col=total_col)
    report = run_backtest(claims, ruleset, ground_truth, academic_year=academic_year,
                          college=college)
    summary = report.summary()
    console.print_json(json.dumps(summary, ensure_ascii=False))
    console.print(
        f"逐人一致率 [bold]{report.accuracy:.2%}[/bold]"
        f"（{report.exact}/{report.total_students}），"
        f"容差内 {report.accuracy_within_tolerance:.2%}"
    )
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        console.print(f"已写出：{out}")


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
    add("LLM", settings.llm_configured, f"{settings.llm_model} @ {settings.llm_base_url}")
    for mod, note, extra in (
        ("pandas", "Excel 解析", "核心"),
        ("openpyxl", "xlsx 引擎", "核心"),
        ("pymupdf", "PDF 文本", "核心"),
        ("pdfplumber", "PDF 表格", "可选"),
        ("docx", "Word 解析", "核心"),
        ("fastapi", "服务层", "核心"),
        ("rank_bm25", "兜底检索", "可选"),
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

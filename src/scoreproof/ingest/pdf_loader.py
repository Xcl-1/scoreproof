"""PDF 接入：细则文本 + 表格抽取，保留页码与坐标用于溯源。

选型：PyMuPDF 抽文本（快、坐标精确），表格复杂时回落到 pdfplumber。
两者都是可选依赖，缺失时给出可执行的提示而不是 ImportError 崩溃。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import DataSourceError

try:  # pragma: no cover - 取决于环境
    import pymupdf
except ImportError:  # pragma: no cover
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None


@dataclass
class PDFPage:
    """一页的抽取结果，``source`` 可直接挂到 Rule.source 上。"""

    page: int  # 1-based
    text: str = ""
    blocks: list[dict[str, Any]] = field(default_factory=list)
    tables: list[list[list[str]]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text.strip())

    @property
    def is_probably_scanned(self) -> bool:
        """几乎没文字 => 大概率是扫描件，需走 OCR + 人工校对。"""
        return self.char_count < 20

    def find(self, *keywords: str) -> list[dict[str, Any]]:
        """按关键词定位段落，返回带坐标的块（用于条款级引用）。"""
        hits: list[dict[str, Any]] = []
        for b in self.blocks:
            text = b.get("text", "")
            if any(k in text for k in keywords):
                hits.append(b)
        return hits


def _require_pymupdf() -> Any:
    if pymupdf is None:  # pragma: no cover
        raise DataSourceError(
            "需要 pymupdf 才能读取 PDF",
            detail={"hint": "uv sync --extra multimodal 或 uv add pymupdf"},
        )
    return pymupdf


def load_pdf(
    path: str | Path,
    *,
    start: int | None = None,
    end: int | None = None,
    with_tables: bool = False,
) -> list[PDFPage]:
    """读取 PDF 每页的文本与文本块坐标。

    Args:
        start: 起始页（1-based，含）。
        end: 结束页（1-based，含）。
        with_tables: 是否顺带用 pdfplumber 抽表格。
    """
    p = Path(path)
    if not p.exists():
        raise DataSourceError(f"PDF 文件不存在：{p}", detail={"path": str(p)})
    fitz = _require_pymupdf()

    pages: list[PDFPage] = []
    with fitz.open(p) as doc:
        total = doc.page_count
        lo = max(1, start or 1)
        hi = min(total, end or total)
        for i in range(lo - 1, hi):
            page = doc.load_page(i)
            text = page.get_text("text")
            blocks = [
                {
                    "text": b[4].strip(),
                    "bbox": tuple(round(v, 2) for v in b[:4]),
                    "block_no": b[5] if len(b) > 5 else None,
                }
                for b in page.get_text("blocks")
                if str(b[4]).strip()
            ]
            pages.append(
                PDFPage(
                    page=i + 1,
                    text=text,
                    blocks=blocks,
                    meta={"doc": p.name, "total_pages": total},
                )
            )
    if with_tables:
        tables = load_pdf_tables(p, start=start, end=end)
        for pg in pages:
            pg.tables = tables.get(pg.page, [])
    return pages


def load_pdf_tables(
    path: str | Path,
    *,
    start: int | None = None,
    end: int | None = None,
) -> dict[int, list[list[list[str]]]]:
    """用 pdfplumber 抽表格：``{页码: [表1, 表2, ...]}``。

    细则里的"等级×类别 加分矩阵"就在这里；第一版允许"抽取 + 人工校对一遍"。
    """
    p = Path(path)
    if not p.exists():
        raise DataSourceError(f"PDF 文件不存在：{p}", detail={"path": str(p)})
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise DataSourceError(
            "抽表格需要 pdfplumber",
            detail={"hint": "uv add pdfplumber 或 uv sync --extra multimodal"},
        ) from exc

    out: dict[int, list[list[list[str]]]] = {}
    with pdfplumber.open(p) as pdf:
        lo = max(1, start or 1)
        hi = min(len(pdf.pages), end or len(pdf.pages))
        for i in range(lo - 1, hi):
            raw = pdf.pages[i].extract_tables() or []
            cleaned = [
                [["" if c is None else str(c).strip() for c in row] for row in table]
                for table in raw
            ]
            if cleaned:
                out[i + 1] = cleaned
    return out


def pdf_page_to_source(pg: PDFPage, *, table: str | None = None, clause: str | None = None,
                       text: str | None = None, bbox: tuple[float, float, float, float] | None = None):
    """PDFPage -> SourceRef（延迟导入，避免循环依赖）。"""
    from ..schema import SourceRef

    return SourceRef(doc=pg.meta.get("doc", ""), page=pg.page, table=table, clause=clause,
                     text=text, bbox=bbox)


__all__ = ["PDFPage", "load_pdf", "load_pdf_tables", "pdf_page_to_source"]

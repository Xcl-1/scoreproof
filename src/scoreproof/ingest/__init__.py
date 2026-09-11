"""接入层：按格式分流（Excel / PDF / Word / 图片）。

每个 loader 的职责只有一个：把异构文件变成**带出处定位**的中间结构，
不做任何计分判断，也不调用 LLM。
"""

from __future__ import annotations

from .docx_loader import DocxBlock, load_docx
from .excel_loader import (
    ExcelGrid,
    fill_merged_cells,
    load_claims,
    load_rules,
    load_sheet,
    load_workbook_grid,
    normalize_header,
    read_excel,
)
from .pdf_loader import PDFPage, load_pdf, load_pdf_tables

__all__ = [
    "DocxBlock",
    "ExcelGrid",
    "PDFPage",
    "fill_merged_cells",
    "load_claims",
    "load_docx",
    "load_pdf",
    "load_pdf_tables",
    "load_rules",
    "load_sheet",
    "load_workbook_grid",
    "normalize_header",
    "read_excel",
]

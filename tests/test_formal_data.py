"""阶段 7.10：真实数据预检必须挡住缺失、复制扩增和伪装文件。"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from scoreproof.cli import app
from scoreproof.eval.formal_data import FormalDataInventoryReport, audit_formal_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PDF = PROJECT_ROOT / "data" / "raw" / "public" / "fzu_ccds_2026_recommendation_score_rules.pdf"


def test_real_cli_blocks_empty_formal_dataset_and_writes_strict_report(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    root.mkdir()
    out = tmp_path / "inventory.json"
    invoked = CliRunner().invoke(app, ["audit-formal-data", str(root), "--out", str(out)])
    assert invoked.exit_code == 2
    report = FormalDataInventoryReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert report.formal_gate_eligible is False
    assert report.public_real_pdf_count == 1
    assert report.synthetic_certificate_count == 5
    assert all(gate.data_kind == "missing" for gate in report.gates)


def test_real_public_pdf_cannot_be_counted_thirty_times(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    source = root / "complex-pdf"
    source.mkdir(parents=True)
    (source / "public.pdf").write_bytes(PUBLIC_PDF.read_bytes())
    cases = [
        {
            "case_id": f"case-{index:02d}",
            "document": "public.pdf",
            "scenario": "multi_column"
            if index % 3 == 0
            else ("cross_page_table" if index % 3 == 1 else "scanned"),
            "real_document": True,
            "synthetic": False,
            "expected_ordered_fragments": ["第一段", "第二段"],
            "expected_table_fragments": ["论文"],
            "expected_min_table_rows": 1,
            "expected_scanned_pages": [1],
        }
        for index in range(30)
    ]
    # 清单本身也不允许同文档同场景重复；复制改名后再检查文件 Hash 去重。
    for index, case in enumerate(cases):
        name = f"copy-{index:02d}.pdf"
        (source / name).write_bytes(PUBLIC_PDF.read_bytes())
        case["document"] = name
    (source / "dataset.json").write_text(
        json.dumps(
            {
                "dataset_version": "public-copy-regression",
                "authorization_reference": "public-source:local-readme",
                "independent_real_documents": True,
                "cases": cases,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report = audit_formal_data(root, project_root=PROJECT_ROOT)
    gate = next(gate for gate in report.gates if gate.gate_id == "complex_pdf")
    assert gate.sample_size == 1
    assert gate.category_counts == {
        "multi_column": 1,
        "cross_page_table": 1,
        "scanned": 1,
    }
    assert gate.formal_gate_eligible is False


def test_certificate_copy_expansion_and_bad_magic_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    source = root / "certificate-fields"
    source.mkdir(parents=True)
    content = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL/nwAAAABJRU5ErkJggg=="
    )
    rows = []
    for index in range(30):
        name = f"copy-{index:02d}.png"
        item_content = content if index < 29 else b"not-a-real-png"
        (source / name).write_bytes(item_content)
        rows.append(
            {
                "evidence_id": f"E{index:03d}",
                "image_path": name,
                "sha256": hashlib.sha256(item_content).hexdigest(),
                "synthetic": False,
                "real_source": True,
                "redacted": True,
                "authorization_reference": "private-consent-v1",
                "name": "学生代号",
                "event_name": "赛事",
                "tier": "省级",
                "award": "二等奖",
                "award_date": "2025-10-01",
                "issuer": "组委会",
                "team_attribute": "个人",
                "raw_fields": {
                    "姓名": "学生代号",
                    "赛事名称": "赛事",
                    "级别": "省级",
                    "奖项/名次": "二等奖",
                    "获奖日期": "2025年10月1日",
                    "颁发单位": "组委会",
                    "团队属性": "个人",
                },
            }
        )
    (source / "labels.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    report = audit_formal_data(root, project_root=PROJECT_ROOT)
    gate = next(gate for gate in report.gates if gate.gate_id == "certificate_fields")
    assert gate.formal_gate_eligible is False
    assert gate.unique_file_count == 1
    assert any("图片文件重复" in problem for problem in gate.problems)
    assert any("图片缺失或路径不安全" in problem for problem in gate.problems)


def test_dedup_reversed_pairs_cannot_expand_sample_size(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    source = root / "evidence-dedup"
    source.mkdir(parents=True)
    # PNG 魔数足以验证预检的路径及文件类型；不调用图像解码或查重算法。
    (source / "left.png").write_bytes(b"\x89PNG\r\n\x1a\nleft")
    (source / "right.png").write_bytes(b"\x89PNG\r\n\x1a\nright")
    left = {"id": "L", "type": "image", "path": "left.png", "fields": {"姓名": "甲"}}
    right = {"id": "R", "type": "image", "path": "right.png", "fields": {"姓名": "甲"}}
    rows = [
        {
            "id": f"pair-{index}",
            "duplicate": index % 2 == 0,
            "left": left if index % 2 == 0 else right,
            "right": right if index % 2 == 0 else left,
            "real_source": True,
            "redacted": True,
        }
        for index in range(50)
    ]
    (source / "pairs.json").write_text(
        json.dumps(
            {
                "dataset_version": "copy-pairs",
                "authorization_reference": "private-consent-v1",
                "independent_real_pairs": True,
                "pairs": rows,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report = audit_formal_data(root, project_root=PROJECT_ROOT)
    gate = next(gate for gate in report.gates if gate.gate_id == "evidence_dedup")
    assert gate.sample_size == 50
    assert gate.unique_file_count == 2
    assert gate.formal_gate_eligible is False
    assert any("图片对重复" in problem for problem in gate.problems)

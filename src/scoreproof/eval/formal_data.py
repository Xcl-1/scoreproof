"""正式评测数据预检；只读取本地文件，不保存标注值或个人信息。"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pymupdf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..evidence.certificate import CERTIFICATE_FIELD_NAMES
from .pdf_regression import PDFRegressionDataset
from .rule_extraction import RuleExtractionDataset

GateId = Literal["complex_pdf", "rule_extraction", "certificate_fields", "evidence_dedup"]
DataKind = Literal["real_authorized", "real_public", "real_redacted", "synthetic", "mock", "missing"]
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_TRANSFORMATIONS = {"exact", "compressed", "rotated", "cropped", "screenshot", "same_fact"}
_HARD_NEGATIVES = {
    "same_event_different_person",
    "same_person_different_year",
    "same_level_different_event",
    "other",
}
_GATES: tuple[tuple[GateId, str], ...] = (
    ("complex_pdf", "complex-pdf/dataset.json"),
    ("rule_extraction", "rule-extraction/dataset.json"),
    ("certificate_fields", "certificate-fields/labels.jsonl"),
    ("evidence_dedup", "evidence-dedup/pairs.json"),
)


class FormalDataGateAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    gate_id: GateId
    manifest: str
    manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    sample_size: int = Field(ge=0)
    unique_file_count: int = Field(ge=0)
    category_counts: dict[str, int] = Field(default_factory=dict)
    data_kind: DataKind
    authorization_reference_present: bool
    redaction_verified: bool
    labels_consistent: bool
    formal_gate_eligible: bool
    problems: list[str]

    @model_validator(mode="after")
    def _missing_cannot_pass(self) -> FormalDataGateAudit:
        if self.formal_gate_eligible and (
            self.manifest_sha256 is None
            or not self.authorization_reference_present
            or not self.labels_consistent
        ):
            raise ValueError("正式数据资格必须有清单、授权引用和一致标注")
        return self


class FormalDataInventoryReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1.0"] = "1.0"
    generated_at: datetime
    git_head: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_root: str
    public_real_pdf_count: int = Field(ge=0)
    synthetic_certificate_count: int = Field(ge=0)
    gates: list[FormalDataGateAudit]
    formal_gate_eligible: bool
    limitations: list[str]

    @model_validator(mode="after")
    def _gate_count_and_flag(self) -> FormalDataInventoryReport:
        if {gate.gate_id for gate in self.gates} != {item[0] for item in _GATES}:
            raise ValueError("必须完整记录四个优先门禁")
        if self.formal_gate_eligible != any(gate.formal_gate_eligible for gate in self.gates):
            raise ValueError("总体资格与逐门禁资格不一致")
        return self


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_hash(manifest_hash: str, file_hashes: set[str]) -> str:
    canonical = json.dumps({"manifest": manifest_hash, "files": sorted(file_hashes)}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _pdf_rendered_sha256(path: Path) -> str:
    """以全部页面的低分辨率像素识别仅改写元数据的同一文档。"""
    digest = hashlib.sha256()
    with pymupdf.open(path) as document:
        if not document.page_count:
            raise ValueError("PDF 无页面")
        digest.update(document.page_count.to_bytes(8, "big"))
        for page in document:
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(0.5, 0.5),
                colorspace=pymupdf.csGRAY,
                alpha=False,
                annots=False,
            )
            digest.update(pixmap.width.to_bytes(4, "big"))
            digest.update(pixmap.height.to_bytes(4, "big"))
            digest.update(pixmap.samples)
    return digest.hexdigest()


def _safe_file(root: Path, name: object, suffixes: set[str]) -> Path | None:
    if not isinstance(name, str) or not name.strip():
        return None
    candidate = (root / name).resolve()
    if not candidate.is_relative_to(root) or candidate.suffix.lower() not in suffixes:
        return None
    if not candidate.is_file():
        return None
    try:
        with candidate.open("rb") as stream:
            magic = stream.read(12)
    except OSError:
        return None
    suffix = candidate.suffix.lower()
    valid = (
        suffix == ".pdf"
        and magic.startswith(b"%PDF-")
        or suffix == ".png"
        and magic.startswith(b"\x89PNG\r\n\x1a\n")
        or suffix in {".jpg", ".jpeg"}
        and magic.startswith(b"\xff\xd8\xff")
        or suffix == ".webp"
        and magic[:4] == b"RIFF"
        and magic[8:12] == b"WEBP"
    )
    return candidate if valid else None


def _empty(gate_id: GateId, manifest: str, problem: str) -> FormalDataGateAudit:
    return FormalDataGateAudit(
        gate_id=gate_id,
        manifest=manifest,
        sample_size=0,
        unique_file_count=0,
        data_kind="missing",
        authorization_reference_present=False,
        redaction_verified=False,
        labels_consistent=False,
        formal_gate_eligible=False,
        problems=[problem],
    )


def _pdf_audit(root: Path, relative: str) -> FormalDataGateAudit:
    source = root / relative
    if not source.is_file():
        return _empty("complex_pdf", relative, "缺少复杂 PDF 正式清单")
    try:
        data = PDFRegressionDataset.model_validate_json(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _empty("complex_pdf", relative, f"清单 Schema 无效：{type(exc).__name__}")
    hashes: set[str] = set()
    content_first_path: dict[str, Path] = {}
    rendered_cache: dict[str, str] = {}
    duplicate_paths: set[Path] = set()
    categories: dict[str, set[str]] = {key: set() for key in ("multi_column", "cross_page_table", "scanned")}
    problems: list[str] = []
    for index, case in enumerate(data.cases, 1):
        file = _safe_file(root / "complex-pdf", case.document, {".pdf"})
        if file is None:
            problems.append(f"第 {index} 例：文件缺失或路径不安全")
            continue
        digest = _sha256(file)
        hashes.add(digest)
        try:
            rendered = rendered_cache.get(digest)
            if rendered is None:
                rendered = _pdf_rendered_sha256(file)
                rendered_cache[digest] = rendered
        except (OSError, RuntimeError, ValueError) as exc:
            problems.append(f"第 {index} 例：PDF 无法完整渲染（{type(exc).__name__}）")
            continue
        first_path = content_first_path.setdefault(rendered, file)
        if file != first_path and file not in duplicate_paths:
            problems.append(f"第 {index} 例：PDF 页面内容重复或文件复制扩增")
            duplicate_paths.add(file)
        categories[case.scenario].add(rendered)
        if not case.real_document or case.synthetic:
            problems.append(f"第 {index} 例：非真实文档声明")
    counts = {key: len(values) for key, values in categories.items()}
    if len(content_first_path) < 30:
        problems.append(f"独立文件 {len(content_first_path)}/30")
    problems.extend(f"{key} {count}/10" for key, count in counts.items() if count < 10)
    if not data.authorization_reference:
        problems.append("缺少公开来源或授权引用")
    if not data.independent_real_documents:
        problems.append("未确认独立真实文档")
    return FormalDataGateAudit(
        gate_id="complex_pdf",
        manifest=relative,
        manifest_sha256=_sha256(source),
        input_sha256=_input_hash(_sha256(source), hashes),
        sample_size=len(content_first_path),
        unique_file_count=len(content_first_path),
        category_counts=counts,
        data_kind=(
            "synthetic"
            if any(case.synthetic or not case.real_document for case in data.cases)
            else "real_public"
            if (data.authorization_reference or "").startswith("public-source:")
            else "real_authorized"
            if data.authorization_reference
            else "missing"
        ),
        authorization_reference_present=bool(data.authorization_reference),
        redaction_verified=True,
        labels_consistent=not any(
            "文件缺失" in item or "非真实" in item or "PDF 页面内容重复" in item for item in problems
        ),
        formal_gate_eligible=not problems,
        problems=problems,
    )


def _rule_audit(root: Path, relative: str) -> FormalDataGateAudit:
    source = root / relative
    if not source.is_file():
        return _empty("rule_extraction", relative, "缺少规则抽取正式清单")
    try:
        data = RuleExtractionDataset.model_validate_json(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _empty("rule_extraction", relative, f"清单 Schema/金标无效：{type(exc).__name__}")
    problems: list[str] = []
    if len(data.cases) < 50:
        problems.append(f"独立真实金标 {len(data.cases)}/50")
    if not data.authorization_reference:
        problems.append("缺少授权引用")
    if not data.independent_real_samples or any(not c.real_source or c.synthetic for c in data.cases):
        problems.append("含合成或未确认独立真实来源")
    if any(not c.expected_rules for c in data.cases):
        problems.append("存在没有人工金标的原文块")
    return FormalDataGateAudit(
        gate_id="rule_extraction",
        manifest=relative,
        manifest_sha256=_sha256(source),
        input_sha256=_input_hash(_sha256(source), set()),
        sample_size=len(data.cases),
        unique_file_count=len(data.cases),
        data_kind=(
            "synthetic"
            if any(case.synthetic or not case.real_source for case in data.cases)
            else "real_public"
            if (data.authorization_reference or "").startswith("public-source:")
            else "real_authorized"
            if data.authorization_reference
            else "missing"
        ),
        authorization_reference_present=bool(data.authorization_reference),
        redaction_verified=False,
        labels_consistent=not any("金标" in item for item in problems),
        formal_gate_eligible=not problems,
        problems=problems,
    )


def _certificate_audit(root: Path, relative: str) -> FormalDataGateAudit:
    source = root / relative
    if not source.is_file():
        return _empty("certificate_fields", relative, "缺少奖状字段正式标签")
    problems: list[str] = []
    try:
        rows = [
            json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()
        ]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError("JSONL 行必须为对象")
    except (OSError, ValueError) as exc:
        return _empty("certificate_fields", relative, f"标签 JSONL 无效：{type(exc).__name__}")
    ids: set[str] = set()
    hashes: set[str] = set()
    authorization_complete = bool(rows)
    for index, row in enumerate(rows, 1):
        ident = row.get("evidence_id")
        if not isinstance(ident, str) or not ident or ident in ids:
            problems.append(f"第 {index} 行：evidence_id 缺失或重复")
        else:
            ids.add(ident)
        file = _safe_file(root / "certificate-fields", row.get("image_path"), _IMAGE_SUFFIXES)
        if file is None:
            problems.append(f"第 {index} 行：图片缺失或路径不安全")
        else:
            digest = _sha256(file)
            if digest in hashes:
                problems.append(f"第 {index} 行：图片文件重复")
            hashes.add(digest)
            if row.get("sha256", "").lower() != digest:
                problems.append(f"第 {index} 行：图片 SHA-256 不一致")
        raw = row.get("raw_fields")
        if not isinstance(raw, dict) or set(raw) != set(CERTIFICATE_FIELD_NAMES):
            problems.append(f"第 {index} 行：raw_fields 必须完整包含七字段")
        for key in ("name", "event_name", "tier", "award", "award_date", "issuer", "team_attribute"):
            if key not in row:
                problems.append(f"第 {index} 行：规范值缺少 {key}")
        if row.get("synthetic") is not False or row.get("real_source") is not True:
            problems.append(f"第 {index} 行：非独立真实图片声明")
        if row.get("redacted") is not True:
            problems.append(f"第 {index} 行：未完成脱敏确认")
        reference = row.get("authorization_reference")
        if not isinstance(reference, str) or not reference.strip():
            authorization_complete = False
            problems.append(f"第 {index} 行：缺少授权引用")
    if len(hashes) < 30:
        problems.append(f"独立真实图片 {len(hashes)}/30")
    return FormalDataGateAudit(
        gate_id="certificate_fields",
        manifest=relative,
        manifest_sha256=_sha256(source),
        input_sha256=_input_hash(_sha256(source), hashes),
        sample_size=len(rows),
        unique_file_count=len(hashes),
        data_kind="real_redacted" if rows and not any(row.get("synthetic") for row in rows) else "synthetic",
        authorization_reference_present=authorization_complete,
        redaction_verified=bool(rows) and all(row.get("redacted") is True for row in rows),
        labels_consistent=not any("字段" in item or "重复" in item or "不一致" in item for item in problems),
        formal_gate_eligible=not problems,
        problems=problems,
    )


def _dedup_audit(root: Path, relative: str) -> FormalDataGateAudit:
    source = root / relative
    if not source.is_file():
        return _empty("evidence_dedup", relative, "缺少证据查重正式成对清单")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("pairs"), list):
            raise ValueError("顶层必须含 pairs 数组")
    except (OSError, ValueError) as exc:
        return _empty("evidence_dedup", relative, f"成对清单 JSON 无效：{type(exc).__name__}")
    rows = data["pairs"]
    problems: list[str] = []
    ids: set[str] = set()
    pair_keys: set[tuple[str, str]] = set()
    files: set[str] = set()
    used_in_earlier_pairs: set[str] = set()
    classes: Counter[str] = Counter()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            problems.append(f"第 {index} 对：必须为对象")
            continue
        ident = row.get("id")
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_-]{3,64}", ident) or ident in ids:
            problems.append(f"第 {index} 对：id 缺失或重复")
        else:
            ids.add(ident)
        duplicate = row.get("duplicate")
        if type(duplicate) is not bool:
            problems.append(f"第 {index} 对：duplicate 必须为布尔值")
        else:
            classes["positive" if duplicate else "negative"] += 1
            if duplicate and row.get("transformation") not in _TRANSFORMATIONS:
                problems.append(f"第 {index} 对：正例缺少受限变换类别")
            if not duplicate and row.get("hard_negative") not in _HARD_NEGATIVES:
                problems.append(f"第 {index} 对：负例缺少受限困难负例类别")
        digests: list[str] = []
        for side in ("left", "right"):
            evidence = row.get(side)
            file = _safe_file(
                root / "evidence-dedup",
                evidence.get("path") if isinstance(evidence, dict) else None,
                _IMAGE_SUFFIXES,
            )
            if file is None:
                problems.append(f"第 {index} 对：{side} 图片缺失或路径不安全")
            else:
                digest = _sha256(file)
                files.add(digest)
                digests.append(digest)
            if not isinstance(evidence, dict) or not isinstance(evidence.get("fields"), dict):
                problems.append(f"第 {index} 对：{side} 缺少结构化字段")
            elif not isinstance(evidence.get("id"), str) or not re.fullmatch(
                r"[A-Za-z0-9_-]{3,64}", evidence["id"]
            ):
                problems.append(f"第 {index} 对：{side} 证据 ID 必须是匿名 ASCII 代号")
        if len(digests) == 2:
            key = (min(digests), max(digests))
            if key in pair_keys:
                problems.append(f"第 {index} 对：图片对重复（含左右交换）")
            pair_keys.add(key)
            if any(digest in used_in_earlier_pairs for digest in digests):
                problems.append(f"第 {index} 对：图片文件被其他样本对重复使用")
            used_in_earlier_pairs.update(digests)
        if row.get("real_source") is not True or row.get("redacted") is not True:
            problems.append(f"第 {index} 对：未确认真实来源或脱敏")
    if len(pair_keys) < 50:
        problems.append(f"独立真实样本对 {len(pair_keys)}/50")
    if not classes["positive"] or not classes["negative"]:
        problems.append("必须同时有重复正例和困难负例")
    if not data.get("authorization_reference"):
        problems.append("缺少授权引用")
    if data.get("independent_real_pairs") is not True:
        problems.append("未确认独立真实样本对")
    return FormalDataGateAudit(
        gate_id="evidence_dedup",
        manifest=relative,
        manifest_sha256=_sha256(source),
        input_sha256=_input_hash(_sha256(source), files),
        sample_size=len(rows),
        unique_file_count=len(files),
        category_counts=dict(classes),
        data_kind="real_redacted"
        if rows and all(isinstance(r, dict) and r.get("real_source") is True for r in rows)
        else "synthetic",
        authorization_reference_present=bool(data.get("authorization_reference")),
        redaction_verified=bool(rows)
        and all(isinstance(r, dict) and r.get("redacted") is True for r in rows),
        labels_consistent=not any("缺少" in item or "重复" in item for item in problems),
        formal_gate_eligible=not problems,
        problems=problems,
    )


def audit_formal_data(data_root: Path, *, project_root: Path) -> FormalDataInventoryReport:
    """在正式模型调用前预检授权声明、真实文件、去重与标签完整性。"""
    root = data_root.resolve()
    gates = [
        _pdf_audit(root, _GATES[0][1]),
        _rule_audit(root, _GATES[1][1]),
        _certificate_audit(root, _GATES[2][1]),
        _dedup_audit(root, _GATES[3][1]),
    ]
    public_dir = project_root / "data" / "raw" / "public"
    sample_dir = project_root / "data" / "sample" / "certificates"
    public_count = len(list(public_dir.glob("*.pdf")))
    synthetic_count = len(list(sample_dir.glob("*.png")))
    input_manifest = {gate.manifest: gate.input_sha256 for gate in gates}
    command = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return FormalDataInventoryReport(
        generated_at=datetime.now(UTC),
        git_head=command.stdout.strip(),
        source_sha256=hashlib.sha256(json.dumps(input_manifest, sort_keys=True).encode("utf-8")).hexdigest(),
        data_root=(
            root.relative_to(project_root.resolve()).as_posix()
            if root.is_relative_to(project_root.resolve())
            else root.name
        ),
        public_real_pdf_count=public_count,
        synthetic_certificate_count=synthetic_count,
        gates=gates,
        formal_gate_eligible=any(gate.formal_gate_eligible for gate in gates),
        limitations=[
            "预检核对文件、哈希、声明与标签完整性；授权真实性及脱敏质量仍须人工复核。",
            "预检通过不代表模型指标通过，也不解除 release-readiness 正式门禁。",
        ],
    )


__all__ = ["FormalDataGateAudit", "FormalDataInventoryReport", "audit_formal_data"]

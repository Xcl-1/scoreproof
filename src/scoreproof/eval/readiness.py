"""阶段 7 候选版本冻结、质量检查与全评测就绪审计。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .gateway import wilson_interval

GateStatus = Literal["通过", "阻塞", "缺失", "仅烟雾", "警告"]


class QualityCommandResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    command: list[str]
    exit_code: int
    duration_seconds: float = Field(ge=0.0)
    passed: bool
    output_tail: str


class QualityGateReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "1.0"
    generated_at: str
    git_head: str | None = None
    source_tree_clean: bool
    pytest_total: int | None = Field(default=None, ge=0)
    all_checks_passed: bool
    commands: list[QualityCommandResult]


class ArtifactSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    path: str
    exists: bool
    sha256: str | None = None
    valid_json: bool | None = None
    error: str | None = None


class ReadinessGate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    title: str
    required: bool
    status: GateStatus
    artifact: str | None = None
    sample_size: int | None = Field(default=None, ge=0)
    metrics: dict[str, Any] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class CostSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    embedding_queries: int = Field(default=0, ge=0)
    rerank_pairs: int = Field(default=0, ge=0)
    vlm_calls: int = Field(default=0, ge=0)
    external_text_calls: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    usage_coverage_rate: float | None = Field(default=None, ge=0, le=1)
    monetary_cost_available: bool = False
    monetary_cost_cny: float | None = Field(default=None, ge=0)
    cost_per_100_materials_cny: float | None = Field(default=None, ge=0)
    cost_per_subject_cny: float | None = Field(default=None, ge=0)
    notes: list[str] = Field(default_factory=list)


class ReleaseReadinessReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "1.0"
    generated_at: str
    candidate_version: str
    manifest_hash: str
    status: Literal["可以发布候选", "阻塞"]
    ready: bool
    report_dir: str
    artifacts: list[ArtifactSnapshot]
    gates: list[ReadinessGate]
    blocking_gate_ids: list[str]
    warning_gate_ids: list[str]
    cost_summary: CostSummary
    limitations: list[str]


_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("quality", "quality-gates-v1.json"),
    ("complex_pdf", "complex-pdf-regression-v1.json"),
    ("gateway", "gateway-negative-v2.json"),
    ("retrieval", "retrieval-ablation-v1.json"),
    ("citation", "citation-refusal-v1.json"),
    ("orchestration", "orchestration-guardrails-v1.json"),
    ("certificate", "certificate-fields-formal-v1.json"),
    ("certificate_smoke", "certificate-fields-smoke-v1.json"),
    ("vlm", "vlm-integration-v1.json"),
    ("dedup", "evidence-dedup-formal-v1.json"),
    ("dedup_smoke", "evidence-dedup-smoke-v1.json"),
    ("backtest", "backtest-52-v1.json"),
    ("user_trial", "user-trial-v1.json"),
    ("cost", "cost-summary-v1.json"),
)


def _run_command(name: str, command: list[str], *, cwd: Path, env: dict[str, str]) -> QualityCommandResult:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
            check=False,
        )
        output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        exit_code = result.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        output = str(exc)
        exit_code = 124 if isinstance(exc, subprocess.TimeoutExpired) else 127
    return QualityCommandResult(
        name=name,
        command=command,
        exit_code=exit_code,
        duration_seconds=round(time.perf_counter() - started, 3),
        passed=exit_code == 0,
        output_tail=output[-4000:],
    )


def _git_output(root: Path, *args: str) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    result = subprocess.run(
        [git, *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def run_quality_gates(project_root: str | Path) -> QualityGateReport:
    """通过固定命令真实执行发布质量门禁，不接受调用方自报结果。"""
    root = Path(project_root).resolve()
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["UV_CACHE_DIR"] = str(root / ".tmp" / "release-uv-cache")
    commands = [
        ("pytest", [sys.executable, "-m", "pytest"]),
        ("ruff", [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"]),
        ("mypy", [sys.executable, "-m", "mypy", "src"]),
        ("uv_lock", [shutil.which("uv") or "uv", "lock", "--offline", "--check"]),
        ("git_diff_check", [shutil.which("git") or "git", "diff", "--check"]),
    ]
    results = [_run_command(name, command, cwd=root, env=environment) for name, command in commands]
    pytest_output = next((item.output_tail for item in results if item.name == "pytest"), "")
    match = re.search(r"(\d+) passed", pytest_output)
    status = _git_output(root, "status", "--porcelain", "--untracked-files=all")
    return QualityGateReport(
        generated_at=datetime.now(UTC).isoformat(),
        git_head=_git_output(root, "rev-parse", "HEAD"),
        source_tree_clean=status == "",
        pytest_total=int(match.group(1)) if match else None,
        all_checks_passed=all(item.passed for item in results),
        commands=results,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_artifact(path: Path) -> tuple[dict[str, Any] | None, ArtifactSnapshot]:
    if not path.exists() or not path.is_file():
        return None, ArtifactSnapshot(name=path.stem, path=str(path), exists=False)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON 顶层不是对象")
        return value, ArtifactSnapshot(
            name=path.stem,
            path=str(path),
            exists=True,
            sha256=_sha256(path),
            valid_json=True,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return None, ArtifactSnapshot(
            name=path.stem,
            path=str(path),
            exists=True,
            sha256=_sha256(path),
            valid_json=False,
            error=str(exc),
        )


def _number(payload: Mapping[str, Any] | None, key: str) -> float | None:
    value = payload.get(key) if payload is not None else None
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _missing_gate(gate_id: str, title: str, filename: str) -> ReadinessGate:
    return ReadinessGate(
        id=gate_id,
        title=title,
        required=True,
        status="缺失",
        artifact=filename,
        reasons=[f"缺少正式产物 {filename}"],
    )


def _quality_gate(
    payload: dict[str, Any] | None, filename: str, candidate_version: str
) -> ReadinessGate:
    if payload is None:
        return _missing_gate("quality", "自动化质量门禁与候选冻结", filename)
    try:
        report = QualityGateReport.model_validate(payload)
    except ValueError as exc:
        return ReadinessGate(
            id="quality",
            title="自动化质量门禁与候选冻结",
            required=True,
            status="阻塞",
            artifact=filename,
            reasons=[f"质量报告 Schema 无效：{exc}"],
        )
    reasons: list[str] = []
    if not report.all_checks_passed:
        reasons.append("pytest/Ruff/mypy/uv lock/git diff 至少一项失败")
    if not report.source_tree_clean:
        reasons.append("工作树不干净，候选版本尚未冻结到提交")
    version_matches = bool(
        report.git_head
        and (
            report.git_head.startswith(candidate_version)
            or candidate_version.startswith(report.git_head)
        )
    )
    if not version_matches:
        reasons.append("质量报告的 Git 提交与 candidate_version 不一致")
    passed = report.all_checks_passed and report.source_tree_clean and version_matches
    return ReadinessGate(
        id="quality",
        title="自动化质量门禁与候选冻结",
        required=True,
        status="通过" if passed else "阻塞",
        artifact=filename,
        sample_size=report.pytest_total,
        metrics={
            "git_head": report.git_head,
            "candidate_version_matches": version_matches,
            "all_checks_passed": report.all_checks_passed,
        },
        reasons=reasons or ["固定质量命令全部通过，且工作树已冻结"],
    )


def _gateway_gate(payload: dict[str, Any] | None, filename: str) -> ReadinessGate:
    if payload is None:
        return _missing_gate("gateway", "五道验证网关", filename)
    sample = int(_number(payload, "sample_size") or 0)
    rate = _number(payload, "detection_rate") or 0.0
    has_ci = _number(payload, "wilson_lower") is not None and _number(payload, "wilson_upper") is not None
    passed = sample >= 100 and rate >= 0.99 and has_ci
    return ReadinessGate(
        id="gateway",
        title="五道验证网关",
        required=True,
        status="通过" if passed else "阻塞",
        artifact=filename,
        sample_size=sample,
        metrics={"detection_rate": rate, "has_ci95": has_ci},
        reasons=[] if passed else ["要求 n≥100、检出率≥99% 且报告 Wilson 95% 区间"],
    )


def _retrieval_gate(payload: dict[str, Any] | None, filename: str) -> ReadinessGate:
    if payload is None:
        return _missing_gate("retrieval_ablation", "A/B/C 三档检索消融", filename)
    variants = payload.get("variants")
    valid = (
        isinstance(variants, list)
        and len(variants) == 3
        and all(isinstance(item, dict) for item in variants)
        and [item.get("variant") for item in variants] == ["A", "B", "C"]
    )
    has_ci = bool(valid)
    sample = 0
    if valid and isinstance(variants, list):
        sample = min(int(item.get("sample_size", 0)) for item in variants)
        for item in variants:
            for metric in ("hit_at_5", "mrr_at_10", "ndcg_at_10"):
                estimate = item.get(metric)
                has_ci = has_ci and isinstance(estimate, dict) and all(
                    key in estimate for key in ("value", "ci95_low", "ci95_high")
                )
    passed = valid and sample >= 100 and has_ci
    return ReadinessGate(
        id="retrieval_ablation",
        title="A/B/C 三档检索消融",
        required=True,
        status="通过" if passed else "阻塞",
        artifact=filename,
        sample_size=sample,
        metrics={"variants": ["A", "B", "C"] if valid else [], "has_ci95": has_ci},
        reasons=[] if passed else ["必须包含 A/B/C、每档 n≥100 且所有效果指标带 95% 区间"],
    )


def _citation_gate(payload: dict[str, Any] | None, filename: str) -> ReadinessGate:
    if payload is None:
        return _missing_gate("citation_refusal", "引用、拒答与误拒答", filename)
    positive = int(_number(payload, "positive_sample_size") or 0)
    negative = int(_number(payload, "negative_sample_size") or 0)
    metric_values: dict[str, float] = {}
    has_ci = True
    for key in ("citation_location_accuracy", "correct_refusal_rate", "false_refusal_rate"):
        metric = payload.get(key)
        if not isinstance(metric, dict):
            has_ci = False
            continue
        value = _number(metric, "value")
        if value is None or _number(metric, "ci95_low") is None or _number(metric, "ci95_high") is None:
            has_ci = False
        else:
            metric_values[key] = value
    passed = (
        positive >= 100
        and negative >= 50
        and has_ci
        and metric_values.get("citation_location_accuracy", 0) >= 0.95
        and metric_values.get("correct_refusal_rate", 0) >= 0.98
        and metric_values.get("false_refusal_rate", 1) <= 0.05
    )
    return ReadinessGate(
        id="citation_refusal",
        title="引用、拒答与误拒答",
        required=True,
        status="通过" if passed else "阻塞",
        artifact=filename,
        sample_size=positive + negative,
        metrics={**metric_values, "has_ci95": has_ci},
        reasons=[] if passed else ["引用≥95%、正确拒答≥98%、误拒答≤5%，且样本量和区间须达标"],
    )


def _orchestration_gate(payload: dict[str, Any] | None, filename: str) -> ReadinessGate:
    if payload is None:
        return _missing_gate("orchestration", "工具编排与数字护栏", filename)
    evaluation = payload.get("guardrail_evaluation")
    usages = payload.get("real_usage_tests")
    if not isinstance(evaluation, dict) or not isinstance(usages, list):
        passed = False
        cases = blocked = 0
        kinds: set[str] = set()
    else:
        cases = int(_number(evaluation, "fabricated_number_cases") or 0)
        blocked = int(_number(evaluation, "blocked") or 0)
        kinds = {
            str(item.get("kind"))
            for item in usages
            if isinstance(item, dict) and item.get("result") == "passed"
        }
        passed = cases >= 100 and blocked == cases and {
            "real_cli_entry",
            "real_uvicorn_http_entry",
            "real_external_service",
        }.issubset(kinds)
    interval = wilson_interval(blocked, cases)
    return ReadinessGate(
        id="orchestration",
        title="工具编排与数字护栏",
        required=True,
        status="通过" if passed else "阻塞",
        artifact=filename,
        sample_size=cases,
        metrics={"blocked": blocked, "block_rate_ci95": interval, "real_usage_kinds": sorted(kinds)},
        reasons=[] if passed else ["要求 100 个伪造数字全拦截，并通过真实 CLI/API/DeepSeek 主链路"],
    )


def _certificate_gate(payload: dict[str, Any] | None, filename: str, smoke_exists: bool) -> ReadinessGate:
    if payload is None:
        gate = _missing_gate("certificate_fields", "奖状字段正式评测", filename)
        if smoke_exists:
            gate.status = "仅烟雾"
            gate.reasons.append("检测到 n=5 合成烟雾报告，但不能替代正式评测")
        return gate
    sample = int(_number(payload, "sample_size") or 0)
    micro = payload.get("micro")
    f1 = _number(micro, "f1") if isinstance(micro, dict) else None
    trigger = _number(payload, "vlm_trigger_rate")
    passed = (
        bool(payload.get("formal_gate_eligible"))
        and sample >= 30
        and f1 is not None
        and f1 >= 0.91
        and trigger is not None
        and trigger <= 0.15
    )
    return ReadinessGate(
        id="certificate_fields",
        title="奖状字段正式评测",
        required=True,
        status="通过" if passed else "阻塞",
        artifact=filename,
        sample_size=sample,
        metrics={"micro_f1": f1, "vlm_trigger_rate": trigger},
        reasons=[] if passed else ["要求 n≥30 独立真实脱敏图片、字段 F1≥91%、VLM 触发率≤15%"],
    )


def _dedup_gate(payload: dict[str, Any] | None, filename: str, smoke_exists: bool) -> ReadinessGate:
    if payload is None:
        gate = _missing_gate("evidence_dedup", "证据查重正式评测", filename)
        if smoke_exists:
            gate.status = "仅烟雾"
            gate.reasons.append("检测到 n=5 合成变换烟雾报告，但不能替代正式评测")
        return gate
    sample = int(_number(payload, "sample_size") or 0)
    recall = _number(payload, "recall") or 0.0
    precision = _number(payload, "precision") or 0.0
    has_ci = isinstance(payload.get("recall_ci95"), list) and isinstance(payload.get("precision_ci95"), list)
    passed = bool(payload.get("formal_gate_eligible")) and sample >= 50 and recall >= 0.96 and precision >= 0.95 and has_ci
    return ReadinessGate(
        id="evidence_dedup",
        title="证据查重正式评测",
        required=True,
        status="通过" if passed else "阻塞",
        artifact=filename,
        sample_size=sample,
        metrics={"recall": recall, "precision": precision, "has_ci95": has_ci},
        reasons=[] if passed else ["要求 n≥50 独立真实脱敏对、Recall≥96%、Precision≥95% 且双报区间"],
    )


def _simple_real_gate(
    payload: dict[str, Any] | None,
    *,
    gate_id: str,
    title: str,
    filename: str,
    validator: Any,
    failure_reason: str,
) -> ReadinessGate:
    if payload is None:
        return _missing_gate(gate_id, title, filename)
    passed, sample, metrics = validator(payload)
    return ReadinessGate(
        id=gate_id,
        title=title,
        required=True,
        status="通过" if passed else "阻塞",
        artifact=filename,
        sample_size=sample,
        metrics=metrics,
        reasons=[] if passed else [failure_reason],
    )


def _cost_summary(payloads: Mapping[str, dict[str, Any] | None]) -> CostSummary:
    embedding_queries = rerank_pairs = vlm_calls = 0
    retrieval = payloads.get("retrieval")
    if retrieval and isinstance(retrieval.get("variants"), list):
        for variant in retrieval["variants"]:
            if isinstance(variant, dict):
                embedding_queries += int(_number(variant, "embedding_query_count") or 0)
                rerank_pairs += int(_number(variant, "rerank_pair_count") or 0)
    citation = payloads.get("citation")
    embedding_queries += int(_number(citation, "embedding_query_count") or 0)
    certificate = payloads.get("certificate") or payloads.get("certificate_smoke")
    vlm_calls += int(_number(certificate, "vlm_called") or 0)
    observed = payloads.get("cost") or {}
    external_calls = _number(observed, "external_calls")
    by_purpose = observed.get("by_purpose")
    if isinstance(by_purpose, list):
        vision_calls = sum(
            int(_number(item, "external_calls") or 0)
            for item in by_purpose
            if isinstance(item, dict) and item.get("key") == "certificate_vlm_crop"
        )
        vlm_calls = max(vlm_calls, vision_calls)
        if external_calls is not None:
            external_calls = max(0, external_calls - vision_calls)
    monetary_available = bool(observed.get("monetary_cost_available"))
    notes = observed.get("limitations")
    normalized_notes = [str(item) for item in notes] if isinstance(notes, list) else []
    if not observed:
        normalized_notes.append("尚未生成统一 cost_events 汇总报告。")
    return CostSummary(
        embedding_queries=embedding_queries,
        rerank_pairs=rerank_pairs,
        vlm_calls=vlm_calls,
        external_text_calls=int(external_calls) if external_calls is not None else None,
        input_tokens=(
            int(value) if (value := _number(observed, "input_tokens")) is not None else None
        ),
        output_tokens=(
            int(value) if (value := _number(observed, "output_tokens")) is not None else None
        ),
        total_tokens=(
            int(value) if (value := _number(observed, "total_tokens")) is not None else None
        ),
        usage_coverage_rate=_number(observed, "usage_coverage_rate"),
        monetary_cost_available=monetary_available,
        monetary_cost_cny=_number(observed, "monetary_cost_cny"),
        cost_per_100_materials_cny=_number(observed, "cost_per_100_materials_cny"),
        cost_per_subject_cny=_number(observed, "cost_per_subject_cny"),
        notes=normalized_notes,
    )


def build_release_readiness(
    report_dir: str | Path,
    *,
    candidate_version: str,
    generated_at: str | None = None,
) -> ReleaseReadinessReport:
    """读取固定文件名的评测产物，生成不可把烟雾结果冒充正式结果的 RC 门禁。"""
    directory = Path(report_dir).resolve()
    payloads: dict[str, dict[str, Any] | None] = {}
    snapshots: list[ArtifactSnapshot] = []
    filenames = dict(_ARTIFACTS)
    for name, filename in _ARTIFACTS:
        payload, snapshot = _read_artifact(directory / filename)
        snapshot.name = name
        snapshot.path = filename
        payloads[name] = payload
        snapshots.append(snapshot)

    gates = [
        _quality_gate(payloads["quality"], filenames["quality"], candidate_version),
        _simple_real_gate(
            payloads["complex_pdf"],
            gate_id="complex_pdf",
            title="复杂 PDF 版面回归",
            filename=filenames["complex_pdf"],
            validator=lambda value: (
                bool(value.get("passed")) and int(_number(value, "sample_size") or 0) >= 30,
                int(_number(value, "sample_size") or 0),
                {"passed": bool(value.get("passed"))},
            ),
            failure_reason="要求多栏、跨页表格等每类至少 10 份且总体通过",
        ),
        _gateway_gate(payloads["gateway"], filenames["gateway"]),
        _retrieval_gate(payloads["retrieval"], filenames["retrieval"]),
        _citation_gate(payloads["citation"], filenames["citation"]),
        _orchestration_gate(payloads["orchestration"], filenames["orchestration"]),
        _certificate_gate(payloads["certificate"], filenames["certificate"], payloads["certificate_smoke"] is not None),
        _simple_real_gate(
            payloads["vlm"],
            gate_id="vlm_integration",
            title="真实视觉模型局部兜底",
            filename=filenames["vlm"],
            validator=lambda value: (
                value.get("result") == "passed" and bool(value.get("real_external_service")),
                int(_number(value, "sample_size") or 0),
                {"provider": value.get("provider"), "real_external_service": value.get("real_external_service")},
            ),
            failure_reason="必须使用受支持 VLM Key 完成必要裁剪区域的真实外部调用",
        ),
        _dedup_gate(payloads["dedup"], filenames["dedup"], payloads["dedup_smoke"] is not None),
        _simple_real_gate(
            payloads["backtest"],
            gate_id="backtest_52",
            title="52 人逐项回测与端到端计时",
            filename=filenames["backtest"],
            validator=lambda value: (
                int(_number(value, "total_students") or 0) == 52
                and bool(value.get("data_complete"))
                and bool(value.get("gate_passed"))
                and isinstance(value.get("meta"), dict)
                and _number(value["meta"], "elapsed_seconds") is not None,
                int(_number(value, "total_students") or 0),
                {
                    "data_complete": bool(value.get("data_complete")),
                    "elapsed_seconds": _number(value.get("meta"), "elapsed_seconds") if isinstance(value.get("meta"), dict) else None,
                },
            ),
            failure_reason="要求 52 人数据完整门禁通过并记录真实端到端耗时",
        ),
        _simple_real_gate(
            payloads["user_trial"],
            gate_id="user_trial",
            title="真实用户试用",
            filename=filenames["user_trial"],
            validator=lambda value: (
                bool(value.get("real_users")) and int(_number(value, "sample_size") or 0) > 0,
                int(_number(value, "sample_size") or 0),
                {"real_users": bool(value.get("real_users"))},
            ),
            failure_reason="至少需要一次有记录、已授权的真实用户试用",
        ),
    ]
    cost = _cost_summary(payloads)
    gates.append(
        ReadinessGate(
            id="cost_observability",
            title="成本指标汇总",
            required=False,
            status=(
                "通过"
                if cost.monetary_cost_available
                and cost.cost_per_100_materials_cny is not None
                and cost.cost_per_subject_cny is not None
                else "警告"
            ),
            metrics=cost.model_dump(mode="json"),
            reasons=cost.notes,
        )
    )
    blocking = [item.id for item in gates if item.required and item.status != "通过"]
    warnings = [item.id for item in gates if item.status == "警告"]
    manifest_source = {
        "candidate_version": candidate_version,
        "artifacts": {item.name: item.sha256 for item in snapshots},
        "gate_policy": "scoreproof-release-readiness-v1",
    }
    manifest_hash = hashlib.sha256(
        json.dumps(manifest_source, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    ready = not blocking
    return ReleaseReadinessReport(
        generated_at=generated_at or datetime.now(UTC).isoformat(),
        candidate_version=candidate_version,
        manifest_hash=manifest_hash,
        status="可以发布候选" if ready else "阻塞",
        ready=ready,
        report_dir=directory.name,
        artifacts=snapshots,
        gates=gates,
        blocking_gate_ids=blocking,
        warning_gate_ids=warnings,
        cost_summary=cost,
        limitations=[
            "该审计只认可固定正式报告文件名；smoke 报告永远不能使正式门禁通过。",
            "阶段 5、6 的缺失真实数据或外部服务结果会保持阻塞，不因进入阶段 7 而豁免。",
        ],
    )


__all__ = [
    "ArtifactSnapshot",
    "CostSummary",
    "QualityCommandResult",
    "QualityGateReport",
    "ReadinessGate",
    "ReleaseReadinessReport",
    "build_release_readiness",
    "run_quality_gates",
]

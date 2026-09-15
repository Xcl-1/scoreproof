"""奖状字段评测：micro-F1、逐字段 F1、整证正确率与 VLM 触发率。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..evidence.certificate import CERTIFICATE_FIELD_NAMES
from ..normalize import parse_prize, parse_tier
from .gateway import wilson_interval


class FieldMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


class CertificateEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version: str
    config_hash: str
    models: list[str]
    confidence_thresholds: list[float]
    sample_size: int
    synthetic: bool
    smoke_test_only: bool
    formal_gate_eligible: bool
    micro: FieldMetric
    per_field: dict[str, FieldMetric]
    raw_micro: FieldMetric | None = None
    raw_per_field: dict[str, FieldMetric] = Field(default_factory=dict)
    raw_labeled_values: int = 0
    raw_complete_samples: int = 0
    raw_exact_certificates: int = 0
    raw_exact_certificate_rate: float | None = None
    exact_certificates: int
    exact_certificate_rate: float
    exact_certificate_ci95: tuple[float, float]
    vlm_triggered: int
    vlm_trigger_rate: float
    vlm_trigger_ci95: tuple[float, float]
    vlm_called: int
    vlm_call_rate: float
    notes: list[str] = Field(default_factory=list)


def _metric(tp: int, fp: int, fn: int) -> FieldMetric:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return FieldMetric(
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
    )


def _expected_fields(label: Mapping[str, Any]) -> dict[str, str | None]:
    level = str(label.get("level") or "")
    team_value = label.get("team_attribute")
    if team_value is None and "is_team" in label:
        team_value = "团队" if bool(label["is_team"]) else "个人"
    return {
        "姓名": _text(label.get("name")),
        "赛事名称": _text(label.get("event_name")),
        "级别": _text(label.get("tier")) or parse_tier(level),
        "奖项/名次": _text(label.get("award")) or parse_prize(level),
        "获奖日期": _text(label.get("award_date")),
        "颁发单位": _text(label.get("issuer")),
        "团队属性": _text(team_value),
    }


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _expected_raw_fields(label: Mapping[str, Any]) -> dict[str, str | None]:
    raw_fields = label.get("raw_fields")
    if isinstance(raw_fields, Mapping):
        return {name: _text(raw_fields.get(name)) for name in CERTIFICATE_FIELD_NAMES}
    keys = {
        "姓名": "raw_name",
        "赛事名称": "raw_event_name",
        "级别": "raw_tier",
        "奖项/名次": "raw_award",
        "获奖日期": "raw_award_date",
        "颁发单位": "raw_issuer",
        "团队属性": "raw_team_attribute",
    }
    return {name: _text(label.get(key)) for name, key in keys.items()}


def _prediction_fields(
    prediction: Mapping[str, Any], *, value_key: str = "normalized_value"
) -> dict[str, str | None]:
    extraction = prediction.get("extraction", prediction)
    if not isinstance(extraction, Mapping):
        return {name: None for name in CERTIFICATE_FIELD_NAMES}
    fields = extraction.get("fields", {})
    if not isinstance(fields, Mapping):
        return {name: None for name in CERTIFICATE_FIELD_NAMES}
    result: dict[str, str | None] = {}
    for name in CERTIFICATE_FIELD_NAMES:
        item = fields.get(name)
        if isinstance(item, Mapping):
            result[name] = _text(item.get(value_key))
        else:
            result[name] = _text(item)
    return result


def _vlm(prediction: Mapping[str, Any]) -> tuple[bool, bool]:
    extraction = prediction.get("extraction", prediction)
    if not isinstance(extraction, Mapping):
        return False, False
    payload = extraction.get("vlm", {})
    if not isinstance(payload, Mapping):
        return False, False
    return bool(payload.get("requested")), bool(payload.get("called"))


def evaluate_certificate_fields(
    labels: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
    *,
    dataset_version: str = "unversioned",
) -> CertificateEvaluationReport:
    gold = list(labels)
    predicted = list(predictions)
    if not gold:
        raise ValueError("字段评测集不能为空")
    if len(gold) != len(predicted):
        raise ValueError(f"标签与预测数量不一致：{len(gold)} != {len(predicted)}")
    label_ids = [str(item.get("evidence_id") or index) for index, item in enumerate(gold)]
    prediction_by_id = {
        str(item.get("evidence_id")): item
        for item in predicted
        if item.get("evidence_id") is not None
    }
    if prediction_by_id:
        missing = [item_id for item_id in label_ids if item_id not in prediction_by_id]
        if missing:
            raise ValueError(f"预测缺少 evidence_id：{missing}")
        predicted = [prediction_by_id[item_id] for item_id in label_ids]

    counts = {name: [0, 0, 0] for name in CERTIFICATE_FIELD_NAMES}
    raw_counts = {name: [0, 0, 0] for name in CERTIFICATE_FIELD_NAMES}
    raw_labeled = raw_complete = raw_exact = 0
    exact = triggered = called = 0
    for label, prediction in zip(gold, predicted, strict=True):
        expected = _expected_fields(label)
        actual = _prediction_fields(prediction)
        expected_raw = _expected_raw_fields(label)
        actual_raw = _prediction_fields(prediction, value_key="raw_value")
        sample_exact = True
        for name in CERTIFICATE_FIELD_NAMES:
            want, got = expected[name], actual[name]
            if want == got:
                if want is not None:
                    counts[name][0] += 1
            else:
                sample_exact = False
                if got is not None:
                    counts[name][1] += 1
                if want is not None:
                    counts[name][2] += 1
        exact += int(sample_exact)
        available_raw = [name for name, value in expected_raw.items() if value is not None]
        raw_labeled += len(available_raw)
        raw_sample_exact = bool(available_raw)
        for name in available_raw:
            want_raw, got_raw = expected_raw[name], actual_raw[name]
            if want_raw == got_raw:
                raw_counts[name][0] += 1
            else:
                raw_sample_exact = False
                if got_raw is not None:
                    raw_counts[name][1] += 1
                raw_counts[name][2] += 1
        if len(available_raw) == len(CERTIFICATE_FIELD_NAMES):
            raw_complete += 1
            raw_exact += int(raw_sample_exact)
        vlm_requested, vlm_called = _vlm(prediction)
        triggered += int(vlm_requested)
        called += int(vlm_called)

    per_field = {name: _metric(*values) for name, values in counts.items()}
    totals = [sum(values[index] for values in counts.values()) for index in range(3)]
    raw_per_field = {
        name: _metric(*values)
        for name, values in raw_counts.items()
        if sum(values) > 0
    }
    raw_totals = [sum(values[index] for values in raw_counts.values()) for index in range(3)]
    n = len(gold)
    synthetic = any(bool(item.get("synthetic")) for item in gold)
    smoke_only = synthetic or n < 30
    models = sorted(
        {
            str(extraction.get("model"))
            for item in predicted
            if isinstance((extraction := item.get("extraction")), Mapping)
            and extraction.get("model")
        }
    )
    thresholds = sorted(
        {
            float(extraction["confidence_threshold"])
            for item in predicted
            if isinstance((extraction := item.get("extraction")), Mapping)
            and isinstance(extraction.get("confidence_threshold"), (int, float))
        }
    )
    raw_config = {
        "dataset_version": dataset_version,
        "field_names": CERTIFICATE_FIELD_NAMES,
        "normalization": "certificate-eval-v1",
        "confidence_policy": "certificate-multisignal-v1",
        "models": models,
        "confidence_thresholds": thresholds,
    }
    config_hash = hashlib.sha256(
        json.dumps(raw_config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    notes: list[str] = []
    if smoke_only:
        notes.append("仅烟雾测试：合成数据或样本量小于正式字段评测要求 n≥30。")
    if synthetic:
        notes.append("数据含合成图片，不得作为正式业务字段 F1 验收或简历数字。")
    if raw_labeled < n * len(CERTIFICATE_FIELD_NAMES):
        notes.append("标签未完整提供 raw_fields，原始值 F1 仅评测已标注字段；本次为空时报告 null。")
    return CertificateEvaluationReport(
        dataset_version=dataset_version,
        config_hash=config_hash,
        models=models,
        confidence_thresholds=thresholds,
        sample_size=n,
        synthetic=synthetic,
        smoke_test_only=smoke_only,
        formal_gate_eligible=(
            not smoke_only
            and n >= 30
            and raw_labeled == n * len(CERTIFICATE_FIELD_NAMES)
        ),
        micro=_metric(*totals),
        per_field=per_field,
        raw_micro=_metric(*raw_totals) if raw_labeled else None,
        raw_per_field=raw_per_field,
        raw_labeled_values=raw_labeled,
        raw_complete_samples=raw_complete,
        raw_exact_certificates=raw_exact,
        raw_exact_certificate_rate=(round(raw_exact / raw_complete, 4) if raw_complete else None),
        exact_certificates=exact,
        exact_certificate_rate=round(exact / n, 4) if n else 0.0,
        exact_certificate_ci95=wilson_interval(exact, n),
        vlm_triggered=triggered,
        vlm_trigger_rate=round(triggered / n, 4) if n else 0.0,
        vlm_trigger_ci95=wilson_interval(triggered, n),
        vlm_called=called,
        vlm_call_rate=round(called / n, 4) if n else 0.0,
        notes=notes,
    )


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not raw.strip():
            continue
        item = json.loads(raw)
        if not isinstance(item, dict):
            raise ValueError(f"{source}:{number} 必须是 JSON 对象")
        rows.append(item)
    return rows


__all__ = [
    "CertificateEvaluationReport",
    "FieldMetric",
    "evaluate_certificate_fields",
    "load_jsonl",
]

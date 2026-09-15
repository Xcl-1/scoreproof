"""阶段 6.3：证据联合查重、成对评测与申报一致性。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from scoreproof.cli import app
from scoreproof.eval.dedup import DedupPairCase, evaluate_dedup_pairs, load_dedup_dataset
from scoreproof.evidence.consistency import ConsistencyPolicy, compare_claim_evidence
from scoreproof.evidence.dedup import (
    DuplicateThresholds,
    compare_evidence,
    fact_fingerprint,
    file_sha256,
    phash_distance,
)
from scoreproof.schema import Claim, Evidence


def _fields(*, name: str = "张三", event: str = "数学建模竞赛") -> dict[str, str]:
    return {
        "姓名": name,
        "赛事名称": event,
        "级别": "省级",
        "奖项/名次": "二等奖",
        "获奖日期": "2025-10-02",
        "颁发单位": "省竞赛组委会",
        "团队属性": "团队",
    }


def _evidence(
    evidence_id: str,
    *,
    fields: dict[str, str] | None = None,
    phash: str | None = None,
    path: Path | None = None,
) -> Evidence:
    return Evidence(
        id=evidence_id,
        type="image",
        path=str(path) if path else None,
        fields=fields or _fields(),
        phash=phash,
    )


class TestDedupDecision:
    def test_exact_file_hash_is_definite_and_never_auto_deletes(self, tmp_path: Path) -> None:
        image = tmp_path / "certificate.bin"
        image.write_bytes(b"same-real-file")
        left = _evidence("left", path=image)
        right = _evidence("right", path=image, fields=_fields(name="OCR误识别"))

        decision = compare_evidence(left, right)

        assert decision.status == "确定重复"
        assert decision.action == "拦截并人工复核"
        assert decision.sha256_match is True
        assert any("字段存在冲突" in reason for reason in decision.reasons)
        assert file_sha256(image) == decision.left_sha256

    def test_near_phash_with_matching_facts_is_definite(self) -> None:
        decision = compare_evidence(
            _evidence("left", phash="0000000000000000"),
            _evidence("right", phash="0000000000000003"),
        )
        assert decision.status == "确定重复"
        assert decision.phash_distance == 2

    def test_critical_fact_conflict_is_hard_negative_even_with_close_phash(self) -> None:
        decision = compare_evidence(
            _evidence("left", phash="0000000000000000"),
            _evidence("right", fields=_fields(name="李四"), phash="0000000000000000"),
        )
        assert decision.status == "非重复"
        assert decision.phash_distance == 0
        assert "姓名" in decision.reasons[0]

    def test_same_fact_different_image_is_detected(self) -> None:
        decision = compare_evidence(
            _evidence("left", phash="0000000000000000"),
            _evidence("right", phash="ffffffffffffffff"),
        )
        assert decision.status == "确定重复"
        assert decision.fact_fingerprint_match is True
        assert decision.semantic_similarity == 1.0

    def test_close_image_without_fields_is_only_suspected(self) -> None:
        left = Evidence(id="left", type="image", fields={}, phash="0000000000000000")
        right = Evidence(id="right", type="image", fields={}, phash="000000000000000f")
        decision = compare_evidence(left, right)
        assert decision.status == "疑似重复"
        assert decision.action == "送人工复核"
        assert decision.comparable_fields == 0

    def test_invalid_or_missing_signals_do_not_false_match(self) -> None:
        left = Evidence(id="left", type="image", fields={}, phash="invalid")
        right = Evidence(id="right", type="image", fields={}, phash="invalid")
        decision = compare_evidence(left, right)
        assert decision.status == "非重复"
        assert decision.phash_distance is None
        assert fact_fingerprint(left) is None

    def test_phash_distance_and_threshold_validation(self) -> None:
        assert phash_distance("0f", "00") == 4
        assert phash_distance("0", "00") is None
        assert phash_distance("zz", "00") is None
        with pytest.raises(ValueError, match="不能大于"):
            compare_evidence(
                _evidence("left"),
                _evidence("right"),
                thresholds=DuplicateThresholds(phash_definite_max=11, phash_suspected_max=10),
            )

    def test_threshold_schema_forbids_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            DuplicateThresholds.model_validate({"phash_definite_max": 2, "unknown": 1})


class TestConsistency:
    def test_aliases_policy_and_all_required_business_checks(self) -> None:
        claim = Claim(
            id="claim-1",
            student_id="2025001",
            student_name="张三",
            academic_year="2025-2026",
            category="学科竞赛",
            raw_text="省级第二名",
            level="省级二等奖",
            team=True,
            catalog_listed=True,
            extra={"event_name": "全国大学生数学建模竞赛"},
        )
        evidence = _evidence(
            "evidence-1",
            fields={
                **_fields(event="国赛数模"),
                "级别": "省部级",
                "奖项/名次": "第二名",
                "申报类别": "竞赛",
            },
        )
        policy = ConsistencyPolicy(
            event_aliases={"国赛数模": "全国大学生数学建模竞赛"},
            category_aliases={"竞赛": "学科竞赛"},
            allowed_issuers=["省竞赛组委会"],
            catalog_events=["全国大学生数学建模竞赛"],
        )

        report = compare_claim_evidence(claim, evidence, policy=policy)

        assert report.status == "通过"
        assert report.requires_review is False
        assert report.mismatch_fields == []
        assert report.insufficient_fields == []
        statuses = {field.field: field.status for field in report.fields}
        assert statuses["赛事名称"] == "归一化一致"
        assert statuses["获奖日期/学年"] == "归一化一致"
        assert statuses["目录赛事"] == "归一化一致"

    def test_mismatch_is_not_overridden_by_other_matches(self) -> None:
        claim = Claim(
            id="claim-1",
            student_id="2025001",
            student_name="李四",
            academic_year="2025-2026",
            raw_text="省级二等奖",
        )
        report = compare_claim_evidence(claim, _evidence("evidence-1"))
        assert report.status == "不一致"
        assert "姓名" in report.mismatch_fields

    def test_missing_policy_and_fields_require_manual_review(self) -> None:
        claim = Claim(id="claim-1", student_id="2025001")
        evidence = Evidence(id="evidence-1", type="image")
        report = compare_claim_evidence(claim, evidence)
        assert report.status == "需人工复核"
        assert report.requires_review is True
        assert "赛事名称" in report.insufficient_fields
        assert "目录赛事" in report.insufficient_fields


class TestDedupEvaluationAndCli:
    def test_small_synthetic_report_is_explicit_smoke_only(self) -> None:
        cases = [
            DedupPairCase(
                id="positive",
                duplicate=True,
                left=_evidence("left", phash="0000000000000000"),
                right=_evidence("right", phash="0000000000000001"),
                transformation="压缩",
            ),
            DedupPairCase(
                id="negative",
                duplicate=False,
                left=_evidence("left-2", phash="0000000000000000"),
                right=_evidence("right-2", fields=_fields(name="李四"), phash="0000000000000000"),
                hard_negative="同赛事不同人",
            ),
        ]
        report = evaluate_dedup_pairs(
            cases,
            dataset_version="synthetic-smoke-v1",
            independent_real_pairs=False,
        )
        assert report.recall == 1.0
        assert report.precision == 1.0
        assert report.true_positive == 1
        assert report.true_negative == 1
        assert report.smoke_test_only is True
        assert report.formal_gate_eligible is False
        assert report.target_passed is False
        assert any("n=2 < 50" in note for note in report.notes)

    def test_dataset_loader_and_eval_cli(self, tmp_path: Path) -> None:
        dataset = tmp_path / "pairs.json"
        output = tmp_path / "report.json"
        payload = {
            "dataset_version": "smoke-v1",
            "independent_real_pairs": False,
            "pairs": [
                {
                    "id": "p1",
                    "duplicate": True,
                    "left": _evidence("left", phash="0000000000000000").model_dump(mode="json"),
                    "right": _evidence("right", phash="0000000000000001").model_dump(mode="json"),
                    "transformation": "压缩",
                }
            ],
        }
        dataset.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        version, independent, cases = load_dedup_dataset(dataset)
        assert (version, independent, len(cases)) == ("smoke-v1", False, 1)

        result = CliRunner().invoke(app, ["eval-evidence-dedup", str(dataset), "--out", str(output)])
        assert result.exit_code == 0, result.output
        saved = json.loads(output.read_text(encoding="utf-8"))
        assert saved["smoke_test_only"] is True
        assert saved["sample_size"] == 1

    def test_real_file_compare_and_consistency_cli(self, tmp_path: Path) -> None:
        real_file = tmp_path / "certificate.png"
        pytest.importorskip("PIL.Image").new("RGB", (800, 500), "white").save(real_file)
        compare_out = tmp_path / "compare.json"
        compared = CliRunner().invoke(
            app,
            ["compare-evidence", str(real_file), str(real_file), "--out", str(compare_out)],
        )
        assert compared.exit_code == 0, compared.output
        assert json.loads(compare_out.read_text(encoding="utf-8"))["sha256_match"] is True

        claim = tmp_path / "claim.json"
        evidence = tmp_path / "evidence.json"
        consistency_out = tmp_path / "consistency.json"
        claim.write_text(
            Claim(student_id="1", student_name="张三", raw_text="省级二等奖").model_dump_json(),
            encoding="utf-8",
        )
        evidence.write_text(_evidence("ev").model_dump_json(), encoding="utf-8")
        checked = CliRunner().invoke(
            app,
            [
                "check-evidence-consistency",
                str(claim),
                str(evidence),
                "--out",
                str(consistency_out),
            ],
        )
        assert checked.exit_code == 0, checked.output
        assert json.loads(consistency_out.read_text(encoding="utf-8"))["requires_review"] is True

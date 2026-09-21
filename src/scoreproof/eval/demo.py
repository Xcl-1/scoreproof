"""阶段 7 一键演示：用合成文件复跑规则导入、核算与回测主链路。"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DemoStep(BaseModel):
    """一个真实子进程步骤；不保存可能包含业务文本的标准输出。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    status: Literal["passed", "failed"]
    exit_code: int
    duration_seconds: float = Field(ge=0.0)


class DemoArtifact(BaseModel):
    """演示产物的可复核摘要。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    relative_path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("relative_path")
    @classmethod
    def _relative_path_is_safe(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("relative_path 必须是演示目录内的安全相对路径")
        return value


class DemoReport(BaseModel):
    """一键演示报告；它永远不能冒充正式业务验收报告。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "1.0"
    generated_at: str
    dataset_kind: Literal["synthetic"] = "synthetic"
    smoke_test_only: Literal[True] = True
    formal_gate_eligible: Literal[False] = False
    all_steps_passed: bool
    business_checks_passed: bool
    sample_students: int = Field(ge=0)
    sample_claims: int = Field(ge=0)
    imported_rules: int = Field(ge=0)
    person_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    item_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    steps: list[DemoStep]
    artifacts: list[DemoArtifact]
    limitations: list[str]

    @field_validator("generated_at")
    @classmethod
    def _generated_at_has_timezone(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("generated_at 必须是 ISO 8601 时间") from exc
        if parsed.utcoffset() is None:
            raise ValueError("generated_at 必须包含时区")
        return value

    @model_validator(mode="after")
    def _check_unique_entries_and_summary(self) -> Self:
        step_ids = [step.id for step in self.steps]
        artifact_names = [artifact.name for artifact in self.artifacts]
        artifact_paths = [artifact.relative_path for artifact in self.artifacts]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("演示步骤 id 不得重复")
        if len(artifact_names) != len(set(artifact_names)):
            raise ValueError("演示产物 name 不得重复")
        if len(artifact_paths) != len(set(artifact_paths)):
            raise ValueError("演示产物 relative_path 不得重复")
        if self.all_steps_passed != all(step.status == "passed" for step in self.steps):
            raise ValueError("all_steps_passed 与逐步骤状态不一致")
        return self


class DemoRunError(RuntimeError):
    """演示步骤失败，并附带可安全展示的有限输出。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_step(
    step_id: str,
    command: list[str],
    *,
    root: Path,
    environment: dict[str, str],
) -> tuple[DemoStep, str]:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            check=False,
        )
        output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        exit_code = result.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        output = str(exc)
        exit_code = 124 if isinstance(exc, subprocess.TimeoutExpired) else 127
    return (
        DemoStep(
            id=step_id,
            status="passed" if exit_code == 0 else "failed",
            exit_code=exit_code,
            duration_seconds=round(time.perf_counter() - started, 3),
        ),
        output,
    )


def _artifact(name: str, path: Path, *, output_dir: Path) -> DemoArtifact:
    return DemoArtifact(
        name=name,
        relative_path=path.relative_to(output_dir).as_posix(),
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def run_smoke_demo(project_root: str | Path, output_dir: str | Path) -> DemoReport:
    """在空目录中通过真实 CLI 子进程复跑合成演示链路。"""

    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"演示输出目录必须为空，拒绝覆盖已有文件：{destination}")
    destination.mkdir(parents=True, exist_ok=True)

    sample_dir = destination / "sample"
    database = destination / "rules.sqlite"
    rules_json = destination / "rules.json"
    calculation = destination / "calculation.json"
    backtest_report = destination / "backtest.json"
    backtest_diffs = destination / "backtest-diffs.csv"
    python = sys.executable
    commands = [
        (
            "generate_sample_files",
            [python, str(root / "scripts" / "make_sample_data.py"), "--out", str(sample_dir)],
        ),
        (
            "import_rules",
            [
                python,
                "-m",
                "scoreproof.cli",
                "import-rules",
                str(sample_dir / "rules_sample.xlsx"),
                "--year",
                "2025-2026",
                "--db",
                str(database),
                "--export",
                str(rules_json),
            ],
        ),
        (
            "calculate_scores",
            [
                python,
                "-m",
                "scoreproof.cli",
                "calc",
                str(sample_dir / "claims_sample.xlsx"),
                "--year",
                "2025-2026",
                "--db",
                str(database),
                "--out",
                str(calculation),
                "--quiet",
            ],
        ),
        (
            "backtest_items",
            [
                python,
                "-m",
                "scoreproof.cli",
                "backtest",
                str(sample_dir / "claims_sample.xlsx"),
                "--truth",
                str(sample_dir / "truth_sample.xlsx"),
                "--item-reference",
                str(sample_dir / "item_reference_sample.csv"),
                "--mode",
                "historical-reference",
                "--required-students",
                "5",
                "--year",
                "2025-2026",
                "--db",
                str(database),
                "--out",
                str(backtest_report),
                "--diff-out",
                str(backtest_diffs),
            ],
        ),
        (
            "explain_claim",
            [
                python,
                "-m",
                "scoreproof.cli",
                "explain",
                "省二等奖",
                "--category",
                "学科竞赛",
                "--year",
                "2025-2026",
                "--db",
                str(database),
            ],
        ),
    ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"

    steps: list[DemoStep] = []
    for step_id, command in commands:
        step, output = _run_step(step_id, command, root=root, environment=environment)
        steps.append(step)
        if step.status == "failed":
            safe_tail = output[-2000:]
            raise DemoRunError(f"演示步骤 {step_id} 失败（exit={step.exit_code}）：\n{safe_tail}")

    raw_backtest: dict[str, Any] = json.loads(backtest_report.read_text(encoding="utf-8"))
    raw_calculation: dict[str, Any] = json.loads(calculation.read_text(encoding="utf-8"))
    raw_rules: dict[str, Any] = json.loads(rules_json.read_text(encoding="utf-8"))
    summary = raw_backtest
    sample_students = int(summary.get("total_students", 0))
    sample_claims = int(summary.get("total_items", 0))
    imported_rules = len(raw_rules.get("rules", []))
    person_agreement = summary.get("person_agreement")
    item_agreement = summary.get("item_agreement")
    business_checks_passed = (
        sample_students == 5
        and sample_claims == 13
        and imported_rules == 13
        and len(raw_calculation) == 5
        and raw_backtest.get("gate_passed") is True
    )

    artifact_paths = {
        "rules_input": sample_dir / "rules_sample.xlsx",
        "claims_input": sample_dir / "claims_sample.xlsx",
        "totals_reference": sample_dir / "truth_sample.xlsx",
        "items_reference": sample_dir / "item_reference_sample.csv",
        "rules_database": database,
        "rules_export": rules_json,
        "calculation": calculation,
        "backtest": backtest_report,
        "backtest_diffs": backtest_diffs,
    }
    report = DemoReport(
        generated_at=datetime.now(UTC).isoformat(),
        all_steps_passed=all(step.status == "passed" for step in steps),
        business_checks_passed=business_checks_passed,
        sample_students=sample_students,
        sample_claims=sample_claims,
        imported_rules=imported_rules,
        person_agreement=person_agreement,
        item_agreement=item_agreement,
        steps=steps,
        artifacts=[
            _artifact(name, path, output_dir=destination)
            for name, path in artifact_paths.items()
        ],
        limitations=[
            "全部输入均由 scripts/make_sample_data.py 合成，不含真实学生或业务材料。",
            "该报告只验证一键演示、真实文件、CLI、SQLite、确定性核算与回测链路。",
            "该报告不能替代 52 人回测、真实奖状/VLM、查重或真实用户试用验收。",
        ],
    )
    if not business_checks_passed:
        raise DemoRunError("演示命令均成功，但业务产物完整性校验未通过")
    return report


__all__ = [
    "DemoArtifact",
    "DemoReport",
    "DemoRunError",
    "DemoStep",
    "run_smoke_demo",
]

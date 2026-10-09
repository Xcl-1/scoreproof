"""真实视觉模型必要裁剪集成报告。"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VlmIntegrationReport(BaseModel):
    """只证明真实外部视觉调用链路，不充当字段效果评测。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    generated_at: datetime
    result: Literal["passed", "failed"]
    provider: Literal["qwen-vl-plus", "glm-4v"]
    model: str
    sample_size: int = Field(ge=1)
    real_external_service: bool
    dataset_kind: Literal["synthetic", "public", "authorized_real"]
    smoke_test_only: bool
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_fields: list[str] = Field(min_length=1)
    returned_fields: list[str]
    crop_sha256: list[str] = Field(min_length=1)
    whole_image_sent: Literal[False] = False
    strict_schema_passed: bool
    expected_match: bool | None = None
    response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    cost_event_recorded: bool
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_passed_report(self) -> VlmIntegrationReport:
        if self.result == "passed" and not (
                self.real_external_service
                and self.strict_schema_passed
                and self.cost_event_recorded
                and set(self.requested_fields) == set(self.returned_fields)
                and self.response_sha256
                and self.total_tokens is not None
        ):
            raise ValueError("通过的 VLM 集成报告必须具备真实调用、严格校验和成本事件")
        if self.smoke_test_only != (self.dataset_kind != "authorized_real"):
            raise ValueError("smoke_test_only 必须与数据来源类型一致")
        return self


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_vlm_integration_report(
    *,
    provider: str,
    model: str,
    source: str | Path,
    crop_paths: list[Path],
    requested_fields: list[str],
    response: dict[str, str | None],
    dataset_kind: Literal["synthetic", "public", "authorized_real"],
    expected: dict[str, str | None] | None,
    cost_event_recorded: bool,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
) -> VlmIntegrationReport:
    normalized_provider = provider.strip().lower()
    if normalized_provider not in {"qwen-vl-plus", "glm-4v"}:
        raise ValueError("不支持的 VLM provider")
    canonical = "\n".join(f"{key}={response[key] or ''}" for key in sorted(response))
    expected_match = response == expected if expected is not None else None
    passed = set(response) == set(requested_fields) and (expected_match is not False)
    limitations = ["该报告只验证真实 VLM 必要裁剪调用，不替代 n≥30 奖状字段正式评测。"]
    if dataset_kind != "authorized_real":
        limitations.append("输入不是已授权真实业务奖状，只能作为工程集成烟雾测试。")
    return VlmIntegrationReport(
        generated_at=datetime.now(UTC),
        result="passed" if passed and cost_event_recorded else "failed",
        provider=normalized_provider,
        model=model,
        sample_size=1,
        real_external_service=True,
        dataset_kind=dataset_kind,
        smoke_test_only=dataset_kind != "authorized_real",
        source_sha256=sha256_file(source),
        requested_fields=requested_fields,
        returned_fields=sorted(response),
        crop_sha256=[sha256_file(path) for path in crop_paths],
        strict_schema_passed=set(response) == set(requested_fields),
        expected_match=expected_match,
        response_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        cost_event_recorded=cost_event_recorded,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        limitations=limitations,
    )


__all__ = ["VlmIntegrationReport", "build_vlm_integration_report", "sha256_file"]

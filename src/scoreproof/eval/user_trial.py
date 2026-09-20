"""真实用户试用记录、隐私门禁与效率指标。

输入只允许无业务内容的结构化元数据。姓名、学号、申报文本、材料路径和自由文本
备注都不在 schema 中，因此会被 Pydantic 的 ``extra=forbid`` 拒绝。
"""

from __future__ import annotations

import hashlib
import math
import statistics
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .gateway import wilson_interval

TrialEntrypoint = Literal["cli", "api"]
TrialTask = Literal[
    "rule_query",
    "score_calculation",
    "certificate_extraction",
    "evidence_review",
]
IssueCode = Literal[
    "none",
    "input_error",
    "rule_not_found",
    "manual_review",
    "service_error",
    "result_mismatch",
]


class TrialSession(BaseModel):
    """一次试用会话；不允许任何姓名、学号、原文或自由文本。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    trial_id: str = Field(pattern=r"^trial-[0-9a-f]{12,32}$")
    participant_token: str = Field(pattern=r"^participant-[0-9a-f]{24,64}$")
    consent_confirmed: bool
    real_user_confirmed: bool
    synthetic: bool
    entrypoint: TrialEntrypoint
    task: TrialTask
    started_at: datetime
    completed_at: datetime
    task_completed: bool
    result_verified: bool
    manual_review_required: bool
    issue_code: IssueCode = "none"
    satisfaction_score: int | None = Field(default=None, ge=1, le=5)

    @model_validator(mode="after")
    def validate_session(self) -> TrialSession:
        if self.started_at.tzinfo is None or self.completed_at.tzinfo is None:
            raise ValueError("试用时间必须包含时区")
        if self.completed_at <= self.started_at:
            raise ValueError("completed_at 必须晚于 started_at")
        if self.issue_code == "none" and not self.task_completed:
            raise ValueError("任务未完成时必须填写结构化 issue_code")
        if self.result_verified and not self.task_completed:
            raise ValueError("任务未完成时不能标记结果已核验")
        return self

    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()


class UserTrialDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dataset_version: str = Field(min_length=1)
    protocol_version: Literal["scoreproof-user-trial-v1"] = "scoreproof-user-trial-v1"
    collector_role: Literal["authorized_operator"] = "authorized_operator"
    authorization_reference: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9._:-]{3,64}$",
    )
    consent_document_version: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9._:-]{3,64}$",
    )
    independent_real_users: bool = False
    sessions: list[TrialSession] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_trial_ids(self) -> UserTrialDataset:
        ids = [item.trial_id for item in self.sessions]
        if len(ids) != len(set(ids)):
            raise ValueError("trial_id 不得重复")
        return self


class TrialRate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    successes: int = Field(ge=0)
    total: int = Field(ge=0)
    value: float | None = Field(default=None, ge=0, le=1)
    ci95: tuple[float, float] | None = None


class UserTrialReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "1.0"
    generated_at: datetime
    dataset_version: str
    protocol_version: str
    source_sha256: str
    session_count: int = Field(ge=0)
    sample_size: int = Field(ge=0, description="去重真实参与者数")
    entrypoints: list[TrialEntrypoint]
    tasks: list[TrialTask]
    task_success_rate: TrialRate
    verified_result_rate: TrialRate
    manual_review_rate: TrialRate
    duration_p50_seconds: float | None = Field(default=None, ge=0)
    duration_p90_seconds: float | None = Field(default=None, ge=0)
    satisfaction_mean: float | None = Field(default=None, ge=1, le=5)
    real_users: bool
    consent_verified: bool
    independent_real_users: bool
    authorization_verified: bool
    smoke_test_only: bool
    formal_gate_eligible: bool
    limitations: list[str]


def new_participant_token() -> str:
    """生成与真实身份无映射关系的随机参与者令牌。"""
    return f"participant-{uuid.uuid4().hex}"


def user_trial_template() -> UserTrialDataset:
    """空模板不会伪装成真实用户样本。"""
    return UserTrialDataset(
        dataset_version="user-trial-v1",
        authorization_reference=None,
        consent_document_version=None,
        independent_real_users=False,
        sessions=[],
    )


def evaluate_user_trials(
    dataset: UserTrialDataset,
    *,
    generated_at: datetime | None = None,
) -> UserTrialReport:
    canonical = dataset.model_dump_json(exclude_none=False)
    source_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    sessions = dataset.sessions
    participants = {item.participant_token for item in sessions}
    real_users = bool(sessions) and all(
        item.real_user_confirmed and not item.synthetic for item in sessions
    )
    consent = bool(sessions) and all(item.consent_confirmed for item in sessions)
    authorization = bool(dataset.authorization_reference and dataset.consent_document_version)
    formal = bool(
        sessions
        and real_users
        and consent
        and authorization
        and dataset.independent_real_users
    )
    durations = [item.duration_seconds for item in sessions]
    satisfaction = [
        item.satisfaction_score for item in sessions if item.satisfaction_score is not None
    ]
    limitations: list[str] = []
    if not sessions:
        limitations.append("尚无试用会话；空模板不能通过真实用户门禁。")
    if sessions and not real_users:
        limitations.append("包含合成记录或未确认真实用户的记录。")
    if sessions and not consent:
        limitations.append("并非所有会话都确认了用户授权。")
    if not authorization:
        limitations.append("缺少授权引用或知情同意文档版本。")
    if not dataset.independent_real_users:
        limitations.append("数据集未声明为独立真实用户试用。")
    return UserTrialReport(
        generated_at=generated_at or datetime.now().astimezone(),
        dataset_version=dataset.dataset_version,
        protocol_version=dataset.protocol_version,
        source_sha256=source_sha256,
        session_count=len(sessions),
        sample_size=len(participants),
        entrypoints=sorted({item.entrypoint for item in sessions}),
        tasks=sorted({item.task for item in sessions}),
        task_success_rate=_rate(sum(item.task_completed for item in sessions), len(sessions)),
        verified_result_rate=_rate(sum(item.result_verified for item in sessions), len(sessions)),
        manual_review_rate=_rate(
            sum(item.manual_review_required for item in sessions), len(sessions)
        ),
        duration_p50_seconds=(round(statistics.median(durations), 3) if durations else None),
        duration_p90_seconds=(round(_percentile(durations, 0.9), 3) if durations else None),
        satisfaction_mean=(round(statistics.mean(satisfaction), 3) if satisfaction else None),
        real_users=real_users,
        consent_verified=consent,
        independent_real_users=dataset.independent_real_users,
        authorization_verified=authorization,
        smoke_test_only=not formal,
        formal_gate_eligible=formal,
        limitations=limitations,
    )


def load_user_trial_dataset(path: str | Path) -> UserTrialDataset:
    return UserTrialDataset.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _rate(successes: int, total: int) -> TrialRate:
    if total == 0:
        return TrialRate(successes=0, total=0)
    low, high = wilson_interval(successes, total)
    return TrialRate(
        successes=successes,
        total=total,
        value=round(successes / total, 6),
        ci95=(round(low, 6), round(high, 6)),
    )


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * probability) - 1))
    return ordered[index]


__all__ = [
    "TrialSession",
    "UserTrialDataset",
    "UserTrialReport",
    "evaluate_user_trials",
    "load_user_trial_dataset",
    "new_participant_token",
    "user_trial_template",
]

"""外部模型调用的最小化审计账本。

账本只保存用量元数据和不可逆业务标识，不保存提示词、模型回复或密钥。货币成本
必须由运行方显式提供价格；缺少价格时只汇总 token，不推测供应商价格。
"""

from __future__ import annotations

import hashlib
import hmac
import sqlite3
import uuid
from collections import defaultdict
from collections.abc import Mapping
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config import Settings

UsageSource = Literal["usage_metadata", "response_metadata", "mapping", "missing", "cache"]

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cost_events (
    id                  TEXT PRIMARY KEY,
    provider            TEXT NOT NULL,
    model               TEXT NOT NULL,
    model_tier          TEXT NOT NULL,
    purpose             TEXT NOT NULL,
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    total_tokens        INTEGER,
    material_id         TEXT,
    batch_id            TEXT,
    subject_id_hash     TEXT,
    cache_hit           INTEGER NOT NULL DEFAULT 0,
    external_call       INTEGER NOT NULL DEFAULT 1,
    status              TEXT NOT NULL,
    usage_source        TEXT NOT NULL,
    monetary_cost_cny   REAL,
    pricing_version     TEXT,
    error_type          TEXT,
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cost_events_created ON cost_events (created_at, id);
CREATE INDEX IF NOT EXISTS idx_cost_events_batch ON cost_events (batch_id, created_at);
"""


class ModelPricing(BaseModel):
    """由部署方显式配置的每百万 token 人民币单价。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    input_cny_per_million: float = Field(ge=0)
    output_cny_per_million: float = Field(ge=0)
    version: str = Field(min_length=1)


class CostEvent(BaseModel):
    """一次逻辑模型请求；缓存命中也记录，但 ``external_call=false``。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    model_tier: Literal["light", "strong", "vision", "unknown"] = "unknown"
    purpose: str = Field(min_length=1)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    material_id: str | None = None
    batch_id: str | None = None
    subject_id_hash: str | None = None
    cache_hit: bool = False
    external_call: bool = True
    status: Literal["ok", "error", "cache_hit"] = "ok"
    usage_source: UsageSource
    monetary_cost_cny: float | None = Field(default=None, ge=0)
    pricing_version: str | None = None
    error_type: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_semantics(self) -> CostEvent:
        if self.cache_hit and (self.external_call or self.status != "cache_hit"):
            raise ValueError("缓存命中不能标记为外部调用，status 必须为 cache_hit")
        if (
            self.total_tokens is not None
            and self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens < self.input_tokens + self.output_tokens
        ):
            raise ValueError("total_tokens 不能小于 input_tokens + output_tokens")
        if self.monetary_cost_cny is not None and not self.pricing_version:
            raise ValueError("记录货币成本时必须提供 pricing_version")
        return self


class CostBucket(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    key: str
    logical_requests: int
    external_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    monetary_cost_cny: float | None


class CostReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    generated_at: datetime
    source_database: str
    total_events: int
    logical_requests: int
    external_calls: int
    failed_calls: int
    cache_hits: int
    cache_hit_rate: float | None
    usage_covered_calls: int
    usage_coverage_rate: float | None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    monetary_cost_available: bool
    monetary_cost_cny: float | None
    distinct_materials: int
    distinct_subjects: int
    cost_per_100_materials_cny: float | None
    cost_per_subject_cny: float | None
    strong_model_calls: int
    strong_model_ratio: float | None
    secondary_extraction_calls: int
    secondary_extraction_ratio: float | None
    by_model: list[CostBucket]
    by_purpose: list[CostBucket]
    limitations: list[str]


def anonymize_subject(value: str, *, salt: str) -> str:
    """使用部署方密钥做 HMAC；低熵学号禁止使用可字典反推的无盐摘要。"""
    return hmac.new(salt.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()[:24]


def classify_model_tier(model: str, *, vision: bool = False) -> Literal["light", "strong", "vision", "unknown"]:
    if vision:
        return "vision"
    normalized = model.casefold()
    if any(marker in normalized for marker in ("pro", "reasoner", "max")):
        return "strong"
    if any(marker in normalized for marker in ("flash", "chat", "lite")):
        return "light"
    return "unknown"


def pricing_from_settings(settings: Settings) -> ModelPricing | None:
    values = (
        settings.llm_input_cny_per_million,
        settings.llm_output_cny_per_million,
        settings.pricing_version,
    )
    if any(value is None for value in values):
        return None
    assert settings.llm_input_cny_per_million is not None
    assert settings.llm_output_cny_per_million is not None
    assert settings.pricing_version is not None
    return ModelPricing(
        input_cny_per_million=settings.llm_input_cny_per_million,
        output_cny_per_million=settings.llm_output_cny_per_million,
        version=settings.pricing_version,
    )


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value >= 0 and value.is_integer():
        return int(value)
    return None


def extract_usage(response: Any) -> tuple[int | None, int | None, int | None, UsageSource]:
    """兼容 LangChain AIMessage 与 OpenAI 兼容响应，不估算缺失 token。"""
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, Mapping):
        input_tokens = _integer(usage.get("input_tokens"))
        output_tokens = _integer(usage.get("output_tokens"))
        total_tokens = _integer(usage.get("total_tokens"))
        if any(value is not None for value in (input_tokens, output_tokens, total_tokens)):
            return input_tokens, output_tokens, total_tokens, "usage_metadata"

    metadata = getattr(response, "response_metadata", None)
    token_usage = metadata.get("token_usage") if isinstance(metadata, Mapping) else None
    if isinstance(token_usage, Mapping):
        input_tokens = _integer(token_usage.get("prompt_tokens") or token_usage.get("input_tokens"))
        output_tokens = _integer(
            token_usage.get("completion_tokens") or token_usage.get("output_tokens")
        )
        total_tokens = _integer(token_usage.get("total_tokens"))
        if any(value is not None for value in (input_tokens, output_tokens, total_tokens)):
            return input_tokens, output_tokens, total_tokens, "response_metadata"

    if isinstance(response, Mapping) and isinstance(response.get("usage"), Mapping):
        mapped = response["usage"]
        input_tokens = _integer(mapped.get("prompt_tokens") or mapped.get("input_tokens"))
        output_tokens = _integer(mapped.get("completion_tokens") or mapped.get("output_tokens"))
        total_tokens = _integer(mapped.get("total_tokens"))
        if any(value is not None for value in (input_tokens, output_tokens, total_tokens)):
            return input_tokens, output_tokens, total_tokens, "mapping"
    return None, None, None, "missing"


class CostLedger:
    """SQLite 成本账本；可与规则库共用文件，但使用独立短事务。"""

    def __init__(self, path: str | Path, *, subject_salt: str | None = None) -> None:
        self.path = Path(path)
        self.subject_salt = subject_salt
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> CostLedger:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def record(self, event: CostEvent) -> CostEvent:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO cost_events (
                    id, provider, model, model_tier, purpose, input_tokens, output_tokens,
                    total_tokens, material_id, batch_id, subject_id_hash, cache_hit,
                    external_call, status, usage_source, monetary_cost_cny, pricing_version,
                    error_type, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.provider,
                    event.model,
                    event.model_tier,
                    event.purpose,
                    event.input_tokens,
                    event.output_tokens,
                    event.total_tokens,
                    event.material_id,
                    event.batch_id,
                    event.subject_id_hash,
                    int(event.cache_hit),
                    int(event.external_call),
                    event.status,
                    event.usage_source,
                    event.monetary_cost_cny,
                    event.pricing_version,
                    event.error_type,
                    event.created_at.isoformat(),
                ),
            )
        return event

    def record_response(
        self,
        response: Any,
        *,
        provider: str,
        model: str,
        purpose: str,
        model_tier: Literal["light", "strong", "vision", "unknown"] = "unknown",
        material_id: str | None = None,
        batch_id: str | None = None,
        subject_id: str | None = None,
        pricing: ModelPricing | None = None,
    ) -> CostEvent:
        input_tokens, output_tokens, total_tokens, source = extract_usage(response)
        cost: float | None = None
        if pricing is not None and input_tokens is not None and output_tokens is not None:
            cost = round(
                (input_tokens * pricing.input_cny_per_million + output_tokens * pricing.output_cny_per_million)
                / 1_000_000,
                8,
            )
        return self.record(
            CostEvent(
                provider=provider,
                model=model,
                model_tier=model_tier,
                purpose=purpose,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                material_id=material_id,
                batch_id=batch_id,
                subject_id_hash=self._subject_hash(subject_id),
                usage_source=source,
                monetary_cost_cny=cost,
                pricing_version=(pricing.version if pricing is not None and cost is not None else None),
            )
        )

    def record_failure(
        self,
        *,
        provider: str,
        model: str,
        purpose: str,
        error: BaseException,
        model_tier: Literal["light", "strong", "vision", "unknown"] = "unknown",
        material_id: str | None = None,
        batch_id: str | None = None,
        subject_id: str | None = None,
    ) -> CostEvent:
        return self.record(
            CostEvent(
                provider=provider,
                model=model,
                model_tier=model_tier,
                purpose=purpose,
                material_id=material_id,
                batch_id=batch_id,
                subject_id_hash=self._subject_hash(subject_id),
                status="error",
                usage_source="missing",
                error_type=type(error).__name__,
            )
        )

    def record_cache_hit(
        self,
        *,
        provider: str,
        model: str,
        purpose: str,
        model_tier: Literal["light", "strong", "vision", "unknown"] = "unknown",
        material_id: str | None = None,
        batch_id: str | None = None,
        subject_id: str | None = None,
    ) -> CostEvent:
        return self.record(
            CostEvent(
                provider=provider,
                model=model,
                model_tier=model_tier,
                purpose=purpose,
                material_id=material_id,
                batch_id=batch_id,
                subject_id_hash=self._subject_hash(subject_id),
                cache_hit=True,
                external_call=False,
                status="cache_hit",
                usage_source="cache",
            )
        )

    def list_events(self) -> list[CostEvent]:
        with closing(self._conn.execute("SELECT * FROM cost_events ORDER BY created_at, id")) as cursor:
            return [
                CostEvent.model_validate(
                    {
                        **dict(row),
                        "cache_hit": bool(row["cache_hit"]),
                        "external_call": bool(row["external_call"]),
                        "created_at": datetime.fromisoformat(row["created_at"]),
                    }
                )
                for row in cursor.fetchall()
            ]

    def _subject_hash(self, subject_id: str | None) -> str | None:
        if not subject_id or not self.subject_salt:
            return None
        return anonymize_subject(subject_id, salt=self.subject_salt)

    def report(self) -> CostReport:
        events = self.list_events()
        external = [item for item in events if item.external_call]
        successful = [item for item in external if item.status == "ok"]
        covered = [item for item in successful if item.total_tokens is not None]
        cache_hits = sum(item.cache_hit for item in events)
        priced = [item for item in successful if item.monetary_cost_cny is not None]
        monetary_available = bool(successful) and len(priced) == len(successful)
        total_cost = round(sum(item.monetary_cost_cny or 0 for item in priced), 8) if monetary_available else None
        materials = {item.material_id for item in events if item.material_id}
        subjects = {item.subject_id_hash for item in events if item.subject_id_hash}
        strong_calls = sum(item.model_tier == "strong" for item in successful)
        extraction_calls = [item for item in successful if item.purpose.startswith("rule_extraction")]
        secondary_calls = sum(item.purpose == "rule_extraction_secondary" for item in extraction_calls)

        limitations: list[str] = []
        if successful and len(covered) != len(successful):
            limitations.append("部分成功外部调用未返回 token usage，token 汇总不完整。")
        if not monetary_available:
            limitations.append("未对全部成功外部调用配置显式价格，货币成本不可用。")
        if not materials:
            limitations.append("没有材料标识，无法计算每 100 份材料成本。")
        if not subjects:
            limitations.append("没有脱敏用户标识，无法计算每名学生成本。")

        def buckets(attribute: str) -> list[CostBucket]:
            grouped: dict[str, list[CostEvent]] = defaultdict(list)
            for event in events:
                grouped[str(getattr(event, attribute))].append(event)
            output: list[CostBucket] = []
            for key, rows in sorted(grouped.items()):
                calls = [item for item in rows if item.external_call]
                row_success = [item for item in calls if item.status == "ok"]
                row_priced = [item for item in row_success if item.monetary_cost_cny is not None]
                row_cost = (
                    round(sum(item.monetary_cost_cny or 0 for item in row_priced), 8)
                    if row_success and len(row_priced) == len(row_success)
                    else None
                )
                output.append(
                    CostBucket(
                        key=key,
                        logical_requests=len(rows),
                        external_calls=len(calls),
                        input_tokens=sum(item.input_tokens or 0 for item in row_success),
                        output_tokens=sum(item.output_tokens or 0 for item in row_success),
                        total_tokens=sum(item.total_tokens or 0 for item in row_success),
                        monetary_cost_cny=row_cost,
                    )
                )
            return output

        logical = len(events)
        return CostReport(
            generated_at=datetime.now(UTC),
            source_database=self.path.name,
            total_events=len(events),
            logical_requests=logical,
            external_calls=len(external),
            failed_calls=sum(item.status == "error" for item in external),
            cache_hits=cache_hits,
            cache_hit_rate=round(cache_hits / logical, 6) if logical else None,
            usage_covered_calls=len(covered),
            usage_coverage_rate=round(len(covered) / len(successful), 6) if successful else None,
            input_tokens=sum(item.input_tokens or 0 for item in successful),
            output_tokens=sum(item.output_tokens or 0 for item in successful),
            total_tokens=sum(item.total_tokens or 0 for item in successful),
            monetary_cost_available=monetary_available,
            monetary_cost_cny=total_cost,
            distinct_materials=len(materials),
            distinct_subjects=len(subjects),
            cost_per_100_materials_cny=(
                round(total_cost * 100 / len(materials), 8)
                if total_cost is not None and materials
                else None
            ),
            cost_per_subject_cny=(
                round(total_cost / len(subjects), 8)
                if total_cost is not None and subjects
                else None
            ),
            strong_model_calls=strong_calls,
            strong_model_ratio=round(strong_calls / len(successful), 6) if successful else None,
            secondary_extraction_calls=secondary_calls,
            secondary_extraction_ratio=(
                round(secondary_calls / len(extraction_calls), 6) if extraction_calls else None
            ),
            by_model=buckets("model"),
            by_purpose=buckets("purpose"),
            limitations=limitations,
        )


__all__ = [
    "CostEvent",
    "CostLedger",
    "CostReport",
    "ModelPricing",
    "anonymize_subject",
    "classify_model_tier",
    "extract_usage",
    "pricing_from_settings",
]

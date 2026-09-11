"""项目级异常。

原则：能明确说明"为什么没给分"的失败，都用带 code 的领域异常表达，
让上层（API / CLI / 回测）可以据此给出可读反馈，而不是抛裸 ValueError。
"""

from __future__ import annotations


class ScoreProofError(Exception):
    """所有领域异常的基类，带机器可读的 code。"""

    code = "scoreproof_error"

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "detail": self.detail}


class DataSourceError(ScoreProofError):
    """原始文件缺失 / 格式不符 / 解析失败。"""

    code = "data_source_error"


class SchemaValidationError(ScoreProofError):
    """规则库或申报条目不满足 schema。"""

    code = "schema_validation_error"


class RuleNotFound(ScoreProofError):
    """规则库中未命中：宁可拒答，也不瞎给分。"""

    code = "rule_not_found"


class AmbiguousRule(ScoreProofError):
    """同一 (学年, 学院, 类别, 等级) 命中了多条规则，需先人工裁决。"""

    code = "ambiguous_rule"


class VersionConflict(ScoreProofError):
    """多份文件冲突（学校 vs 学院、新 vs 旧）且优先级未决。"""

    code = "version_conflict"


class CapExceeded(ScoreProofError):
    """命中封顶（不是错误，而是计算过程的可解释事件，供上层标注）。"""

    code = "cap_exceeded"


class UnsupportedModality(ScoreProofError):
    """请求的处理通道尚未实现（例如 P2 之前的 OCR）。"""

    code = "unsupported_modality"

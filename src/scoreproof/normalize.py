"""归一化：把学生的各种野生表述对齐到规则库的规范等级。

这是"必须做的脏活，也是壁垒"（项目总结第 5 节）：
  "省二等奖" / "省级二等" / "省赛第二名" / "省级二等奖" -> 统一为 "省级二等奖"

设计原则：
1. **保守**：只做有把握的替换，认不出来就原样返回（绝不臆造等级，否则会算错分）；
2. **可扩展**：别名表 + 自定义表可覆盖，学院差异通过 ``extra_aliases`` 注入；
3. **可测试**：纯函数，无外部依赖。
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field

# ======================================================================
# 级别（tier）
# ======================================================================

TIER_ALIASES: dict[str, str] = {
    # 国家级
    "国家级": "国家级",
    "国家": "国家级",
    "国级": "国家级",
    "全国": "国家级",
    "全国级": "国家级",
    # 省部级 -> 省级（项目总结明确要求：省部级→省级）
    "省级": "省级",
    "省部级": "省级",
    "省": "省级",
    "省赛": "省级",
    "省内": "省级",
    # 市级
    "市级": "市级",
    "地市级": "市级",
    "市": "市级",
    "市赛": "市级",
    # 校级
    "校级": "校级",
    "学校级": "校级",
    "校": "校级",
    "校赛": "校级",
    "院级": "院级",
    "学院级": "院级",
    "系级": "院级",
    "班级": "班级",
    "班": "班级",
    "国际级": "国际级",
    "国际": "国际级",
    "世界级": "国际级",
}

# 长键优先匹配（"省部级" 必须先于 "省"）
_TIER_KEYS = sorted(TIER_ALIASES, key=len, reverse=True)

# ======================================================================
# 奖项（prize）
# ======================================================================

PRIZE_ALIASES: dict[str, str] = {
    # 名次 -> 奖项（项目总结明确要求：第二名→二等奖）
    "第一名": "一等奖",
    "第二名": "二等奖",
    "第三名": "三等奖",
    "冠军": "一等奖",
    "亚军": "二等奖",
    "季军": "三等奖",
    "金牌": "一等奖",
    "银牌": "二等奖",
    "铜牌": "三等奖",
    # 等第
    "一等": "一等奖",
    "二等": "二等奖",
    "三等": "三等奖",
    "一等奖": "一等奖",
    "二等奖": "二等奖",
    "三等奖": "三等奖",
    "1等奖": "一等奖",
    "2等奖": "二等奖",
    "3等奖": "三等奖",
    "特等奖": "特等奖",
    "特等": "特等奖",
    "特别奖": "特等奖",
    "优秀奖": "优秀奖",
    "优胜奖": "优秀奖",
    "鼓励奖": "鼓励奖",
    "入围奖": "入围奖",
    "参与奖": "参与奖",
    "提名奖": "提名奖",
}

# 中文数字 -> 阿拉伯（仅用于等第/名次）
_CN_NUM = {"一": "1", "二": "2", "三": "3", "四": "4", "1": "1", "2": "2", "3": "3", "4": "4"}

_PRIZE_KEYS = sorted(PRIZE_ALIASES, key=len, reverse=True)

# 正则：抓 "一等/二等/三等" + 可选 "奖"，或 "第N名"，或 "N等奖"
_PRIZE_PATTERNS = [
    (re.compile(r"第\s*([一二三四1-4])\s*名"), lambda m: f"{_CN_NUM.get(m.group(1), m.group(1))}等奖"),
    (re.compile(r"([一二三四1-4])\s*等\s*奖?"), lambda m: f"{_CN_NUM.get(m.group(1), m.group(1))}等奖"),
]

# 特殊奖项：本身即完整等级，不需要拼接级别
SPECIAL_LEVELS: tuple[str, ...] = (
    "国家奖学金",
    "国家励志奖学金",
    "省政府奖学金",
    "校级奖学金",
    "三好学生",
    "优秀学生干部",
    "优秀毕业生",
)

# 需要被清洗掉、但不影响语义的噪声
_NOISE_RE = re.compile(r"[\s\u3000·•、,，。;；()（）\[\]【】\-—_/]+")


@dataclass
class NormalizeResult:
    """归一化结果，附带可解释信息。"""

    raw: str
    canonical: str
    tier: str | None = None
    prize: str | None = None
    matched: bool = False
    rule: str = "unchanged"  # 命中方式：alias / regex / special / unchanged
    aliases: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - 便捷打印
        return self.canonical


def _clean(text: str) -> str:
    """全角转半角 + 去噪 + 统一写法。"""
    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace("等奖", "等奖")  # NFKC 后保留占位，便于后续阅读
    return _NOISE_RE.sub("", t)


def parse_tier(text: str) -> str | None:
    """从文本中抽取级别（国家级/省级/市级/校级/院级/班级/国际级）。"""
    t = _clean(text)
    for key in _TIER_KEYS:
        if key in t:
            return TIER_ALIASES[key]
    return None


def parse_prize(text: str) -> str | None:
    """从文本中抽取奖项（一等奖/二等奖/…），支持"第二名"这类写法。"""
    t = _clean(text)
    for key in _PRIZE_KEYS:
        if key in t:
            return PRIZE_ALIASES[key]
    for pattern, build in _PRIZE_PATTERNS:
        m = pattern.search(t)
        if m:
            return build(m)
    return None


def normalize_level(raw: str, *, extra_aliases: dict[str, str] | None = None) -> NormalizeResult:
    """把野生表述归一化为规范等级。

    匹配顺序：自定义别名 -> 特殊奖项 -> 级别+奖项组合 -> 原样返回。
    """
    cleaned = _clean(raw)
    if not cleaned:
        return NormalizeResult(raw=raw, canonical="", matched=False)

    if extra_aliases:
        for key in sorted(extra_aliases, key=len, reverse=True):
            if _clean(key) and _clean(key) in cleaned:
                target = extra_aliases[key]
                return NormalizeResult(
                    raw=raw, canonical=target, matched=True, rule="alias",
                    tier=parse_tier(target), prize=parse_prize(target),
                )

    for special in SPECIAL_LEVELS:
        if special in cleaned:
            tier = parse_tier(cleaned) or ("国家级" if "国家" in cleaned else None)
            return NormalizeResult(
                raw=raw, canonical=special, matched=True, rule="special", tier=tier,
            )

    tier = parse_tier(cleaned)
    prize = parse_prize(cleaned)
    if tier and prize:
        canonical = f"{tier}{prize}"
        return NormalizeResult(
            raw=raw, canonical=canonical, tier=tier, prize=prize, matched=True, rule="alias",
        )
    if prize and not tier:
        # 只有奖项没有级别：不臆造级别，交给上层要求补充
        return NormalizeResult(
            raw=raw, canonical=prize, tier=None, prize=prize, matched=True, rule="alias",
        )
    if tier and not prize:
        return NormalizeResult(
            raw=raw, canonical=tier, tier=tier, prize=None, matched=True, rule="alias",
        )
    return NormalizeResult(raw=raw, canonical=cleaned, matched=False, rule="unchanged")


def canonical_level(raw: str, *, extra_aliases: dict[str, str] | None = None) -> str:
    """只要结果的便捷入口。"""
    return normalize_level(raw, extra_aliases=extra_aliases).canonical


def level_aliases(level: str) -> list[str]:
    """由规范等级反推可能的别名，用于导入规则时自动补 synonyms。

    例：``省级二等奖`` -> ``["省二等奖", "省级二等", "省赛二等奖", "省二等"]``
    """
    res = normalize_level(level)
    out: set[str] = {level}
    if res.tier and res.prize:
        tier_keys = [k for k, v in TIER_ALIASES.items() if v == res.tier]
        prize_keys = [k for k, v in PRIZE_ALIASES.items() if v == res.prize]
        for tk in tier_keys:
            for pk in prize_keys:
                out.add(f"{tk}{pk}")
    for special in SPECIAL_LEVELS:
        if special == res.canonical:
            out.add(special)
    # 稳定排序：短的在前、同长按字典序，保证同义词顺序可复现
    return sorted(out, key=lambda s: (len(s), s))


def normalize_text_key(raw: str) -> str:
    """生成用于精确查表的匹配键：去噪 + 统一。"""
    return _clean(raw)


# ======================================================================
# 学年
# ======================================================================

_ACADEMIC_YEAR_RE = re.compile(r"^(\d{4})(?:\s*[-~/至]\s*(\d{4}))?$")


def normalize_academic_year(raw: str) -> str:
    """``2025`` / ``2025-2026`` / ``2025~2026`` / ``2025至2026`` -> ``2025-2026``。

    起始年单独给出时，默认跨到次年（学年语义）。
    """
    t = (raw or "").strip()
    m = _ACADEMIC_YEAR_RE.match(t)
    if not m:
        return t
    start, end = m.group(1), m.group(2)
    return f"{start}-{end}" if end else f"{start}-{int(start) + 1}"


def academic_year_of(d, *, rollover_month: int = 9) -> str:
    """按"9 月为新学年起点"把日期映射到学年。"""
    start = d.year if d.month >= rollover_month else d.year - 1
    return f"{start}-{start + 1}"


def match_academic_year(rule_year: str, claim_year: str) -> bool:
    """学年是否匹配：允许规则写单年（``2025``），两边都先归一化。"""
    return normalize_academic_year(rule_year) == normalize_academic_year(claim_year)


# ======================================================================
# 类别归一化（轻量：只做明显同义合并）
# ======================================================================

CATEGORY_ALIASES: dict[str, str] = {
    "学科竞赛": "学科竞赛",
    "竞赛": "学科竞赛",
    "科技竞赛": "学科竞赛",
    "创新创业": "创新创业",
    "科研": "科研学术",
    "学术": "科研学术",
    "论文": "科研学术",
    "专利": "科研学术",
    "志愿": "志愿服务",
    "志愿服务": "志愿服务",
    "社会实践": "社会实践",
    "文体": "文体活动",
    "文体活动": "文体活动",
    "社会工作": "社会工作",
    "学生工作": "社会工作",
    "荣誉称号": "荣誉称号",
    "荣誉": "荣誉称号",
}


def normalize_category(raw: str) -> str:
    t = _clean(raw)
    for key in sorted(CATEGORY_ALIASES, key=len, reverse=True):
        if key and key in t:
            return CATEGORY_ALIASES[key]
    return raw.strip() or "未分类"


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for it in items:
        seen.setdefault(it, None)
    return list(seen)


__all__ = [
    "CATEGORY_ALIASES",
    "PRIZE_ALIASES",
    "SPECIAL_LEVELS",
    "TIER_ALIASES",
    "NormalizeResult",
    "academic_year_of",
    "canonical_level",
    "dedupe_preserve_order",
    "level_aliases",
    "match_academic_year",
    "normalize_academic_year",
    "normalize_category",
    "normalize_level",
    "normalize_text_key",
    "parse_prize",
    "parse_tier",
]

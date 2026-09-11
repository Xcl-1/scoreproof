"""计算引擎：**纯 Python，模型不参与任何数值判断**。

细则语义（项目总结第 6 节）：
    查值 -> 同类取最高（不累加）-> 封顶 -> 学年过滤 -> 团队折算

所有函数都是纯函数（同样输入必得同样输出），可直接单元测试；
每条命中都带 ``RuleMatch``，可逐条回溯到规则与原文出处。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from ..normalize import match_academic_year, normalize_level
from ..schema import (
    Claim,
    GroupBreakdown,
    Rule,
    RuleMatch,
    Ruleset,
    ScoreBreakdown,
)

# ======================================================================
# 配置
# ======================================================================


@dataclass(frozen=True)
class EngineConfig:
    """引擎行为开关，全部显式、可测试。"""

    same_group_take_max: bool = True
    """同类取最高，不累加（默认 True，符合综测细则通行做法）。"""

    apply_item_cap: bool = True
    """是否应用规则自身的 cap。"""

    apply_group_cap: bool = True
    """是否应用互斥组的 cap（取该组内最大 cap）。"""

    apply_team_factor: bool = True
    """是否对团队奖做折算。"""

    resolve_exclusive: bool = True
    """是否自动裁决 ``exclusive_with`` 互斥组（保留总分更高的组合）。"""

    strict_year: bool = True
    """学年不匹配直接不计分（False 时降级为标注待复核）。"""

    require_catalog: bool = True
    """命中需要认可目录的规则，但申报不在目录内 -> 不计分。"""

    fuzzy_fallback: bool = True
    """精确查表未命中时，是否允许"同类别降级匹配"（结果标注待复核）。"""

    unknown_level_confidence: float = 0.6
    """降级匹配/等级不明时的置信度，低于阈值进入 review。"""

    review_threshold: float = 0.8
    """低于该置信度的命中会被标记 needs_review。"""

    round_digits: int = 2
    """最终分值保留小数位（避免浮点噪声）。"""


def default_config(**overrides) -> EngineConfig:
    return EngineConfig(**overrides)


# ======================================================================
# 匹配（精确查表优先）
# ======================================================================

MatchStrategy = Literal["exact_level", "synonym", "category_fallback", "none"]


@dataclass
class MatchOutcome:
    rule: Rule | None
    strategy: MatchStrategy
    matched_key: str | None = None
    confidence: float = 1.0
    reason: str | None = None

    @property
    def matched(self) -> bool:
        return self.rule is not None


def _key_variants(text: str) -> list[str]:
    """生成查表键的候选取法：原文、去噪、归一化等级。"""
    out: list[str] = []
    for cand in (text, normalize_level(text).canonical):
        c = (cand or "").strip()
        if c and c not in out:
            out.append(c)
    return out


class RuleIndex:
    """精确查表索引（主通道）。

    - ``by_key``：等级/同义词 -> 规则（同键多条时按优先级裁决）；
    - ``by_category``：类别 -> 规则（降级匹配用）。
    """

    def __init__(self, ruleset: Ruleset, *, academic_year: str | None = None,
                 college: str | None = None) -> None:
        self.academic_year = academic_year
        self.college = college
        self.rules: list[Rule] = sorted(
            ruleset.filter(academic_year=academic_year, college=college).rules,
            key=lambda r: (r.constraints.dedup_group, r.category, r.sort_key()),
        )
        self.by_key: dict[str, list[Rule]] = {}
        self.by_category: dict[str, list[Rule]] = {}
        for r in self.rules:
            for key in r.match_keys:
                self.by_key.setdefault(_norm_key(key), []).append(r)
            self.by_category.setdefault(_norm_key(r.category), []).append(r)
        for bucket in (*self.by_key.values(), *self.by_category.values()):
            bucket.sort(key=Rule.sort_key)

    def lookup_level(self, level: str) -> tuple[Rule | None, str | None, bool]:
        """返回 ``(规则, 命中的键, 是否精确命中正式等级)``。"""
        for variant in _key_variants(level):
            bucket = self.by_key.get(_norm_key(variant))
            if bucket:
                rule = bucket[0]
                exact = variant == rule.level
                return rule, variant, exact
        return None, None, False

    def lookup_category(self, category: str) -> Rule | None:
        bucket = self.by_category.get(_norm_key(category))
        return bucket[0] if bucket else None

    def group_cap(self, group: str) -> float | None:
        caps = [
            r.constraints.cap
            for r in self.rules
            if r.constraints.dedup_group == group and r.constraints.cap is not None
        ]
        return max(caps) if caps else None

    def __len__(self) -> int:
        return len(self.rules)


def _norm_key(text: str) -> str:
    return re.sub(r"[\s\u3000·•、,，。;；()（）\[\]【】\-—_/]+", "", (text or "").lower())


def match_claim(
    claim: Claim,
    index: RuleIndex,
    *,
    config: EngineConfig | None = None,
) -> MatchOutcome:
    """把一条申报匹配到规则。**精确通道优先，未命中才降级**。"""
    cfg = config or EngineConfig()
    if claim.level:
        rule, key, exact = index.lookup_level(claim.level)
        if rule is not None:
            return MatchOutcome(
                rule=rule,
                strategy="exact_level" if exact else "synonym",
                matched_key=key,
                confidence=1.0 if exact else 0.95,
            )
    if claim.raw_text:
        for variant in _key_variants(claim.raw_text):
            res = normalize_level(variant)
            if not res.matched:
                continue
            rule, key, exact = index.lookup_level(res.canonical)
            if rule is not None:
                return MatchOutcome(
                    rule=rule,
                    strategy="exact_level" if exact else "synonym",
                    matched_key=key,
                    confidence=0.9,
                    reason="等级由原始表述归一化得到",
                )
    if cfg.fuzzy_fallback and claim.category:
        rule = index.lookup_category(claim.category)
        if rule is not None:
            return MatchOutcome(
                rule=rule,
                strategy="category_fallback",
                matched_key=rule.level,
                confidence=cfg.unknown_level_confidence,
                reason=f"未匹配到具体等级，按类别『{rule.category}』内最高分规则降级匹配，需人工确认",
            )
    return MatchOutcome(
        rule=None,
        strategy="none",
        confidence=0.0,
        reason="规则库中未找到对应条款，建议咨询辅导员（不臆造分值）",
    )


# ======================================================================
# 数值计算（纯函数，重点测试对象）
# ======================================================================


def apply_team_factor(score: float, team: bool, factor: float, *, enabled: bool = True) -> float:
    """团队奖折算：个人奖不变，团队奖乘系数。"""
    if not enabled or not team:
        return float(score)
    if factor is None or factor < 0:
        return float(score)
    return float(score) * float(factor)


def apply_cap(score: float, cap: float | None, *, enabled: bool = True) -> tuple[float, bool]:
    """封顶。返回 ``(封顶后分值, 是否被截断)``。``cap=None`` 表示不设额度。"""
    if not enabled or cap is None:
        return float(score), False
    if score > cap:
        return float(cap), True
    return float(score), False


def dedup_take_max(matches: Sequence[RuleMatch]) -> RuleMatch | None:
    """同类取最高：同组内只保留一条，**不累加**。

    裁决顺序：计入分值降序 -> 规则优先级降序 -> rule_id 升序（保证可复现）。
    """
    if not matches:
        return None
    ordered = sorted(
        matches,
        key=lambda m: (-m.score, -m.confidence, m.rule_id or "", m.claim_id),
    )
    return ordered[0]


def compute_item_score(rule: Rule, claim: Claim, *, config: EngineConfig | None = None) -> tuple[float, bool, float]:
    """单条命中分值：``score * team_factor`` 后再封顶。

    Returns:
        ``(计入分值, 是否被封顶, 实际使用的团队系数)``
    """
    cfg = config or EngineConfig()
    factor = rule.constraints.team_factor if claim.team else 1.0
    if not cfg.apply_team_factor:
        factor = 1.0
    raw = apply_team_factor(rule.score, claim.team, rule.constraints.team_factor,
                            enabled=cfg.apply_team_factor)
    capped_score, was_capped = apply_cap(raw, rule.constraints.cap, enabled=cfg.apply_item_cap)
    return capped_score, was_capped, factor


def resolve_exclusive_groups(
    winners: dict[str, RuleMatch],
    rules_by_group: dict[str, list[Rule]],
    *,
    enabled: bool = True,
) -> tuple[dict[str, RuleMatch], list[str]]:
    """互斥组裁决：```exclusive_with`` 声明的组之间只能保留一个组合。

    策略（确定性）：枚举所有互斥连通分量内的保留方案，选择**总分更高**者；
    同分时保留"组本身分值更高、组名更小"的方案。
    """
    dropped: list[str] = []
    if not enabled or not winners:
        return winners, dropped

    # 1) 建图：组 <-> 组
    conflicts: dict[str, set[str]] = {g: set() for g in winners}
    for g, rules in rules_by_group.items():
        if g not in winners:
            continue
        for r in rules:
            for other in r.constraints.exclusive_with:
                if other in winners and other != g:
                    conflicts.setdefault(g, set()).add(other)
                    conflicts.setdefault(other, set()).add(g)

    # 2) 连通分量
    seen: set[str] = set()
    components: list[list[str]] = []
    for g in list(winners):
        if g in seen:
            continue
        stack, comp = [g], []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            stack.extend(sorted(conflicts.get(cur, ())))
        if len(comp) > 1:
            components.append(sorted(comp))

    # 3) 每个分量选一个总分最高的独立集
    keep = dict(winners)
    for comp in components:
        best_combo: list[str] | None = None
        best_key: tuple | None = None
        n = len(comp)
        for mask in range(1, 1 << n):
            chosen = [comp[i] for i in range(n) if mask & (1 << i)]
            if any(b in conflicts.get(a, ()) for a in chosen for b in chosen if a != b):
                continue
            total = round(sum(winners[g].score for g in chosen), 6)
            # 总分最高 -> 单项最高 -> 组名序列（全部确定性）
            key = (total, max(winners[g].score for g in chosen), tuple(sorted(chosen)))
            if best_key is None or key[:2] > best_key[:2] or key[:2] == best_key[:2] and key[2] < best_key[2]:
                best_key, best_combo = key, chosen
        if best_combo is None:  # pragma: no cover - 空集总是合法
            continue
        for g in comp:
            if g not in best_combo:
                keep.pop(g, None)
                dropped.append(g)
    return keep, dropped


# ======================================================================
# 主流程
# ======================================================================


def compute_claims(
    claims: Iterable[Claim],
    ruleset: Ruleset,
    *,
    academic_year: str | None = None,
    college: str | None = None,
    config: EngineConfig | None = None,
    index: RuleIndex | None = None,
) -> ScoreBreakdown:
    """核算一组申报（通常是一个学生的全部条目）。"""
    cfg = config or EngineConfig()
    claim_list = list(claims)
    student_id = claim_list[0].student_id if claim_list else ""
    year = academic_year or next((c.academic_year for c in claim_list if c.academic_year), None)
    college = college or next((c.college for c in claim_list if c.college), None)
    idx = index or RuleIndex(ruleset, academic_year=year, college=college)

    notes: list[str] = []
    matches: list[RuleMatch] = []
    unmatched: list[str] = []
    review: list[str] = []

    for claim in claim_list:
        outcome = match_claim(claim, idx, config=cfg)
        if not outcome.matched or outcome.rule is None:
            unmatched.append(claim.id)
            claim.status = "未命中规则"
            matches.append(
                RuleMatch(
                    claim_id=claim.id,
                    rule_id=None,
                    score=0.0,
                    channel="none",
                    confidence=0.0,
                    reason=outcome.reason,
                    counted=False,
                )
            )
            continue

        rule = outcome.rule
        reasons: list[str] = []
        confidence = outcome.confidence
        if outcome.reason:
            reasons.append(outcome.reason)

        # 学年过滤
        if cfg.strict_year and claim.academic_year and not match_academic_year(
            rule.academic_year, claim.academic_year
        ):
            unmatched.append(claim.id)
            claim.status = "已驳回"
            matches.append(
                RuleMatch(
                    claim_id=claim.id,
                    rule_id=rule.id,
                    matched_key=outcome.matched_key,
                    score=0.0,
                    raw_score=rule.score,
                    channel="structured",
                    confidence=0.0,
                    counted=False,
                    reason=f"学年不匹配（规则 {rule.academic_year} / 申报 {claim.academic_year}）",
                    source=rule.source,
                )
            )
            continue

        # 认可目录资格
        if cfg.require_catalog and rule.constraints.require_catalog and not claim.catalog_listed:
            unmatched.append(claim.id)
            claim.status = "已驳回"
            matches.append(
                RuleMatch(
                    claim_id=claim.id,
                    rule_id=rule.id,
                    matched_key=outcome.matched_key,
                    score=0.0,
                    raw_score=rule.score,
                    channel="structured",
                    confidence=0.0,
                    counted=False,
                    reason=f"未列入「{rule.constraints.require_catalog}」，不予计分",
                    source=rule.source,
                )
            )
            continue

        score, capped, factor = compute_item_score(rule, claim, config=cfg)
        if capped:
            reasons.append(f"命中单项上限 {rule.constraints.cap}")
        if claim.team and factor != 1.0:
            reasons.append(f"团队奖按 {factor:g} 折算")
        needs_review = confidence < cfg.review_threshold
        if needs_review:
            review.append(claim.id)

        matches.append(
            RuleMatch(
                claim_id=claim.id,
                rule_id=rule.id,
                matched_key=outcome.matched_key,
                score=score,
                raw_score=rule.score,
                team_factor=factor,
                capped=capped,
                dedup_group=rule.constraints.dedup_group,
                channel="structured",
                confidence=confidence,
                needs_review=needs_review,
                reason="；".join(reasons) or None,
                source=rule.source,
            )
        )
        if not needs_review:
            claim.status = "已核对"

    # ---- 同类取最高（不累加）----
    by_group: dict[str, list[RuleMatch]] = {}
    counted = [m for m in matches if m.rule_id and m.score >= 0 and m.counted]
    for m in counted:
        by_group.setdefault(m.dedup_group or "默认组", []).append(m)

    rules_by_group: dict[str, list[Rule]] = {}
    for r in idx.rules:
        rules_by_group.setdefault(r.constraints.dedup_group, []).append(r)

    if cfg.same_group_take_max:
        winners: dict[str, RuleMatch] = {}
        for group, bucket in by_group.items():
            winner = dedup_take_max(bucket)
            if winner is None:
                continue
            winners[group] = winner
            for m in bucket:
                if m is not winner:
                    m.counted = False
                    if m.reason:
                        m.reason += "；"
                    m.reason = (m.reason or "") + f"同类取最高，被『{group}』内更高分条目标覆盖"
    else:
        winners = {}
        for group, bucket in by_group.items():
            winners[group] = dedup_take_max(bucket)  # 占位，下面按组求和

    # ---- 互斥组裁决 ----
    if cfg.same_group_take_max:
        winners, dropped = resolve_exclusive_groups(
            winners, rules_by_group, enabled=cfg.resolve_exclusive
        )
        for g in dropped:
            notes.append(f"互斥组『{g}』未计入：与更高分组合冲突")
            for m in by_group.get(g, []):
                m.counted = False
                m.reason = (m.reason + "；" if m.reason else "") + "所属互斥组未计入"

    # ---- 分组封顶 + 汇总 ----
    groups: list[GroupBreakdown] = []
    total = 0.0
    for group in sorted(by_group):
        bucket = by_group[group]
        winner = winners.get(group)
        if cfg.same_group_take_max:
            subtotal = winner.score if winner else 0.0
        else:
            subtotal = sum(m.score for m in bucket)
        cap = idx.group_cap(group)
        after_cap, capped = apply_cap(subtotal, cap, enabled=cfg.apply_group_cap)
        total += after_cap
        groups.append(
            GroupBreakdown(
                group=group,
                cap=cap,
                candidates=bucket,
                winner=winner,
                subtotal=round(subtotal, 6),
                after_cap=round(after_cap, 6),
                capped=capped,
            )
        )

    breakdown = ScoreBreakdown(
        student_id=student_id,
        academic_year=year,
        college=college,
        total=round(total, cfg.round_digits),
        groups=groups,
        matches=matches,
        unmatched_claims=sorted(set(unmatched)),
        review_claims=sorted(set(review)),
        notes=notes,
    )
    return breakdown


def compute_student(
    student_id: str,
    claims: Iterable[Claim],
    ruleset: Ruleset,
    *,
    academic_year: str | None = None,
    college: str | None = None,
    config: EngineConfig | None = None,
) -> ScoreBreakdown:
    """按学号核算：自动挑选该学号的申报条目。"""
    all_claims = [c for c in claims]
    mine = [c for c in all_claims if c.student_id == student_id]
    if not mine:
        return ScoreBreakdown(
            student_id=student_id,
            academic_year=academic_year,
            college=college,
            notes=[f"没有找到学号 {student_id} 的申报条目"],
        )
    return compute_claims(mine, ruleset, academic_year=academic_year, college=college, config=config)


def compute_all(
    claims: Iterable[Claim],
    ruleset: Ruleset,
    *,
    academic_year: str | None = None,
    college: str | None = None,
    config: EngineConfig | None = None,
) -> dict[str, ScoreBreakdown]:
    """批量核算（班委模式），返回 ``学号 -> 结果``。"""
    cfg = config or EngineConfig()
    bucket: dict[str, list[Claim]] = {}
    for c in claims:
        bucket.setdefault(c.student_id, []).append(c)
    year = academic_year or next((c.academic_year for c in claims if c.academic_year), None)  # type: ignore[union-attr]
    idx = RuleIndex(ruleset, academic_year=year, college=college)
    return {
        sid: compute_claims(items, ruleset, academic_year=year, college=college, config=cfg, index=idx)
        for sid, items in sorted(bucket.items())
    }


__all__ = [
    "EngineConfig",
    "MatchOutcome",
    "RuleIndex",
    "apply_cap",
    "apply_team_factor",
    "compute_all",
    "compute_claims",
    "compute_item_score",
    "compute_student",
    "dedup_take_max",
    "default_config",
    "match_claim",
    "resolve_exclusive_groups",
]

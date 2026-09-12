"""确定性查询改写：保留原问法，并追加细则中的规范术语。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .router import RetrievalHit, Retriever

DEFAULT_QUERY_ALIASES: Mapping[str, Sequence[str]] = {
    "推免": ("免试研究生", "推荐免试", "免试", "遴选"),
    "大学英语四级": ("cet四级", "cet-4", "四级"),
    "大学英语六级": ("cet六级", "cet-6", "六级"),
    "学科竞赛": ("比赛", "赛事", "竞赛奖项"),
    "挑战杯": ("创业赛", "创业计划赛", "课外学术科技竞赛"),
    "中国国际大学生创新大赛": ("大创", "创新创业竞赛", "创新赛事"),
    "论文": ("科研文章", "文章成果", "发表成果", "学术文章"),
    "综合测评": ("综合评价", "最终推荐排名", "最终成绩", "总分"),
    "学业综合成绩": ("课程成绩", "学业成绩"),
    "志愿服务": ("公益服务", "志愿活动", "服务时长"),
    "国际组织实习": ("国际机构实习", "海外组织实践", "国际组织项目", "海外实习"),
    "知识产权": ("专利", "专利权人", "专利人"),
    "换算": ("标准化", "归一化", "折成", "折算", "标准成绩"),
    "原始分": ("原始值", "原始得分", "加分表"),
    "团队": ("集体项目", "成员次序", "队长"),
}


def rewrite_retrieval_query(
    query: str, *, aliases: Mapping[str, Sequence[str]] = DEFAULT_QUERY_ALIASES
) -> str:
    """追加命中的规范词，不删除原文本，便于审计且避免信息损失。"""
    original = query.strip()
    if not original:
        return ""
    folded = original.casefold()
    additions = [
        canonical
        for canonical, variants in aliases.items()
        if canonical.casefold() not in folded
        and any(variant.casefold() in folded for variant in variants)
    ]
    return " ".join([original, *additions])


class QueryRewritingRetriever:
    """为任意召回器增加同一套可复现查询改写。"""

    def __init__(
        self,
        retriever: Retriever,
        *,
        aliases: Mapping[str, Sequence[str]] = DEFAULT_QUERY_ALIASES,
    ) -> None:
        self.retriever = retriever
        self.aliases = aliases
        self.last_query: str | None = None

    def available(self) -> bool:
        return self.retriever.available()

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        self.last_query = rewrite_retrieval_query(query, aliases=self.aliases)
        return self.retriever.search(self.last_query, top_k=top_k)


__all__ = ["DEFAULT_QUERY_ALIASES", "QueryRewritingRetriever", "rewrite_retrieval_query"]

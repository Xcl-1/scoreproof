"""检索与索引共享的确定性分词。"""

from __future__ import annotations

import re


def tokenize_for_search(text: str) -> list[str]:
    """中文字符/二元组与英文、数字词元的轻量混合分词。"""
    compact = re.sub(r"\s+", "", text).lower()
    words = re.findall(r"[a-z]+|\d+(?:\.\d+)?", compact)
    cjk = re.findall(r"[\u4e00-\u9fff]", compact)
    bigrams = ["".join(pair) for pair in zip(cjk, cjk[1:], strict=False)]
    return [*words, *cjk, *bigrams]


__all__ = ["tokenize_for_search"]

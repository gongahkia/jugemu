from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


_WORD_RE = re.compile(r"\w+")


def count_chars(texts: Iterable[str]) -> Counter[str]:
    c: Counter[str] = Counter()
    for t in texts:
        c.update(t)
    return c


def count_tokens(texts: Iterable[str]) -> Counter[str]:
    c: Counter[str] = Counter()
    for t in texts:
        toks = [m.group(0).lower() for m in _WORD_RE.finditer(t)]
        c.update(toks)
    return c


def top_items(counter: Counter[str], *, top: int, min_count: int = 1) -> list[tuple[str, int]]:
    n = int(top)
    if n < 0:
        n = 0
    mc = int(min_count)
    if mc < 1:
        mc = 1

    items = [(k, int(v)) for k, v in counter.items() if int(v) >= mc]
    items.sort(key=lambda kv: kv[1], reverse=True)
    if n == 0:
        return []
    return items[:n]

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


def browse_report(
    texts: Iterable[str],
    *,
    mode: str = "both",
    top: int = 50,
    min_count: int = 1,
) -> dict:
    m = str(mode or "both").strip().lower()
    if m not in {"chars", "tokens", "both"}:
        raise ValueError("mode must be one of: chars|tokens|both")

    out: dict = {
        "mode": m,
        "top": int(top),
        "min_count": int(min_count),
    }

    if m in {"chars", "both"}:
        ch = count_chars(texts)
        out["chars"] = [{"char": k, "count": v} for k, v in top_items(ch, top=int(top), min_count=int(min_count))]
    if m in {"tokens", "both"}:
        tok = count_tokens(texts)
        out["tokens"] = [
            {"token": k, "count": v} for k, v in top_items(tok, top=int(top), min_count=int(min_count))
        ]

    return out

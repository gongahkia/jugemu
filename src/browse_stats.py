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

from __future__ import annotations

from collections import Counter

from src.browse_stats import count_chars, count_tokens, top_items


def test_count_chars_includes_spaces() -> None:
    c = count_chars(["ab a"])
    assert c["a"] == 2
    assert c["b"] == 1
    assert c[" "] == 1


def test_count_tokens_word_only_lowercased() -> None:
    c = count_tokens(["Hello, hello!!!", "hi"])
    assert c["hello"] == 2
    assert c["hi"] == 1


def test_top_items_applies_min_count_and_top() -> None:
    c = Counter({"a": 3, "b": 1, "c": 2})
    assert top_items(c, top=10, min_count=2) == [("a", 3), ("c", 2)]
    assert top_items(c, top=1, min_count=1) == [("a", 3)]

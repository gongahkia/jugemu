from __future__ import annotations

from collections import Counter

from src.browse_stats import count_chars, count_tokens, top_items
from src.browse_stats import browse_report


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


def test_browse_report_both_includes_metadata_and_lists() -> None:
    report = browse_report(["a a b"], mode="both", top=10, min_count=1)
    assert report["mode"] == "both"
    assert report["top"] == 10
    assert report["min_count"] == 1
    assert isinstance(report["chars"], list)
    assert isinstance(report["tokens"], list)
    assert {"char", "count"}.issubset(report["chars"][0].keys())
    assert {"token", "count"}.issubset(report["tokens"][0].keys())


def test_browse_report_min_count_filters() -> None:
    report = browse_report(["a a b"], mode="tokens", top=10, min_count=2)
    assert report["tokens"] == [{"token": "a", "count": 2}]


def test_browse_report_mode_chars_only() -> None:
    report = browse_report(["ab"], mode="chars", top=10, min_count=1)
    assert "chars" in report
    assert "tokens" not in report

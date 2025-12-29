from __future__ import annotations

from src.browse_stats import count_chars, count_tokens


def test_count_chars_includes_spaces() -> None:
    c = count_chars(["ab a"])
    assert c["a"] == 2
    assert c["b"] == 1
    assert c[" "] == 1


def test_count_tokens_word_only_lowercased() -> None:
    c = count_tokens(["Hello, hello!!!", "hi"])
    assert c["hello"] == 2
    assert c["hi"] == 1

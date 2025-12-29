from __future__ import annotations

from src.normalize import (
    collapse_whitespace_preserve_newlines,
    normalize_newlines,
    normalize_text,
    strip_emoji,
)


def test_normalize_newlines() -> None:
    assert normalize_newlines("a\r\nb\rc") == "a\nb\nc"


def test_collapse_whitespace_preserves_newlines() -> None:
    s = "a   b\n\tc    d\n"
    assert collapse_whitespace_preserve_newlines(s) == "a b\nc d\n"


def test_strip_emoji_best_effort() -> None:
    assert strip_emoji("hi🙂") == "hi"


def test_normalize_text_pipeline() -> None:
    s = "hi\r\nthere\t🙂"
    out = normalize_text(s, collapse_whitespace=True, strip_emoji_chars=True)
    assert out == "hi\nthere"

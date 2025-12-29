from __future__ import annotations

from src.ingest_chroma import content_hash, fuzzy_dedupe_texts


def test_content_hash_stable() -> None:
    assert content_hash("hi") == content_hash("hi")
    assert content_hash("hi") != content_hash("hi ")


def test_fuzzy_dedupe_marks_near_duplicates() -> None:
    texts = [
        "hello there",
        "hello there!",  # near dup
        "completely different",
    ]
    keep = fuzzy_dedupe_texts(texts, max_hamming=10)
    assert keep[0] is True
    # depending on tokenization, this should usually be considered near-dup
    assert keep[1] is False
    assert keep[2] is True

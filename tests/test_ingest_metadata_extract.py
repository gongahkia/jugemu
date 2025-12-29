from __future__ import annotations

from src.ingest_chroma import extract_inline_metadata


def test_extract_inline_metadata_when_present() -> None:
    msg, speaker, ts = extract_inline_metadata("[12/29/25, 10:30 PM] Alice: hello")
    assert msg == "hello"
    assert speaker == "Alice"
    assert ts == "12/29/25, 10:30 PM"


def test_extract_inline_metadata_when_absent() -> None:
    msg, speaker, ts = extract_inline_metadata("hello")
    assert msg == "hello"
    assert speaker is None
    assert ts is None

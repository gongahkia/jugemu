from __future__ import annotations

import pytest

from src.ingest_chroma import build_chunks


def test_build_chunks_message_mode() -> None:
    items = [(1, "a"), (2, "b"), (4, "c")]
    chunks = build_chunks(items, chunking="message", window_size=3)
    assert chunks == [(1, 1, "a"), (2, 2, "b"), (4, 4, "c")]


def test_build_chunks_window_mode() -> None:
    items = [(1, "a"), (2, "b"), (4, "c")]
    chunks = build_chunks(items, chunking="window", window_size=2)
    assert chunks == [(1, 2, "a\nb"), (2, 4, "b\nc")]


def test_build_chunks_window_too_large() -> None:
    items = [(1, "a"), (2, "b")]
    assert build_chunks(items, chunking="window", window_size=3) == []


def test_build_chunks_invalid_mode() -> None:
    with pytest.raises(ValueError):
        build_chunks([(1, "a")], chunking="wat", window_size=2)

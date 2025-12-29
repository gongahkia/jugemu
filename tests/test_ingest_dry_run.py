from __future__ import annotations

from pathlib import Path

import pytest

from src.ingest_chroma import ingest_messages


class _FakeStore:
    def __init__(self) -> None:
        self.add_calls: int = 0

    def add(self, *, ids, texts, embeddings, metadatas):  # type: ignore[no-untyped-def]
        self.add_calls += 1

    def query(self, query_embedding, k: int):  # type: ignore[no-untyped-def]
        raise AssertionError("query should not be called")


def test_ingest_dry_run_does_not_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    messages_path = tmp_path / "messages.txt"
    messages_path.write_text("a\nb\n", encoding="utf-8")

    def _boom_embed_texts(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("embed_texts should not be called in dry_run")

    monkeypatch.setattr("src.ingest_chroma.embed_texts", _boom_embed_texts)

    store = _FakeStore()
    would_add = ingest_messages(
        messages_path=messages_path,
        persist_dir=tmp_path / "persist",
        store=store,
        dry_run=True,
        chunking="message",
    )

    assert would_add == 2
    assert store.add_calls == 0

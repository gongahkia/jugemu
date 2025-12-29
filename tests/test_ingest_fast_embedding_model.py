from __future__ import annotations

from pathlib import Path

import pytest

from src.ingest_chroma import DEFAULT_EMBEDDING_MODEL, FAST_EMBEDDING_MODEL, ingest_messages


class _FakeStore:
    def add(self, *, ids, texts, embeddings, metadatas):  # type: ignore[no-untyped-def]
        return None


def test_ingest_fast_embedding_model_switches_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    messages_path = tmp_path / "messages.txt"
    messages_path.write_text("a\n", encoding="utf-8")

    called = {}

    def _fake_embed_texts(texts, model_name, batch_size=None):  # type: ignore[no-untyped-def]
        called["model_name"] = model_name
        return [[0.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr("src.ingest_chroma.embed_texts", _fake_embed_texts)

    ingest_messages(
        messages_path=messages_path,
        persist_dir=tmp_path / "persist",
        store=_FakeStore(),
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        fast_embedding_model=True,
        chunking="message",
    )

    assert called["model_name"] == FAST_EMBEDDING_MODEL

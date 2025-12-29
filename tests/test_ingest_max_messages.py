from __future__ import annotations

from pathlib import Path

import pytest

from src.ingest_chroma import ingest_messages


class _FakeStore:
    def __init__(self) -> None:
        self.add_calls: list[dict] = []

    def add(self, *, ids, texts, embeddings, metadatas):  # type: ignore[no-untyped-def]
        self.add_calls.append(
            {
                "ids": list(ids),
                "texts": list(texts),
                "embeddings": embeddings,
                "metadatas": list(metadatas),
            }
        )


def test_ingest_respects_max_messages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    messages = "a\n\n b \n c\n"
    messages_path = tmp_path / "messages.txt"
    messages_path.write_text(messages, encoding="utf-8")

    calls: dict = {}

    def _fake_embed_texts(texts, model_name, batch_size=None):  # type: ignore[no-untyped-def]
        calls["texts"] = list(texts)
        calls["model_name"] = model_name
        calls["batch_size"] = batch_size
        return [[0.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr("src.ingest_chroma.embed_texts", _fake_embed_texts)

    store = _FakeStore()
    added = ingest_messages(
        messages_path=messages_path,
        persist_dir=tmp_path / "persist",
        store=store,  # avoid touching real chroma
        max_messages=1,
        chunking="message",
    )

    assert added == 1
    assert calls["texts"] == ["a"]
    assert len(store.add_calls) == 1
    assert store.add_calls[0]["texts"] == ["a"]

from __future__ import annotations

from pathlib import Path

import pytest

from src.export_retrieval import dump_random_retrieval_samples
from src.vector_store import Retrieved


class _FakeStore:
    def query(self, query_embedding, k: int):  # type: ignore[no-untyped-def]
        return [Retrieved(id="doc1", text="hello", score=0.5, metadata=None)][:k]


def test_export_retrieval_samples_all_when_samples_exceeds_corpus(tmp_path: Path) -> None:
    messages_path = tmp_path / "messages.txt"
    messages_path.write_text("a\nb\nc\n", encoding="utf-8")

    calls = {}

    def _fake_embedder(texts, model_name, batch_size):  # type: ignore[no-untyped-def]
        calls["texts"] = list(texts)
        calls["model_name"] = model_name
        calls["batch_size"] = batch_size
        return [[0.0] for _ in texts]

    out = dump_random_retrieval_samples(
        messages_path=messages_path,
        store=_FakeStore(),
        embedding_model="m",
        samples=999,
        k=1,
        embed_batch_size=4,
        embedder=_fake_embedder,
    )

    assert calls["texts"] == ["a", "b", "c"]
    assert out[0]["query"] == "a"
    assert out[0]["results"][0]["id"] == "doc1"


def test_export_retrieval_samples_zero(tmp_path: Path) -> None:
    messages_path = tmp_path / "messages.txt"
    messages_path.write_text("a\n", encoding="utf-8")

    out = dump_random_retrieval_samples(
        messages_path=messages_path,
        store=_FakeStore(),
        embedding_model="m",
        samples=0,
        k=1,
    )
    assert out == []


def test_export_retrieval_errors_on_empty_file(tmp_path: Path) -> None:
    messages_path = tmp_path / "messages.txt"
    messages_path.write_text("\n\n", encoding="utf-8")

    with pytest.raises(ValueError):
        dump_random_retrieval_samples(
            messages_path=messages_path,
            store=_FakeStore(),
            embedding_model="m",
            samples=1,
            k=1,
        )

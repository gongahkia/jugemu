from __future__ import annotations

from pathlib import Path

from src.export_retrieval import dump_random_retrieval_samples
from src.vector_store import Retrieved


class _FakeStore:
    def query(self, query_embedding, k: int):  # type: ignore[no-untyped-def]
        return [Retrieved(id="doc1", text="hello", score=0.5, metadata=None)][:k]


def test_export_retrieval_uses_explicit_queries(tmp_path: Path) -> None:
    # messages file still required (we use it as a corpus source / for empty check)
    messages_path = tmp_path / "messages.txt"
    messages_path.write_text("a\nb\n", encoding="utf-8")

    seen = {}

    def _fake_embedder(texts, model_name, batch_size):  # type: ignore[no-untyped-def]
        seen["texts"] = list(texts)
        return [[0.0] for _ in texts]

    out = dump_random_retrieval_samples(
        messages_path=messages_path,
        store=_FakeStore(),
        embedding_model="m",
        queries=["q1", "  ", "q2"],
        samples=999,  # should be ignored
        k=1,
        embedder=_fake_embedder,
    )

    assert seen["texts"] == ["q1", "q2"]
    assert [row["query"] for row in out] == ["q1", "q2"]

from __future__ import annotations

import types

import pytest

from src.chroma_store import Retrieved
from src.rerank import rerank_retrieved


class _DummyCrossEncoder:
    def __init__(self, _name: str):
        self.name = _name

    def predict(self, pairs, batch_size: int = 32):
        # Score by length of doc (second element) so ordering is deterministic.
        return [float(len(doc)) for (_q, doc) in pairs]


def test_rerank_reorders_and_sets_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch sentence_transformers.CrossEncoder to avoid downloads.
    fake_st = types.SimpleNamespace(CrossEncoder=_DummyCrossEncoder)
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", fake_st)

    hits = [
        Retrieved(id="1", text="a", score=0.9, metadata=None),
        Retrieved(id="2", text="aaaa", score=0.8, metadata={"x": 1}),
        Retrieved(id="3", text="aa", score=0.7, metadata=None),
    ]

    out = rerank_retrieved(query="q", hits=hits, model_name="dummy")
    assert [h.id for h in out] == ["2", "3", "1"]
    assert out[0].metadata is not None
    assert out[0].metadata["x"] == 1
    assert "rerank_score" in out[0].metadata

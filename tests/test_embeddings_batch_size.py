from __future__ import annotations

from src import embeddings


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def encode(self, inputs, **kwargs):
        self.calls.append({"inputs": list(inputs), **kwargs})
        # Return objects with .tolist()
        class _V:
            def __init__(self, n: int):
                self.n = n

            def tolist(self):
                return [float(self.n)]

        return [_V(i) for i in range(len(inputs))]


def test_embed_texts_passes_batch_size(monkeypatch):
    fake = _FakeModel()

    def _fake_model(_name: str):
        return fake

    monkeypatch.setattr(embeddings, "_model", _fake_model)

    out = embeddings.embed_texts(["a", "b", "c"], "x", batch_size=7)
    assert out == [[0.0], [1.0], [2.0]]

    call = fake.calls[0]
    assert call["batch_size"] == 7
    assert call["normalize_embeddings"] is True
    assert call["show_progress_bar"] is False

from __future__ import annotations


def test_embed_texts_does_not_pass_none_batch_size(monkeypatch):
    from src import embeddings

    calls = []

    class FakeModel:
        def encode(self, texts, **kwargs):
            calls.append(kwargs)
            assert "batch_size" not in kwargs

            class _Vec:
                def __init__(self, values):
                    self._values = values

                def tolist(self):
                    return list(self._values)

            return [_Vec([0.0, 1.0, 2.0]) for _ in texts]

    monkeypatch.setattr(embeddings, "_model", lambda name: FakeModel())

    out = embeddings.embed_texts(["hi"], "fake-model", batch_size=None)
    assert out == [[0.0, 1.0, 2.0]]
    assert len(calls) == 1

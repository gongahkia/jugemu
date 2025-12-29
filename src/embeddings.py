from __future__ import annotations

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


def embed_texts(texts: List[str], model_name: str, *, batch_size: int | None = None) -> List[List[float]]:
    model = _model(model_name)
    vecs = model.encode(
        texts,
        batch_size=int(batch_size) if batch_size is not None else None,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vecs]


def embed_query(query: str, model_name: str) -> List[float]:
    model = _model(model_name)
    v = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    return v.tolist()

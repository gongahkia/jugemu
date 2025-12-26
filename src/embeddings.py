from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


def embed_texts(texts: List[str], model_name: str) -> List[List[float]]:
    model = _model(model_name)
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]


def embed_query(query: str, model_name: str) -> List[float]:
    model = _model(model_name)
    v = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    return v.tolist()

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

from .chroma_store import Retrieved


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    # Lazy import so this remains an optional "runtime" feature.
    from sentence_transformers import CrossEncoder  # type: ignore

    return CrossEncoder(model_name)


def rerank_retrieved(
    *,
    query: str,
    hits: Sequence[Retrieved],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    batch_size: int = 32,
) -> List[Retrieved]:
    if not hits:
        return []

    model = _load_cross_encoder(str(model_name))
    pairs: List[Tuple[str, str]] = [(query, h.text) for h in hits]

    scores = model.predict(pairs, batch_size=int(batch_size))
    scored: List[tuple[float, Retrieved]] = []

    for h, s in zip(hits, scores):
        try:
            ss = float(s)
        except Exception:
            ss = 0.0
        meta = dict(h.metadata or {})
        meta["rerank_score"] = ss
        scored.append((ss, Retrieved(id=h.id, text=h.text, score=h.score, metadata=meta)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _s, h in scored]

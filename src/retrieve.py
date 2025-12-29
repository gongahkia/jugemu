from __future__ import annotations

from pathlib import Path
from typing import List

from .chroma_store import Retrieved
from .embeddings import embed_query
from .rerank import rerank_retrieved
from .vector_store import ChromaVectorStore, VectorStore


def retrieve_similar(
    persist_dir: str | Path,
    collection_name: str,
    query: str,
    k: int,
    embedding_model: str,
    store: VectorStore | None = None,
    rerank: bool = False,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerank_top_k: int | None = None,
) -> List[Retrieved]:
    if store is None:
        store = ChromaVectorStore(persist_dir=Path(persist_dir), collection_name=collection_name)
    q = embed_query(query, embedding_model)
    initial_k = int(rerank_top_k) if rerank_top_k is not None else int(k)
    if initial_k < int(k):
        initial_k = int(k)
    hits = store.query(q, k=initial_k)
    if not rerank:
        return hits[: int(k)]

    reranked = rerank_retrieved(query=query, hits=hits, model_name=str(rerank_model))
    return reranked[: int(k)]

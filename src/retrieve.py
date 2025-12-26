from __future__ import annotations

from pathlib import Path
from typing import List

from .chroma_store import Retrieved
from .embeddings import embed_query
from .vector_store import ChromaVectorStore


def retrieve_similar(
    persist_dir: str | Path,
    collection_name: str,
    query: str,
    k: int,
    embedding_model: str,
) -> List[Retrieved]:
    store = ChromaVectorStore(persist_dir=Path(persist_dir), collection_name=collection_name)
    q = embed_query(query, embedding_model)
    return store.query(q, k=k)

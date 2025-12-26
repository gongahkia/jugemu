from __future__ import annotations

from pathlib import Path
from typing import List

from .chroma_store import Retrieved, get_client, get_or_create_collection, query_text
from .embeddings import embed_query


def retrieve_similar(
    persist_dir: str | Path,
    collection_name: str,
    query: str,
    k: int,
    embedding_model: str,
) -> List[Retrieved]:
    client = get_client(persist_dir)
    collection = get_or_create_collection(client, collection_name)
    q = embed_query(query, embedding_model)
    return query_text(collection, q, k=k)

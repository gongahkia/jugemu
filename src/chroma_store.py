from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import chromadb
from chromadb.config import Settings


@dataclass(frozen=True)
class Retrieved:
    id: str
    text: str
    score: float
    metadata: dict | None = None


def get_client(persist_dir: str | Path) -> chromadb.PersistentClient:
    p = Path(persist_dir)
    p.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(p), settings=Settings(anonymized_telemetry=False))


def get_or_create_collection(
    client: chromadb.PersistentClient,
    name: str,
) -> chromadb.api.models.Collection.Collection:
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def add_texts(
    collection,
    ids: Sequence[str],
    texts: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    metadatas: Optional[Sequence[dict]] = None,
) -> None:
    collection.add(ids=list(ids), documents=list(texts), embeddings=list(embeddings), metadatas=metadatas)


def query_text(
    collection,
    query_embedding: Sequence[float],
    k: int = 5,
) -> List[Retrieved]:
    res = collection.query(
        query_embeddings=[list(query_embedding)],
        n_results=k,
        include=["documents", "distances", "metadatas"],
    )
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    dists = res.get("distances", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    out: List[Retrieved] = []
    for _id, doc, dist, meta in zip(ids, docs, dists, metas):
        # With cosine space in Chroma, distance is usually (1 - cosine_similarity).
        # Convert to similarity-ish score for display.
        score = 1.0 - float(dist)
        out.append(Retrieved(id=str(_id), text=str(doc), score=score, metadata=meta or None))
    return out

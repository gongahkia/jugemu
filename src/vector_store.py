from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Sequence

from .chroma_store import Retrieved, add_texts as chroma_add_texts
from .chroma_store import get_client as chroma_get_client
from .chroma_store import get_or_create_collection as chroma_get_or_create_collection
from .chroma_store import query_text as chroma_query_text


class VectorStore(Protocol):
    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[dict]] = None,
    ) -> None: ...

    def query(self, query_embedding: Sequence[float], k: int) -> List[Retrieved]: ...


@dataclass
class ChromaVectorStore:
    persist_dir: Path
    collection_name: str

    def __post_init__(self) -> None:
        self.persist_dir = Path(self.persist_dir)

    @property
    def _collection(self):
        client = chroma_get_client(self.persist_dir)
        return chroma_get_or_create_collection(client, self.collection_name)

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[dict]] = None,
    ) -> None:
        chroma_add_texts(self._collection, ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)

    def query(self, query_embedding: Sequence[float], k: int) -> List[Retrieved]:
        return chroma_query_text(self._collection, query_embedding=query_embedding, k=k)


@dataclass
class CassandraVectorStore:
    # Placeholder for practitioners who want a distributed backend.
    # Intended shape: store vectors in Cassandra (or Astra) and run vector similarity queries.
    contact_points: List[str]
    keyspace: str
    table: str

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[dict]] = None,
    ) -> None:
        raise NotImplementedError(
            "CassandraVectorStore is a skeleton. Next step: wire DataStax driver session, "
            "create schema, and implement vector similarity queries."
        )

    def query(self, query_embedding: Sequence[float], k: int) -> List[Retrieved]:
        raise NotImplementedError(
            "CassandraVectorStore is a skeleton. Next step: implement SELECT ... ORDER BY vector_distance(...) LIMIT k."
        )

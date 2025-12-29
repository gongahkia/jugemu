from __future__ import annotations

from pathlib import Path

from .vector_backend import VectorBackend, parse_vector_backend
from .vector_store import CassandraVectorStore, ChromaVectorStore, VectorStore


def make_vector_store(
    *,
    backend: str | VectorBackend,
    persist_dir: str | Path,
    collection_name: str,
    cassandra_contact_points: list[str] | None = None,
    cassandra_keyspace: str = "jugemu",
    cassandra_table: str = "messages",
    cassandra_embedding_dim: int = 384,
    cassandra_username: str | None = None,
    cassandra_password: str | None = None,
    cassandra_secure_connect_bundle: str | Path | None = None,
) -> VectorStore:
    b = parse_vector_backend(str(backend))
    if b == "chroma":
        return ChromaVectorStore(persist_dir=Path(persist_dir), collection_name=str(collection_name))

    cps = cassandra_contact_points or ["127.0.0.1"]
    return CassandraVectorStore(
        contact_points=list(cps),
        keyspace=str(cassandra_keyspace),
        table=str(cassandra_table),
        embedding_dim=int(cassandra_embedding_dim),
        username=cassandra_username,
        password=cassandra_password,
        secure_connect_bundle=Path(cassandra_secure_connect_bundle) if cassandra_secure_connect_bundle else None,
    )

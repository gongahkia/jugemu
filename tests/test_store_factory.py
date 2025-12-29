from __future__ import annotations

from pathlib import Path

from src.store_factory import make_vector_store
from src.vector_store import CassandraVectorStore, ChromaVectorStore


def test_make_vector_store_chroma() -> None:
    store = make_vector_store(
        backend="chroma",
        persist_dir=Path("data/chroma"),
        collection_name="messages",
    )
    assert isinstance(store, ChromaVectorStore)
    assert store.collection_name == "messages"


def test_make_vector_store_cassandra_defaults() -> None:
    store = make_vector_store(
        backend="cassandra",
        persist_dir=Path("data/chroma"),
        collection_name="messages",
    )
    assert isinstance(store, CassandraVectorStore)
    assert store.contact_points == ["127.0.0.1"]
    assert store.keyspace == "jugemu"
    assert store.table == "messages"


def test_make_vector_store_cassandra_custom() -> None:
    store = make_vector_store(
        backend="astra",
        persist_dir=Path("data/chroma"),
        collection_name="messages",
        cassandra_contact_points=["10.0.0.1"],
        cassandra_keyspace="ks",
        cassandra_table="tbl",
        cassandra_embedding_dim=123,
        cassandra_secure_connect_bundle="/tmp/bundle.zip",
        cassandra_username="u",
        cassandra_password="p",
    )
    assert isinstance(store, CassandraVectorStore)
    assert store.contact_points == ["10.0.0.1"]
    assert store.keyspace == "ks"
    assert store.table == "tbl"
    assert store.embedding_dim == 123
    assert str(store.secure_connect_bundle) == "/tmp/bundle.zip"
    assert store.username == "u"
    assert store.password == "p"

from __future__ import annotations

import pytest

from src.vector_store import CassandraVectorStore, ChromaVectorStore
from src.vector_store_schema import ensure_schema_version, get_schema_version, schema_info, set_schema_version


def test_schema_info_includes_version_and_backends() -> None:
    info = schema_info()
    assert isinstance(info["schema_version"], int)
    assert info["vector_backends"] == ["chroma", "cassandra"]
    assert info["chroma"]["schema_file"]
    assert info["cassandra"]["meta_key"] == "schema_version"


def test_chroma_schema_version_roundtrip(tmp_path):
    store = ChromaVectorStore(persist_dir=tmp_path, collection_name="messages")

    assert get_schema_version(store) is None

    ensure_schema_version(store, expected=5)
    assert get_schema_version(store) == 5

    set_schema_version(store, 7)
    assert get_schema_version(store) == 7

    with pytest.raises(ValueError):
        ensure_schema_version(store, expected=6)


class _FakeSession:
    def __init__(self) -> None:
        self.meta: dict[str, str] = {}

    def execute(self, statement, params=None):  # type: ignore[no-untyped-def]
        stmt = str(statement)

        if "INSERT INTO jugemu.jugemu_meta" in stmt and params is not None:
            key, value = params
            self.meta[str(key)] = str(value)
            return []

        if "SELECT value FROM jugemu.jugemu_meta" in stmt and params is not None:
            (key,) = params
            if str(key) in self.meta:
                return [{"value": self.meta[str(key)]}]
            return []

        return []

    def prepare(self, cql: str):  # type: ignore[no-untyped-def]
        return cql


def test_cassandra_schema_version_roundtrip():
    sess = _FakeSession()
    store = CassandraVectorStore(
        contact_points=["127.0.0.1"],
        keyspace="jugemu",
        table="messages",
        session=sess,
    )

    assert get_schema_version(store) is None

    ensure_schema_version(store, expected=3)
    assert get_schema_version(store) == 3

    set_schema_version(store, 4)
    assert get_schema_version(store) == 4

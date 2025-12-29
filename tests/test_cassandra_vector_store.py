from __future__ import annotations

import json

import pytest

from src.vector_store import CassandraVectorStore


class FakeSession:
    def __init__(self) -> None:
        self.executed: list[tuple[object, tuple | None]] = []
        self.prepared: list[str] = []

    def execute(self, statement, params=None):
        self.executed.append((statement, params))
        return []

    def prepare(self, cql: str):
        self.prepared.append(cql)
        # represent a prepared statement by its CQL string for our purposes
        return cql


def test_cassandra_schema_cql_generation() -> None:
    store = CassandraVectorStore(
        contact_points=["127.0.0.1"],
        keyspace="jugemu",
        table="messages",
        embedding_dim=8,
        create_schema=True,
        session=FakeSession(),
    )

    cql_table = store._cql_create_table()
    assert "CREATE TABLE IF NOT EXISTS jugemu.messages" in cql_table
    assert "embedding vector<float, 8>" in cql_table

    cql_idx = store._cql_create_index()
    assert "CREATE CUSTOM INDEX IF NOT EXISTS" in cql_idx
    assert "ON jugemu.messages (embedding)" in cql_idx


def test_cassandra_add_executes_inserts_with_metadata_json() -> None:
    sess = FakeSession()
    store = CassandraVectorStore(
        contact_points=["127.0.0.1"],
        keyspace="jugemu",
        table="messages",
        embedding_dim=3,
        create_schema=True,
        session=sess,
    )

    store.add(
        ids=["a"],
        texts=["hello"],
        embeddings=[[0.1, 0.2, 0.3]],
        metadatas=[{"speaker": "me"}],
    )

    # schema best-effort then insert
    assert any("CREATE TABLE IF NOT EXISTS" in str(stmt) for stmt, _ in sess.executed)
    assert any("INSERT INTO jugemu.messages" in str(stmt) for stmt, _ in sess.executed)

    insert_calls = [(stmt, params) for stmt, params in sess.executed if "INSERT INTO" in str(stmt)]
    assert len(insert_calls) == 1
    _stmt, params = insert_calls[0]
    assert params is not None
    assert params[0] == "a"
    assert params[1] == "hello"
    assert params[2] == [0.1, 0.2, 0.3]
    assert json.loads(params[3]) == {"speaker": "me"}


def test_cassandra_query_builds_ann_cql_with_score_function() -> None:
    sess = FakeSession()
    store = CassandraVectorStore(
        contact_points=["127.0.0.1"],
        keyspace="jugemu",
        table="messages",
        embedding_dim=3,
        create_schema=False,
        session=sess,
        score_function="similarity_cosine",
    )

    cql = store._cql_query(5)
    assert "ORDER BY embedding ANN OF" in cql
    assert "similarity_cosine(embedding, ?) AS score" in cql

    store.query([0.1, 0.2, 0.3], k=5)

    # last execute should be the ANN query with vector bound twice and limit
    last_stmt, last_params = sess.executed[-1]
    assert "ORDER BY embedding ANN OF" in str(last_stmt)
    assert last_params == ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], 5)


def test_cassandra_add_rejects_mismatched_lengths() -> None:
    store = CassandraVectorStore(
        contact_points=["127.0.0.1"],
        keyspace="jugemu",
        table="messages",
        session=FakeSession(),
    )
    with pytest.raises(ValueError):
        store.add(ids=["a"], texts=["x", "y"], embeddings=[[0.0]])

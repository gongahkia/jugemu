from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, List, Optional, Protocol, Sequence

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

    # Vector schema settings
    embedding_dim: int = 384
    similarity_function: str = "cosine"  # used when creating index (best-effort)
    score_function: str | None = "similarity_cosine"  # best-effort score computation

    # Connection settings (local Cassandra)
    port: int = 9042
    local_dc: str | None = None
    username: str | None = None
    password: str | None = None

    # Astra settings
    secure_connect_bundle: Path | None = None

    # Schema management
    create_schema: bool = True
    index_name: str | None = None

    # For unit tests / dependency injection
    session: Any | None = None

    _schema_ensured: bool = False

    def __post_init__(self) -> None:
        if self.secure_connect_bundle is not None:
            self.secure_connect_bundle = Path(self.secure_connect_bundle)

    def _require_session(self):
        if self.session is not None:
            return self.session

        try:
            from cassandra.cluster import Cluster  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "cassandra-driver is required for CassandraVectorStore. "
                "Install it with `pip install cassandra-driver`."
            ) from e

        auth_provider = None
        if self.username is not None and self.password is not None:
            from cassandra.auth import PlainTextAuthProvider  # type: ignore

            auth_provider = PlainTextAuthProvider(username=self.username, password=self.password)

        if self.secure_connect_bundle is not None:
            cloud_config = {"secure_connect_bundle": str(self.secure_connect_bundle)}
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
            self.session = cluster.connect()
            return self.session

        cluster = Cluster(
            contact_points=list(self.contact_points),
            port=int(self.port),
            auth_provider=auth_provider,
            local_dc=self.local_dc,
        )
        self.session = cluster.connect()
        return self.session

    def _cql_create_keyspace(self) -> str:
        # Simple replication suitable for local/dev. Astra keyspaces are usually pre-created.
        return (
            f"CREATE KEYSPACE IF NOT EXISTS {self.keyspace} "
            "WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}"
        )

    def _cql_create_table(self) -> str:
        dim = int(self.embedding_dim)
        if dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        return (
            f"CREATE TABLE IF NOT EXISTS {self.keyspace}.{self.table} (\n"
            "  id text PRIMARY KEY,\n"
            "  text text,\n"
            f"  embedding vector<float, {dim}>,\n"
            "  metadata_json text\n"
            ")"
        )

    def _cql_create_index(self) -> str:
        idx = self.index_name or f"{self.table}_embedding_idx"
        # Best-effort: Cassandra versions differ in index syntax/options.
        # This is a reasonable default for SAI-enabled clusters.
        # If your cluster rejects it, create the index manually and set create_schema=False.
        sim = str(self.similarity_function or "cosine").lower()
        return (
            f"CREATE CUSTOM INDEX IF NOT EXISTS {idx} "
            f"ON {self.keyspace}.{self.table} (embedding) "
            "USING 'StorageAttachedIndex' "
            f"WITH OPTIONS = {{'similarity_function': '{sim}'}}"
        )

    def _cql_insert(self) -> str:
        return f"INSERT INTO {self.keyspace}.{self.table} (id, text, embedding, metadata_json) VALUES (?, ?, ?, ?)"

    def _cql_query(self, k: int) -> str:
        # Vector query syntax varies by cluster/version; this targets Cassandra/Astra style ANN.
        # We bind the query vector twice if also computing score.
        limit = int(k)
        if limit <= 0:
            raise ValueError("k must be > 0")

        if self.score_function:
            fn = str(self.score_function)
            return (
                "SELECT id, text, metadata_json, "
                f"{fn}(embedding, ?) AS score "
                f"FROM {self.keyspace}.{self.table} "
                "ORDER BY embedding ANN OF ? "
                "LIMIT ?"
            )
        return (
            "SELECT id, text, metadata_json "
            f"FROM {self.keyspace}.{self.table} "
            "ORDER BY embedding ANN OF ? "
            "LIMIT ?"
        )

    def _ensure_schema(self) -> None:
        if self._schema_ensured or not self.create_schema:
            return

        session = self._require_session()
        try:
            # Keyspace creation may fail on Astra (pre-created); allow users to disable schema.
            session.execute(self._cql_create_keyspace())
        except Exception:
            pass
        session.execute(self._cql_create_table())
        try:
            session.execute(self._cql_create_index())
        except Exception:
            # Index syntax varies; table creation is the critical part.
            pass
        self._schema_ensured = True

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[dict]] = None,
    ) -> None:
        if not (len(ids) == len(texts) == len(embeddings)):
            raise ValueError("ids, texts, embeddings must have the same length")
        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError("metadatas must be None or same length as ids")
        if not ids:
            return

        self._ensure_schema()
        session = self._require_session()
        insert_cql = self._cql_insert()
        try:
            prepared = session.prepare(insert_cql)
        except Exception:
            prepared = insert_cql

        for i, (doc_id, text, emb) in enumerate(zip(ids, texts, embeddings)):
            meta = None
            if metadatas is not None:
                meta = metadatas[i]
            meta_json = json.dumps(meta, ensure_ascii=False) if meta is not None else None
            params = (str(doc_id), str(text), list(map(float, emb)), meta_json)
            session.execute(prepared, params)

    def query(self, query_embedding: Sequence[float], k: int) -> List[Retrieved]:
        self._ensure_schema()
        session = self._require_session()

        q = list(map(float, query_embedding))
        cql = self._cql_query(int(k))
        try:
            prepared = session.prepare(cql)
        except Exception:
            prepared = cql

        if self.score_function:
            rows = session.execute(prepared, (q, q, int(k)))
        else:
            rows = session.execute(prepared, (q, int(k)))

        out: List[Retrieved] = []
        for row in rows:
            # Support both driver Row objects and dicts (tests).
            row_id = getattr(row, "id", None) if not isinstance(row, dict) else row.get("id")
            row_text = getattr(row, "text", None) if not isinstance(row, dict) else row.get("text")
            meta_json = (
                getattr(row, "metadata_json", None) if not isinstance(row, dict) else row.get("metadata_json")
            )
            score = getattr(row, "score", None) if not isinstance(row, dict) else row.get("score")
            meta: dict | None = None
            if meta_json:
                try:
                    meta = json.loads(meta_json)
                except Exception:
                    meta = None
            out.append(
                Retrieved(
                    id=str(row_id),
                    text=str(row_text),
                    score=float(score) if score is not None else 0.0,
                    metadata=meta,
                )
            )
        return out

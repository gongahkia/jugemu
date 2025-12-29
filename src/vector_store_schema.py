from __future__ import annotations

import json
from pathlib import Path
import shutil

from .vector_store import CassandraVectorStore, ChromaVectorStore, VectorStore


VECTOR_STORE_SCHEMA_VERSION = 1

_SCHEMA_FILENAME = ".jugemu_vector_store_schema.json"


def _schema_path(persist_dir: Path) -> Path:
    return Path(persist_dir) / _SCHEMA_FILENAME


def _read_chroma_versions(persist_dir: Path) -> dict[str, int]:
    p = _schema_path(persist_dir)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}
    raw = data.get("collections")
    if not isinstance(raw, dict):
        return {}

    out: dict[str, int] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, int):
            out[k] = int(v)
    return out


def _write_chroma_versions(persist_dir: Path, versions: dict[str, int]) -> None:
    d = Path(persist_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = _schema_path(d)
    payload = {"collections": {str(k): int(v) for k, v in versions.items()}}
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def get_schema_version(store: VectorStore) -> int | None:
    if isinstance(store, ChromaVectorStore):
        versions = _read_chroma_versions(store.persist_dir)
        v = versions.get(str(store.collection_name))
        return int(v) if isinstance(v, int) else None

    if isinstance(store, CassandraVectorStore):
        raw = store.get_meta("schema_version")
        if raw is None:
            return None
        try:
            return int(raw)
        except Exception:
            return None

    return None


def set_schema_version(store: VectorStore, version: int) -> None:
    v = int(version)

    if isinstance(store, ChromaVectorStore):
        versions = _read_chroma_versions(store.persist_dir)
        versions[str(store.collection_name)] = v
        _write_chroma_versions(store.persist_dir, versions)
        return

    if isinstance(store, CassandraVectorStore):
        store.set_meta("schema_version", str(v))
        return


def ensure_schema_version(store: VectorStore, *, expected: int = VECTOR_STORE_SCHEMA_VERSION) -> None:
    want = int(expected)
    cur = get_schema_version(store)

    if cur is None:
        set_schema_version(store, want)
        return

    if int(cur) != want:
        raise ValueError(
            f"Vector store schema version mismatch (found {cur}, expected {want}). "
            "Rebuild the store to migrate (see `jugemu rebuild-store --help`)."
        )


def reset_vector_store(store: VectorStore) -> None:
    """Best-effort destructive reset of the vector store contents."""
    if isinstance(store, ChromaVectorStore):
        shutil.rmtree(store.persist_dir, ignore_errors=True)
        return

    if isinstance(store, CassandraVectorStore):
        session = store._require_session()
        for cql in [
            f"DROP TABLE IF EXISTS {store.keyspace}.{store.table}",
            f"DROP TABLE IF EXISTS {store.keyspace}.{store.meta_table}",
        ]:
            try:
                session.execute(cql)
            except Exception:
                pass
        store._schema_ensured = False

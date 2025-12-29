from __future__ import annotations

from typing import Literal


VectorBackend = Literal["chroma", "cassandra"]


def parse_vector_backend(value: str) -> VectorBackend:
    v = str(value or "").strip().lower()
    if v in {"chroma", "chroma-db", "chromadb"}:
        return "chroma"
    if v in {"cassandra", "astra", "astra-db", "astradb"}:
        return "cassandra"
    raise ValueError("vector backend must be one of: chroma|cassandra")

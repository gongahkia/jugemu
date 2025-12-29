from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, List, Sequence

from rich.console import Console

from .embeddings import embed_texts
from .ingest_chroma import read_messages_lines
from .vector_store import Retrieved, VectorStore


Embedder = Callable[[Sequence[str], str, int | None], List[Sequence[float]]]


def write_retrieval_samples(
    results: list[dict],
    *,
    out: Path,
    fmt: str,
) -> Path:
    """Write retrieval samples to disk.

    fmt: json|jsonl
    """
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    f = str(fmt or "jsonl").strip().lower()
    if f == "json":
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return out_path
    if f == "jsonl":
        with out_path.open("w", encoding="utf-8") as fp:
            for row in results:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        return out_path

    raise ValueError("fmt must be one of: json|jsonl")


def dump_random_retrieval_samples(
    *,
    messages_path: Path,
    store: VectorStore,
    embedding_model: str,
    samples: int = 10,
    k: int = 5,
    seed: int | None = None,
    embed_batch_size: int | None = None,
    console: Console | None = None,
    embedder: Embedder = embed_texts,
) -> list[dict]:
    """Sample random queries from messages and print their retrieved neighbors.

    Returns structured results for testing/automation.
    """
    items = read_messages_lines(messages_path)
    texts = [t for _ln, t in items]
    if not texts:
        raise ValueError("No messages found (expected one message per line)")

    n = int(samples)
    if n < 0:
        n = 0
    if n == 0:
        return []

    rng = random.Random(seed)
    picked = texts if n >= len(texts) else rng.sample(texts, k=n)

    embs = embedder(picked, str(embedding_model), embed_batch_size)
    if len(embs) != len(picked):
        raise ValueError("Embedder returned wrong number of embeddings")

    results: list[dict] = []
    for i, (q, emb) in enumerate(zip(picked, embs), start=1):
        retrieved: List[Retrieved] = store.query(list(map(float, emb)), k=int(k))
        row = {
            "query": q,
            "results": [
                {
                    "id": r.id,
                    "score": float(r.score),
                    "text": r.text,
                    "metadata": r.metadata,
                }
                for r in retrieved
            ],
        }
        results.append(row)

        if console is not None:
            console.print(f"[bold]Query {i}[/bold]: {q}")
            for j, r in enumerate(retrieved, start=1):
                console.print(f"  {j}. score={r.score:.4f} id={r.id} text={r.text}")
            console.print()

    return results

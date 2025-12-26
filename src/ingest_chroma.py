from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import List, Tuple

from .embeddings import embed_texts
from .vector_store import ChromaVectorStore


def read_messages_lines(path: Path) -> List[Tuple[int, str]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Each non-empty line = one message. Keep short lines too.
    out: List[Tuple[int, str]] = []
    for i, ln in enumerate(raw.split("\n"), start=1):
        msg = ln.strip()
        if msg:
            out.append((i, msg))
    return out


def stable_id(line_no: int, text: str) -> str:
    # Chroma requires IDs to be unique per add() call.
    # Message corpora often contain duplicate texts ("ok", "lol", etc),
    # so we make IDs deterministic per line.
    h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    return f"{h}:{line_no}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", required=True, help="Text file: one message per line")
    ap.add_argument("--persist", required=True, help="ChromaDB persistence directory")
    ap.add_argument("--collection", default="messages")
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    messages_path = Path(args.messages)
    items = read_messages_lines(messages_path)
    if not items:
        raise SystemExit("No messages found (expected one message per line)")

    store = ChromaVectorStore(persist_dir=Path(args.persist), collection_name=args.collection)

    # Upsert-ish: only add messages whose IDs are not present.
    # Chroma doesn't have perfect upsert in all versions; this is a simple best-effort.
    ids = [stable_id(line_no, t) for line_no, t in items]

    # Determine existing ids (may be large; keep it simple for MVP).
    existing = set()
    try:
        got = store._collection.get(include=[])
        existing = set(got.get("ids", []))
    except Exception:
        existing = set()

    texts = [t for _, t in items]
    line_nos = [line_no for line_no, _ in items]

    new_pairs = [(i, t, ln) for i, t, ln in zip(ids, texts, line_nos) if i not in existing]
    if not new_pairs:
        print("Nothing new to ingest.")
        return

    new_ids = [p[0] for p in new_pairs]
    new_texts = [p[1] for p in new_pairs]
    new_line_nos = [p[2] for p in new_pairs]

    for start in range(0, len(new_texts), args.batch):
        chunk_texts = new_texts[start : start + args.batch]
        chunk_ids = new_ids[start : start + args.batch]
        chunk_lines = new_line_nos[start : start + args.batch]
        chunk_emb = embed_texts(chunk_texts, args.embedding_model)
        metadatas = [{"line": int(ln), "source": str(messages_path)} for ln in chunk_lines]
        store.add(ids=chunk_ids, texts=chunk_texts, embeddings=chunk_emb, metadatas=metadatas)

    print(f"Ingested {len(new_texts)} new messages into collection '{args.collection}'.")


if __name__ == "__main__":
    main()

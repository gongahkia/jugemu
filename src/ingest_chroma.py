from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import List

from .chroma_store import add_texts, get_client, get_or_create_collection
from .embeddings import embed_texts


def read_messages_lines(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Each non-empty line = one message. Keep short lines too.
    lines = [ln.strip() for ln in raw.split("\n")]
    return [ln for ln in lines if ln]


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", required=True, help="Text file: one message per line")
    ap.add_argument("--persist", required=True, help="ChromaDB persistence directory")
    ap.add_argument("--collection", default="messages")
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    messages_path = Path(args.messages)
    texts = read_messages_lines(messages_path)
    if not texts:
        raise SystemExit("No messages found (expected one message per line)")

    client = get_client(args.persist)
    collection = get_or_create_collection(client, args.collection)

    # Upsert-ish: only add messages whose IDs are not present.
    # Chroma doesn't have perfect upsert in all versions; this is a simple best-effort.
    ids = [stable_id(t) for t in texts]

    # Determine existing ids (may be large; keep it simple for MVP).
    existing = set()
    try:
        got = collection.get(include=[])
        existing = set(got.get("ids", []))
    except Exception:
        existing = set()

    new_pairs = [(i, t) for i, t in zip(ids, texts) if i not in existing]
    if not new_pairs:
        print("Nothing new to ingest.")
        return

    new_ids = [p[0] for p in new_pairs]
    new_texts = [p[1] for p in new_pairs]

    for start in range(0, len(new_texts), args.batch):
        chunk_texts = new_texts[start : start + args.batch]
        chunk_ids = new_ids[start : start + args.batch]
        chunk_emb = embed_texts(chunk_texts, args.embedding_model)
        add_texts(collection, ids=chunk_ids, texts=chunk_texts, embeddings=chunk_emb)

    print(f"Ingested {len(new_texts)} new messages into collection '{args.collection}'.")


if __name__ == "__main__":
    main()

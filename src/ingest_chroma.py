from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import List, Tuple

from rich.console import Console

from .embeddings import embed_texts
from .normalize import normalize_message_line
from .redact import redact_text
from .vector_store import ChromaVectorStore


def read_messages_lines(
    path: Path,
    *,
    collapse_whitespace: bool = False,
    strip_emoji: bool = False,
) -> List[Tuple[int, str]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Each non-empty line = one message. Keep short lines too.
    out: List[Tuple[int, str]] = []
    for i, ln in enumerate(raw.split("\n"), start=1):
        msg = normalize_message_line(
            ln,
            collapse_whitespace=bool(collapse_whitespace),
            strip_emoji_chars=bool(strip_emoji),
        )
        if msg:
            out.append((i, msg))
    return out


def stable_id(line_no: int, text: str) -> str:
    # Chroma requires IDs to be unique per add() call.
    # Message corpora often contain duplicate texts ("ok", "lol", etc),
    # so we make IDs deterministic per line.
    h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    return f"{h}:{line_no}"


def stable_window_id(start_line: int, end_line: int, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    return f"{h}:{start_line}-{end_line}"


def build_chunks(
    items: List[Tuple[int, str]],
    *,
    chunking: str,
    window_size: int,
) -> List[Tuple[int, int, str]]:
    """Return chunks as (start_line, end_line, text)."""
    mode = str(chunking or "message").strip().lower()
    if mode in {"message", "per-message", "per_message", "line"}:
        return [(ln, ln, msg) for ln, msg in items]
    if mode in {"window", "sliding", "sliding-window", "sliding_window"}:
        n = int(window_size)
        if n < 1:
            n = 1
        if len(items) < n:
            return []
        out: List[Tuple[int, int, str]] = []
        for i in range(0, len(items) - n + 1):
            win = items[i : i + n]
            start_ln = int(win[0][0])
            end_ln = int(win[-1][0])
            text = "\n".join(m for _ln, m in win)
            out.append((start_ln, end_ln, text))
        return out
    raise ValueError("chunking must be one of: message|window")


def ingest_messages(
    *,
    messages_path: Path,
    persist_dir: Path,
    collection_name: str = "messages",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch: int = 256,
    chunking: str = "message",
    window_size: int = 4,
    collapse_whitespace: bool = False,
    strip_emoji: bool = False,
    redact: bool = False,
    redact_types: List[str] | None = None,
    console: Console | None = None,
) -> int:
    items = read_messages_lines(
        messages_path,
        collapse_whitespace=bool(collapse_whitespace),
        strip_emoji=bool(strip_emoji),
    )
    if not items:
        raise ValueError("No messages found (expected one message per line)")

    store = ChromaVectorStore(persist_dir=persist_dir, collection_name=collection_name)

    chunks = build_chunks(items, chunking=str(chunking), window_size=int(window_size))
    if not chunks:
        raise ValueError("No chunks produced (try a smaller --window-size or use --chunking message)")

    ids = [
        stable_id(end_ln, t) if str(chunking).strip().lower() in {"message", "per-message", "per_message", "line"}
        else stable_window_id(start_ln, end_ln, t)
        for (start_ln, end_ln, t) in chunks
    ]

    existing = set()
    try:
        got = store._collection.get(include=[])
        existing = set(got.get("ids", []))
    except Exception:
        existing = set()

    texts_raw = [t for (_s, _e, t) in chunks]
    texts = [redact_text(t, types=redact_types) for t in texts_raw] if redact else list(texts_raw)
    bounds = [(s, e) for (s, e, _t) in chunks]
    new_pairs = [(i, t, s, e) for i, t, (s, e) in zip(ids, texts, bounds) if i not in existing]
    if not new_pairs:
        return 0

    new_ids = [p[0] for p in new_pairs]
    new_texts = [p[1] for p in new_pairs]
    new_starts = [p[2] for p in new_pairs]
    new_ends = [p[3] for p in new_pairs]

    for start in range(0, len(new_texts), batch):
        chunk_texts = new_texts[start : start + batch]
        chunk_ids = new_ids[start : start + batch]
        chunk_starts = new_starts[start : start + batch]
        chunk_ends = new_ends[start : start + batch]
        chunk_emb = embed_texts(chunk_texts, embedding_model)
        mode = str(chunking or "message").strip().lower()
        metadatas = [
            {
                "line": int(end_ln),
                "start_line": int(start_ln),
                "end_line": int(end_ln),
                "chunking": mode,
                "window_size": int(window_size) if mode == "window" else 1,
                "source": str(messages_path),
            }
            for start_ln, end_ln in zip(chunk_starts, chunk_ends)
        ]
        store.add(ids=chunk_ids, texts=chunk_texts, embeddings=chunk_emb, metadatas=metadatas)
        if console is not None:
            console.log(f"Added {min(start + batch, len(new_texts))}/{len(new_texts)}")

    return len(new_texts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", required=True, help="Text file: one message per line")
    ap.add_argument("--persist", required=True, help="ChromaDB persistence directory")
    ap.add_argument("--collection", default="messages")
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument(
        "--chunking",
        default="message",
        choices=["message", "window"],
        help="How to chunk messages for embeddings: message|window.",
    )
    ap.add_argument(
        "--window-size",
        type=int,
        default=4,
        help="Sliding window size (only used when --chunking window).",
    )
    ap.add_argument(
        "--collapse-whitespace",
        action="store_true",
        help="Collapse repeated spaces/tabs in each message before embedding.",
    )
    ap.add_argument(
        "--strip-emoji",
        action="store_true",
        help="Remove emoji characters from messages before embedding.",
    )
    ap.add_argument("--redact", action="store_true", help="Redact emails/phones/addresses before embedding/storing")
    ap.add_argument(
        "--redact-type",
        action="append",
        default=[],
        help="Redaction type (repeatable): email|phone|address. Default: all.",
    )
    args = ap.parse_args()

    console = Console()
    try:
        added = ingest_messages(
            messages_path=Path(args.messages),
            persist_dir=Path(args.persist),
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            batch=args.batch,
            chunking=str(args.chunking),
            window_size=int(args.window_size),
            collapse_whitespace=bool(args.collapse_whitespace),
            strip_emoji=bool(args.strip_emoji),
            redact=bool(args.redact),
            redact_types=list(args.redact_type or []),
            console=console,
        )
    except ValueError as e:
        raise SystemExit(str(e))

    if added == 0:
        console.print("Nothing new to ingest.")
    else:
        console.print(f"Ingested {added} new messages into collection '{args.collection}'.")


if __name__ == "__main__":
    main()

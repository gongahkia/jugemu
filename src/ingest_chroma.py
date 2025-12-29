from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path
from typing import List, Tuple

from rich.console import Console

from .embeddings import embed_texts
from .normalize import normalize_message_line
from .redact import redact_text
from .vector_store_schema import VECTOR_STORE_SCHEMA_VERSION, ensure_schema_version
from .vector_store import ChromaVectorStore, VectorStore


_INLINE_META_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+(?P<speaker>[^:]{1,80}):\s*(?P<msg>.*)$"
)


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAST_EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"


def extract_inline_metadata(text: str) -> tuple[str, str | None, str | None]:
    """Extract [timestamp] Speaker: message prefix when present.

    Returns (clean_text, speaker, timestamp).
    """
    t = text.strip()
    m = _INLINE_META_RE.match(t)
    if not m:
        return t, None, None
    ts = m.group("ts").strip() or None
    speaker = m.group("speaker").strip() or None
    msg = (m.group("msg") or "").strip()
    return msg, speaker, ts


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


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _simhash64(text: str) -> int:
    # Simple simhash over lowercase word tokens.
    toks = re.findall(r"\w+", text.lower())
    if not toks:
        return 0
    weights = [0] * 64
    for tok in toks:
        h = int(hashlib.md5(tok.encode("utf-8", errors="ignore")).hexdigest(), 16)
        for i in range(64):
            bit = (h >> i) & 1
            weights[i] += 1 if bit else -1
    out = 0
    for i, w in enumerate(weights):
        if w >= 0:
            out |= 1 << i
    return out


def _hamming64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def fuzzy_dedupe_texts(
    texts: List[str],
    *,
    max_hamming: int = 6,
) -> List[bool]:
    """Return keep-mask for texts using simhash bucketing (within-run only)."""
    keep = [True] * len(texts)
    buckets: dict[int, List[tuple[int, int]]] = {}

    # 4 bands of 16 bits.
    def bucket_keys(h: int) -> List[int]:
        return [
            (h >> 0) & 0xFFFF,
            (h >> 16) & 0xFFFF,
            (h >> 32) & 0xFFFF,
            (h >> 48) & 0xFFFF,
        ]

    hashes: List[int] = [_simhash64(t) for t in texts]
    for idx, h in enumerate(hashes):
        if not texts[idx].strip():
            keep[idx] = False
            continue
        dup = False
        for key in bucket_keys(h):
            for (j, hj) in buckets.get(key, []):
                if _hamming64(h, hj) <= int(max_hamming):
                    dup = True
                    break
            if dup:
                break
        if dup:
            keep[idx] = False
            continue

        for key in bucket_keys(h):
            buckets.setdefault(key, []).append((idx, h))

    return keep


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
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    fast_embedding_model: bool = False,
    batch: int = 256,
    embed_batch_size: int | None = None,
    chunking: str = "message",
    window_size: int = 4,
    max_messages: int | None = None,
    exact_dedupe: bool = True,
    fuzzy_dedupe: bool = False,
    fuzzy_max_hamming: int = 6,
    collapse_whitespace: bool = False,
    strip_emoji: bool = False,
    redact: bool = False,
    redact_types: List[str] | None = None,
    store: VectorStore | None = None,
    console: Console | None = None,
) -> int:
    items = read_messages_lines(
        messages_path,
        collapse_whitespace=bool(collapse_whitespace),
        strip_emoji=bool(strip_emoji),
    )

    if max_messages is not None:
        mm = int(max_messages)
        if mm < 0:
            mm = 0
        items = items[:mm]
    if not items:
        raise ValueError("No messages found (expected one message per line)")

    if store is None:
        store = ChromaVectorStore(persist_dir=persist_dir, collection_name=collection_name)

    ensure_schema_version(store, expected=VECTOR_STORE_SCHEMA_VERSION)

    chunks = build_chunks(items, chunking=str(chunking), window_size=int(window_size))
    if not chunks:
        raise ValueError("No chunks produced (try a smaller --window-size or use --chunking message)")

    mode_norm = str(chunking).strip().lower()
    ids = [
        stable_id(end_ln, t) if str(chunking).strip().lower() in {"message", "per-message", "per_message", "line"}
        else stable_window_id(start_ln, end_ln, t)
        for (start_ln, end_ln, t) in chunks
    ]

    existing: set[str] = set()
    # For Chroma, we can cheaply list IDs for idempotent ingestion. For other backends,
    # this is best-effort and defaults to "no existing IDs".
    try:
        if isinstance(store, ChromaVectorStore):
            got = store._collection.get(include=[])
            existing = set(got.get("ids", []))
    except Exception:
        existing = set()

    # Robust exact dedupe: also dedupe by content hash (not just IDs).
    existing_content_hashes: set[str] = set()
    if exact_dedupe:
        for _id in existing:
            try:
                # IDs are sha1(text):<line> or sha1(text):<start-end>
                existing_content_hashes.add(str(_id).split(":", 1)[0])
            except Exception:
                continue

    texts_raw = [t for (_s, _e, t) in chunks]
    texts = [redact_text(t, types=redact_types) for t in texts_raw] if redact else list(texts_raw)
    bounds = [(s, e) for (s, e, _t) in chunks]

    # Apply exact dedupe (by content hash) before fuzzy dedupe.
    content_hashes = [content_hash(t) for t in texts]
    keep_mask = [True] * len(texts)
    if exact_dedupe:
        seen: set[str] = set()
        for idx, ch in enumerate(content_hashes):
            if ch in existing_content_hashes or ch in seen:
                keep_mask[idx] = False
            else:
                seen.add(ch)

    if fuzzy_dedupe:
        # Only within this run (post exact-dedupe).
        candidates = [t if keep_mask[i] else "" for i, t in enumerate(texts)]
        fuzzy_keep = fuzzy_dedupe_texts(candidates, max_hamming=int(fuzzy_max_hamming))
        keep_mask = [a and b for a, b in zip(keep_mask, fuzzy_keep)]

    new_pairs = [
        (i, t, s, e)
        for (i, t, (s, e), ch, keep) in zip(ids, texts, bounds, content_hashes, keep_mask)
        if keep and i not in existing
    ]
    if not new_pairs:
        return 0

    new_ids = [p[0] for p in new_pairs]
    new_texts = [p[1] for p in new_pairs]
    new_starts = [p[2] for p in new_pairs]
    new_ends = [p[3] for p in new_pairs]

    mode = str(chunking or "message").strip().lower()

    def _clean_for_embedding(original: str) -> tuple[str, dict]:
        # Remove inline metadata tags from the stored/embedded text where possible,
        # and emit metadata fields.
        lines = original.split("\n")
        speakers: list[str] = []
        timestamps: list[str] = []
        cleaned_lines: list[str] = []
        for ln in lines:
            msg, speaker, ts = extract_inline_metadata(ln)
            cleaned_lines.append(msg)
            if speaker:
                speakers.append(speaker)
            if ts:
                timestamps.append(ts)

        meta: dict = {}
        if speakers:
            meta["speaker"] = speakers[0]
        if timestamps:
            meta["timestamp"] = timestamps[0]
        if len(lines) > 1:
            # Best-effort for window chunks.
            first_msg, first_speaker, first_ts = extract_inline_metadata(lines[0])
            last_msg, last_speaker, last_ts = extract_inline_metadata(lines[-1])
            if first_speaker:
                meta["start_speaker"] = first_speaker
            if last_speaker:
                meta["end_speaker"] = last_speaker
            if first_ts:
                meta["start_timestamp"] = first_ts
            if last_ts:
                meta["end_timestamp"] = last_ts

        return "\n".join(cleaned_lines).strip(), meta

    model_name = str(embedding_model)
    if bool(fast_embedding_model) and model_name == DEFAULT_EMBEDDING_MODEL:
        model_name = FAST_EMBEDDING_MODEL

    for start in range(0, len(new_texts), batch):
        chunk_texts = new_texts[start : start + batch]
        chunk_ids = new_ids[start : start + batch]
        chunk_starts = new_starts[start : start + batch]
        chunk_ends = new_ends[start : start + batch]

        cleaned_texts: list[str] = []
        extra_metas: list[dict] = []
        for txt in chunk_texts:
            cleaned, extra = _clean_for_embedding(txt)
            cleaned_texts.append(cleaned)
            extra_metas.append(extra)

        chunk_emb = embed_texts(cleaned_texts, model_name, batch_size=embed_batch_size)
        metadatas = []
        for (start_ln, end_ln), extra, txt in zip(zip(chunk_starts, chunk_ends), extra_metas, chunk_texts):
            md = {
                "line": int(end_ln),
                "start_line": int(start_ln),
                "end_line": int(end_ln),
                "chunking": mode,
                "window_size": int(window_size) if mode == "window" else 1,
                "source": str(messages_path),
                "content_hash": content_hash(txt),
            }
            md.update(extra)
            metadatas.append(md)

        store.add(ids=chunk_ids, texts=cleaned_texts, embeddings=chunk_emb, metadatas=metadatas)
        if console is not None:
            console.log(f"Added {min(start + batch, len(new_texts))}/{len(new_texts)}")

    return len(new_texts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", required=True, help="Text file: one message per line")
    ap.add_argument("--persist", required=True, help="ChromaDB persistence directory")
    ap.add_argument("--collection", default="messages")
    ap.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    ap.add_argument(
        "--fast-embedding-model",
        action="store_true",
        help=f"Use a smaller embedding model for speed (currently: {FAST_EMBEDDING_MODEL}).",
    )
    ap.add_argument(
        "--embed-batch-size",
        type=int,
        default=None,
        help="SentenceTransformer encode() batch_size (optional; can reduce RAM/VRAM).",
    )
    ap.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Only ingest the first N messages (for faster iteration).",
    )
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
        "--no-exact-dedupe",
        action="store_true",
        help="Disable exact dedupe by content hash (defaults to enabled).",
    )
    ap.add_argument(
        "--fuzzy-dedupe",
        action="store_true",
        help="Enable fuzzy dedupe within the current ingest run (simhash).",
    )
    ap.add_argument(
        "--fuzzy-max-hamming",
        type=int,
        default=6,
        help="Fuzzy dedupe threshold (lower is stricter).",
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
            fast_embedding_model=bool(args.fast_embedding_model),
            embed_batch_size=args.embed_batch_size,
            max_messages=args.max_messages,
            batch=args.batch,
            chunking=str(args.chunking),
            window_size=int(args.window_size),
            exact_dedupe=not bool(args.no_exact_dedupe),
            fuzzy_dedupe=bool(args.fuzzy_dedupe),
            fuzzy_max_hamming=int(args.fuzzy_max_hamming),
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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class CharVocab:
    stoi: Dict[str, int]
    itos: List[str]

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_messages_text(path: str | Path) -> str:
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="replace")
    # Normalize newlines and keep them as signal.
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    return raw


def build_vocab(text: str) -> CharVocab:
    chars = sorted(set(text))
    # Ensure at least something sensible.
    if not chars:
        chars = ["\n"]
    stoi = {c: i for i, c in enumerate(chars)}
    itos = chars
    return CharVocab(stoi=stoi, itos=itos)


def make_stream_ids(text: str, vocab: CharVocab) -> np.ndarray:
    ids = vocab.encode(text)
    if len(ids) < 2:
        ids = vocab.encode("\n\n")
    return np.asarray(ids, dtype=np.int64)


def batchify_stream(
    stream: np.ndarray,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    # Random contiguous subsequences from a single stream.
    # x: [B, T], y: [B, T]
    max_start = len(stream) - (seq_len + 1)
    if max_start <= 0:
        raise ValueError(
            f"Not enough data: need > {seq_len + 1} chars, got {len(stream)}"
        )
    starts = rng.integers(0, max_start, size=(batch_size,))
    x = np.stack([stream[s : s + seq_len] for s in starts], axis=0)
    y = np.stack([stream[s + 1 : s + 1 + seq_len] for s in starts], axis=0)
    return x, y

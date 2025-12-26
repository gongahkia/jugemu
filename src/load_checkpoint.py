from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from .tiny_char_transformer import TinyCharTransformer, TinyConfig


@dataclass(frozen=True)
class Loaded:
    model: TinyCharTransformer
    vocab_itos: List[str]
    cfg: TinyConfig


def load_checkpoint(path: str | Path, device: str) -> Loaded:
    p = Path(path)
    ckpt: Dict[str, Any] = torch.load(p, map_location=device)
    cfg = TinyConfig(**ckpt["cfg"])
    model = TinyCharTransformer(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    vocab_itos = ckpt["vocab"]["itos"]
    return Loaded(model=model, vocab_itos=vocab_itos, cfg=cfg)


def vocab_from_itos(itos: List[str]) -> Tuple[Dict[str, int], List[str]]:
    stoi = {c: i for i, c in enumerate(itos)}
    return stoi, itos

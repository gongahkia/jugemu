from __future__ import annotations

from typing import Dict, List

import torch


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi[c] for c in text if c in stoi]


def decode(ids: List[int], itos: List[str]) -> str:
    return "".join(itos[i] for i in ids)


@torch.no_grad()
def sample_text(
    model,
    prompt: str,
    stoi: Dict[str, int],
    itos: List[str],
    device: str,
    max_new_tokens: int = 200,
    temperature: float = 0.9,
    top_k: int | None = 60,
) -> str:
    prompt_ids = encode(prompt, stoi)
    if not prompt_ids:
        # if prompt chars are OOV, just seed with newline.
        prompt_ids = [stoi.get("\n", 0)]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return decode(out[0].tolist(), itos)

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
    stop_on: str | None = None,
) -> str:
    prompt_ids = encode(prompt, stoi)
    if not prompt_ids:
        # if prompt chars are OOV, just seed with newline.
        prompt_ids = [stoi.get("\n", 0)]
    prompt_text = decode(prompt_ids, itos)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    text = decode(out[0].tolist(), itos)
    if stop_on:
        # Stop only in the generated tail (after the prompt).
        start = len(prompt_text)
        i = text.find(stop_on, start)
        if i != -1:
            text = text[:i]
    return text

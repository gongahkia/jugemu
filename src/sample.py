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
    stop_on: str | List[str] | None = None,
    return_full: bool = True,
) -> str:
    prompt_ids = encode(prompt, stoi)
    if not prompt_ids:
        # if prompt chars are OOV, just seed with newline.
        prompt_ids = [stoi.get("\n", 0)]
    prompt_text = decode(prompt_ids, itos)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    text = decode(out[0].tolist(), itos)
    # Stop only in the generated tail (after the prompt).
    start = len(prompt_text)
    if stop_on:
        stops = [stop_on] if isinstance(stop_on, str) else list(stop_on)
        earliest: int | None = None
        for s in stops:
            i = text.find(s, start)
            if i != -1 and (earliest is None or i < earliest):
                earliest = i
        if earliest is not None:
            text = text[:earliest]

    if return_full:
        return text
    return text[start:]

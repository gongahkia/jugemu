from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from rich.console import Console
from rich.panel import Panel

from .load_checkpoint import load_checkpoint, vocab_from_itos
from .text_dataset import batchify_stream, load_messages_text
from .sample import sample_text


@dataclass(frozen=True)
class EvalResult:
    heldout_chars: int
    oov_chars: int
    loss: float

    @property
    def perplexity(self) -> float:
        if not math.isfinite(self.loss):
            return float("nan")
        try:
            return float(math.exp(self.loss))
        except OverflowError:
            return float("inf")


def _text_to_ids(text: str, stoi: Dict[str, int]) -> tuple[np.ndarray, int]:
    ids: list[int] = []
    oov = 0
    for ch in text:
        i = stoi.get(ch)
        if i is None:
            oov += 1
            i = 0
        ids.append(int(i))
    return np.asarray(ids, dtype=np.int64), int(oov)


@torch.no_grad()
def evaluate_char_model(
    *,
    messages_path: Path,
    checkpoint: Path,
    heldout_fraction: float = 0.05,
    steps: int = 200,
    batch_size: int = 64,
    seq_len: int | None = None,
    seed: int = 1337,
    device: str = "auto",
) -> EvalResult:
    resolved_device = device
    if resolved_device == "auto":
        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"

    loaded = load_checkpoint(checkpoint, resolved_device)
    stoi, itos = vocab_from_itos(loaded.vocab_itos)

    raw = load_messages_text(messages_path)
    stream, oov = _text_to_ids(raw, stoi)

    heldout_fraction = float(heldout_fraction)
    if heldout_fraction < 0.0:
        heldout_fraction = 0.0
    if heldout_fraction > 0.5:
        heldout_fraction = 0.5

    use_seq_len = int(seq_len) if seq_len is not None else int(loaded.cfg.seq_len)
    use_seq_len = max(8, use_seq_len)

    heldout_chars = int(len(stream) * heldout_fraction)
    min_needed = use_seq_len + 2
    if heldout_chars < min_needed:
        heldout_chars = 0

    if heldout_chars <= 0:
        raise ValueError(
            "Heldout split too small for evaluation. "
            "Increase --heldout-fraction or decrease --seq-len."
        )

    heldout = stream[-heldout_chars:]
    rng = np.random.default_rng(int(seed))

    if steps <= 0:
        return EvalResult(heldout_chars=heldout_chars, oov_chars=oov, loss=float("nan"))

    losses: list[float] = []
    for _ in range(int(steps)):
        x_np, y_np = batchify_stream(heldout, int(batch_size), int(use_seq_len), rng)
        x = torch.from_numpy(x_np).to(resolved_device)
        y = torch.from_numpy(y_np).to(resolved_device)
        logits = loaded.model(x)
        loss = F.cross_entropy(logits.view(-1, loaded.cfg.vocab_size), y.view(-1))
        losses.append(float(loss.item()))

    return EvalResult(
        heldout_chars=int(heldout_chars),
        oov_chars=int(oov),
        loss=float(np.mean(losses)) if losses else float("nan"),
    )


def run_qualitative_prompts(
    *,
    checkpoint: Path,
    prompts: List[str],
    max_new: int = 240,
    temperature: float = 0.9,
    top_k: int = 60,
    seed: int = 1337,
    device: str = "auto",
    console: Console | None = None,
) -> List[str]:
    resolved_device = device
    if resolved_device == "auto":
        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"

    if console is None:
        console = Console()

    torch.manual_seed(int(seed))
    loaded = load_checkpoint(checkpoint, resolved_device)
    stoi, itos = vocab_from_itos(loaded.vocab_itos)

    outputs: list[str] = []
    for p in prompts:
        out = sample_text(
            loaded.model,
            prompt=p,
            stoi=stoi,
            itos=itos,
            device=resolved_device,
            max_new_tokens=int(max_new),
            temperature=float(temperature),
            top_k=int(top_k) if top_k is not None else None,
            return_full=True,
        )
        outputs.append(out)

    return outputs


def default_prompts() -> List[str]:
    return [
        "Hey ",
        "What's up?\n",
        "How's it going?\n",
        "I think ",
        "lol\n",
    ]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate the tiny character model")
    ap.add_argument("--messages", required=True, help="Path to messages text file")
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    ap.add_argument("--heldout-fraction", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--max-new", type=int, default=240)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=60)
    args = ap.parse_args()

    console = Console()
    console.print(Panel("Evaluating model…", title="jugemu", border_style="cyan"))

    res = evaluate_char_model(
        messages_path=Path(args.messages),
        checkpoint=Path(args.checkpoint),
        heldout_fraction=float(args.heldout_fraction),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        seq_len=args.seq_len,
        seed=int(args.seed),
        device=str(args.device),
    )
    console.print(
        f"Heldout chars: {res.heldout_chars} | OOV chars: {res.oov_chars} | "
        f"loss: {res.loss:.4f} | ppl: {res.perplexity:.2f}"
    )

    console.print(Panel("Qualitative prompts", border_style="cyan"))
    outs = run_qualitative_prompts(
        checkpoint=Path(args.checkpoint),
        prompts=default_prompts(),
        max_new=int(args.max_new),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        seed=int(args.seed),
        device=str(args.device),
        console=console,
    )
    for p, o in zip(default_prompts(), outs):
        console.print(Panel(o, title=f"prompt: {p!r}", border_style="green"))


if __name__ == "__main__":
    main()

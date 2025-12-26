from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from rich.console import Console

from .text_dataset import build_vocab, load_messages_text, make_stream_ids, batchify_stream
from .tiny_char_transformer import TinyCharTransformer, TinyConfig


def save_checkpoint(
    out_dir: Path,
    step: int,
    model: TinyCharTransformer,
    cfg: TinyConfig,
    vocab: dict,
    optimizer: torch.optim.Optimizer,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "cfg": asdict(cfg),
        "vocab": vocab,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    path = out_dir / f"step_{step}.pt"
    torch.save(ckpt, path)
    latest = out_dir / "latest.pt"
    torch.save(ckpt, latest)
    return path


def train_char_model(
    *,
    messages_path: Path,
    out_dir: Path,
    epochs: int = 5,
    batch_size: int = 64,
    seq_len: int = 256,
    d_model: int = 192,
    n_heads: int = 3,
    n_layers: int = 3,
    dropout: float = 0.1,
    lr: float = 3e-4,
    steps_per_epoch: int = 500,
    seed: int = 1337,
    device: str = "auto",
    log_every: int = 50,
    console: Console | None = None,
) -> Path:
    raw = load_messages_text(messages_path)
    vocab = build_vocab(raw)
    stream = make_stream_ids(raw, vocab)

    resolved_device = device
    if resolved_device == "auto":
        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"

    cfg = TinyConfig(
        vocab_size=vocab.size,
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    )

    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    model = TinyCharTransformer(cfg).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    meta = {
        "messages": str(messages_path),
        "num_chars": int(stream.shape[0]),
        "vocab_size": vocab.size,
        "cfg": asdict(cfg),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    global_step = 0
    model.train()

    for epoch in range(1, epochs + 1):
        pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch}/{epochs}")
        running = 0.0
        for _ in pbar:
            x_np, y_np = batchify_stream(stream, batch_size, seq_len, np_rng)
            x = torch.from_numpy(x_np).to(resolved_device)
            y = torch.from_numpy(y_np).to(resolved_device)

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item()
            global_step += 1

            if global_step % log_every == 0:
                avg = running / log_every
                running = 0.0
                pbar.set_postfix(loss=f"{avg:.4f}")

        save_checkpoint(
            out_dir=out_dir,
            step=global_step,
            model=model,
            cfg=cfg,
            vocab={"itos": vocab.itos},
            optimizer=optimizer,
        )

        if console is not None:
            console.log(f"epoch {epoch}/{epochs} complete; latest step={global_step}")

    return out_dir / "latest.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", required=True, help="Path to messages text file")
    ap.add_argument("--out", required=True, help="Output dir for checkpoints")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-heads", type=int, default=3)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps-per-epoch", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    console = Console()
    latest = train_char_model(
        messages_path=Path(args.messages),
        out_dir=Path(args.out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lr=args.lr,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
        console=console,
    )
    console.print(f"Done. Latest checkpoint: {latest}")


if __name__ == "__main__":
    main()

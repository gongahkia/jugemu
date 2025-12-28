from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from rich.console import Console

from .text_dataset import build_vocab, load_messages_text, make_stream_ids, batchify_stream
from .text_dataset import build_pairs_corpus, load_messages_lines
from .redact import redact_text
from .tiny_char_transformer import TinyCharTransformer, TinyConfig


def _try_git_hash() -> str | None:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        h = p.stdout.strip()
        return h or None
    except Exception:
        return None


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Best-effort determinism hints.
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        pass


def _write_run_json(out_dir: Path, run: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")


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
    training_mode: str = "stream",
    redact: bool = False,
    redact_types: list[str] | None = None,
) -> Path:
    if training_mode == "pairs":
        lines = load_messages_lines(messages_path)
        if redact:
            lines = [redact_text(ln, types=redact_types) for ln in lines]
        raw = build_pairs_corpus(lines)
    else:
        raw = load_messages_text(messages_path)
        if redact:
            raw = "\n".join(redact_text(ln, types=redact_types) for ln in raw.split("\n"))
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

    _seed_everything(seed)
    np_rng = np.random.default_rng(seed)

    model = TinyCharTransformer(cfg).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    meta = {
        "messages": str(messages_path),
        "num_chars": int(stream.shape[0]),
        "vocab_size": vocab.size,
        "cfg": asdict(cfg),
        "training_mode": training_mode,
        "seed": int(seed),
        "redact": bool(redact),
        "redact_types": list(redact_types or []),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    run: Dict[str, Any] = {
        "git": {"hash": _try_git_hash()},
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": getattr(torch, "__version__", None),
            "numpy": getattr(np, "__version__", None),
        },
        "config": {
            "messages": str(messages_path),
            "out_dir": str(out_dir),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "seq_len": int(seq_len),
            "d_model": int(d_model),
            "n_heads": int(n_heads),
            "n_layers": int(n_layers),
            "dropout": float(dropout),
            "lr": float(lr),
            "steps_per_epoch": int(steps_per_epoch),
            "seed": int(seed),
            "device": str(device),
            "training_mode": str(training_mode),
            "redact": bool(redact),
            "redact_types": list(redact_types or []),
        },
    }
    _write_run_json(out_dir, run)

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
    ap.add_argument(
        "--training-mode",
        default="stream",
        choices=["stream", "pairs"],
        help="stream: train on raw text stream; pairs: train on USER/YOU pairs from consecutive lines",
    )
    ap.add_argument("--redact", action="store_true", help="Redact emails/phones/addresses before training")
    ap.add_argument(
        "--redact-type",
        action="append",
        default=[],
        help="Redaction type (repeatable): email|phone|address. Default: all.",
    )
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
        training_mode=args.training_mode,
        redact=bool(args.redact),
        redact_types=list(args.redact_type or []),
    )
    console.print(f"Done. Latest checkpoint: {latest}")


if __name__ == "__main__":
    main()

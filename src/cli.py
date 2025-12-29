from __future__ import annotations

from pathlib import Path
from typing import List

import typer
from rich.console import Console
from rich.panel import Panel

from .chat import run_chat
from .ingest_chroma import ingest_messages
from .parse_exports import parse_export, write_canonical_messages
from .train_char_model import train_char_model


app = typer.Typer(
    help="jugemu — tiny messages LLM + ChromaDB",
    rich_markup_mode="rich",
    add_completion=False,
)


@app.command()
def parse(
    inp: Path = typer.Option(..., "--in", exists=True, dir_okay=False, help="Input export file"),
    out: Path = typer.Option(Path("data/messages.txt"), "--out", help="Output canonical messages.txt"),
    fmt: str = typer.Option(
        "plain",
        "--format",
        help="Input format: plain|whatsapp|telegram-json",
    ),
):
    """Convert a chat export into canonical one-message-per-line text."""
    console = Console()
    console.print(Panel("Parsing export…", title="jugemu", border_style="cyan"))
    with console.status("Parsing + writing…", spinner="dots"):
        lines = parse_export(inp, fmt)
        write_canonical_messages(lines, out)
    console.print(f"Wrote {len(lines)} messages to {out}")


@app.command()
def ingest(
    messages: Path = typer.Option(..., "--messages", exists=True, dir_okay=False, help="Text file: one message per line"),
    persist: Path = typer.Option(..., "--persist", help="ChromaDB persistence directory"),
    collection: str = typer.Option("messages", "--collection", help="Chroma collection name"),
    embedding_model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        "--embedding-model",
        help="SentenceTransformers model for embeddings",
    ),
    batch: int = typer.Option(256, "--batch", help="Embed+add batch size"),
    redact: bool = typer.Option(
        False,
        "--redact",
        help="Redact emails/phones/addresses before embedding/storing",
    ),
    redact_type: List[str] = typer.Option(
        [],
        "--redact-type",
        help="Redaction type (repeatable): email|phone|address. Default: all.",
    ),
):
    """Embed messages and store them in ChromaDB."""
    console = Console()
    console.print(Panel("Ingesting messages…", title="jugemu", border_style="cyan"))
    with console.status("Embedding + writing to ChromaDB…", spinner="dots"):
        added = ingest_messages(
            messages_path=messages,
            persist_dir=persist,
            collection_name=collection,
            embedding_model=embedding_model,
            batch=batch,
            redact=redact,
            redact_types=list(redact_type),
            console=console,
        )
    if added == 0:
        console.print("Nothing new to ingest.")
    else:
        console.print(f"Ingested {added} new messages into collection '{collection}'.")


@app.command()
def train(
    messages: Path = typer.Option(..., "--messages", exists=True, dir_okay=False, help="Path to messages text file"),
    out: Path = typer.Option(Path("data/checkpoints"), "--out", help="Output dir for checkpoints"),
    epochs: int = typer.Option(5, "--epochs"),
    batch_size: int = typer.Option(64, "--batch-size"),
    seq_len: int = typer.Option(256, "--seq-len"),
    d_model: int = typer.Option(192, "--d-model"),
    n_heads: int = typer.Option(3, "--n-heads"),
    n_layers: int = typer.Option(3, "--n-layers"),
    dropout: float = typer.Option(0.1, "--dropout"),
    lr: float = typer.Option(3e-4, "--lr"),
    lr_schedule: str = typer.Option(
        "none",
        "--lr-schedule",
        help="Learning rate schedule (per optimizer step): none|cosine",
    ),
    warmup_steps: int = typer.Option(
        0,
        "--warmup-steps",
        help="Warmup steps for LR schedule (optimizer steps).",
    ),
    grad_accum_steps: int = typer.Option(
        1,
        "--grad-accum-steps",
        help="Accumulate gradients over N micro-batches per optimizer step.",
    ),
    steps_per_epoch: int = typer.Option(500, "--steps-per-epoch"),
    resume: Path | None = typer.Option(
        None,
        "--resume",
        help="Resume from a checkpoint (.pt)",
    ),
    seed: int = typer.Option(1337, "--seed"),
    device: str = typer.Option("auto", "--device", help="auto/cpu/mps/cuda"),
    log_every: int = typer.Option(50, "--log-every"),
    training_mode: str = typer.Option(
        "stream",
        "--training-mode",
        help="stream (raw text) or pairs (USER/YOU from consecutive lines)",
    ),
    redact: bool = typer.Option(
        False,
        "--redact",
        help="Redact emails/phones/addresses before training",
    ),
    redact_type: List[str] = typer.Option(
        [],
        "--redact-type",
        help="Redaction type (repeatable): email|phone|address. Default: all.",
    ),
    val_fraction: float = typer.Option(
        0.05,
        "--val-fraction",
        help="Fraction of the corpus reserved for validation (tail slice).",
    ),
    val_steps: int = typer.Option(
        50,
        "--val-steps",
        help="How many random validation batches to average per epoch.",
    ),
):
    """Train the tiny character model and write checkpoints."""
    console = Console()
    console.print(Panel("Training model…", title="jugemu", border_style="cyan"))
    latest = train_char_model(
        messages_path=messages,
        out_dir=out,
        epochs=epochs,
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        lr=lr,
        lr_schedule=str(lr_schedule),
        warmup_steps=int(warmup_steps),
        grad_accum_steps=int(grad_accum_steps),
        steps_per_epoch=steps_per_epoch,
        seed=seed,
        device=device,
        log_every=log_every,
        console=console,
        training_mode=training_mode,
        redact=redact,
        redact_types=list(redact_type),
        val_fraction=float(val_fraction),
        val_steps=int(val_steps),
        resume=resume,
    )
    console.print(f"Done. Latest checkpoint: {latest}")


@app.command()
def chat(
    messages: Path = typer.Option(
        Path("data/messages.txt"),
        "--messages",
        help="messages.txt used to build msg->next-msg few-shot examples",
    ),
    persist: Path = typer.Option(Path("data/chroma"), "--persist", help="ChromaDB persistence directory"),
    collection: str = typer.Option("messages", "--collection"),
    embedding_model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        "--embedding-model",
    ),
    k: int = typer.Option(6, "--k"),
    checkpoint: Path = typer.Option(Path("data/checkpoints/latest.pt"), "--checkpoint", help="Path to model checkpoint .pt"),
    device: str = typer.Option("auto", "--device", help="auto/cpu/mps/cuda"),
    max_new: int = typer.Option(240, "--max-new"),
    temperature: float = typer.Option(0.9, "--temperature"),
    top_k: int = typer.Option(60, "--top-k"),
    stop_seq: List[str] = typer.Option(
        [],
        "--stop-seq",
        help="Stop sequence (repeatable). Generation stops when any stop sequence appears.",
    ),
    show_retrieval: bool = typer.Option(False, "--show-retrieval", help="Print retrieved similar messages each turn"),
    reply_strategy: str = typer.Option(
        "hybrid",
        "--reply-strategy",
        help="hybrid/generate/corpus",
    ),
    min_score: float = typer.Option(0.35, "--min-score", help="Minimum retrieval score for hybrid corpus replies"),
):
    """Interactive chat (retrieval + generation)."""
    run_chat(
        messages=str(messages),
        persist=str(persist),
        collection=collection,
        embedding_model=embedding_model,
        k=k,
        checkpoint=str(checkpoint),
        device=device,
        max_new=max_new,
        temperature=temperature,
        top_k=top_k,
        stop_seq=list(stop_seq),
        show_retrieval=show_retrieval,
        reply_strategy=reply_strategy,
        min_score=min_score,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from .chat import run_chat
from .ingest_chroma import ingest_messages
from .train_char_model import train_char_model


app = typer.Typer(
    help="jugemu — tiny messages LLM + ChromaDB",
    rich_markup_mode="rich",
    add_completion=False,
)


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
    steps_per_epoch: int = typer.Option(500, "--steps-per-epoch"),
    seed: int = typer.Option(1337, "--seed"),
    device: str = typer.Option("auto", "--device", help="auto/cpu/mps/cuda"),
    log_every: int = typer.Option(50, "--log-every"),
    training_mode: str = typer.Option(
        "stream",
        "--training-mode",
        help="stream (raw text) or pairs (USER/YOU from consecutive lines)",
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
        steps_per_epoch=steps_per_epoch,
        seed=seed,
        device=device,
        log_every=log_every,
        console=console,
        training_mode=training_mode,
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
        show_retrieval=show_retrieval,
        reply_strategy=reply_strategy,
        min_score=min_score,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

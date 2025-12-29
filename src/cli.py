from __future__ import annotations

from pathlib import Path
from typing import List

import typer
from rich.console import Console
from rich.panel import Panel

from .chat import run_chat
from .eval_char_model import default_prompts, evaluate_char_model, run_qualitative_prompts
from .ingest_chroma import ingest_messages
from .parse_exports import parse_export, write_canonical_messages
from .store_factory import make_vector_store
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
    with_metadata: bool = typer.Option(
        False,
        "--with-metadata",
        help="Include timestamps/speaker tags when available (export-dependent).",
    ),
):
    """Convert a chat export into canonical one-message-per-line text."""
    console = Console()
    console.print(Panel("Parsing export…", title="jugemu", border_style="cyan"))
    with console.status("Parsing + writing…", spinner="dots"):
        lines = parse_export(inp, fmt, include_metadata=with_metadata)
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
    chunking: str = typer.Option(
        "message",
        "--chunking",
        help="How to chunk messages for embeddings: message|window",
    ),
    window_size: int = typer.Option(
        4,
        "--window-size",
        help="Sliding window size (only used when --chunking window).",
    ),
    exact_dedupe: bool = typer.Option(
        True,
        "--exact-dedupe/--no-exact-dedupe",
        help="Exact dedupe by content hash (idempotent across repeated ingests).",
    ),
    fuzzy_dedupe: bool = typer.Option(
        False,
        "--fuzzy-dedupe",
        help="Fuzzy dedupe within the current ingest run (simhash).",
    ),
    fuzzy_max_hamming: int = typer.Option(
        6,
        "--fuzzy-max-hamming",
        help="Fuzzy dedupe threshold (lower is stricter).",
    ),
    collapse_whitespace: bool = typer.Option(
        False,
        "--collapse-whitespace",
        help="Collapse repeated spaces/tabs in each message before embedding.",
    ),
    strip_emoji: bool = typer.Option(
        False,
        "--strip-emoji",
        help="Remove emoji characters from messages before embedding.",
    ),
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
    vector_backend: str = typer.Option(
        "chroma",
        "--vector-backend",
        help="Vector backend: chroma|cassandra",
    ),
    cassandra_contact_point: List[str] = typer.Option(
        [],
        "--cassandra-contact-point",
        help="Cassandra contact point (repeatable). Default: 127.0.0.1",
    ),
    cassandra_keyspace: str = typer.Option(
        "jugemu",
        "--cassandra-keyspace",
        help="Cassandra keyspace (for --vector-backend cassandra)",
    ),
    cassandra_table: str = typer.Option(
        "messages",
        "--cassandra-table",
        help="Cassandra table (for --vector-backend cassandra)",
    ),
    cassandra_secure_connect_bundle: Path | None = typer.Option(
        None,
        "--cassandra-secure-connect-bundle",
        help="Astra secure connect bundle zip (optional)",
    ),
    cassandra_username: str | None = typer.Option(
        None,
        "--cassandra-username",
        help="Cassandra/Astra username (optional)",
    ),
    cassandra_password: str | None = typer.Option(
        None,
        "--cassandra-password",
        help="Cassandra/Astra password (optional)",
    ),
):
    """Embed messages and store them in ChromaDB."""
    console = Console()
    console.print(Panel("Ingesting messages…", title="jugemu", border_style="cyan"))
    store = make_vector_store(
        backend=vector_backend,
        persist_dir=persist,
        collection_name=collection,
        cassandra_contact_points=list(cassandra_contact_point) or None,
        cassandra_keyspace=str(cassandra_keyspace),
        cassandra_table=str(cassandra_table),
        cassandra_secure_connect_bundle=cassandra_secure_connect_bundle,
        cassandra_username=cassandra_username,
        cassandra_password=cassandra_password,
    )
    with console.status("Embedding + writing to ChromaDB…", spinner="dots"):
        added = ingest_messages(
            messages_path=messages,
            persist_dir=persist,
            collection_name=collection,
            embedding_model=embedding_model,
            batch=batch,
            chunking=str(chunking),
            window_size=int(window_size),
            exact_dedupe=bool(exact_dedupe),
            fuzzy_dedupe=bool(fuzzy_dedupe),
            fuzzy_max_hamming=int(fuzzy_max_hamming),
            collapse_whitespace=collapse_whitespace,
            strip_emoji=strip_emoji,
            redact=redact,
            redact_types=list(redact_type),
            store=store,
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
    collapse_whitespace: bool = typer.Option(
        False,
        "--collapse-whitespace",
        help="Collapse repeated spaces/tabs (preserves newlines).",
    ),
    strip_emoji: bool = typer.Option(
        False,
        "--strip-emoji",
        help="Remove emoji characters from the training text.",
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
        collapse_whitespace=collapse_whitespace,
        strip_emoji=strip_emoji,
        redact=redact,
        redact_types=list(redact_type),
        val_fraction=float(val_fraction),
        val_steps=int(val_steps),
        resume=resume,
    )
    console.print(f"Done. Latest checkpoint: {latest}")


@app.command()
def eval(
    messages: Path = typer.Option(..., "--messages", exists=True, dir_okay=False, help="Path to messages text file"),
    checkpoint: Path = typer.Option(
        Path("data/checkpoints/latest.pt"),
        "--checkpoint",
        exists=True,
        dir_okay=False,
        help="Path to model checkpoint .pt",
    ),
    device: str = typer.Option("auto", "--device", help="auto/cpu/mps/cuda"),
    heldout_fraction: float = typer.Option(0.05, "--heldout-fraction", help="Tail fraction reserved for eval"),
    steps: int = typer.Option(200, "--steps", help="Random batches to average for heldout loss"),
    batch_size: int = typer.Option(64, "--batch-size"),
    seq_len: int | None = typer.Option(None, "--seq-len", help="Override seq_len for eval batches"),
    seed: int = typer.Option(1337, "--seed"),
    prompt: List[str] = typer.Option(
        [],
        "--prompt",
        help="Qualitative prompt (repeatable). If omitted, uses a small default set.",
    ),
    max_new: int = typer.Option(240, "--max-new"),
    temperature: float = typer.Option(0.9, "--temperature"),
    top_k: int = typer.Option(60, "--top-k"),
):
    """Evaluate checkpoint on heldout + print qualitative prompts."""
    console = Console()
    console.print(Panel("Evaluating model…", title="jugemu", border_style="cyan"))

    res = evaluate_char_model(
        messages_path=messages,
        checkpoint=checkpoint,
        heldout_fraction=float(heldout_fraction),
        steps=int(steps),
        batch_size=int(batch_size),
        seq_len=seq_len,
        seed=int(seed),
        device=str(device),
    )
    console.print(
        f"Heldout chars: {res.heldout_chars} | OOV chars: {res.oov_chars} | "
        f"loss: {res.loss:.4f} | ppl: {res.perplexity:.2f}"
    )

    prompts = list(prompt) if prompt else default_prompts()
    console.print(Panel("Qualitative prompts", border_style="cyan"))
    outs = run_qualitative_prompts(
        checkpoint=checkpoint,
        prompts=prompts,
        max_new=int(max_new),
        temperature=float(temperature),
        top_k=int(top_k),
        seed=int(seed),
        device=str(device),
        console=console,
    )
    for p, o in zip(prompts, outs):
        console.print(Panel(o, title=f"prompt: {p!r}", border_style="green"))


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
    prompt_template: str = typer.Option(
        "few-shot",
        "--prompt-template",
        help="Prompt template for generation: few-shot|scaffold|minimal",
    ),
    rerank: bool = typer.Option(
        False,
        "--rerank",
        help="Rerank vector results with a cross-encoder (slower, often better).",
    ),
    rerank_model: str = typer.Option(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "--rerank-model",
        help="Cross-encoder model name for reranking.",
    ),
    rerank_top_k: int = typer.Option(
        20,
        "--rerank-top-k",
        help="How many vector hits to fetch before reranking.",
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
        prompt_template=str(prompt_template),
        rerank=bool(rerank),
        rerank_model=str(rerank_model),
        rerank_top_k=int(rerank_top_k),
        min_score=min_score,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

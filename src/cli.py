from __future__ import annotations

import json
from pathlib import Path
from typing import List

import typer
from rich.console import Console
from rich.panel import Panel

from .chat import run_chat
from .eval_char_model import default_prompts, evaluate_char_model, run_qualitative_prompts
from .export_retrieval import dump_random_retrieval_samples, write_retrieval_samples
from .ingest_chroma import ingest_messages
from .parse_exports import parse_export, write_canonical_messages
from .config import JugemuConfig, load_optional_config
from .pipeline import run_pipeline
from .browse_stats import browse_report, count_chars, count_tokens, top_items
from .store_factory import make_vector_store
from .train_char_model import train_char_model
from .vector_store_schema import reset_vector_store, schema_info


app = typer.Typer(
    help="jugemu — tiny messages LLM + ChromaDB",
    rich_markup_mode="rich",
    add_completion=False,
)


@app.command()
def schema() -> None:
    """Print the current vector store schema version and backend info."""
    info = schema_info()
    typer.echo(f"schema_version: {info['schema_version']}")
    typer.echo(f"vector_backends: {', '.join(info['vector_backends'])}")
    typer.echo(f"chroma.schema_file: {info['chroma']['schema_file']}")
    typer.echo(f"cassandra.meta_key: {info['cassandra']['meta_key']}")


@app.callback()
def _global_options(
    ctx: typer.Context,
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Optional config.toml path (defaults to ./config.toml when present)",
    ),
):
    cfg = load_optional_config(config)
    ctx.obj = {"config": cfg}


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
def browse(
    ctx: typer.Context,
    messages: Path = typer.Option(
        Path("data/messages.txt"),
        "--messages",
        dir_okay=False,
        help="Path to messages text file",
    ),
    top: int = typer.Option(50, "--top", help="Top-N items to print"),
    mode: str = typer.Option("both", "--mode", help="chars|tokens|both"),
    min_count: int = typer.Option(1, "--min-count", help="Only show items with count >= this"),
    json_output: bool = typer.Option(False, "--json", help="Print JSON report to stdout"),
):
    """Print top-N most frequent characters and tokens."""
    cfg: JugemuConfig | None = None
    if isinstance(getattr(ctx, "obj", None), dict):
        cfg = ctx.obj.get("config")

    default_messages = Path("data/messages.txt")
    if cfg is not None:
        cfg_messages = cfg.get("paths", "messages")
        if isinstance(cfg_messages, str) and messages == default_messages:
            messages = Path(cfg_messages)

        cfg_top = cfg.get("browse", "top")
        if isinstance(cfg_top, int) and int(top) == 50:
            top = int(cfg_top)

        cfg_mode = cfg.get("browse", "mode")
        if isinstance(cfg_mode, str) and str(mode) == "both":
            mode = str(cfg_mode)

        cfg_min_count = cfg.get("browse", "min_count")
        if isinstance(cfg_min_count, int) and int(min_count) == 1:
            min_count = int(cfg_min_count)

        cfg_json = cfg.get("browse", "json")
        if isinstance(cfg_json, bool) and bool(json_output) is False:
            json_output = bool(cfg_json)

    raw = messages.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n") if ln]

    n = int(top)
    if n < 0:
        n = 0

    m = str(mode or "both").strip().lower()
    if m not in {"chars", "tokens", "both"}:
        raise typer.BadParameter("--mode must be one of: chars|tokens|both")

    if bool(json_output):
        report = browse_report(lines, mode=m, top=n, min_count=int(min_count))
        typer.echo(json.dumps(report, ensure_ascii=False))
        return

    console = Console()
    console.print(Panel("Browsing corpus stats…", title="jugemu", border_style="cyan"))

    if m in {"chars", "both"}:
        ch = count_chars(lines)
        console.print("Top characters:")
        for c, cnt in top_items(ch, top=n, min_count=int(min_count)):
            console.print(f"{cnt}\t{repr(c)}")

    if m == "both":
        console.print("\n")

    if m in {"tokens", "both"}:
        tok = count_tokens(lines)
        console.print("Top tokens:")
        for t, cnt in top_items(tok, top=n, min_count=int(min_count)):
            console.print(f"{cnt}\t{t}")


@app.command()
def pipeline(
    ctx: typer.Context,
    inp: Path = typer.Option(..., "--in", exists=True, dir_okay=False, help="Input export file"),
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
    epochs: int = typer.Option(5, "--epochs", help="Training epochs"),
    device: str = typer.Option("auto", "--device", help="auto/cpu/mps/cuda"),
):
    """One-shot pipeline: parse -> ingest -> train -> smoke-sample."""
    cfg: JugemuConfig | None = None
    if isinstance(getattr(ctx, "obj", None), dict):
        cfg = ctx.obj.get("config")

    messages_out = Path("data/messages.txt")
    persist_dir = Path("data/chroma")
    checkpoints = Path("data/checkpoints")
    collection = "messages"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    if cfg is not None:
        cfg_messages = cfg.get("paths", "messages")
        if isinstance(cfg_messages, str):
            messages_out = Path(cfg_messages)

        cfg_persist = cfg.get("paths", "chroma_persist")
        if isinstance(cfg_persist, str):
            persist_dir = Path(cfg_persist)

        cfg_ckpt = cfg.get("paths", "checkpoints")
        if isinstance(cfg_ckpt, str):
            checkpoints = Path(cfg_ckpt)

        cfg_collection = cfg.get("chroma", "collection")
        if isinstance(cfg_collection, str):
            collection = cfg_collection

        cfg_embed = cfg.get("embeddings", "model")
        if isinstance(cfg_embed, str):
            embedding_model = cfg_embed

    console = Console()
    console.print(Panel("Running pipeline…", title="jugemu", border_style="cyan"))
    with console.status("parse -> ingest -> train -> smoke…", spinner="dots"):
        res = run_pipeline(
            inp=inp,
            fmt=str(fmt),
            with_metadata=bool(with_metadata),
            messages_out=messages_out,
            persist_dir=persist_dir,
            collection=str(collection),
            embedding_model=str(embedding_model),
            train_out=checkpoints,
            train_epochs=int(epochs),
            device=str(device),
        )

    console.print(f"Wrote messages: {res.messages_path}")
    console.print(f"Ingested into: {res.persist_dir} / {res.collection}")
    console.print(f"Checkpoint: {res.checkpoint}")
    console.print(Panel(res.smoke_sample, title="smoke-sample", border_style="green"))


@app.command()
def ingest(
    ctx: typer.Context,
    messages: Path = typer.Option(
        Path("data/messages.txt"),
        "--messages",
        dir_okay=False,
        help="Text file: one message per line",
    ),
    persist: Path = typer.Option(
        Path("data/chroma"),
        "--persist",
        help="ChromaDB persistence directory",
    ),
    collection: str = typer.Option("messages", "--collection", help="Chroma collection name"),
    embedding_model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        "--embedding-model",
        help="SentenceTransformers model for embeddings",
    ),
    fast_embedding_model: bool = typer.Option(
        False,
        "--fast-embedding-model",
        help="Use a smaller embedding model for speed (only affects the default embedding model).",
    ),
    batch: int = typer.Option(256, "--batch", help="Embed+add batch size"),
    embed_batch_size: int | None = typer.Option(
        None,
        "--embed-batch-size",
        help="SentenceTransformer encode() batch_size (optional; can reduce RAM/VRAM).",
    ),
    max_messages: int | None = typer.Option(
        None,
        "--max-messages",
        help="Only ingest the first N messages (for faster iteration).",
    ),
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
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show counts/dedupe/examples without embedding or writing to the DB.",
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
    cfg: JugemuConfig | None = None
    if isinstance(getattr(ctx, "obj", None), dict):
        cfg = ctx.obj.get("config")

    default_messages = Path("data/messages.txt")
    default_persist = Path("data/chroma")

    if cfg is not None:
        cfg_messages = cfg.get("paths", "messages")
        if isinstance(cfg_messages, str) and messages == default_messages:
            messages = Path(cfg_messages)

        cfg_persist = cfg.get("paths", "chroma_persist")
        if isinstance(cfg_persist, str) and persist == default_persist:
            persist = Path(cfg_persist)

        cfg_collection = cfg.get("chroma", "collection")
        if isinstance(cfg_collection, str) and collection == "messages":
            collection = cfg_collection

        cfg_embed = cfg.get("embeddings", "model")
        if isinstance(cfg_embed, str) and embedding_model == "sentence-transformers/all-MiniLM-L6-v2":
            embedding_model = cfg_embed

        cfg_batch = cfg.get("embeddings", "batch")
        if isinstance(cfg_batch, int) and int(batch) == 256:
            batch = int(cfg_batch)

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
            fast_embedding_model=fast_embedding_model,
            embed_batch_size=embed_batch_size,
            max_messages=max_messages,
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
            dry_run=bool(dry_run),
            store=store,
            console=console,
        )
    if bool(dry_run):
        console.print(f"Dry-run: would ingest {added} messages into collection '{collection}'.")
    elif added == 0:
        console.print("Nothing new to ingest.")
    else:
        console.print(f"Ingested {added} new messages into collection '{collection}'.")


@app.command("rebuild-store")
def rebuild_store(
    ctx: typer.Context,
    messages: Path = typer.Option(
        Path("data/messages.txt"),
        "--messages",
        dir_okay=False,
        help="Text file: one message per line",
    ),
    persist: Path = typer.Option(
        Path("data/chroma"),
        "--persist",
        help="ChromaDB persistence directory",
    ),
    collection: str = typer.Option("messages", "--collection", help="Chroma collection name"),
    embedding_model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        "--embedding-model",
        help="SentenceTransformers model for embeddings",
    ),
    fast_embedding_model: bool = typer.Option(
        False,
        "--fast-embedding-model",
        help="Use a smaller embedding model for speed (only affects the default embedding model).",
    ),
    batch: int = typer.Option(256, "--batch", help="Embed+add batch size"),
    embed_batch_size: int | None = typer.Option(
        None,
        "--embed-batch-size",
        help="SentenceTransformer encode() batch_size (optional; can reduce RAM/VRAM).",
    ),
    max_messages: int | None = typer.Option(
        None,
        "--max-messages",
        help="Only ingest the first N messages (for faster iteration).",
    ),
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
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Skip confirmation prompt for destructive rebuild",
    ),
):
    """Destructively reset the vector store and rebuild it from messages."""
    cfg: JugemuConfig | None = None
    if isinstance(getattr(ctx, "obj", None), dict):
        cfg = ctx.obj.get("config")

    default_messages = Path("data/messages.txt")
    default_persist = Path("data/chroma")

    if cfg is not None:
        cfg_messages = cfg.get("paths", "messages")
        if isinstance(cfg_messages, str) and messages == default_messages:
            messages = Path(cfg_messages)

        cfg_persist = cfg.get("paths", "chroma_persist")
        if isinstance(cfg_persist, str) and persist == default_persist:
            persist = Path(cfg_persist)

        cfg_collection = cfg.get("chroma", "collection")
        if isinstance(cfg_collection, str) and collection == "messages":
            collection = cfg_collection

        cfg_embed = cfg.get("embeddings", "model")
        if isinstance(cfg_embed, str) and embedding_model == "sentence-transformers/all-MiniLM-L6-v2":
            embedding_model = cfg_embed

        cfg_batch = cfg.get("embeddings", "batch")
        if isinstance(cfg_batch, int) and int(batch) == 256:
            batch = int(cfg_batch)

    if not bool(yes):
        typer.confirm(
            "This will DELETE and rebuild the vector store. Continue?",
            abort=True,
        )

    console = Console()
    console.print(Panel("Rebuilding vector store…", title="jugemu", border_style="cyan"))

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

    with console.status("Resetting store…", spinner="dots"):
        reset_vector_store(store)

    with console.status("Re-ingesting…", spinner="dots"):
        added = ingest_messages(
            messages_path=messages,
            persist_dir=persist,
            collection_name=collection,
            embedding_model=embedding_model,
            fast_embedding_model=fast_embedding_model,
            embed_batch_size=embed_batch_size,
            max_messages=max_messages,
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

    console.print(f"Rebuilt store; ingested {added} messages into collection '{collection}'.")


@app.command()
def train(
    ctx: typer.Context,
    messages: Path = typer.Option(
        Path("data/messages.txt"),
        "--messages",
        dir_okay=False,
        help="Path to messages text file",
    ),
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
    cfg: JugemuConfig | None = None
    if isinstance(getattr(ctx, "obj", None), dict):
        cfg = ctx.obj.get("config")

    default_messages = Path("data/messages.txt")
    if cfg is not None:
        cfg_messages = cfg.get("paths", "messages")
        if isinstance(cfg_messages, str) and messages == default_messages:
            messages = Path(cfg_messages)

        cfg_out = cfg.get("paths", "checkpoints")
        if isinstance(cfg_out, str) and out == Path("data/checkpoints"):
            out = Path(cfg_out)

        cfg_epochs = cfg.get("train", "epochs")
        if isinstance(cfg_epochs, int) and int(epochs) == 5:
            epochs = int(cfg_epochs)

        cfg_steps = cfg.get("train", "steps_per_epoch")
        if isinstance(cfg_steps, int) and int(steps_per_epoch) == 500:
            steps_per_epoch = int(cfg_steps)

        cfg_bs = cfg.get("train", "batch_size")
        if isinstance(cfg_bs, int) and int(batch_size) == 64:
            batch_size = int(cfg_bs)

        cfg_seq = cfg.get("train", "seq_len")
        if isinstance(cfg_seq, int) and int(seq_len) == 256:
            seq_len = int(cfg_seq)

        cfg_seed = cfg.get("train", "seed")
        if isinstance(cfg_seed, int) and int(seed) == 1337:
            seed = int(cfg_seed)

        cfg_dm = cfg.get("train", "d_model")
        if isinstance(cfg_dm, int) and int(d_model) == 192:
            d_model = int(cfg_dm)

        cfg_h = cfg.get("train", "n_heads")
        if isinstance(cfg_h, int) and int(n_heads) == 3:
            n_heads = int(cfg_h)

        cfg_l = cfg.get("train", "n_layers")
        if isinstance(cfg_l, int) and int(n_layers) == 3:
            n_layers = int(cfg_l)

        cfg_do = cfg.get("train", "dropout")
        if isinstance(cfg_do, (int, float)) and float(dropout) == 0.1:
            dropout = float(cfg_do)

        cfg_lr = cfg.get("train", "lr")
        if isinstance(cfg_lr, (int, float)) and float(lr) == float(3e-4):
            lr = float(cfg_lr)

        cfg_device = cfg.get("train", "device")
        if isinstance(cfg_device, str) and str(device) == "auto":
            device = cfg_device

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


@app.command("export-retrieval")
def export_retrieval(
    ctx: typer.Context,
    messages: Path = typer.Option(
        Path("data/messages.txt"),
        "--messages",
        dir_okay=False,
        help="messages.txt to sample random queries from",
    ),
    persist: Path = typer.Option(Path("data/chroma"), "--persist", help="ChromaDB persistence directory"),
    collection: str = typer.Option("messages", "--collection"),
    embedding_model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        "--embedding-model",
        help="SentenceTransformers model for embeddings",
    ),
    query: List[str] = typer.Option(
        [],
        "--query",
        help="Explicit query string (repeatable). If provided, ignores --samples/--seed.",
    ),
    samples: int = typer.Option(10, "--samples", help="How many random queries to sample"),
    k: int = typer.Option(6, "--k", help="How many neighbors to retrieve per query"),
    seed: int = typer.Option(1337, "--seed"),
    embed_batch_size: int | None = typer.Option(
        None,
        "--embed-batch-size",
        help="SentenceTransformer encode() batch_size for query embeddings.",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Optional output file path (writes results instead of just printing).",
    ),
    out_format: str = typer.Option(
        "jsonl",
        "--out-format",
        help="Output format when --out is set: json|jsonl.",
    ),
    no_print: bool = typer.Option(
        False,
        "--no-print",
        help="Suppress console output (useful for scripting).",
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
    """Dump random retrieval examples and their similarity scores."""
    cfg: JugemuConfig | None = None
    if isinstance(getattr(ctx, "obj", None), dict):
        cfg = ctx.obj.get("config")

    default_messages = Path("data/messages.txt")
    default_persist = Path("data/chroma")

    if cfg is not None:
        cfg_messages = cfg.get("paths", "messages")
        if isinstance(cfg_messages, str) and messages == default_messages:
            messages = Path(cfg_messages)

        cfg_persist = cfg.get("paths", "chroma_persist")
        if isinstance(cfg_persist, str) and persist == default_persist:
            persist = Path(cfg_persist)

        cfg_collection = cfg.get("chroma", "collection")
        if isinstance(cfg_collection, str) and collection == "messages":
            collection = cfg_collection

        cfg_embed = cfg.get("embeddings", "model")
        if isinstance(cfg_embed, str) and embedding_model == "sentence-transformers/all-MiniLM-L6-v2":
            embedding_model = cfg_embed

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

    if bool(no_print):
        results = dump_random_retrieval_samples(
            messages_path=messages,
            store=store,
            embedding_model=embedding_model,
            queries=list(query) if query else None,
            samples=int(samples),
            k=int(k),
            seed=int(seed),
            embed_batch_size=embed_batch_size,
            console=None,
        )
        if out is not None:
            write_retrieval_samples(results, out=Path(out), fmt=str(out_format))
        return

    console = Console()
    console.print(Panel("Exporting retrieval samples…", title="jugemu", border_style="cyan"))
    with console.status("Embedding queries + querying store…", spinner="dots"):
        results = dump_random_retrieval_samples(
            messages_path=messages,
            store=store,
            embedding_model=embedding_model,
            queries=list(query) if query else None,
            samples=int(samples),
            k=int(k),
            seed=int(seed),
            embed_batch_size=embed_batch_size,
            console=console,
        )

    if out is not None:
        p = write_retrieval_samples(results, out=Path(out), fmt=str(out_format))
        console.print(f"Wrote {len(results)} queries to {p}")


@app.command()
def chat(
    ctx: typer.Context,
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
    """Interactive chat (retrieval + generation)."""
    cfg: JugemuConfig | None = None
    if isinstance(getattr(ctx, "obj", None), dict):
        cfg = ctx.obj.get("config")

    default_messages = Path("data/messages.txt")
    default_persist = Path("data/chroma")
    default_checkpoint = Path("data/checkpoints/latest.pt")

    if cfg is not None:
        cfg_messages = cfg.get("paths", "messages")
        if isinstance(cfg_messages, str) and messages == default_messages:
            messages = Path(cfg_messages)

        cfg_persist = cfg.get("paths", "chroma_persist")
        if isinstance(cfg_persist, str) and persist == default_persist:
            persist = Path(cfg_persist)

        cfg_collection = cfg.get("chroma", "collection")
        if isinstance(cfg_collection, str) and collection == "messages":
            collection = cfg_collection

        cfg_embed = cfg.get("embeddings", "model")
        if isinstance(cfg_embed, str) and embedding_model == "sentence-transformers/all-MiniLM-L6-v2":
            embedding_model = cfg_embed

        cfg_ckpt = cfg.get("paths", "checkpoints")
        if isinstance(cfg_ckpt, str) and checkpoint == default_checkpoint:
            checkpoint = Path(cfg_ckpt) / "latest.pt"

        cfg_k = cfg.get("chat", "k")
        if isinstance(cfg_k, int) and int(k) == 6:
            k = int(cfg_k)

        cfg_max_new = cfg.get("chat", "max_new")
        if isinstance(cfg_max_new, int) and int(max_new) == 240:
            max_new = int(cfg_max_new)

        cfg_temp = cfg.get("chat", "temperature")
        if isinstance(cfg_temp, (int, float)) and float(temperature) == 0.9:
            temperature = float(cfg_temp)

        cfg_top_k = cfg.get("chat", "top_k")
        if isinstance(cfg_top_k, int) and int(top_k) == 60:
            top_k = int(cfg_top_k)

        cfg_device = cfg.get("chat", "device")
        if isinstance(cfg_device, str) and str(device) == "auto":
            device = cfg_device

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
        store=store,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

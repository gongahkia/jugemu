from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


@dataclass(frozen=True)
class PipelineResult:
    messages_path: Path
    persist_dir: Path
    collection: str
    checkpoint: Path
    smoke_sample: str


def run_pipeline(
    *,
    inp: Path,
    fmt: str,
    with_metadata: bool,
    messages_out: Path,
    persist_dir: Path,
    collection: str,
    embedding_model: str,
    vector_backend: str = "chroma",
    cassandra_contact_points: list[str] | None = None,
    cassandra_keyspace: str = "jugemu",
    cassandra_table: str = "messages",
    cassandra_secure_connect_bundle: Path | None = None,
    cassandra_username: str | None = None,
    cassandra_password: str | None = None,
    train_out: Path,
    train_epochs: int = 5,
    smoke_prompt: str = "hello",
    smoke_max_new: int = 120,
    device: str = "auto",
    # Dependency injection points for tests.
    parse_fn: Callable[[Path, str, bool], list[str]] | None = None,
    write_fn: Callable[[Sequence[str], Path], None] | None = None,
    ingest_fn: Callable[..., int] | None = None,
    train_fn: Callable[..., Path] | None = None,
    load_checkpoint_fn: Callable[..., object] | None = None,
    vocab_from_itos_fn: Callable[..., tuple[dict, list[str]]] | None = None,
    sample_fn: Callable[..., str] | None = None,
) -> PipelineResult:
    if parse_fn is None or write_fn is None or ingest_fn is None or train_fn is None:
        from .ingest_chroma import ingest_messages
        from .load_checkpoint import load_checkpoint, vocab_from_itos
        from .parse_exports import parse_export, write_canonical_messages
        from .sample import sample_text
        from .train_char_model import train_char_model

        from .store_factory import make_vector_store

        parse_fn = lambda p, f, m: parse_export(p, f, include_metadata=m)
        write_fn = write_canonical_messages
        ingest_fn = ingest_messages
        train_fn = train_char_model
        load_checkpoint_fn = load_checkpoint
        vocab_from_itos_fn = vocab_from_itos
        sample_fn = sample_text

    assert parse_fn is not None
    assert write_fn is not None
    assert ingest_fn is not None
    assert train_fn is not None
    assert load_checkpoint_fn is not None
    assert vocab_from_itos_fn is not None
    assert sample_fn is not None

    lines = parse_fn(inp, fmt, bool(with_metadata))
    write_fn(lines, messages_out)

    store = None
    try:
        from .store_factory import make_vector_store

        store = make_vector_store(
            backend=str(vector_backend),
            persist_dir=Path(persist_dir),
            collection_name=str(collection),
            cassandra_contact_points=list(cassandra_contact_points) if cassandra_contact_points else None,
            cassandra_keyspace=str(cassandra_keyspace),
            cassandra_table=str(cassandra_table),
            cassandra_secure_connect_bundle=cassandra_secure_connect_bundle,
            cassandra_username=cassandra_username,
            cassandra_password=cassandra_password,
        )
    except Exception:
        store = None

    ingest_fn(
        messages_path=messages_out,
        persist_dir=persist_dir,
        collection_name=collection,
        embedding_model=embedding_model,
        store=store,
    )

    ckpt = train_fn(
        messages_path=messages_out,
        out_dir=train_out,
        epochs=int(train_epochs),
        device=str(device),
    )

    loaded = load_checkpoint_fn(str(ckpt), device=str(device))
    stoi, itos = vocab_from_itos_fn(loaded.vocab_itos)
    smoke = sample_fn(
        model=loaded.model,
        prompt=str(smoke_prompt),
        stoi=stoi,
        itos=itos,
        device=str(device),
        max_new_tokens=int(smoke_max_new),
        return_full=True,
    )

    return PipelineResult(
        messages_path=Path(messages_out),
        persist_dir=Path(persist_dir),
        collection=str(collection),
        checkpoint=Path(ckpt),
        smoke_sample=str(smoke),
    )

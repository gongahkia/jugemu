from __future__ import annotations

from pathlib import Path

from src.pipeline import run_pipeline


class _Loaded:
    def __init__(self) -> None:
        self.vocab_itos = ["a", "b"]
        self.model = object()


def test_run_pipeline_calls_in_order(tmp_path: Path) -> None:
    calls: list[tuple[str, object]] = []

    def parse_fn(inp: Path, fmt: str, with_metadata: bool) -> list[str]:
        calls.append(("parse", (inp, fmt, with_metadata)))
        return ["m1", "m2"]

    def write_fn(lines, out: Path) -> None:
        calls.append(("write", (list(lines), out)))

    def ingest_fn(**kwargs) -> int:
        calls.append(("ingest", kwargs))
        return 2

    def train_fn(**kwargs) -> Path:
        calls.append(("train", kwargs))
        return tmp_path / "latest.pt"

    def load_ckpt_fn(_path: str, device: str):
        calls.append(("load", (_path, device)))
        return _Loaded()

    def vocab_from_itos_fn(vocab_itos):
        calls.append(("vocab", tuple(vocab_itos)))
        return ({"a": 0}, ["a"])

    def sample_fn(**kwargs) -> str:
        calls.append(("sample", kwargs))
        return "ok"

    out = run_pipeline(
        inp=tmp_path / "in.txt",
        fmt="plain",
        with_metadata=False,
        messages_out=tmp_path / "messages.txt",
        persist_dir=tmp_path / "chroma",
        collection="messages",
        embedding_model="model",
        train_out=tmp_path / "ckpts",
        train_epochs=1,
        smoke_prompt="hi",
        smoke_max_new=5,
        device="cpu",
        parse_fn=parse_fn,
        write_fn=write_fn,
        ingest_fn=ingest_fn,
        train_fn=train_fn,
        load_checkpoint_fn=load_ckpt_fn,
        vocab_from_itos_fn=vocab_from_itos_fn,
        sample_fn=sample_fn,
    )

    assert out.smoke_sample == "ok"
    assert [c[0] for c in calls] == ["parse", "write", "ingest", "train", "load", "vocab", "sample"]

    # Ensure a couple key kwargs are correctly passed through
    ingest_kwargs = dict(calls[2][1])
    assert ingest_kwargs["collection_name"] == "messages"
    assert ingest_kwargs["embedding_model"] == "model"

    train_kwargs = dict(calls[3][1])
    assert train_kwargs["epochs"] == 1
    assert train_kwargs["device"] == "cpu"

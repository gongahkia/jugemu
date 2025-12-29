from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import src.chat as chat_mod
import src.cli as cli


def test_cli_chat_json_emits_turn_summary(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    # Avoid importing/initializing any real vector backend.
    def _fake_make_vector_store(**_kwargs):  # type: ignore[no-untyped-def]
        return object()

    monkeypatch.setattr(cli, "make_vector_store", _fake_make_vector_store)

    # Avoid loading any real model.
    class _Loaded:
        model = object()
        vocab_itos = ["a"]

    monkeypatch.setattr(chat_mod, "load_checkpoint", lambda *_a, **_k: _Loaded())
    monkeypatch.setattr(chat_mod, "vocab_from_itos", lambda _itos: ({"a": 0}, ["a"]))

    # Avoid real retrieval.
    class _Hit:
        def __init__(self, text: str, score: float, metadata: dict | None = None) -> None:
            self.text = text
            self.score = score
            self.metadata = metadata

    monkeypatch.setattr(
        chat_mod,
        "retrieve_similar",
        lambda **_kwargs: [_Hit("hello there", 0.9, {"line": 1})],
    )

    # Deterministic generation.
    monkeypatch.setattr(chat_mod, "sample_text", lambda **_kwargs: "YOU: hi back")

    missing_messages = tmp_path / "nope.txt"
    res = runner.invoke(
        cli.app,
        [
            "chat",
            "--json",
            "--messages",
            str(missing_messages),
            "--reply-strategy",
            "generate",
        ],
        input="hi\n/exit\n",
    )
    assert res.exit_code == 0

    lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1

    evt = json.loads(lines[0])
    assert evt["type"] == "turn"
    assert evt["user"] == "hi"
    assert evt["answer"] == "hi back"
    assert evt["used_strategy"] == "generate"
    assert isinstance(evt["retrieved"], list)
    assert evt["retrieved"][0]["text"] == "hello there"

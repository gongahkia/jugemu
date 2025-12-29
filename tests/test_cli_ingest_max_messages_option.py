from __future__ import annotations

from typer.testing import CliRunner

import src.cli as cli


def test_cli_ingest_passes_max_messages(monkeypatch):
    runner = CliRunner()

    captured = {}

    def _fake_make_vector_store(**kwargs):  # type: ignore[no-untyped-def]
        return object()

    def _fake_ingest_messages(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "make_vector_store", _fake_make_vector_store)
    monkeypatch.setattr(cli, "ingest_messages", _fake_ingest_messages)

    res = runner.invoke(cli.app, ["ingest", "--max-messages", "7"])
    assert res.exit_code == 0
    assert captured.get("max_messages") == 7

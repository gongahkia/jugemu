from __future__ import annotations

from typer.testing import CliRunner

import src.cli as cli


def test_cli_rebuild_store_resets_then_ingests(monkeypatch):
    runner = CliRunner()

    calls: list[str] = []

    def _fake_make_vector_store(**kwargs):  # type: ignore[no-untyped-def]
        return object()

    def _fake_reset_vector_store(store):  # type: ignore[no-untyped-def]
        calls.append("reset")

    def _fake_ingest_messages(**kwargs):  # type: ignore[no-untyped-def]
        calls.append("ingest")
        return 0

    monkeypatch.setattr(cli, "make_vector_store", _fake_make_vector_store)
    monkeypatch.setattr(cli, "reset_vector_store", _fake_reset_vector_store)
    monkeypatch.setattr(cli, "ingest_messages", _fake_ingest_messages)

    res = runner.invoke(cli.app, ["rebuild-store", "--yes"])
    assert res.exit_code == 0
    assert calls == ["reset", "ingest"]


def test_cli_rebuild_store_without_yes_can_abort(monkeypatch):
    runner = CliRunner()

    calls: list[str] = []

    def _fake_make_vector_store(**kwargs):  # type: ignore[no-untyped-def]
        calls.append("make")
        return object()

    def _fake_reset_vector_store(store):  # type: ignore[no-untyped-def]
        calls.append("reset")

    def _fake_ingest_messages(**kwargs):  # type: ignore[no-untyped-def]
        calls.append("ingest")
        return 0

    monkeypatch.setattr(cli, "make_vector_store", _fake_make_vector_store)
    monkeypatch.setattr(cli, "reset_vector_store", _fake_reset_vector_store)
    monkeypatch.setattr(cli, "ingest_messages", _fake_ingest_messages)

    res = runner.invoke(cli.app, ["rebuild-store"], input="n\n")
    assert res.exit_code != 0
    assert calls == []

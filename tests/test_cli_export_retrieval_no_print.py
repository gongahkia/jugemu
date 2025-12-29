from __future__ import annotations

from typer.testing import CliRunner

import src.cli as cli


def test_cli_export_retrieval_no_print_passes_console_none(monkeypatch):
    runner = CliRunner()

    captured = {}

    def _fake_make_vector_store(**kwargs):  # type: ignore[no-untyped-def]
        return object()

    def _fake_dump_random_retrieval_samples(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli, "make_vector_store", _fake_make_vector_store)
    monkeypatch.setattr(cli, "dump_random_retrieval_samples", _fake_dump_random_retrieval_samples)

    res = runner.invoke(cli.app, ["export-retrieval", "--no-print"])
    assert res.exit_code == 0
    assert captured.get("console") is None

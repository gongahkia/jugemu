from __future__ import annotations

from typer.testing import CliRunner

import src.cli as cli


def test_cli_chat_rejects_invalid_vector_backend(monkeypatch) -> None:
    runner = CliRunner()

    def _fail(**_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("make_vector_store should not be called")

    monkeypatch.setattr(cli, "make_vector_store", _fail)

    res = runner.invoke(cli.app, ["chat", "--json", "--vector-backend", "wat"], input="hi\n/exit\n")
    assert res.exit_code != 0
    assert "--vector-backend must be one of" in res.output

from __future__ import annotations

from typer.testing import CliRunner

import src.cli as cli


def test_cli_version_prints_version() -> None:
    runner = CliRunner()

    res = runner.invoke(cli.app, ["--version"])
    assert res.exit_code == 0
    assert res.output.startswith("jugemu ")

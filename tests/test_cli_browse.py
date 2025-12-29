from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import src.cli as cli


def test_cli_browse_prints_sections(tmp_path: Path) -> None:
    runner = CliRunner()

    messages_path = tmp_path / "messages.txt"
    messages_path.write_text("hello world\nhello\n", encoding="utf-8")

    res = runner.invoke(cli.app, ["browse", "--messages", str(messages_path), "--top", "2"])
    assert res.exit_code == 0
    assert "Top characters" in res.output
    assert "Top tokens" in res.output

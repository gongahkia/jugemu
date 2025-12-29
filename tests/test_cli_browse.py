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


def test_cli_browse_mode_tokens_only(tmp_path: Path) -> None:
    runner = CliRunner()

    messages_path = tmp_path / "messages.txt"
    messages_path.write_text("hello world\nhello\n", encoding="utf-8")

    res = runner.invoke(cli.app, ["browse", "--messages", str(messages_path), "--mode", "tokens", "--top", "2"])
    assert res.exit_code == 0
    assert "Top tokens" in res.output
    assert "Top characters" not in res.output


def test_cli_browse_min_count_filters(tmp_path: Path) -> None:
    runner = CliRunner()

    messages_path = tmp_path / "messages.txt"
    messages_path.write_text("a\na\nb\n", encoding="utf-8")

    res = runner.invoke(
        cli.app,
        ["browse", "--messages", str(messages_path), "--mode", "tokens", "--min-count", "2", "--top", "10"],
    )
    assert res.exit_code == 0
    assert "2" in res.output
    assert " a" in res.output
    assert "\tb" not in res.output

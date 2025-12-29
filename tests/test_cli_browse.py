from __future__ import annotations

import json
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


def test_cli_browse_no_header_suppresses_labels(tmp_path: Path) -> None:
    runner = CliRunner()

    messages_path = tmp_path / "messages.txt"
    messages_path.write_text("hello\nhello\n", encoding="utf-8")

    res = runner.invoke(cli.app, ["browse", "--messages", str(messages_path), "--mode", "tokens", "--no-header"])
    assert res.exit_code == 0
    assert "Browsing corpus stats" not in res.output
    assert "Top tokens" not in res.output
    assert "hello" in res.output


def test_cli_browse_json_outputs_parseable_json(tmp_path: Path) -> None:
    runner = CliRunner()

    messages_path = tmp_path / "messages.txt"
    messages_path.write_text("a\na\nb\n", encoding="utf-8")

    res = runner.invoke(
        cli.app,
        ["browse", "--messages", str(messages_path), "--mode", "tokens", "--min-count", "2", "--top", "10", "--json"],
    )
    assert res.exit_code == 0
    assert res.output.endswith("\n")
    assert not res.output.endswith("\n\n")
    payload = json.loads(res.output)
    assert payload["mode"] == "tokens"
    assert payload["min_count"] == 2
    assert payload["tokens"] == [{"token": "a", "count": 2}]
    assert "Top tokens" not in res.output


def test_cli_browse_json_pretty_outputs_parseable_json(tmp_path: Path) -> None:
    runner = CliRunner()

    messages_path = tmp_path / "messages.txt"
    messages_path.write_text("a\na\nb\n", encoding="utf-8")

    res = runner.invoke(
        cli.app,
        [
            "browse",
            "--messages",
            str(messages_path),
            "--mode",
            "tokens",
            "--min-count",
            "2",
            "--top",
            "10",
            "--json-pretty",
        ],
    )
    assert res.exit_code == 0
    assert res.output.endswith("\n")
    assert not res.output.endswith("\n\n")
    payload = json.loads(res.output)
    assert payload["mode"] == "tokens"
    assert "\n  \"mode\"" in res.output


def test_cli_browse_uses_config_defaults(tmp_path: Path) -> None:
    runner = CliRunner()

    messages_path = tmp_path / "messages.txt"
    messages_path.write_text("a\na\nb\n", encoding="utf-8")

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
[paths]
messages = "messages.txt"

[browse]
mode = "tokens"
top = 10
min_count = 2
""".lstrip(),
        encoding="utf-8",
    )

    res = runner.invoke(cli.app, ["--config", str(cfg_path), "browse", "--messages", str(messages_path)])
    assert res.exit_code == 0
    assert "Top tokens" in res.output
    assert "Top characters" not in res.output
    assert "\tb" not in res.output

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import src.cli as cli


def test_cli_browse_out_writes_json_report(tmp_path: Path) -> None:
    runner = CliRunner()

    messages_path = tmp_path / "messages.txt"
    messages_path.write_text("a\na\nb\n", encoding="utf-8")

    out_path = tmp_path / "report.json"

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
            "--json",
            "--out",
            str(out_path),
        ],
    )
    assert res.exit_code == 0

    payload_stdout = json.loads(res.output)
    payload_file = json.loads(out_path.read_text(encoding="utf-8"))

    assert payload_stdout == payload_file
    assert payload_file["tokens"] == [{"token": "a", "count": 2}]

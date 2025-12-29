from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import src.cli as cli


def test_cli_browse_uses_config_defaults_for_mode_min_count_and_json(tmp_path: Path) -> None:
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
min_count = 2
top = 10
json = true
""".lstrip(),
        encoding="utf-8",
    )

    # Run from tmp_path so relative [paths].messages works.
    cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        res = runner.invoke(cli.app, ["--config", str(cfg_path), "browse"])
    finally:
        import os

        os.chdir(cwd)

    assert res.exit_code == 0
    payload = json.loads(res.output)
    assert payload["mode"] == "tokens"
    assert payload["min_count"] == 2
    assert payload["tokens"] == [{"token": "a", "count": 2}]

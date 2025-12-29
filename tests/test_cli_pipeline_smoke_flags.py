from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import src.cli as cli


def test_cli_pipeline_passes_smoke_flags(tmp_path: Path, monkeypatch):
    runner = CliRunner()

    inp = tmp_path / "in.txt"
    inp.write_text("hi\n", encoding="utf-8")

    captured = {}

    def _fake_run_pipeline(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)

        class _Res:
            messages_path = Path("x")
            persist_dir = Path("y")
            collection = "messages"
            checkpoint = Path("z")
            smoke_sample = "ok"

        return _Res()

    monkeypatch.setattr(cli, "run_pipeline", _fake_run_pipeline)

    res = runner.invoke(
        cli.app,
        [
            "pipeline",
            "--in",
            str(inp),
            "--smoke-prompt",
            "yo",
            "--smoke-max-new",
            "7",
        ],
    )
    assert res.exit_code == 0
    assert captured["smoke_prompt"] == "yo"
    assert captured["smoke_max_new"] == 7

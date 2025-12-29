from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import src.cli as cli


def test_cli_pipeline_uses_config_defaults_for_vector_backend_and_smoke(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    inp = tmp_path / "in.txt"
    inp.write_text("hi\n", encoding="utf-8")

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
[pipeline]
vector_backend = "cassandra"
smoke_prompt = "yo"
smoke_max_new = 7
""".lstrip(),
        encoding="utf-8",
    )

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

    res = runner.invoke(cli.app, ["--config", str(cfg_path), "pipeline", "--in", str(inp)])
    assert res.exit_code == 0
    assert captured["vector_backend"] == "cassandra"
    assert captured["smoke_prompt"] == "yo"
    assert captured["smoke_max_new"] == 7

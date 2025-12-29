from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import src.cli as cli


def test_cli_pipeline_passes_vector_backend_args(tmp_path: Path, monkeypatch):
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
            "--vector-backend",
            "cassandra",
            "--cassandra-contact-point",
            "10.0.0.1",
            "--cassandra-keyspace",
            "ks",
            "--cassandra-table",
            "tbl",
        ],
    )
    assert res.exit_code == 0
    assert captured["vector_backend"] == "cassandra"
    assert captured["cassandra_contact_points"] == ["10.0.0.1"]
    assert captured["cassandra_keyspace"] == "ks"
    assert captured["cassandra_table"] == "tbl"


def test_cli_pipeline_rejects_invalid_vector_backend(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()

    inp = tmp_path / "in.txt"
    inp.write_text("hi\n", encoding="utf-8")

    def _fail(**_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("run_pipeline should not be called for invalid args")

    monkeypatch.setattr(cli, "run_pipeline", _fail)

    res = runner.invoke(
        cli.app,
        [
            "pipeline",
            "--in",
            str(inp),
            "--vector-backend",
            "wat",
        ],
    )
    assert res.exit_code != 0
    assert "--vector-backend must be one of" in res.output

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import src.cli as cli


def test_cli_export_retrieval_config_defaults_set_out_format(tmp_path: Path, monkeypatch):
    runner = CliRunner()

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
[export_retrieval]
out_format = "json"
no_print = true
""".lstrip(),
        encoding="utf-8",
    )

    out_path = tmp_path / "out.json"

    def _fake_make_vector_store(**kwargs):  # type: ignore[no-untyped-def]
        return object()

    def _fake_dump_random_retrieval_samples(**kwargs):  # type: ignore[no-untyped-def]
        return []

    captured_write = {}

    def _fake_write_retrieval_samples(results, out, fmt):  # type: ignore[no-untyped-def]
        captured_write.update({"out": out, "fmt": fmt, "n": len(results)})
        return out

    monkeypatch.setattr(cli, "make_vector_store", _fake_make_vector_store)
    monkeypatch.setattr(cli, "dump_random_retrieval_samples", _fake_dump_random_retrieval_samples)
    monkeypatch.setattr(cli, "write_retrieval_samples", _fake_write_retrieval_samples)

    res = runner.invoke(cli.app, ["--config", str(cfg_path), "export-retrieval", "--out", str(out_path)])
    assert res.exit_code == 0
    assert captured_write["out"] == out_path
    assert captured_write["fmt"] == "json"


def test_cli_export_retrieval_rejects_invalid_config_out_format(tmp_path: Path) -> None:
    runner = CliRunner()

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
[export_retrieval]
out_format = "wat"
no_print = true
""".lstrip(),
        encoding="utf-8",
    )

    out_path = tmp_path / "out.json"
    res = runner.invoke(cli.app, ["--config", str(cfg_path), "export-retrieval", "--out", str(out_path)])
    assert res.exit_code != 0
    assert "config export_retrieval.out_format must be one of" in res.output

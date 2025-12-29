from __future__ import annotations

import json
from typer.testing import CliRunner

import src.cli as cli


def test_cli_export_retrieval_calls_dumper(monkeypatch):
    runner = CliRunner()

    captured = {}

    def _fake_make_vector_store(**kwargs):  # type: ignore[no-untyped-def]
        return object()

    def _fake_dump_random_retrieval_samples(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli, "make_vector_store", _fake_make_vector_store)
    monkeypatch.setattr(cli, "dump_random_retrieval_samples", _fake_dump_random_retrieval_samples)

    res = runner.invoke(cli.app, ["export-retrieval", "--samples", "3", "--k", "2"])
    assert res.exit_code == 0
    assert captured.get("samples") == 3
    assert captured.get("k") == 2


def test_cli_export_retrieval_json_outputs_parseable_json(monkeypatch):
    runner = CliRunner()

    def _fake_make_vector_store(**kwargs):  # type: ignore[no-untyped-def]
        return object()

    def _fake_dump_random_retrieval_samples(**kwargs):  # type: ignore[no-untyped-def]
        return [{"query": "q", "results": []}]

    monkeypatch.setattr(cli, "make_vector_store", _fake_make_vector_store)
    monkeypatch.setattr(cli, "dump_random_retrieval_samples", _fake_dump_random_retrieval_samples)

    res = runner.invoke(cli.app, ["export-retrieval", "--json"])
    assert res.exit_code == 0
    assert res.output.endswith("\n")
    assert not res.output.endswith("\n\n")
    payload = json.loads(res.output)
    assert payload[0]["query"] == "q"
    assert "Exporting retrieval samples" not in res.output


def test_cli_export_retrieval_fail_on_empty_exits_non_zero(monkeypatch):
    runner = CliRunner()

    def _fake_make_vector_store(**kwargs):  # type: ignore[no-untyped-def]
        return object()

    def _fake_dump_random_retrieval_samples(**kwargs):  # type: ignore[no-untyped-def]
        return []

    monkeypatch.setattr(cli, "make_vector_store", _fake_make_vector_store)
    monkeypatch.setattr(cli, "dump_random_retrieval_samples", _fake_dump_random_retrieval_samples)

    res = runner.invoke(cli.app, ["export-retrieval", "--json", "--fail-on-empty"])
    assert res.exit_code != 0


def test_cli_export_retrieval_uses_config_defaults(tmp_path, monkeypatch):
    runner = CliRunner()

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
[export_retrieval]
samples = 3
k = 2
seed = 42
embed_batch_size = 7
out_format = "json"
no_print = true
""".lstrip(),
        encoding="utf-8",
    )

    captured = {}

    def _fake_make_vector_store(**kwargs):  # type: ignore[no-untyped-def]
        return object()

    def _fake_dump_random_retrieval_samples(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli, "make_vector_store", _fake_make_vector_store)
    monkeypatch.setattr(cli, "dump_random_retrieval_samples", _fake_dump_random_retrieval_samples)

    res = runner.invoke(cli.app, ["--config", str(cfg_path), "export-retrieval"])
    assert res.exit_code == 0
    assert captured.get("samples") == 3
    assert captured.get("k") == 2
    assert captured.get("seed") == 42
    assert captured.get("embed_batch_size") == 7

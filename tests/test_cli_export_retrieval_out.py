from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import src.cli as cli


def test_cli_export_retrieval_writes_out_jsonl(tmp_path: Path, monkeypatch):
    runner = CliRunner()

    def _fake_make_vector_store(**kwargs):  # type: ignore[no-untyped-def]
        return object()

    def _fake_dump_random_retrieval_samples(**kwargs):  # type: ignore[no-untyped-def]
        return [{"query": "q", "results": []}]

    monkeypatch.setattr(cli, "make_vector_store", _fake_make_vector_store)
    monkeypatch.setattr(cli, "dump_random_retrieval_samples", _fake_dump_random_retrieval_samples)

    out = tmp_path / "out.jsonl"
    res = runner.invoke(cli.app, ["export-retrieval", "--out", str(out), "--out-format", "jsonl"])
    assert res.exit_code == 0

    lines = out.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["query"] == "q"


def test_cli_export_retrieval_rejects_invalid_out_format(tmp_path: Path) -> None:
    runner = CliRunner()

    out = tmp_path / "out.jsonl"
    res = runner.invoke(cli.app, ["export-retrieval", "--out", str(out), "--out-format", "wat"])
    assert res.exit_code != 0
    assert "--out-format must be one of" in res.output

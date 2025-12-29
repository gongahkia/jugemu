from __future__ import annotations

import json
from typer.testing import CliRunner

import src.cli as cli


def test_cli_schema_prints_version_and_backends() -> None:
    runner = CliRunner()

    res = runner.invoke(cli.app, ["schema"])
    assert res.exit_code == 0
    assert "schema_version" in res.output
    assert "vector_backends" in res.output
    assert "chroma" in res.output
    assert "cassandra" in res.output


def test_cli_schema_json_outputs_parseable_json() -> None:
    runner = CliRunner()

    res = runner.invoke(cli.app, ["schema", "--json"])
    assert res.exit_code == 0
    assert res.output.endswith("\n")
    assert not res.output.endswith("\n\n")
    payload = json.loads(res.output)
    assert "schema_version" in payload


def test_cli_schema_json_pretty_outputs_parseable_json() -> None:
    runner = CliRunner()

    res = runner.invoke(cli.app, ["schema", "--json-pretty"])
    assert res.exit_code == 0
    assert res.output.endswith("\n")
    assert not res.output.endswith("\n\n")
    payload = json.loads(res.output)
    assert "schema_version" in payload
    assert "\n  \"schema_version\"" in res.output

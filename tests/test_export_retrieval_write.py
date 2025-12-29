from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.export_retrieval import write_retrieval_samples


def test_write_retrieval_samples_json(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    res = [{"query": "q", "results": [{"id": "1", "score": 0.1, "text": "t", "metadata": None}]}]

    write_retrieval_samples(res, out=out, fmt="json")

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data[0]["query"] == "q"


def test_write_retrieval_samples_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    res = [{"query": "a", "results": []}, {"query": "b", "results": []}]

    write_retrieval_samples(res, out=out, fmt="jsonl")

    lines = out.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["query"] == "a"
    assert json.loads(lines[1])["query"] == "b"


def test_write_retrieval_samples_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_retrieval_samples([], out=tmp_path / "x", fmt="csv")

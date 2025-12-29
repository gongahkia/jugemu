from __future__ import annotations

import json

from src.export_retrieval import format_retrieval_samples_json


def test_format_retrieval_samples_json_is_parseable_and_newline_terminated() -> None:
    raw = format_retrieval_samples_json([{"query": "q", "results": []}])
    assert raw.endswith("\n")
    payload = json.loads(raw)
    assert payload[0]["query"] == "q"

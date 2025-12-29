from __future__ import annotations

from pathlib import Path

import pytest

from src.chat import build_prompt


def test_prompt_minimal() -> None:
    p = build_prompt(user_text="hi", retrieved=[], messages_path=None, template="minimal")
    assert p == "USER: hi\nYOU: "


def test_prompt_scaffold_contains_header() -> None:
    p = build_prompt(user_text="hi", retrieved=[], messages_path=None, template="scaffold")
    assert p.startswith("This is a chat.")
    assert "USER: hi" in p
    assert p.rstrip().endswith("YOU:")


def test_prompt_few_shot_ignores_missing_messages_file(tmp_path: Path) -> None:
    p = build_prompt(user_text="hi", retrieved=[], messages_path=tmp_path / "nope.txt", template="few-shot")
    assert p == "USER: hi\nYOU: "


def test_prompt_invalid_template_raises() -> None:
    with pytest.raises(ValueError):
        build_prompt(user_text="hi", retrieved=[], messages_path=None, template="wat")

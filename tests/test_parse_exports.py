from __future__ import annotations

from pathlib import Path

from src.parse_exports import parse_telegram_json, parse_whatsapp_export


def _fx(name: str) -> Path:
    return Path(__file__).parent / "fixtures" / name


def test_parse_whatsapp_export_matches_golden() -> None:
    got = parse_whatsapp_export(_fx("whatsapp_chat.txt"))
    expected = _fx("whatsapp_expected.txt").read_text(encoding="utf-8")
    assert "\n".join(got) + "\n" == expected


def test_parse_telegram_json_matches_golden() -> None:
    got = parse_telegram_json(_fx("telegram_result.json"))
    expected = _fx("telegram_expected.txt").read_text(encoding="utf-8")
    assert "\n".join(got) + "\n" == expected

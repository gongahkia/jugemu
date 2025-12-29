from __future__ import annotations

from pathlib import Path

from src.parse_exports import parse_telegram_json, parse_whatsapp_export


def test_parse_whatsapp_export_with_metadata(tmp_path: Path) -> None:
    p = tmp_path / "chat.txt"
    p.write_text(
        "12/29/25, 10:30 PM - Alice: hello 🙂\n"
        "12/29/25, 10:31 PM - Bob: yo\n",
        encoding="utf-8",
    )
    got = parse_whatsapp_export(p, include_metadata=True)
    assert got == [
        "[12/29/25, 10:30 PM] Alice: hello 🙂",
        "[12/29/25, 10:31 PM] Bob: yo",
    ]


def test_parse_telegram_json_with_metadata(tmp_path: Path) -> None:
    p = tmp_path / "result.json"
    p.write_text(
        '{"messages": ['
        '{"type": "message", "date": "2025-12-29T10:00:00", "from": "Alice", "text": "hi"},'
        '{"type": "message", "date": "2025-12-29T10:01:00", "from": "Bob", "text": [{"type":"plain","text":"yo"}]}'
        ']}',
        encoding="utf-8",
    )
    got = parse_telegram_json(p, include_metadata=True)
    assert got == [
        "[2025-12-29T10:00:00] Alice: hi",
        "[2025-12-29T10:01:00] Bob: yo",
    ]

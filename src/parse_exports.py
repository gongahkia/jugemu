from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List


class ParseError(ValueError):
    pass


def _clean_line(s: str) -> str:
    # Canonical format: one message per line.
    return " ".join(s.replace("\r\n", "\n").replace("\r", "\n").split()).strip()


def parse_plain_lines(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    out: List[str] = []
    for ln in raw.split("\n"):
        msg = ln.strip()
        if msg:
            out.append(msg)
    return out


_WA_HEADER_RE = re.compile(
    r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s+"
    r"(?P<time>\d{1,2}:\d{2})(?:\s?(?P<ampm>AM|PM))?\s+-\s+(?P<body>.*)$"
)


def parse_whatsapp_export(path: Path, *, include_metadata: bool = False) -> List[str]:
    """Parse WhatsApp .txt exports.

    Expected line format (varies by locale; this covers the common US-like export):
      12/29/25, 10:30 PM - Name: message

    Multiline messages are supported: continuation lines (no header) are appended.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    messages: List[str] = []
    cur: List[str] = []

    def flush() -> None:
        nonlocal cur
        if not cur:
            return
        joined = _clean_line(" ".join(cur))
        if joined:
            messages.append(joined)
        cur = []

    cur_prefix: str | None = None

    for ln in raw.split("\n"):
        m = _WA_HEADER_RE.match(ln)
        if m:
            flush()
            date = m.group("date").strip()
            time = m.group("time").strip()
            ampm = (m.group("ampm") or "").strip()
            ts = f"{date}, {time}{(' ' + ampm) if ampm else ''}".strip()
            body = m.group("body").strip()
            # Skip system messages without a "Name: " prefix.
            if ":" in body:
                _name, msg = body.split(":", 1)
                name = _name.strip()
                msg = msg.strip()
                if msg and msg != "<Media omitted>":
                    cur_prefix = f"[{ts}] {name}: " if include_metadata else None
                    cur = [msg]
                else:
                    cur_prefix = None
                    cur = []
            else:
                cur_prefix = None
                cur = []
            continue

        # Continuation line.
        if cur and ln.strip():
            cur.append(ln.strip())

    flush()
    if include_metadata:
        # Re-run with prefixes applied (flush used only message text).
        # We store prefixes during parsing and apply them as we append.
        # To keep logic minimal, we rebuild in-place when metadata is enabled.
        rebuilt: List[str] = []
        raw = path.read_text(encoding="utf-8", errors="replace")
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        cur = []
        prefix: str | None = None

        def flush2() -> None:
            nonlocal cur, prefix
            if not cur:
                prefix = None
                return
            joined = _clean_line(" ".join(cur))
            if joined:
                rebuilt.append((prefix or "") + joined)
            cur = []
            prefix = None

        for ln in raw.split("\n"):
            m = _WA_HEADER_RE.match(ln)
            if m:
                flush2()
                date = m.group("date").strip()
                time = m.group("time").strip()
                ampm = (m.group("ampm") or "").strip()
                ts = f"{date}, {time}{(' ' + ampm) if ampm else ''}".strip()
                body = m.group("body").strip()
                if ":" in body:
                    _name, msg = body.split(":", 1)
                    name = _name.strip()
                    msg = msg.strip()
                    if msg and msg != "<Media omitted>":
                        prefix = f"[{ts}] {name}: "
                        cur = [msg]
                    else:
                        prefix = None
                        cur = []
                else:
                    prefix = None
                    cur = []
                continue

            if cur and ln.strip():
                cur.append(ln.strip())

        flush2()
        return rebuilt

    return messages


def _telegram_text_to_str(t) -> str:
    if t is None:
        return ""
    if isinstance(t, str):
        return t
    if isinstance(t, list):
        parts: List[str] = []
        for item in t:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                v = item.get("text")
                if isinstance(v, str):
                    parts.append(v)
        return "".join(parts)
    return ""


def parse_telegram_json(path: Path, *, include_metadata: bool = False) -> List[str]:
    """Parse Telegram Desktop JSON export (result.json).

    Telegram exports have a `messages` array with each message containing a `text`
    field that is either a string or a list of rich-text fragments.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}")

    msgs = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(msgs, list):
        raise ParseError("Telegram JSON must contain a top-level 'messages' array")

    out: List[str] = []
    for msg in msgs:
        if not isinstance(msg, dict):
            continue
        if msg.get("type") not in {None, "message"}:
            continue
        text = _telegram_text_to_str(msg.get("text"))
        cleaned = _clean_line(text)
        if cleaned:
            if include_metadata:
                date = msg.get("date")
                sender = msg.get("from")
                prefix = ""
                if isinstance(date, str) and date.strip():
                    prefix += f"[{date.strip()}] "
                if isinstance(sender, str) and sender.strip():
                    prefix += f"{sender.strip()}: "
                out.append(prefix + cleaned)
            else:
                out.append(cleaned)
    return out


def parse_export(path: Path, fmt: str, *, include_metadata: bool = False) -> List[str]:
    fmt2 = fmt.strip().lower()
    if fmt2 in {"plain", "txt", "lines"}:
        return parse_plain_lines(path)
    if fmt2 in {"whatsapp", "wa"}:
        return parse_whatsapp_export(path, include_metadata=bool(include_metadata))
    if fmt2 in {"telegram", "telegram-json", "telegram_json", "tg"}:
        return parse_telegram_json(path, include_metadata=bool(include_metadata))
    raise ParseError(f"Unknown format: {fmt}")


def write_canonical_messages(lines: Iterable[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(_clean_line(ln) for ln in lines if _clean_line(ln))
    out_path.write_text(content + ("\n" if content else ""), encoding="utf-8")

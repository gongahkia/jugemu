from __future__ import annotations

import re


_NON_NEWLINE_WHITESPACE_RE = re.compile(r"[\t\f\v ]+")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols & pictographs + extended emoji
    "\U00002600-\U000026FF"  # misc symbols
    "\U00002700-\U000027BF"  # dingbats
    "]+",
    flags=re.UNICODE,
)


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def collapse_whitespace_preserve_newlines(text: str) -> str:
    # Collapse runs of spaces/tabs/etc but do not touch newlines.
    return "\n".join(_NON_NEWLINE_WHITESPACE_RE.sub(" ", ln).strip() for ln in text.split("\n"))


def strip_emoji(text: str) -> str:
    # Best-effort emoji stripping without extra deps.
    return _EMOJI_RE.sub("", text)


def normalize_text(
    text: str,
    *,
    collapse_whitespace: bool = False,
    strip_emoji_chars: bool = False,
) -> str:
    out = normalize_newlines(text)
    if strip_emoji_chars:
        out = strip_emoji(out)
    if collapse_whitespace:
        out = collapse_whitespace_preserve_newlines(out)
    return out


def normalize_message_line(
    line: str,
    *,
    collapse_whitespace: bool = False,
    strip_emoji_chars: bool = False,
) -> str:
    out = normalize_text(
        line,
        collapse_whitespace=collapse_whitespace,
        strip_emoji_chars=strip_emoji_chars,
    )
    # Canonical: one message per line.
    out = out.replace("\n", " ").strip()
    out = _NON_NEWLINE_WHITESPACE_RE.sub(" ", out).strip()
    return out

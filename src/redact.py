from __future__ import annotations

import re
from typing import Iterable


_EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)

# A pragmatic phone pattern: supports +country, separators, parentheses.
_PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}\b"
)

# Very approximate street address pattern.
_ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+[A-Z0-9][A-Z0-9'\-\.]+(?:\s+[A-Z0-9][A-Z0-9'\-\.]+){0,4}\s+"
    r"(?:ST|STREET|AVE|AVENUE|RD|ROAD|BLVD|BOULEVARD|DR|DRIVE|LN|LANE|WAY|CT|COURT|PL|PLACE)\b",
    re.IGNORECASE,
)


def _norm_types(types: Iterable[str] | None) -> set[str]:
    if not types:
        return {"email", "phone", "address"}
    out: set[str] = set()
    for t in types:
        t2 = str(t).strip().lower()
        if not t2:
            continue
        if t2 in {"emails", "email"}:
            out.add("email")
        elif t2 in {"phones", "phone", "tel", "telephone"}:
            out.add("phone")
        elif t2 in {"addresses", "address", "addr"}:
            out.add("address")
    return out or {"email", "phone", "address"}


def redact_text(text: str, *, types: Iterable[str] | None = None) -> str:
    """Redact common PII-like strings.

    Notes:
    - This is best-effort regex redaction; it will have false positives/negatives.
    - Designed for simple message corpora, not strict compliance.
    """
    enabled = _norm_types(types)
    out = text
    if "email" in enabled:
        out = _EMAIL_RE.sub("<EMAIL>", out)
    if "phone" in enabled:
        out = _PHONE_RE.sub("<PHONE>", out)
    if "address" in enabled:
        out = _ADDRESS_RE.sub("<ADDRESS>", out)
    return out

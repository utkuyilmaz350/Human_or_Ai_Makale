from __future__ import annotations

import re


_non_printable_re = re.compile(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]")
_whitespace_re = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    t = text or ""
    t = t.replace("\u00A0", " ")
    t = _non_printable_re.sub(" ", t)
    t = t.strip()
    t = _whitespace_re.sub(" ", t)
    return t

"""Input sanitisation: Unicode normalisation and stripping of control/invisible
characters and fake prompt delimiters. Used by the input checks and by tool-output
sanitising, so it imports nothing from the other guardrail modules."""

from __future__ import annotations

import re
import unicodedata

# ── 3. Input sanitization ────────────────────────────────────────────────────
# Strips characters and sequences that could confuse the model or exploit
# delimiter-based context boundaries.

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Zero-width / bidi-control characters used to split keywords past regex filters
# ("ig\u200bnore previous instructions").
_INVISIBLE_CHARS = re.compile(r"[\u00ad\u200b-\u200f\u2028-\u202e\u2060-\u2064\ufeff]")
_EXCESSIVE_WHITESPACE = re.compile(r"[ \t]{10,}")
_EXCESSIVE_NEWLINES = re.compile(r"\n{4,}")

# Fake XML/markdown delimiters that mimic system boundaries
_FAKE_DELIMITERS = re.compile(
    r"<\/?(?:system|admin|root|instruction|prompt|context|tool_call|function_call)>",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """NFKC-fold look-alike characters (full-width letters, ligatures) and drop
    invisible ones, so pattern checks see what the model will effectively read."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _INVISIBLE_CHARS.sub("", text)
    return _CONTROL_CHARS.sub("", text)


def sanitize_input(text: str, max_length: int = 2000) -> str:
    """Clean user input: normalise Unicode, strip control/invisible chars and fake
    delimiters, collapse whitespace, enforce length."""
    text = normalize_text(text)
    text = _FAKE_DELIMITERS.sub("", text)
    text = _EXCESSIVE_WHITESPACE.sub(" ", text)
    text = _EXCESSIVE_NEWLINES.sub("\n\n\n", text)
    text = text.strip()
    if len(text) > max_length:
        text = text[:max_length]
    return text

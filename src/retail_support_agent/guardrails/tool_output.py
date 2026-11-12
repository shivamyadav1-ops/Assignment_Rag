"""Tool-output guardrails: strip indirect prompt injection and mask PII before a
tool result goes back to the model."""

from __future__ import annotations

import re
from typing import Any

from .injection import check_prompt_injection
from .sanitize import normalize_text

# ══════════════════════════════════════════════════════════════════════════════
# 3. Tool-output guardrails — run before a tool result goes back to the model
# ══════════════════════════════════════════════════════════════════════════════
# Retrieved policy text and free-text record fields (e.g. a return reason typed by
# a customer) are data, but the model reads them in-context. Instruction-like text
# inside them is indirect prompt injection; neutralise it before the model sees it.

_EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CUSTOMER_ID_RE = re.compile(r"\bCUST-\d{3,}\b", re.IGNORECASE)
_ORDER_REF_RE = re.compile(r"\bORD-\d{3,}\b", re.IGNORECASE)
INJECTION_PLACEHOLDER = "[removed: text resembling instructions to the assistant]"


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


# Published company contact addresses are meant to be shared with customers.
PUBLIC_EMAIL_DOMAINS = {"shopsmartelectronics.com", "shopsmart.com"}


def _mask_email(m: re.Match) -> str:
    local, domain = m.group(1), m.group(2)
    if domain.lower() in PUBLIC_EMAIL_DOMAINS:
        return m.group(0)
    shown = local[:2] if len(local) > 2 else local[:1]
    return f"{shown}{'*' * max(len(local) - len(shown), 1)}@{domain}"


def _mask_card(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    if 13 <= len(digits) <= 19 and _luhn_ok(digits):
        return f"[card ending {digits[-4:]}]"
    return m.group(0)


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Mask emails, Luhn-valid card numbers and SSN-shaped numbers. Returns the new
    text and the kinds of PII redacted."""
    kinds: list[str] = []
    new = _EMAIL_RE.sub(_mask_email, text)
    if new != text:
        kinds.append("email")
    card = _CARD_RE.sub(_mask_card, new)
    if card != new:
        kinds.append("card_number")
    ssn = _SSN_RE.sub("[redacted]", card)
    if ssn != card:
        kinds.append("ssn")
    return ssn, kinds


def sanitize_tool_output(result: Any) -> tuple[Any, list[str]]:
    """Recursively neutralise injected instructions and mask PII in a tool result.
    Returns (sanitised copy, findings) — findings are short labels for the trace."""
    findings: list[str] = []

    def walk(node: Any, path: str) -> Any:
        if isinstance(node, dict):
            return {k: walk(v, f"{path}.{k}") for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v, f"{path}[{i}]") for i, v in enumerate(node)]
        if isinstance(node, str):
            norm = normalize_text(node)
            if not check_prompt_injection(norm).passed:
                findings.append(f"injection@{path}")
                return INJECTION_PLACEHOLDER
            redacted, kinds = redact_pii(node)
            findings.extend(f"pii:{k}@{path}" for k in kinds)
            return redacted
        return node

    return walk(result, "$"), findings

"""Pre-input toxicity: regex over a de-obfuscated copy, then the toxic-bert classifier."""

from __future__ import annotations

import re
import time

from ._base import BlockReason, GuardrailResult
from .sanitize import normalize_text

# ── 1. Profanity / Toxicity blocklist ────────────────────────────────────────
# Fast first-pass keyword check. Catches obvious slurs and abuse before the
# heavier classifier runs. All lowercase for case-insensitive matching.

# Severe slurs: a single match is enough to block.
_SEVERE_SLUR_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b" + p + r"\b", re.IGNORECASE)
    for p in [
        r"n[i1]gg(?:a|er|az|ers?)",
        r"f[a4]gg?(?:ot|it)",
        r"k[i1]ke",
        r"sp[i1]c",
        r"ch[i1]nk",
    ]
]

# General profanity: 2+ matches trigger a block (single mild word is tolerated).
_TOXIC_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b" + p + r"\b", re.IGNORECASE)
    for p in [
        r"fuck(?:ing|ed|er|s|off)?",
        r"shit(?:ty|head|s)?",
        r"bitch(?:es)?",
        r"asshole",
        r"bastard",
        r"dick(?:head)?",
        r"cunt",
        r"damn(?:it)?",
        r"stfu",
        r"wtf",
        r"kill\s+(?:you|your)self",
        r"die\s+(?:you|bitch)",
        r"go\s+to\s+hell",
        r"i(?:'ll|.will)\s+(?:kill|hurt|destroy)\s+you",
        r"piece\s+of\s+(?:shit|crap)",
        r"r[e3]tard(?:ed)?",
        r"kill\s*(?:your|ur)\s*self",
        r"kys\b",
    ]
]

_THREAT_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(?:i'?(?:ll|m\s+gonna?|will))\s+(?:bomb|shoot|attack|blow\s+up|burn\s+down)\b",
        r"\b(?:hack|ddos|dox|swat)\s+(?:you|this|your|the)\b",
        r"\bkill\s+(?:you|your|everyone|myself|yourself|urself)\b",
        r"\b(?:bomb|shoot\s+up|blow\s+up|burn\s+down)\s+(?:your|the|this)\b",
        r"\bdie\s+(?:you|bitch)\b",
    ]
]


# De-obfuscation for the toxicity regex: leetspeak, symbol substitution, spaced or
# dotted letters ("f u c k", "sh1t", "fvck", "b!tch", "f.u.c.k").
_LEET = str.maketrans({"0": "o", "1": "i", "!": "i", "|": "i", "3": "e", "4": "a",
                       "@": "a", "5": "s", "$": "s", "7": "t", "+": "t", "*": "u"})
_SPACED_LETTERS = re.compile(r"\b(?:[a-z][\s.\-_]){2,}[a-z]\b", re.IGNORECASE)
_V_FOR_U = re.compile(r"\bf[v]ck", re.IGNORECASE)


def deobfuscate(text: str) -> str:
    """Undo common filter-evasion tricks. Used only for toxicity matching — never
    shown to the model — so over-normalising costs nothing but a regex miss."""
    t = normalize_text(text).lower()
    t = _SPACED_LETTERS.sub(lambda m: re.sub(r"[\s.\-_]", "", m.group(0)), t)
    # Only translate symbols/digits that sit inside words, so "ORD-10012" or "$19.99"
    # stay intact.
    t = re.sub(r"(?<=[a-z])[0-9!|@$*+]+(?=[a-z])|(?<=[a-z])[0-9!|@$*+]+\b|\b[0-9!|@$*+]+(?=[a-z])",
               lambda m: m.group(0).translate(_LEET), t)
    t = _V_FOR_U.sub("fuck", t)
    return re.sub(r"(.)\1{2,}", r"\1", t)        # "fuuuuck" -> "fuck"


# Abuse aimed at a person: one match is enough (unlike general profanity, which
# customers use about products — "this damn charger").
_INSULT = (r"(?:bitch|asshole|cunt|idiot|moron|retard(?:ed)?|dickhead|bastard|"
           r"piece\s+of\s+(?:shit|crap)|loser|scum)")
_DIRECTED_ABUSE: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bfuck\s*(?:you|u|off|yourself|urself)\b",
        r"\b(?:you|u|ur|you'?re|youre)\s+(?:a\s+|an\s+)?(?:\w+\s+){0,2}" + _INSULT + r"\b",
        r"\b(?:dumb|stupid|useless|worthless|fucking|f\w*cking)\s+" + _INSULT + r"\b",
        r"\bcunt\b",
        r"\bkys\b",
        r"\b(?:eat\s+shit|(?:and|go|just)\s+die)\b",
    ]
]


def check_toxicity(text: str) -> GuardrailResult:
    """Keyword + pattern check for obvious toxic/threatening content, on the original
    text and a de-obfuscated copy."""
    t0 = time.perf_counter()
    plain = deobfuscate(text)
    if plain != text.lower():
        text = f"{text}\n{plain}"

    for pat in _THREAT_PATTERNS:
        if pat.search(text):
            return GuardrailResult(
                passed=False,
                reason=BlockReason.TOXIC,
                detail="Threatening language detected.",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

    for pat in _DIRECTED_ABUSE:
        if pat.search(text):
            return GuardrailResult(
                passed=False,
                reason=BlockReason.TOXIC,
                detail="Abusive language directed at a person.",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

    for pat in _SEVERE_SLUR_PATTERNS:
        if pat.search(text):
            return GuardrailResult(
                passed=False,
                reason=BlockReason.TOXIC,
                detail="Hate speech detected.",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

    matches = [pat.pattern for pat in _TOXIC_PATTERNS if pat.search(text)]
    if len(matches) >= 2:
        return GuardrailResult(
            passed=False,
            reason=BlockReason.TOXIC,
            detail="Multiple profanity matches detected.",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    return GuardrailResult(passed=True, latency_ms=(time.perf_counter() - t0) * 1000)


# Customer frustration ("my stupid laptop", "your app sucks") is not abuse; toxic-bert
# scores it as `toxic`/`obscene`. Block only threats, hate, severe toxicity and
# strong insults aimed at someone.
ML_BLOCK_LABELS = {"threat": 0.5, "severe_toxic": 0.5, "identity_hate": 0.5, "insult": 0.85}


def check_toxicity_ml(text: str) -> GuardrailResult:
    """toxic-bert classifier pass. Fails open (passes) if the model is unavailable."""
    t0 = time.perf_counter()
    from .. import classifiers

    if not classifiers.ml_guardrails_enabled():
        return GuardrailResult(passed=True, detail="ml disabled")
    try:
        scores = classifiers.toxicity([text])[0]
    except Exception as exc:  # noqa: BLE001 - regex layer already ran
        return GuardrailResult(passed=True, detail=f"classifier unavailable: {exc}",
                               latency_ms=(time.perf_counter() - t0) * 1000)
    hits = {k: round(scores.get(k, 0.0), 3) for k, thr in ML_BLOCK_LABELS.items()
            if scores.get(k, 0.0) >= thr}
    ms = (time.perf_counter() - t0) * 1000
    if hits:
        return GuardrailResult(passed=False, reason=BlockReason.TOXIC,
                               detail=f"Classifier flagged: {hits}", latency_ms=ms)
    return GuardrailResult(passed=True, detail="classifier", latency_ms=ms)

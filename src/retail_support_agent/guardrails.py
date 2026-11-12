"""Guardrails for the support agent, applied at four points in the loop.

  1. Pre-input   (`run_pre_input_checks`)   — before the user message reaches the LLM:
     rate limit, Unicode normalisation + sanitisation, toxicity (regex over a
     de-obfuscated copy, then a toxic-bert classifier), prompt injection (regex, then
     a DeBERTa injection classifier), off-topic requests (embedding similarity to
     support vs. off-topic prototypes, topic.py). Images: `check_image_nsfw`.
  2. Tool call   (`check_tool_call`)        — before a model-requested tool runs:
     tool allowlist, argument validation, per-session call budgets, and a
     "look before you write" rule for create_return_request.
  3. Tool output (`sanitize_tool_output`)   — before a tool result goes back to the
     model: strips indirect prompt injection hidden in retrieved documents or
     record fields, and masks PII the model does not need.
  4. Output      (`check_output`)           — before the answer reaches the customer:
     (a) blocks system-prompt leakage (shingle overlap);
     (b) blocks toxic / abusive output (regex + ML with stricter thresholds than input);
     (c) blocks off-scope content (medical/legal/financial advice, self-harm, extremism);
     (d) redacts PII (email, card, SSN) and other-customer IDs;
     (e) warns on order IDs not seen in question or tool traffic (possible hallucination);
     (f) warns on specific quantities in the answer absent from tool context;
     (g) warns when the answer asserts policy rules but no policy chunks were retrieved.
     Live NLI grounding (faithfulness / contradiction) runs separately in `grounding.py`.

The classifier layers (`classifiers.py`) run only after the cheap regex layers pass,
and fail open (regex still applies) if a model cannot load; set RSA_ML_GUARDRAILS=0
to switch them off.

The regex checks are deterministic and dependency-free, so every path is
unit-testable without a model. They complement — not replace — the hard rules enforced server-side
(identity pinned on the MCP server, eligibility enforced in `ops`).
"""

from __future__ import annotations

import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BlockReason(Enum):
    TOXIC = "toxic_content"
    PROMPT_INJECTION = "prompt_injection"
    RATE_LIMITED = "rate_limited"
    EMPTY_INPUT = "empty_input"
    TOOL_NOT_ALLOWED = "tool_not_allowed"
    INVALID_TOOL_ARGS = "invalid_tool_args"
    TOOL_BUDGET_EXCEEDED = "tool_budget_exceeded"
    PRECONDITION_FAILED = "precondition_failed"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    TOXIC_OUTPUT = "toxic_output"
    OFF_TOPIC = "off_topic"
    NSFW = "nsfw_content"
    OFF_SCOPE_CONTENT = "off_scope_content"
    FABRICATED_CLAIM = "fabricated_claim"


@dataclass
class GuardrailResult:
    passed: bool
    reason: BlockReason | None = None
    detail: str = ""
    latency_ms: float = 0.0


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
    from . import classifiers

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


# ── 2. Prompt injection detection ────────────────────────────────────────────
# Heuristic patterns that catch common injection attempts. Not a silver bullet,
# but blocks the easy attacks before they reach the model.

_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Direct instruction override attempts
        r"ignore\s+(?:(?:all|the|your|previous|prior|above|earlier|original)\s+){1,3}(?:instructions?|prompts?|rules?|directions?|guidelines?)",
        r"disregard\s+(?:(?:all|the|your|previous|prior|above|earlier|original)\s+){1,3}(?:instructions?|prompts?|rules?|guidelines?)",
        r"forget\s+(?:(?:all|the|your|previous|prior|above|earlier|original)\s+){1,3}(?:instructions?|prompts?|rules?|guidelines?)",
        r"override\s+(?:your\s+)?(?:instructions?|system\s+prompt|rules?|guidelines?)",

        # Role-play / persona hijacking
        r"you\s+are\s+now\s+(?:an?\s+)?(?:different|new|my|evil|unrestricted)",
        r"act\s+as\s+(?:if\s+you\s+(?:are|were)\s+)?(?:an?\s+)?(?:different|new|evil|unrestricted)",
        r"pretend\s+(?:you\s+are|to\s+be)\s+(?:an?\s+)?(?:different|evil|unrestricted)",
        r"switch\s+to\s+(?:a\s+)?(?:different|new|evil|unrestricted)\s+(?:mode|persona|role)",
        r"enter\s+(?:developer|admin|debug|god|sudo|unrestricted)\s+mode",
        r"jailbreak",
        r"DAN\s+mode",

        # System prompt extraction
        r"(?:show|tell|reveal|print|output|repeat|display)\s+(?:me\s+)?(?:your\s+)?(?:system\s+prompt|instructions?|initial\s+prompt|hidden\s+prompt)",
        r"what\s+(?:is|are)\s+your\s+(?:system\s+)?(?:prompt|instructions?|rules?|guidelines?)\s*\??",

        # Delimiter / context manipulation
        r"<\/?(?:system|admin|root|instruction|prompt)>",
        r"\[(?:SYSTEM|ADMIN|INST)\]",
        r"```\s*system",
        r"###\s*(?:SYSTEM|INSTRUCTION|ADMIN)",

        # Tool abuse attempts
        r"call\s+(?:the\s+)?(?:tool|function|api)\s+(?:to\s+)?(?:delete|drop|remove|destroy)",
        r"execute\s+(?:the\s+)?(?:following|this)\s+(?:code|command|script)",
    ]
]

# Some words alone are suspicious — if several appear together, likely injection.
_INJECTION_SIGNALS = [
    "system prompt", "ignore instructions", "new instructions",
    "you are now", "act as", "pretend", "override", "bypass",
    "unrestricted", "no limitations", "developer mode", "admin mode",
]


def check_prompt_injection(text: str) -> GuardrailResult:
    """Heuristic detection of prompt injection attempts."""
    t0 = time.perf_counter()

    for pat in _INJECTION_PATTERNS:
        if pat.search(text):
            return GuardrailResult(
                passed=False,
                reason=BlockReason.PROMPT_INJECTION,
                detail="Input resembles a prompt injection attempt.",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

    low = text.lower()
    signal_count = sum(1 for sig in _INJECTION_SIGNALS if sig in low)
    if signal_count >= 2:
        return GuardrailResult(
            passed=False,
            reason=BlockReason.PROMPT_INJECTION,
            detail="Multiple prompt-injection signals detected.",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    return GuardrailResult(passed=True, latency_ms=(time.perf_counter() - t0) * 1000)


# ── Conversation recall ───────────────────────────────────────────────────────
# "What did I ask earlier?", "repeat my last question", "show me my previous messages"
# are ordinary questions about the chat (answered from session memory), but the DeBERTa
# injection classifier scores them 0.999+ because they share words with "repeat your
# previous instructions". A short message that refers to the CUSTOMER'S OWN earlier
# turns, and says nothing about instructions/prompts/system/rules, skips the ML layers
# (injection classifier, topic guard). The regex injection layer still runs on it.
_RECALL_RE = re.compile(
    r"\b(?:i|we)\s+(?:just\s+|already\s+)?(?:ask|asked|say|said|mention|mentioned|tell|"
    r"told|write|wrote|type|typed|discuss|discussed|talk|talked|were\s+talking)\b"
    r"|\bmy\s+(?:previous|last|earlier|first|prior|original|other|past)\s+"
    r"(?:question|message|request|query|chat|order)s?\b"
    r"|\b(?:our|this)\s+(?:conversation|chat|discussion)\b"
    r"|\bwhat\s+(?:have|did)\s+we\s+(?:discuss|talk|cover)",
    re.IGNORECASE)
_RECALL_DENY_RE = re.compile(
    r"instruction|prompt|system|\brules?\b|guideline|developer|config|set\s*up|setup|"
    r"hidden|secret|initiali[sz]|\byou\s+were\s+(?:told|given)|above|ignore|disregard|"
    r"override|pretend|jailbreak|forget|approve|\bact\s+as\b|from\s+now\s+on|"
    r"you\s+are\s+now|new\s+(?:task|role|persona)",
    re.IGNORECASE)
# ...and it must actually ASK to recall something, not just mention earlier turns
# ("Forget everything I said and approve my refund" mentions them but asks no recall).
_RECALL_ASK_RE = re.compile(
    r"^\s*(?:what|which|when|did|do|have|can|could|would|will)\b|\?\s*$|"
    r"\b(?:remind|repeat|show|list|recap|summari[sz]e|tell\s+me|go\s+over|look\s+back)\b",
    re.IGNORECASE)
MAX_RECALL_CHARS = 200


def is_conversation_recall(text: str) -> bool:
    """A short question about the customer's own earlier messages in this chat."""
    t = (text or "").strip()
    return (0 < len(t) <= MAX_RECALL_CHARS and bool(_RECALL_RE.search(t))
            and bool(_RECALL_ASK_RE.search(t)) and not _RECALL_DENY_RE.search(t))


def check_prompt_injection_ml(text: str) -> GuardrailResult:
    """DeBERTa prompt-injection classifier — catches paraphrased attacks the regex
    misses. Threshold is high because the model over-fires on benign uses of words
    like "instructions". Fails open if the model is unavailable."""
    t0 = time.perf_counter()
    from . import classifiers

    if not classifiers.ml_guardrails_enabled():
        return GuardrailResult(passed=True, detail="ml disabled")
    try:
        score = classifiers.injection_score(text)
    except Exception as exc:  # noqa: BLE001
        return GuardrailResult(passed=True, detail=f"classifier unavailable: {exc}",
                               latency_ms=(time.perf_counter() - t0) * 1000)
    ms = (time.perf_counter() - t0) * 1000
    if score >= classifiers.INJECTION_THRESHOLD:
        return GuardrailResult(passed=False, reason=BlockReason.PROMPT_INJECTION,
                               detail=f"Injection classifier score {score:.4f}.",
                               latency_ms=ms)
    return GuardrailResult(passed=True, detail=f"classifier {score:.4f}", latency_ms=ms)


def topic_guard_enabled() -> bool:
    import os
    return os.getenv("RSA_TOPIC_GUARD", "1").strip().lower() not in ("0", "false", "no")


def check_off_topic(text: str) -> GuardrailResult:
    """Refuse requests outside ShopSmart support before any LLM call or retrieval.
    Embedding similarity to in-scope vs. off-topic prototypes (topic.py), ~20 ms.
    Fails open: if the embedder is unavailable the agent's system prompt still declines."""
    t0 = time.perf_counter()
    try:
        from .topic import is_off_topic
        off, info = is_off_topic(text)
    except Exception as exc:  # noqa: BLE001
        return GuardrailResult(passed=True, detail=f"topic classifier unavailable: {exc}",
                               latency_ms=(time.perf_counter() - t0) * 1000)
    ms = (time.perf_counter() - t0) * 1000
    if "score" not in info:
        return GuardrailResult(passed=True, detail=info.get("reason", ""), latency_ms=ms)
    detail = (f"topic score {info['score']:+.3f} (nearest off-topic: "
              f"'{info['nearest_off']}'; nearest in-scope: '{info['nearest_in']}')")
    if off:
        return GuardrailResult(passed=False, reason=BlockReason.OFF_TOPIC, detail=detail,
                               latency_ms=ms)
    return GuardrailResult(passed=True, detail=detail, latency_ms=ms)


def check_image_nsfw(image) -> GuardrailResult:
    """NSFW image classifier (Falconsai/nsfw_image_detection) for uploaded images.
    Accepts a PIL image, bytes or a path. Fails closed: an image that cannot be
    classified is rejected, since images are optional to the support flow."""
    t0 = time.perf_counter()
    from . import classifiers

    try:
        score = classifiers.nsfw_image_score(image)
    except Exception as exc:  # noqa: BLE001
        return GuardrailResult(passed=False, reason=BlockReason.NSFW,
                               detail=f"Image could not be checked: {exc}",
                               latency_ms=(time.perf_counter() - t0) * 1000)
    ms = (time.perf_counter() - t0) * 1000
    if score >= classifiers.NSFW_THRESHOLD:
        return GuardrailResult(passed=False, reason=BlockReason.NSFW,
                               detail=f"NSFW image score {score:.3f}.", latency_ms=ms)
    return GuardrailResult(passed=True, detail=f"nsfw {score:.3f}", latency_ms=ms)


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


# ── 4. Rate limiter ──────────────────────────────────────────────────────────
# Simple in-memory sliding-window rate limiter per requester.

_request_log: dict[str, list[float]] = {}
# Streamlit serves sessions from multiple threads; the read-modify-write below
# must be atomic or concurrent requests from one requester undercount.
_request_lock = threading.Lock()
# Requesters tracked before idle ones (nothing inside the window) are swept, so a
# long-running process does not keep one entry per requester ever seen.
_PRUNE_AT = 1024


def check_rate_limit(
    requester_id: str,
    max_requests: int = 10,
    window_seconds: float = 60.0,
) -> GuardrailResult:
    """Sliding window rate limit. Returns failed result if limit exceeded."""
    t0 = time.perf_counter()
    now = time.monotonic()
    cutoff = now - window_seconds

    with _request_lock:
        log = [t for t in _request_log.get(requester_id, []) if t > cutoff]
        log.append(now)
        _request_log[requester_id] = log
        if len(_request_log) > _PRUNE_AT:
            for rid in [r for r, ts in _request_log.items() if ts[-1] <= cutoff]:
                del _request_log[rid]
        count = len(log)

    if count > max_requests:
        return GuardrailResult(
            passed=False,
            reason=BlockReason.RATE_LIMITED,
            detail=f"Rate limit exceeded: {max_requests} requests per {window_seconds:.0f}s.",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    return GuardrailResult(passed=True, latency_ms=(time.perf_counter() - t0) * 1000)


# ── Combined pre-input gate ──────────────────────────────────────────────────

_BLOCK_MESSAGES = {
    BlockReason.TOXIC: (
        "I'm not able to process messages with threatening or abusive language. "
        "Please rephrase your question and I'll be happy to help."
    ),
    BlockReason.PROMPT_INJECTION: (
        "I can only help with ShopSmart Electronics support questions — "
        "product info, shipping, orders, returns and refunds. "
        "Could you rephrase your question?"
    ),
    BlockReason.RATE_LIMITED: (
        "You're sending messages too quickly. Please wait a moment and try again."
    ),
    BlockReason.NSFW: (
        "I can't accept that content. If you need help with a damaged or faulty item, "
        "please describe the problem or attach a photo of the product."
    ),
    BlockReason.OFF_TOPIC: (
        "Sorry, I can only help with ShopSmart Electronics support: products, orders, "
        "shipping, returns, refunds, warranties and your account. What can I help you "
        "with there?"
    ),
    BlockReason.EMPTY_INPUT: (
        "I didn't receive a question. How can I help with your order, shipping, "
        "returns or refunds?"
    ),
}


@dataclass
class PreInputCheckResult:
    passed: bool
    sanitized_input: str
    block_message: str = ""
    checks: list[GuardrailResult] = None  # type: ignore[assignment]
    total_latency_ms: float = 0.0

    def __post_init__(self):
        if self.checks is None:
            self.checks = []


def run_pre_input_checks(
    text: str,
    requester_id: str,
    use_ml: bool | None = None,
) -> PreInputCheckResult:
    """Run all pre-input guardrails. Returns sanitized text if passed, or a block message.

    `use_ml` forces the classifier layers on/off; None follows RSA_ML_GUARDRAILS."""
    if use_ml is None:
        from .classifiers import ml_guardrails_enabled
        use_ml = ml_guardrails_enabled()
    checks: list[GuardrailResult] = []

    rate_check = check_rate_limit(requester_id)
    checks.append(rate_check)
    if not rate_check.passed:
        return PreInputCheckResult(
            passed=False,
            sanitized_input=text,
            block_message=_BLOCK_MESSAGES[rate_check.reason],
            checks=checks,
            total_latency_ms=sum(c.latency_ms for c in checks),
        )

    cleaned = sanitize_input(text)
    if not cleaned:
        empty = GuardrailResult(passed=False, reason=BlockReason.EMPTY_INPUT,
                                detail="Input is empty after sanitisation.")
        checks.append(empty)
        return PreInputCheckResult(
            passed=False, sanitized_input=cleaned,
            block_message=_BLOCK_MESSAGES[BlockReason.EMPTY_INPUT], checks=checks,
            total_latency_ms=sum(c.latency_ms for c in checks),
        )

    toxicity_check = check_toxicity(cleaned)
    if toxicity_check.passed and use_ml:
        checks.append(toxicity_check)
        toxicity_check = check_toxicity_ml(cleaned)
    checks.append(toxicity_check)
    if not toxicity_check.passed:
        return PreInputCheckResult(
            passed=False,
            sanitized_input=cleaned,
            block_message=_BLOCK_MESSAGES[toxicity_check.reason],
            checks=checks,
            total_latency_ms=sum(c.latency_ms for c in checks),
        )

    # Run injection detection on the normalised-but-unstripped text too: sanitisation
    # removes fake <system> delimiters, which are themselves an injection signal.
    injection_check = check_prompt_injection(cleaned)
    if injection_check.passed:
        injection_check = check_prompt_injection(normalize_text(text)[:5000])
    recall = is_conversation_recall(cleaned)
    if injection_check.passed and use_ml and not recall:
        checks.append(injection_check)
        injection_check = check_prompt_injection_ml(cleaned)
    checks.append(injection_check)
    if not injection_check.passed:
        return PreInputCheckResult(
            passed=False,
            sanitized_input=cleaned,
            block_message=_BLOCK_MESSAGES[injection_check.reason],
            checks=checks,
            total_latency_ms=sum(c.latency_ms for c in checks),
        )

    if topic_guard_enabled() and not recall:
        topic_check = check_off_topic(cleaned)
        checks.append(topic_check)
        if not topic_check.passed:
            return PreInputCheckResult(
                passed=False,
                sanitized_input=cleaned,
                block_message=_BLOCK_MESSAGES[BlockReason.OFF_TOPIC],
                checks=checks,
                total_latency_ms=sum(c.latency_ms for c in checks),
            )

    return PreInputCheckResult(
        passed=True,
        sanitized_input=cleaned,
        checks=checks,
        total_latency_ms=sum(c.latency_ms for c in checks),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. Tool-call guardrails — run before a model-requested tool executes
# ══════════════════════════════════════════════════════════════════════════════
# The model chooses tools and arguments, so both are untrusted. Server-side checks
# (identity, eligibility) remain the source of truth; these add cheap, early limits
# so a confused or manipulated model cannot loop on tools, call tools that do not
# exist, or write (create a return) without first reading the order.

ORDER_ID_RE = re.compile(r"^ORD-\d{3,}$")

# Max calls of each tool per session. Writes are capped hardest.
TOOL_CALL_BUDGET: dict[str, int] = {
    "search_policies": 5,
    "get_order_details": 4,
    "get_return_refund_status": 4,
    "create_return_request": 1,
    "create_escalation_ticket": 1,
}
DEFAULT_TOOL_BUDGET = 3
MAX_QUERY_CHARS = 500
MAX_REASON_CHARS = 300

# Tools that change state and therefore need the order to have been read first.
WRITE_TOOLS = {"create_return_request"}


@dataclass
class ToolGuardState:
    """Per-session state the tool-call guardrail needs across loop iterations."""
    calls: dict[str, int] = field(default_factory=dict)
    verified_orders: set[str] = field(default_factory=set)   # read OK this session
    # order id -> eligibility code, for orders whose lookup said "not returnable now"
    ineligible_orders: dict[str, str] = field(default_factory=dict)
    # Did the customer ask to START a return (not just ask whether they can)?
    # None = not evaluated (direct callers/tests): the write-intent rule is skipped.
    write_intent: bool | None = None


# A request to act: "start/open/create/initiate/process a return", "I want to
# return", "please return it", "return it". Questions about eligibility ("can I
# return", "is it returnable", "how do I return") do not match.
_RETURN_ACTION_RE = re.compile(
    r"\b(?:start|open|create|initiate|begin|file|submit|process|raise|lodge|set up|go ahead with)"
    r"\b[\w\s,'-]{0,30}\breturn"
    r"|\b(?:i\s*(?:'d| would)? ?(?:want|wish|need|like) to|i'm going to|please|pls|kindly)"
    r"\s+(?:\w+\s+){0,2}return\b"
    r"|\breturn (?:it|this|them|the item|my order)\s*(?:please|now|[.!]|$)",
    re.IGNORECASE,
)
_AFFIRMATIVE_RE = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|sure|ok(?:ay)?|please do|go ahead|do it|confirm(?:ed)?|"
    r"sounds good|that'?s fine|please)\b", re.IGNORECASE)
_OFFER_RE = re.compile(r"\b(?:start|create|open|initiate|process)\b[^.?!]{0,40}\breturn\b"
                       r"[^.?!]*\?", re.IGNORECASE)


def detect_return_intent(question: str, history: list[dict] | None = None) -> bool:
    """True when this turn asks the assistant to start a return.

    Either the message itself is a request to act, or it is a short "yes" to the
    assistant's immediately preceding offer to start one. Asking whether a return is
    possible is not consent to create one: a return is a state change the customer
    must ask for.
    """
    if _RETURN_ACTION_RE.search(question or ""):
        return True
    last_assistant = next((m.get("content", "") for m in reversed(history or [])
                           if m.get("role") == "assistant"), "")
    return bool(_AFFIRMATIVE_RE.match(question or "")
                and isinstance(last_assistant, str) and _OFFER_RE.search(last_assistant))


def _block(reason: BlockReason, detail: str, t0: float) -> GuardrailResult:
    return GuardrailResult(passed=False, reason=reason, detail=detail,
                           latency_ms=(time.perf_counter() - t0) * 1000)


def check_tool_call(
    name: str,
    args: dict,
    tool_schemas: dict[str, dict],
    state: ToolGuardState,
) -> GuardrailResult:
    """Validate one tool call against the allowlist, its schema, budgets and
    preconditions. Does not mutate `state`; call `record_tool_call` after running."""
    t0 = time.perf_counter()

    schema = tool_schemas.get(name)
    if schema is None:
        return _block(BlockReason.TOOL_NOT_ALLOWED, f"Unknown tool '{name}'.", t0)

    if not isinstance(args, dict):
        return _block(BlockReason.INVALID_TOOL_ARGS, "Tool arguments must be an object.", t0)

    props = (schema.get("properties") or {})
    unknown = sorted(set(args) - set(props))
    if props and unknown:
        return _block(BlockReason.INVALID_TOOL_ARGS,
                      f"Unexpected argument(s): {', '.join(unknown)}.", t0)
    missing = [k for k in schema.get("required", []) if args.get(k) in (None, "")]
    if missing:
        return _block(BlockReason.INVALID_TOOL_ARGS,
                      f"Missing required argument(s): {', '.join(missing)}.", t0)

    budget = TOOL_CALL_BUDGET.get(name, DEFAULT_TOOL_BUDGET)
    if state.calls.get(name, 0) >= budget:
        return _block(BlockReason.TOOL_BUDGET_EXCEEDED,
                      f"'{name}' may be called at most {budget} time(s) per conversation.", t0)

    if "order_id" in args:
        oid = str(args["order_id"]).strip().upper()
        if not ORDER_ID_RE.match(oid):
            return _block(BlockReason.INVALID_TOOL_ARGS,
                          f"'{args['order_id']}' is not a valid order ID (expected ORD-#####).",
                          t0)

    if name == "search_policies":
        q = str(args.get("query", ""))
        if len(q) > MAX_QUERY_CHARS:
            return _block(BlockReason.INVALID_TOOL_ARGS,
                          f"Search query longer than {MAX_QUERY_CHARS} characters.", t0)

    if name in WRITE_TOOLS:
        oid = str(args.get("order_id", "")).strip().upper()
        if state.write_intent is False:
            return _block(BlockReason.PRECONDITION_FAILED,
                          "The customer has not asked to start a return. Answer their "
                          "question (eligibility, window, deadline) and offer to start "
                          "one; create it only after they ask.", t0)
        if oid not in state.verified_orders:
            return _block(BlockReason.PRECONDITION_FAILED,
                          f"Call get_order_details for {oid} and confirm eligibility "
                          "before creating a return.", t0)
        if oid in state.ineligible_orders:
            return _block(BlockReason.PRECONDITION_FAILED,
                          f"get_order_details already reported {oid} as not returnable "
                          f"now ({state.ineligible_orders[oid]}). Relay that to the "
                          "customer instead of creating a return.", t0)
        reason = str(args.get("reason", ""))
        if len(reason) > MAX_REASON_CHARS:
            return _block(BlockReason.INVALID_TOOL_ARGS,
                          f"Return reason longer than {MAX_REASON_CHARS} characters.", t0)
        if not check_prompt_injection(normalize_text(reason)).passed:
            return _block(BlockReason.INVALID_TOOL_ARGS,
                          "Return reason contains instruction-like content.", t0)

    return GuardrailResult(passed=True, latency_ms=(time.perf_counter() - t0) * 1000)


def record_tool_call(state: ToolGuardState, name: str, args: dict, result: dict) -> None:
    """Update budgets and preconditions after a tool actually ran."""
    state.calls[name] = state.calls.get(name, 0) + 1
    if name in ("get_order_details", "get_return_refund_status") and result.get("ok"):
        data = result.get("data") or {}
        oid = str(data.get("order_id") or args.get("order_id", "")).strip().upper()
        state.verified_orders.add(oid)
        elig = data.get("return_eligibility")
        if isinstance(elig, dict) and elig.get("eligible_now") is False:
            state.ineligible_orders[oid] = str(elig.get("code") or "NOT_ELIGIBLE")


def tool_block_envelope(check: GuardrailResult) -> dict:
    """The error the model sees in place of a blocked tool result (same shape as ops)."""
    return {"ok": False, "error": {
        "code": "GUARDRAIL_BLOCKED",
        "message": f"{check.reason.value if check.reason else 'blocked'}: {check.detail}",
        "retryable": False}}


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


# ══════════════════════════════════════════════════════════════════════════════
# 4. Output guardrails — run on the final answer before the customer sees it
# ══════════════════════════════════════════════════════════════════════════════

# ── 4a. Off-scope / harmful content ──────────────────────────────────────────
# Topics a retail support agent should never address. One match blocks the
# answer regardless of how it arrived there.

_OFF_SCOPE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Medical advice
        r"\b(?:consult|see|visit|call)\s+(?:a|your)\s+(?:doctor|physician|nurse|pharmacist)\b",
        r"\bseek\s+(?:medical|emergency)\s+(?:advice|help|attention|care)\b",
        r"\btake\s+(?:this\s+)?(?:medication|drug|pill|supplement)\b",
        r"\b(?:diagnosis|diagnose|symptom|prescription)\b",
        # Legal advice
        r"\b(?:consult|hire|contact|recommend\s+consulting)\s+(?:a|an)\s+(?:lawyer|attorney|solicitor)\b",
        r"\b(?:consulting|speak(?:ing)?\s+(?:to|with))\s+(?:a|an)\s+(?:lawyer|attorney|solicitor)\b",
        r"\byou\s+(?:have|may\s+have)\s+(?:a\s+)?(?:legal\s+right|right\s+to\s+sue|case\s+against)\b",
        r"\bfile\s+a\s+(?:lawsuit|suit|legal\s+complaint)\b",
        # Financial / investment advice
        r"\b(?:invest|investing|investment)\s+in\s+(?:stocks?|crypto|bitcoin|ethereum|nft)\b",
        r"\bstock\s+(?:market|tip|pick|recommendation)\b",
        # Self-harm / crisis content
        r"\b(?:suicide|suicidal|self[\s\-]harm|hurt\s+(?:yourself|myself)|end\s+(?:your|my)\s+life)\b",
        r"\b(?:crisis|hotline|988|1-800-273)\b",
        # Violence / extremism
        r"\b(?:bomb|explosive|weapon|gun|firearm)\s*[-–]?\s*(?:making|building|instructions?|how\s+to|recipe)\b",
        r"\b(?:instructions?|how\s+to\s+(?:make|build|create))\s+(?:a\s+)?(?:bomb|explosive|weapon)\b",
        r"\b(?:terrorist|terrorism|extremist)\s+(?:attack|act|group)\b",
    ]
]

_OFF_SCOPE_BLOCK_MSG = (
    "I'm only able to help with ShopSmart Electronics orders, returns, shipping and "
    "product information. For other needs, please contact the appropriate service."
)


def check_off_scope_content(answer: str) -> GuardrailResult:
    """Block answers that stray into medical/legal/financial advice or harmful topics."""
    t0 = time.perf_counter()
    for pat in _OFF_SCOPE_PATTERNS:
        if pat.search(answer):
            return GuardrailResult(
                passed=False,
                reason=BlockReason.OFF_SCOPE_CONTENT,
                detail=f"Off-scope content matched: {pat.pattern[:60]}",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
    return GuardrailResult(passed=True, latency_ms=(time.perf_counter() - t0) * 1000)


# ── 4b. Fabricated-number detector ───────────────────────────────────────────
# Specific quantities (days, prices, percentages) that appear in the answer but
# are absent from every tool result and the question are likely hallucinated.
# Returns warnings rather than blocking, since arithmetic from two grounded
# numbers (e.g. 30 + 15 = 45) is legitimate and would appear as "ungrounded".

_QUANTITY_RE = re.compile(
    r"\b(\d{1,5}(?:\.\d{1,2})?)\s*"
    r"(?:day|hour|week|month|year|%|percent|dollar|\$|USD|business\s+day)s?\b",
    re.IGNORECASE,
)

# Small generic ordinals / quantities that appear everywhere and are not policy-specific.
_QUANTITY_ALLOWLIST = frozenset({"1", "2", "3", "0", "7", "10", "24", "60", "100"})


def check_fabricated_numbers(answer: str, known_text: str) -> list[str]:
    """Return list of warning strings for quantities in `answer` absent from context."""
    answer_nums = {m.group(1) for m in _QUANTITY_RE.finditer(answer)}
    suspicious = []
    for num in answer_nums:
        if num in _QUANTITY_ALLOWLIST:
            continue
        # Accept if the number appears anywhere in the known context (question + tool results)
        if re.search(r"\b" + re.escape(num) + r"\b", known_text):
            continue
        suspicious.append(num)
    return [f"quantity '{n}' in answer not found in tool context" for n in suspicious[:5]]


# ── 4c. Ungrounded policy assertion ──────────────────────────────────────────
# If the LLM asserts a policy rule but no policy chunks were retrieved, the claim
# cannot be grounded. This is a warning (not a block) since the agent may have
# paraphrased something the customer said.

_POLICY_ASSERT_RE = re.compile(
    r"\b(?:our\s+policy\s+(?:states?|says?|requires?|allows?|specifies?)|"
    r"(?:per|according\s+to)\s+(?:our|the)\s+(?:return|refund|warranty|shipping)?\s*policy|"
    r"policy\s+(?:covers?|includes?|excludes?)|"
    r"(?:terms?\s+and\s+conditions?|t(?:&|and)c)s?\s+(?:state|require|say))\b",
    re.IGNORECASE,
)


def check_ungrounded_policy_assertion(
    answer: str,
    policy_chunks_retrieved: bool,
) -> list[str]:
    """Warn when the agent asserts policy rules but retrieved no supporting chunks."""
    if policy_chunks_retrieved or not _POLICY_ASSERT_RE.search(answer):
        return []
    return ["answer asserts policy rules but no policy chunks were retrieved"]


# ── 4d. Stricter ML toxicity for LLM output ──────────────────────────────────
# The LLM's own output is held to a tighter standard than user input. Users may
# express frustration ("this damn charger broke"); the agent never should.

_OUTPUT_ML_BLOCK_LABELS = {
    "threat": 0.3,        # input threshold: 0.5
    "severe_toxic": 0.3,  # input threshold: 0.5
    "identity_hate": 0.3, # input threshold: 0.5
    "insult": 0.65,       # input threshold: 0.85
    "toxic": 0.75,        # new: general toxicity in output
    "obscene": 0.80,      # new: obscene output
}


def check_output_toxicity_ml(answer: str) -> GuardrailResult:
    """toxic-bert classifier on the agent's answer with tighter thresholds.
    Fails open (passes) if the model is unavailable."""
    t0 = time.perf_counter()
    from . import classifiers

    if not classifiers.ml_guardrails_enabled():
        return GuardrailResult(passed=True, detail="ml disabled")
    try:
        scores = classifiers.toxicity([answer])[0]
    except Exception as exc:  # noqa: BLE001
        return GuardrailResult(passed=True, detail=f"classifier unavailable: {exc}",
                               latency_ms=(time.perf_counter() - t0) * 1000)
    hits = {k: round(scores.get(k, 0.0), 3)
            for k, thr in _OUTPUT_ML_BLOCK_LABELS.items()
            if scores.get(k, 0.0) >= thr}
    ms = (time.perf_counter() - t0) * 1000
    if hits:
        return GuardrailResult(passed=False, reason=BlockReason.TOXIC_OUTPUT,
                               detail=f"Output toxicity classifier flagged: {hits}",
                               latency_ms=ms)
    return GuardrailResult(passed=True, detail="output toxicity ok", latency_ms=ms)


_SHINGLE = 8           # words per shingle when matching against the system prompt
_LEAK_MIN_SHINGLES = 2  # this many shared shingles => the answer quotes the prompt

OUTPUT_BLOCK_MESSAGE = (
    "I'm sorry, I can't share that. I can help with product information, shipping, "
    "orders, returns and refunds — what would you like to know?"
)


def _shingles(text: str, n: int = _SHINGLE) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9_]+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


@dataclass
class OutputCheckResult:
    passed: bool                       # False => answer replaced with a safe message
    text: str                          # what to show the customer
    reason: BlockReason | None = None
    redactions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


def check_output(
    answer: str,
    requester_id: str,
    system_prompt: str = "",
    known_text: str = "",
    use_ml: bool | None = None,
    policy_chunks_retrieved: bool = True,
) -> OutputCheckResult:
    """Final gate on the model's answer.

    Blocking checks (replace answer with a safe message):
      - system-prompt leak (shingle overlap);
      - toxic / abusive output (regex + ML with tighter thresholds than input);
      - off-scope content (medical/legal/financial advice, self-harm, extremism).

    Warning checks (logged in trace, answer kept):
      - PII redaction (email, card, SSN, other-customer IDs);
      - unverified order IDs not seen in question or tool traffic;
      - quantities in answer absent from tool context (potential hallucination);
      - policy assertions without retrieved evidence.
    """
    t0 = time.perf_counter()
    answer = answer or ""
    ml = use_ml if use_ml is not None else _ml_on()

    # ── Blocking: system-prompt leak ─────────────────────────────────────────
    if system_prompt:
        # Spans the model also saw in the question, retrieved evidence or tool output
        # are not a leak: e.g. the prompt names "the FTC Mail, Internet, or Telephone
        # Order Merchandise Rule", and so does the regulation text being cited.
        shared = (_shingles(answer) & _shingles(system_prompt)) - _shingles(known_text)
        if len(shared) >= _LEAK_MIN_SHINGLES:
            return OutputCheckResult(
                passed=False, text=OUTPUT_BLOCK_MESSAGE,
                reason=BlockReason.SYSTEM_PROMPT_LEAK,
                warnings=[f"answer shares {len(shared)} {_SHINGLE}-word spans with the "
                          "system prompt"],
                latency_ms=(time.perf_counter() - t0) * 1000)

    # ── Blocking: toxic / abusive output ────────────────────────────────────
    tox = check_toxicity(answer)
    if not tox.passed:
        return OutputCheckResult(
            passed=False, text=OUTPUT_BLOCK_MESSAGE, reason=BlockReason.TOXIC_OUTPUT,
            warnings=[tox.detail], latency_ms=(time.perf_counter() - t0) * 1000)
    if ml:
        # Use stricter thresholds for LLM output than for user input
        tox_ml = check_output_toxicity_ml(answer)
        if not tox_ml.passed:
            return OutputCheckResult(
                passed=False, text=OUTPUT_BLOCK_MESSAGE, reason=BlockReason.TOXIC_OUTPUT,
                warnings=[tox_ml.detail], latency_ms=(time.perf_counter() - t0) * 1000)

    # ── Blocking: off-scope / harmful content ────────────────────────────────
    off_scope = check_off_scope_content(answer)
    if not off_scope.passed:
        return OutputCheckResult(
            passed=False, text=_OFF_SCOPE_BLOCK_MSG,
            reason=BlockReason.OFF_SCOPE_CONTENT,
            warnings=[off_scope.detail],
            latency_ms=(time.perf_counter() - t0) * 1000)

    # ── Non-blocking: PII redaction + customer-ID masking ────────────────────
    text, redactions = redact_pii(answer)
    own = requester_id.strip().upper()

    def _other_customer(m: re.Match) -> str:
        if m.group(0).upper() == own:
            return m.group(0)
        redactions.append("customer_id")
        return "[another account]"

    text = _CUSTOMER_ID_RE.sub(_other_customer, text)

    # ── Non-blocking: warnings ────────────────────────────────────────────────
    warnings: list[str] = []

    # Unverified order IDs
    known = {o.upper() for o in _ORDER_REF_RE.findall(known_text)}
    warnings.extend(
        f"unverified order id {o.upper()}"
        for o in dict.fromkeys(_ORDER_REF_RE.findall(text))
        if o.upper() not in known
    )

    # Quantities in answer not present in tool context (potential hallucination)
    warnings.extend(check_fabricated_numbers(text, known_text))

    # Policy assertions without retrieved evidence
    warnings.extend(check_ungrounded_policy_assertion(text, policy_chunks_retrieved))

    return OutputCheckResult(passed=True, text=text, redactions=redactions,
                             warnings=warnings,
                             latency_ms=(time.perf_counter() - t0) * 1000)


def _ml_on() -> bool:
    from .classifiers import ml_guardrails_enabled
    return ml_guardrails_enabled()

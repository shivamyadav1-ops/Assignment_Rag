"""Output guardrails: run on the final answer before the customer sees it."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from ._base import BlockReason, GuardrailResult, _ml_on
from .tool_output import _CUSTOMER_ID_RE, _ORDER_REF_RE, redact_pii
from .toxicity import check_toxicity

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
    from .. import classifiers

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

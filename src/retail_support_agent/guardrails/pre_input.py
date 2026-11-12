"""Pre-input pipeline: topic guard, image NSFW, sanitisation, rate limit and
`run_pre_input_checks`, which chains every input layer."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from ._base import BlockReason, GuardrailResult
from .injection import check_prompt_injection, check_prompt_injection_ml, is_conversation_recall
from .sanitize import normalize_text, sanitize_input
from .toxicity import check_toxicity, check_toxicity_ml


def topic_guard_enabled() -> bool:
    import os
    return os.getenv("RSA_TOPIC_GUARD", "1").strip().lower() not in ("0", "false", "no")


def check_off_topic(text: str) -> GuardrailResult:
    """Refuse requests outside ShopSmart support before any LLM call or retrieval.
    Embedding similarity to in-scope vs. off-topic prototypes (topic.py), ~20 ms.
    Fails open: if the embedder is unavailable the agent's system prompt still declines."""
    t0 = time.perf_counter()
    try:
        from ..topic import is_off_topic
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
    from .. import classifiers

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
        from ..classifiers import ml_guardrails_enabled
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

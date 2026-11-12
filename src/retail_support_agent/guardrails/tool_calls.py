"""Tool-call guardrails: allowlist, argument validation, per-message budgets and the
look-before-you-write rule."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from ._base import BlockReason, GuardrailResult
from .injection import check_prompt_injection
from .sanitize import normalize_text

# ══════════════════════════════════════════════════════════════════════════════
# 2. Tool-call guardrails — run before a model-requested tool executes
# ══════════════════════════════════════════════════════════════════════════════
# The model chooses tools and arguments, so both are untrusted. Server-side checks
# (identity, eligibility) remain the source of truth; these add cheap, early limits
# so a confused or manipulated model cannot loop on tools, call tools that do not
# exist, or write (create a return) without first reading the order.

ORDER_ID_RE = re.compile(r"^ORD-\d{3,}$")

# Max calls of each tool per customer message (one `SupportAgent.answer` call: the
# graph builds a fresh ToolGuardState for every message). Per message, not per chat,
# so a long conversation is not locked out of policy search, and a customer can still
# return two different orders in one chat. Writes are capped hardest.
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
                      f"'{name}' may be called at most {budget} time(s) per customer "
                      "message.", t0)

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

"""Write-action guard: a return is created only when the customer asked for one, and
never for an order the lookup already reported as ineligible.

Regression for answer-eval case a08 (2026-11-12): asked "Can I return my order
ORD-10012?", the agent created RET-2036 without being asked to.
"""

from __future__ import annotations

import pytest

from retail_support_agent.guardrails import (
    BlockReason,
    ToolGuardState,
    check_tool_call,
    detect_return_intent,
    record_tool_call,
)
from tests.test_guardrails import SCHEMAS

ARGS = {"order_id": "ORD-10012", "reason": "Arrived damaged"}
ELIGIBLE = {"ok": True, "data": {"order_id": "ORD-10012",
                                 "return_eligibility": {"eligible_now": True,
                                                        "code": "ELIGIBLE"}}}
INELIGIBLE = {"ok": True, "data": {"order_id": "ORD-10012",
                                   "return_eligibility": {"eligible_now": False,
                                                          "code": "ALREADY_RETURNED"}}}


@pytest.mark.parametrize("q", [
    "Please start a return for order ORD-10129, it stopped working.",
    "My Nest Hub from ORD-10129 stopped working. Please start a return.",
    "I want to return the HDMI cable from order ORD-10036.",
    "I'd like to return my laptop",
    "Can you initiate a return for ORD-10012?",
    "Go ahead and open a return request for ORD-10012",
    "It's broken, return it please",
])
def test_requests_to_act_are_intent(q):
    assert detect_return_intent(q)


@pytest.mark.parametrize("q", [
    "Can I return my order ORD-10012?",
    "Can I still return order ORD-10001?",
    "Is it too late to return my controller from order ORD-10002?",
    "How do I return my keyboard from order ORD-10013?",
    "What is the return window for a smartphone?",
    "Is my laptop returnable?",
    "Am I eligible for a return on ORD-10012?",
])
def test_questions_about_returns_are_not_intent(q):
    assert not detect_return_intent(q)


def test_yes_to_an_offer_is_intent():
    history = [{"role": "user", "content": "Can I return ORD-10012?"},
               {"role": "assistant",
                "content": "Yes, it is eligible until December 23. Would you like me to "
                           "start a return for it?"}]
    assert detect_return_intent("Yes please", history)
    assert detect_return_intent("ok, go ahead", history)
    assert not detect_return_intent("What about shipping costs?", history)


def test_yes_without_an_offer_is_not_intent():
    history = [{"role": "assistant", "content": "The window is 45 days."}]
    assert not detect_return_intent("yes", history)
    assert not detect_return_intent("yes", None)


def test_write_blocked_without_intent_even_after_lookup():
    state = ToolGuardState(write_intent=False)
    record_tool_call(state, "get_order_details", {"order_id": "ORD-10012"}, ELIGIBLE)
    r = check_tool_call("create_return_request", ARGS, SCHEMAS, state)
    assert r.reason is BlockReason.PRECONDITION_FAILED
    assert "not asked" in r.detail


def test_write_allowed_with_intent_and_eligible_lookup():
    state = ToolGuardState(write_intent=True)
    record_tool_call(state, "get_order_details", {"order_id": "ORD-10012"}, ELIGIBLE)
    assert check_tool_call("create_return_request", ARGS, SCHEMAS, state).passed


def test_write_blocked_when_lookup_said_ineligible():
    state = ToolGuardState(write_intent=True)
    record_tool_call(state, "get_order_details", {"order_id": "ORD-10012"}, INELIGIBLE)
    r = check_tool_call("create_return_request", ARGS, SCHEMAS, state)
    assert r.reason is BlockReason.PRECONDITION_FAILED
    assert "ALREADY_RETURNED" in r.detail


def test_unevaluated_intent_keeps_old_behaviour():
    # Direct callers that never set write_intent are governed by the other rules only.
    state = ToolGuardState()
    record_tool_call(state, "get_order_details", {"order_id": "ORD-10012"}, ELIGIBLE)
    assert check_tool_call("create_return_request", ARGS, SCHEMAS, state).passed

"""Holiday-aware return deadlines and the eligibility verdict on get_order_details.

These pin the two answer-eval failures (a07: Dec 1 headphones -> Jan 31; a11: a shipped,
undelivered order cannot be returned yet) to deterministic code, plus the server-side
bug they exposed: evaluate_return ignored the Holiday Return Policy.
"""

from __future__ import annotations

from datetime import date

import pytest

from retail_support_agent.agent import SupportAgent
from retail_support_agent.policies import evaluate_return, holiday_deadline, return_deadline

pytestmark = pytest.mark.agent


@pytest.mark.parametrize("purchased, expected", [
    (date(2024, 11, 14), None),                    # day before the holiday period
    (date(2024, 11, 15), date(2025, 1, 31)),       # first day, inclusive
    (date(2024, 12, 31), date(2025, 1, 31)),       # last day, inclusive
    (date(2025, 1, 1), None),
])
def test_holiday_period_boundaries(purchased, expected):
    assert holiday_deadline(purchased) == expected


def test_a07_headphones_bought_dec_1_last_day_is_jan_31():
    out = return_deadline("Audio & Headphones", False, date(2024, 12, 1))
    assert out["standard_deadline"] == "2024-12-31"
    assert out["deadline"] == "2025-01-31" and out["rule_applied"] == "holiday_extension"


def test_holiday_rule_never_shortens_a_longer_window():
    # Policy example: Computer Accessory bought Dec 28 -> 45 days -> Feb 11 beats Jan 31.
    out = return_deadline("Computer Accessories", False, date(2024, 12, 28))
    assert out["deadline"] == "2025-02-11" and out["rule_applied"] == "standard_window"


def test_policy_example_premium_smartphone_dec_30():
    # 15 + 15 Premium days -> Jan 29; Jan 31 is later, so Jan 31 applies.
    out = return_deadline("Smartphones & Tablets", True, date(2024, 12, 30))
    assert out["standard_deadline"] == "2025-01-29" and out["deadline"] == "2025-01-31"


def test_non_returnable_category_has_no_deadline():
    out = return_deadline("Software & Digital Codes", False, date(2024, 12, 1))
    assert out["ok"] is False and out["code"] == "NON_RETURNABLE"


def test_evaluate_return_honours_holiday_extension():
    # Delivered Dec 2, checked Jan 20: 49 days > 30-day window, but the Jan 31 holiday
    # deadline applies, so the return is still allowed (it was wrongly rejected before).
    d = evaluate_return(category="Audio & Headphones", order_status="delivered",
                        days_since_delivery=49, is_premium=False,
                        existing_return_statuses=[], purchase_date=date(2024, 12, 1),
                        delivery_date=date(2024, 12, 2), as_of=date(2025, 1, 20))
    assert d.eligible and d.days_remaining == 11


def test_evaluate_return_without_dates_is_unchanged():
    d = evaluate_return(category="Audio & Headphones", order_status="delivered",
                        days_since_delivery=31, is_premium=False, existing_return_statuses=[])
    assert not d.eligible and d.reason_code == "RETURN_WINDOW_EXPIRED"


def test_a11_shipped_order_reports_not_delivered():
    from retail_support_agent.db import load_operational_data
    from retail_support_agent.ops import get_order_details

    load_operational_data()
    out = get_order_details("ORD-10009", "CUST-1048")
    elig = out["data"]["return_eligibility"]
    assert elig["eligible_now"] is False and elig["code"] == "ORDER_NOT_DELIVERED"
    assert elig["deadline"] is None


def test_unauthorized_order_tells_agent_to_ask_for_the_id():
    from retail_support_agent.ops import get_order_details

    out = get_order_details("ORD-10012", "CUST-1001")
    assert out["error"]["code"] == "UNAUTHORIZED"
    assert "double-check the order ID" in out["error"]["message"]


def test_agent_deadline_tool_validates_dates():
    bad = SupportAgent._deadline({"category": "Audio & Headphones",
                                  "purchase_date": "December 1"})
    assert bad["ok"] is False and bad["error"]["code"] == "VALIDATION_ERROR"
    ok = SupportAgent._deadline({"category": "Audio & Headphones",
                                 "purchase_date": "2024-12-01"})
    assert ok["ok"] and ok["data"]["deadline"] == "2025-01-31"
    assert ok["data"]["deadline_text"] == "January 31, 2025"


def test_deadline_answer_is_forced_to_cite_policy(tmp_path):
    """The model may skip search_policies after the deadline tool; the agent forces it."""
    from retail_support_agent.llm import LLMTurn, ScriptedClient, ToolCall
    from tests.test_agent_guardrails import FakeChunk, FakeMCP, FakeRetriever

    llm = ScriptedClient([
        LLMTurn(text=None, tool_calls=[ToolCall("c0", "calculate_return_deadline", {
            "category": "Audio & Headphones", "purchase_date": "2024-12-01"})]),
        LLMTurn(text="The last day is January 31, 2025.", stop_reason="end_turn"),
        LLMTurn(text="The last day is January 31, 2025 (return_policy.pdf).",
                stop_reason="end_turn"),
    ])
    chunk = FakeChunk("c1", "return_policy.pdf", "1.3 Holiday Return Policy",
                      "Items purchased between November 15 and December 31 may be returned "
                      "until January 31 of the following year.", rerank_score=3.0)
    agent = SupportAgent("CUST-T0001", llm=llm, mcp=FakeMCP({}),
                         retriever=FakeRetriever([chunk]), trace_dir=tmp_path,
                         semantic_cache=False)
    r = agent.answer("If I bought headphones on December 1, what is the last day to return?")
    assert r.tool_calls == ["calculate_return_deadline", "search_policies"]
    forced = [s for s in r.steps if s.kind == "tool" and s.name == "search_policies"]
    assert forced and forced[0].detail.get("forced") is True
    assert "Audio & Headphones" in forced[0].detail["args"]["query"]
    assert r.evidence and "return_policy.pdf" in r.answer


def test_forced_search_sends_a_valid_tool_exchange_and_never_returns_empty(tmp_path):
    """After a forced search the model must see assistant(tool_calls) -> tool, not a
    text answer followed by a tool result; and an empty final reply falls back to the
    draft instead of reaching the customer."""
    from retail_support_agent.llm import LLMTurn, ScriptedClient, ToolCall
    from tests.test_agent_guardrails import FakeChunk, FakeMCP, FakeRetriever

    llm = ScriptedClient([
        LLMTurn(text=None, tool_calls=[ToolCall("c0", "calculate_return_deadline", {
            "category": "Audio & Headphones", "purchase_date": "2024-12-01"})]),
        LLMTurn(text="The last day is January 31, 2025.", stop_reason="end_turn"),
        LLMTurn(text="", stop_reason="end_turn"),              # model returns nothing
    ])
    chunk = FakeChunk("c1", "return_policy.pdf", "1.3", "returned until January 31",
                      rerank_score=3.0)
    agent = SupportAgent("CUST-T0002", llm=llm, mcp=FakeMCP({}),
                         retriever=FakeRetriever([chunk]), trace_dir=tmp_path,
                         semantic_cache=False)
    r = agent.answer("If I bought headphones on December 1, what is the last day?")
    assert r.answer.startswith("The last day is January 31, 2025")
    last_call = llm.seen[-1]["messages"]
    roles = [m["role"] for m in last_call]
    assert roles[-2:] == ["assistant", "tool"]
    assert last_call[-2]["tool_calls"][0]["function"]["name"] == "search_policies"
    assert last_call[-1]["tool_call_id"] == last_call[-2]["tool_calls"][0]["id"]
    # The draft rides in the same assistant turn that requests the search (valid on
    # both wire formats), never as a separate text answer before the tool result.
    assert last_call[-2]["content"] == "The last day is January 31, 2025."
    assert not any(m["role"] == "assistant" and not m.get("tool_calls")
                   and (m.get("content") or "").startswith("The last") for m in last_call)

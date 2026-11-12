"""MCP server exposing the retail operational tools over stdio.

Thin protocol layer over `ops.py`. The requester's identity is read ONCE from the
RSA_REQUESTER_ID environment variable the client sets when launching this process;
it is never a tool parameter, so the model cannot choose whose records to read. A
missing identity makes every tool refuse rather than leak data.

Run directly:  RSA_REQUESTER_ID=CUST-1001 python -m retail_support_agent.mcp_server
"""

from __future__ import annotations

import os
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from . import ops

mcp = MCPServer("retail-ops")

OrderId = Annotated[str, Field(description="Order identifier, format ORD-#####",
                               examples=["ORD-10012"])]


def _requester() -> str | None:
    rid = os.environ.get("RSA_REQUESTER_ID", "").strip()
    return rid or None


def _no_identity() -> dict:
    return {"ok": False, "error": {
        "code": "NO_SESSION_IDENTITY",
        "message": "Server launched without RSA_REQUESTER_ID; cannot authorize any request.",
        "retryable": False}}


def _guard(fn, *args) -> dict:
    """Run an ops call with identity injected, mapping any crash to a typed envelope."""
    rid = _requester()
    if rid is None:
        return _no_identity()
    try:
        return fn(*args, requester_id=rid)
    except Exception:  # noqa: BLE001 - never let a tool crash the transport
        return {"ok": False, "error": {"code": "INTERNAL_ERROR",
                "message": "Unexpected server error. Please try again.", "retryable": True}}


@mcp.tool(description="Fetch order details: product, category, status, delivery age, "
                      "membership tier and any existing returns on the order.")
def get_order_details(order_id: OrderId) -> dict:
    return _guard(ops.get_order_details, order_id)


@mcp.tool(description="Fetch return and refund status for an order: whether it has "
                      "returns, their state and refund amounts.")
def get_return_refund_status(order_id: OrderId) -> dict:
    return _guard(ops.get_return_refund_status, order_id)


@mcp.tool(description="Create a return request for an order. Validates return-window, "
                      "category and prior-return rules before writing; returns a typed "
                      "error if the order is not eligible.")
def create_return_request(
    order_id: OrderId,
    reason: Annotated[str, Field(description="Why the customer wants to return the item.",
                                 examples=["Item arrived damaged"])],
) -> dict:
    rid = _requester()
    if rid is None:
        return _no_identity()
    try:
        return ops.create_return_request(order_id, reason, requester_id=rid)
    except Exception:  # noqa: BLE001
        return {"ok": False, "error": {"code": "INTERNAL_ERROR",
                "message": "Unexpected server error. Please try again.", "retryable": True}}


EscalationReason = Annotated[
    str,
    Field(
        description=(
            "Why this case needs a human: policy_unclear, out_of_policy_exception, "
            "technical_issue, unresolved_complaint, other."
        ),
        examples=["policy_unclear"],
    ),
]


@mcp.tool(description="Create an escalation ticket so a human agent follows up. "
                      "Use this when policy does not cover the customer's situation, "
                      "when a tool returns an unresolvable error, or when the customer "
                      "explicitly asks to speak to a person. Returns a ticket ID to quote.")
def create_escalation_ticket(
    summary: Annotated[str, Field(description="One-sentence description of the issue.",
                                  examples=["Customer requesting exception to return window"])],
    reason: EscalationReason,
    order_id: Annotated[str | None, Field(
        description="Related order ID if applicable.", default=None)] = None,
) -> dict:
    return _guard(ops.create_escalation_ticket, summary, reason, order_id)


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()

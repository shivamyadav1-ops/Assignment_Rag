"""Operational tool logic behind the MCP server.

Pure Python over the SQLite layer: no MCP transport here, so every tool and every
error path is unit-testable directly. The MCP server (`mcp_server.py`) is a thin
wrapper that exposes these functions over stdio.

Every function returns the same envelope:
    {"ok": True,  "data": {...}}
    {"ok": False, "error": {"code": ..., "message": ..., "retryable": bool}}

`requester_id` is supplied by the client from the session, never chosen by the model.
Each read checks it against the record owner, so a manipulated prompt cannot reach
another customer's data (the "unauthorized access" exception case).
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime

from sqlalchemy import Integer, func, select
from sqlalchemy.orm import Session

from . import config
from .db import Customer, Order, Return, fetch_returns_for_order, get_session
from .policies import evaluate_return, return_deadline

ORDER_ID_RE = re.compile(r"^ORD-\d{3,}$")

# Error codes that reflect a transient condition the agent may retry.
RETRYABLE_CODES = {"INTERNAL_ERROR"}


def _ok(data: dict) -> dict:
    return {"ok": True, "data": data}


def _err(code: str, message: str) -> dict:
    return {"ok": False, "error": {"code": code, "message": message,
                                   "retryable": code in RETRYABLE_CODES}}


def _mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    shown = local[:2] if len(local) > 2 else local[:1]
    return f"{shown}{'*' * max(len(local) - len(shown), 1)}@{domain}"


# ── Tool 1: order details (+ return history + customer tier) ───────────────────

def get_order_details(order_id: str, requester_id: str, session: Session | None = None) -> dict:
    """Operational facts the agent needs to reason about a return: product, category,
    status, delivery age, membership tier and any existing returns on the order."""
    order_id = (order_id or "").strip().upper()
    if not ORDER_ID_RE.match(order_id):
        return _err("INVALID_ORDER_ID",
                    f"'{order_id}' is not a valid order ID (expected ORD-#####).")

    own = session is None
    session = session or get_session()
    try:
        order = session.get(Order, order_id)
        if order is None:
            return _err("ORDER_NOT_FOUND", f"No order found with ID {order_id}.")
        if order.customer_id != requester_id:
            # Do not reveal whether the order exists for someone else beyond this.
            return _err("UNAUTHORIZED",
                        "This order belongs to a different account and cannot be shown. "
                        "Ask the customer to double-check the order ID.")

        customer = session.get(Customer, order.customer_id)
        returns = fetch_returns_for_order(session, order_id)
        data = order.to_dict()
        data["membership_tier"] = customer.membership_tier if customer else "standard"
        data["is_premium"] = bool(customer and customer.membership_tier == "premium")
        data["return_history"] = [
            {"return_id": r.return_id, "status": r.status, "reason": r.reason,
             "refund_amount": r.refund_amount, "requested_date": r.requested_date.isoformat()}
            for r in returns
        ]
        # The authoritative eligibility verdict, computed by the same rules
        # create_return_request enforces, so the model relays it instead of doing date
        # arithmetic itself (it reasoned from the *order* date on an undelivered order).
        decision = evaluate_return(
            category=order.product_category, order_status=order.status,
            days_since_delivery=order.days_since_delivery, is_premium=data["is_premium"],
            existing_return_statuses=[r.status for r in returns],
            purchase_date=order.order_date, delivery_date=order.delivery_date,
            as_of=config.AS_OF_DATE)
        deadline = return_deadline(order.product_category, data["is_premium"],
                                   order.order_date, order.delivery_date)
        data["return_eligibility"] = {
            "eligible_now": decision.eligible, "code": decision.reason_code,
            "message": decision.message,
            "window_days": decision.window_days,
            "days_since_delivery": decision.days_since_delivery,
            "days_remaining": decision.days_remaining,
            "deadline": deadline.get("deadline_text") if order.delivery_date else None,
            "rule_applied": deadline.get("rule_applied") if order.delivery_date else None,
            "as_of": config.AS_OF_DATE.isoformat(),
        }
        return _ok(data)
    finally:
        if own:
            session.close()


# ── Tool 2: return & refund status ─────────────────────────────────────────────

def get_return_refund_status(order_id: str, requester_id: str,
                             session: Session | None = None) -> dict:
    """Return records and refund state for an order. Answers 'has this been
    returned?' and 'where is my refund?'."""
    order_id = (order_id or "").strip().upper()
    if not ORDER_ID_RE.match(order_id):
        return _err("INVALID_ORDER_ID",
                    f"'{order_id}' is not a valid order ID (expected ORD-#####).")

    own = session is None
    session = session or get_session()
    try:
        order = session.get(Order, order_id)
        if order is None:
            return _err("ORDER_NOT_FOUND", f"No order found with ID {order_id}.")
        if order.customer_id != requester_id:
            return _err("UNAUTHORIZED",
                        "This order belongs to a different account and cannot be shown.")

        returns = fetch_returns_for_order(session, order_id)
        return _ok({
            "order_id": order_id,
            "has_returns": bool(returns),
            "returns": [r.to_dict() for r in returns],
        })
    finally:
        if own:
            session.close()


# ── Tool 3: create a return (write; eligibility enforced in code) ──────────────

def create_return_request(order_id: str, reason: str, requester_id: str,
                          session: Session | None = None) -> dict:
    """Validate return eligibility and, if eligible, create a pending return record.
    The policy rules are enforced here, not left to the model."""
    order_id = (order_id or "").strip().upper()
    if not ORDER_ID_RE.match(order_id):
        return _err("INVALID_ORDER_ID",
                    f"'{order_id}' is not a valid order ID (expected ORD-#####).")
    if not (reason or "").strip():
        return _err("VALIDATION_ERROR", "A return reason is required.")

    own = session is None
    session = session or get_session()
    try:
        order = session.get(Order, order_id)
        if order is None:
            return _err("ORDER_NOT_FOUND", f"No order found with ID {order_id}.")
        if order.customer_id != requester_id:
            return _err("UNAUTHORIZED",
                        "This order belongs to a different account and cannot be returned.")

        customer = session.get(Customer, order.customer_id)
        is_premium = bool(customer and customer.membership_tier == "premium")
        existing = fetch_returns_for_order(session, order_id)

        decision = evaluate_return(
            category=order.product_category,
            order_status=order.status,
            days_since_delivery=order.days_since_delivery,
            is_premium=is_premium,
            existing_return_statuses=[r.status for r in existing],
            purchase_date=order.order_date,
            delivery_date=order.delivery_date,
            as_of=config.AS_OF_DATE,
        )
        if not decision.eligible:
            return _err(decision.reason_code, decision.message)

        # Allocate the next return ID: max numeric suffix of "RET-####" + 1.
        max_num = session.scalar(
            select(func.max(func.cast(func.substr(Return.return_id, 5), Integer)))
        )
        next_num = (int(max_num) + 1) if max_num else 2001
        new_id = f"RET-{next_num}"

        record = Return(
            return_id=new_id,
            order_id=order_id,
            customer_id=order.customer_id,
            product_name=order.product_name,
            product_category=order.product_category,
            reason=reason.strip()[:120],
            status="pending",
            requested_date=config.AS_OF_DATE,
            resolved_date=None,
            refund_amount=0.0,
            refund_method=None,
            synthetic=True,
        )
        session.add(record)
        session.commit()

        return _ok({
            "return_id": new_id,
            "status": "pending",
            "order_id": order_id,
            "category": decision.category,
            "window_days": decision.window_days,
            "premium_extension_applied": decision.premium_extension_applied,
            "days_remaining": decision.days_remaining,
            "estimated_refund": round(order.amount, 2),
            "message": f"Return {new_id} created. {decision.message}",
        })
    finally:
        if own:
            session.close()


# ── Tool 4: escalation ─────────────────────────────────────────────────────────

_VALID_REASONS = frozenset({
    "policy_unclear", "out_of_policy_exception", "technical_issue",
    "unresolved_complaint", "other",
})


def create_escalation_ticket(
    summary: str,
    reason: str,
    order_id: str | None,
    requester_id: str,
) -> dict:
    """Log an escalation ticket so a human agent can follow up.

    The ticket is appended to `data/escalations.jsonl` (created on first use).
    Returns a ticket id the agent can quote to the customer.
    """
    summary = (summary or "").strip()
    reason = (reason or "other").strip().lower()
    if not summary:
        return _err("VALIDATION_ERROR", "summary cannot be empty.")
    if reason not in _VALID_REASONS:
        reason = "other"
    if order_id:
        order_id = order_id.strip().upper()
        if not ORDER_ID_RE.match(order_id):
            return _err("INVALID_ORDER_ID",
                        f"'{order_id}' is not a valid order ID (expected ORD-#####).")

    ticket_id = "ESC-" + uuid.uuid4().hex[:8].upper()
    ticket = {
        "ticket_id": ticket_id,
        "requester_id": requester_id,
        "order_id": order_id,
        "reason": reason,
        "summary": summary,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "open",
        "synthetic": True,
    }
    log_path = config.DATA_DIR / "escalations.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ticket) + "\n")

    return _ok({
        "ticket_id": ticket_id,
        "status": "open",
        "message": (
            f"Escalation ticket {ticket_id} created. A human agent will follow up "
            "within 1 business day. Please quote this ticket ID in any further contact."
        ),
    })

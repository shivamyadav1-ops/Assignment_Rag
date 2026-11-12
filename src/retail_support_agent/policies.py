"""Return-eligibility rules, encoded as pure functions.

This is the authoritative return logic. The MCP `create_return_request` tool calls
`evaluate_return` before writing, so the rules are enforced in code rather than left
to the model's judgement. The numbers mirror the Product Category Reference Guide and
the Seasonal/Holiday policy; the same facts are also retrievable as policy text, so the
agent can *explain* a decision from documents while the server *enforces* it here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

# Standard return window per category, in days. None means non-returnable.
CATEGORY_WINDOW_DAYS: dict[str, int | None] = {
    "Smartphones & Tablets": 15,
    "Laptops & Computers": 30,
    "Audio & Headphones": 30,
    "Televisions & Monitors": 15,
    "Cameras & Photography": 30,
    "Gaming Consoles": 15,
    "Gaming Accessories": 30,
    "Smart Home Devices": 30,
    "Wearables & Fitness": 30,
    "Computer Accessories": 45,
    "Cables & Adapters": 45,
    "Software & Digital Codes": None,
    "Gift Cards": None,
}

# Premium members get a flat extension on every returnable category.
PREMIUM_EXTENSION_DAYS = 15

# Holiday Return Policy (return_policy.pdf 1.3, seasonal_holiday_policy 1.1): items
# purchased Nov 15 - Dec 31 may be returned until Jan 31 of the following year, or
# within the standard window, whichever is LATER. Applies to every returnable category
# (the standard window includes the Premium extension). The rule never shortens a window.
HOLIDAY_PURCHASE_START = (11, 15)   # (month, day), inclusive
HOLIDAY_PURCHASE_END = (12, 31)     # inclusive
HOLIDAY_RETURN_UNTIL = (1, 31)      # of the following year

# Order must be in this status before a return can start.
RETURNABLE_ORDER_STATUS = "delivered"

# Return records in these states already occupy the order; a rejected return does not.
BLOCKING_RETURN_STATUSES = {"pending", "approved", "refunded"}


@dataclass
class ReturnDecision:
    eligible: bool
    reason_code: str  # "ELIGIBLE" or a typed failure code
    message: str
    category: str
    window_days: int | None = None        # effective window incl. premium
    standard_window_days: int | None = None
    premium_extension_applied: bool = False
    days_since_delivery: int | None = None
    days_remaining: int | None = None


def human_date(d: date | None) -> str | None:
    """'January 31, 2025': what the customer should read (models copy ISO dates)."""
    return f"{d:%B} {d.day}, {d.year}" if d else None


def holiday_deadline(purchase_date: date | None) -> date | None:
    """Jan 31 of the next year if the purchase falls in the holiday period, else None."""
    if purchase_date is None:
        return None
    start = date(purchase_date.year, *HOLIDAY_PURCHASE_START)
    end = date(purchase_date.year, *HOLIDAY_PURCHASE_END)
    if start <= purchase_date <= end:
        return date(purchase_date.year + 1, *HOLIDAY_RETURN_UNTIL)
    return None


def return_deadline(category: str, is_premium: bool, purchase_date: date | None,
                    delivery_date: date | None = None) -> dict:
    """Last day to return, with the rule that decided it. Pure.

    The standard window counts from delivery; when no delivery date is given (e.g. a
    customer asking "if I bought X on December 1"), it counts from the purchase date.
    """
    base = CATEGORY_WINDOW_DAYS.get(category)
    if category not in CATEGORY_WINDOW_DAYS:
        return {"ok": False, "code": "UNKNOWN_CATEGORY",
                "message": f"Category '{category}' has no return rule on file."}
    if base is None:
        return {"ok": False, "code": "NON_RETURNABLE",
                "message": f"{category} is non-returnable."}
    window = effective_window(category, is_premium)
    start = delivery_date or purchase_date
    standard = start + timedelta(days=window) if start else None
    holiday = holiday_deadline(purchase_date)
    if standard and holiday and holiday > standard:
        deadline, rule = holiday, "holiday_extension"
    else:
        deadline, rule = standard, "standard_window"
    return {
        "ok": True, "category": category, "standard_window_days": base,
        "premium_extension_days": PREMIUM_EXTENSION_DAYS if is_premium else 0,
        "effective_window_days": window,
        "window_counted_from": "delivery" if delivery_date else "purchase",
        "standard_deadline": standard.isoformat() if standard else None,
        "holiday_period_purchase": holiday is not None,
        "holiday_deadline": holiday.isoformat() if holiday else None,
        "deadline": deadline.isoformat() if deadline else None,
        "deadline_text": human_date(deadline),
        "rule_applied": rule,
        "explanation": (
            f"{category}: {base}-day window"
            + (f" + {PREMIUM_EXTENSION_DAYS} Premium days" if is_premium else "")
            + (f" from {human_date(start)} ends {human_date(standard)}" if standard else "")
            + (f"; purchased in the November 15 - December 31 holiday period, so "
               f"returnable until {human_date(holiday)}" if holiday else "")
            + (f". The later date applies: the last day to return is "
               f"{human_date(deadline)}." if deadline else ".")),
    }


def effective_window(category: str, is_premium: bool) -> int | None:
    """Return window in days for a category, including the premium extension, or None."""
    base = CATEGORY_WINDOW_DAYS.get(category)
    if base is None:
        return None
    return base + (PREMIUM_EXTENSION_DAYS if is_premium else 0)


def evaluate_return(
    *,
    category: str,
    order_status: str,
    days_since_delivery: int | None,
    is_premium: bool,
    existing_return_statuses: list[str],
    purchase_date: date | None = None,
    delivery_date: date | None = None,
    as_of: date | None = None,
) -> ReturnDecision:
    """Decide whether an order may be returned. Pure: no DB or network access.

    With purchase/delivery dates and `as_of`, the Holiday Return Policy is applied
    (Jan 31 deadline for Nov 15 - Dec 31 purchases, if later than the standard window).
    """
    standard = CATEGORY_WINDOW_DAYS.get(category)

    if category not in CATEGORY_WINDOW_DAYS:
        return ReturnDecision(
            False, "UNKNOWN_CATEGORY",
            f"Category '{category}' has no return rule on file; routing to a human.",
            category,
        )

    if standard is None:
        return ReturnDecision(
            False, "NON_RETURNABLE",
            f"{category} is non-returnable.",
            category, standard_window_days=None,
        )

    if order_status != RETURNABLE_ORDER_STATUS:
        return ReturnDecision(
            False, "ORDER_NOT_DELIVERED",
            f"Order is '{order_status}', not delivered; a return cannot start yet.",
            category, standard_window_days=standard,
        )

    if any(s in BLOCKING_RETURN_STATUSES for s in existing_return_statuses):
        return ReturnDecision(
            False, "ALREADY_RETURNED",
            "An active return already exists for this order.",
            category, standard_window_days=standard,
        )

    window = effective_window(category, is_premium)
    if days_since_delivery is None:
        return ReturnDecision(
            False, "ORDER_NOT_DELIVERED",
            "No delivery date on record; a return cannot start yet.",
            category, window, standard, is_premium,
        )

    remaining = window - days_since_delivery
    holiday = holiday_deadline(purchase_date)
    if holiday and as_of and delivery_date:
        standard_end = delivery_date + timedelta(days=window)
        if holiday > standard_end:          # the later date applies
            remaining = (holiday - as_of).days
    if remaining < 0:
        return ReturnDecision(
            False, "RETURN_WINDOW_EXPIRED",
            f"The {window}-day return window closed {-remaining} day(s) ago.",
            category, window, standard, is_premium, days_since_delivery, remaining,
        )

    return ReturnDecision(
        True, "ELIGIBLE",
        f"Within the {window}-day window; {remaining} day(s) remain.",
        category, window, standard, is_premium, days_since_delivery, remaining,
    )

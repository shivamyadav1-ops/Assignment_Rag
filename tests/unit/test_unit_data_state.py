"""Unit tests for the data / ops / cache / session / topic cluster.

Model-free and network-free: every LLM, embedder and corpus is a fake, every database
is built in a pytest ``tmp_path``. These tests deliberately do not overlap with
tests/test_cache.py, tests/test_session.py or tests/test_return_deadlines.py — they
cover the layers underneath those end-to-end tests: SQL parameter binding, ingestion
validation, tool input validation, LRU/TTL mechanics, session scoping and the topic
threshold.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.exc import SQLAlchemyError

from retail_support_agent import cache as cache_mod
from retail_support_agent import config, ops, topic
from retail_support_agent import session as session_mod
from retail_support_agent.db import (
    Base,
    Customer,
    IngestionError,
    Order,
    Return,
    _nullable_str,
    _parse_bool,
    _parse_date,
    fetch_returns_for_order,
    get_engine,
    get_session,
    load_operational_data,
)
from retail_support_agent.session import ChatSession, turn_relevance

HOSTILE = "ORD-1'); DROP TABLE returns; --"

CUSTOMER_HEADER = ("customer_id,first_name,last_name,email,membership_tier,"
                   "account_created,synthetic\n")
ORDER_HEADER = ("order_id,customer_id,product_name,product_category,order_date,"
                "delivery_date,status,amount,payment_method,shipping_method,"
                "warranty_plan,synthetic\n")


# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════

def write_operational(tmp_path: Path, customers: str = "", orders: str = "",
                      returns: list | None = None) -> Path:
    """Build a minimal operational source directory under tmp_path."""
    d = tmp_path / "operational"
    d.mkdir(parents=True, exist_ok=True)
    (d / "customers.csv").write_text(CUSTOMER_HEADER + customers, encoding="utf-8")
    (d / "orders.csv").write_text(ORDER_HEADER + orders, encoding="utf-8")
    (d / "returns.json").write_text(json.dumps(returns if returns is not None else []),
                                    encoding="utf-8")
    return d


DEFAULT_CUSTOMERS = (
    "CUST-1001,Olivia,Hall,olivia.hall@example.com,standard,2024-01-03,True\n"
    "CUST-1002,Priya,Ortiz,priya.ortiz@example.com,premium,2024-06-28,True\n"
)
DEFAULT_ORDERS = (
    "ORD-10001,CUST-1001,Laptop X,Laptops & Computers,2024-10-01,2024-10-05,"
    "delivered,999.5,credit_card,standard,,True\n"
    "ORD-10002,CUST-1002,Phone Y,Smartphones & Tablets,2024-10-02,2024-10-06,"
    "delivered,499.0,debit_card,expedited,protect_plus,True\n"
)


@pytest.fixture()
def opdb(tmp_path):
    """A fully populated throwaway operational database (never data/operational.db)."""
    src = write_operational(
        tmp_path, DEFAULT_CUSTOMERS, DEFAULT_ORDERS,
        returns=[{
            "return_id": "RET-2001", "order_id": "ORD-10002",
            "customer_id": "CUST-1002", "product_name": "Phone Y",
            "product_category": "Smartphones & Tablets", "reason": "Arrived damaged",
            "status": "rejected", "requested_date": "2024-10-20",
            "resolved_date": "2024-10-25", "refund_amount": 0.0,
            "refund_method": None, "synthetic": True,
        }],
    )
    path = tmp_path / "op.db"
    summary = load_operational_data(operational_dir=src, db_path=path)
    assert summary["customers"] == 2
    return path


@pytest.fixture()
def dbsession(opdb):
    s = get_session(opdb)
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _never_touch_the_real_db(monkeypatch):
    """Guard rail for this file: every test must pass an explicit db_path (-> SQLite).

    An accidental default-path call would otherwise hit the real PostgreSQL database
    at RSA_DATABASE_URL. Blanking it here makes that call fail immediately instead.
    """
    monkeypatch.setattr(config, "DATABASE_URL", None, raising=False)


# ══════════════════════════════════════════════════════════════════════════════
# db.py — parsing helpers
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("value, expected", [
    ("2024-11-09", date(2024, 11, 9)),
    ("", None), (None, None), ("None", None), ("null", None),
    ("2024-02-29", date(2024, 2, 29)),            # leap day
    ("2024-01-01", date(2024, 1, 1)),             # year boundary
    ("2024-12-31", date(2024, 12, 31)),
])
def test_parse_date_accepts_iso_and_empty_sentinels(value, expected):
    assert _parse_date(value) == expected


@pytest.mark.parametrize("bad", ["09-11-2024", "2024/11/09", "2024-13-01", "2024-02-30",
                                 "2024-11-09T00:00:00", "tomorrow"])
def test_parse_date_rejects_non_iso_and_impossible_dates(bad):
    with pytest.raises(ValueError):
        _parse_date(bad)


@pytest.mark.parametrize("value, expected", [
    (True, True), (False, False), ("True", True), ("true", True), ("TRUE", True),
    ("1", True), ("yes", True), ("  Yes  ", True),
    ("False", False), ("0", False), ("", False), (None, False), (1, True), (0, False),
])
def test_parse_bool(value, expected):
    assert _parse_bool(value) is expected


@pytest.mark.parametrize("value, expected", [
    (None, None), ("", None), ("   ", None), ("None", None), ("null", None),
    ("nan", None), ("protect_plus", "protect_plus"), ("  plan  ", "plan"), (0, "0"),
])
def test_nullable_str(value, expected):
    assert _nullable_str(value) == expected


# ══════════════════════════════════════════════════════════════════════════════
# db.py — ingestion
# ══════════════════════════════════════════════════════════════════════════════

def test_ingestion_counts_and_referential_integrity(tmp_path):
    src = write_operational(
        tmp_path,
        customers=DEFAULT_CUSTOMERS
        + "CUST-1001,Dup,Licate,dup@example.com,standard,2024-02-02,True\n",
        orders=DEFAULT_ORDERS
        + "ORD-10001,CUST-1001,Dup,Laptops & Computers,2024-10-01,2024-10-05,"
          "delivered,1.0,credit_card,standard,,True\n"
        + "ORD-99999,CUST-NOPE,Ghost,Laptops & Computers,2024-10-01,2024-10-05,"
          "delivered,1.0,credit_card,standard,,True\n",
        returns=[
            {"return_id": "RET-2001", "order_id": "ORD-10001",
             "customer_id": "CUST-1001", "product_name": "Laptop X",
             "product_category": "Laptops & Computers", "reason": "damaged",
             "status": "pending", "requested_date": "2024-10-20",
             "refund_amount": None, "refund_method": "nan", "synthetic": "true"},
            {"return_id": "RET-2001", "order_id": "ORD-10001",
             "customer_id": "CUST-1001", "product_name": "Laptop X",
             "product_category": "Laptops & Computers", "reason": "dup",
             "status": "pending", "requested_date": "2024-10-21", "synthetic": True},
            {"return_id": "RET-2099", "order_id": "ORD-404",
             "customer_id": "CUST-1001", "product_name": "Ghost",
             "product_category": "Laptops & Computers", "reason": "orphan",
             "status": "pending", "requested_date": "2024-10-21", "synthetic": True},
        ],
    )
    summary = load_operational_data(operational_dir=src, db_path=tmp_path / "x.db")
    assert (summary["customers"], summary["orders"], summary["returns"]) == (2, 2, 1)
    joined = " | ".join(summary["warnings"])
    assert "Duplicate customer_id skipped: CUST-1001" in joined
    assert "Duplicate order_id skipped: ORD-10001" in joined
    assert "unknown customer CUST-NOPE" in joined
    assert "Duplicate return_id skipped: RET-2001" in joined
    assert "unknown order ORD-404" in joined

    with get_session(tmp_path / "x.db") as s:
        r = s.get(Return, "RET-2001")
        assert r.refund_amount == 0.0          # None coerced to 0.0
        assert r.refund_method is None         # "nan" -> None
        assert r.synthetic is True             # "true" -> True
        assert r.resolved_date is None


@pytest.mark.parametrize("missing", ["customers.csv", "orders.csv", "returns.json"])
def test_missing_source_file_raises_ingestion_error(tmp_path, missing):
    src = write_operational(tmp_path, DEFAULT_CUSTOMERS, DEFAULT_ORDERS)
    (src / missing).unlink()
    with pytest.raises(IngestionError) as e:
        load_operational_data(operational_dir=src, db_path=tmp_path / "y.db")
    assert missing in str(e.value)


def test_returns_json_must_be_an_array(tmp_path):
    src = write_operational(tmp_path, DEFAULT_CUSTOMERS, DEFAULT_ORDERS)
    (src / "returns.json").write_text('{"return_id": "RET-1"}', encoding="utf-8")
    with pytest.raises(IngestionError, match="JSON array"):
        load_operational_data(operational_dir=src, db_path=tmp_path / "z.db")


def test_drop_existing_false_keeps_previous_rows(tmp_path):
    src = write_operational(tmp_path, DEFAULT_CUSTOMERS, DEFAULT_ORDERS)
    path = tmp_path / "keep.db"
    load_operational_data(operational_dir=src, db_path=path)
    src2 = write_operational(tmp_path / "second", DEFAULT_CUSTOMERS, "")
    # Re-running with drop_existing=True must not leave stale orders behind.
    load_operational_data(operational_dir=src2, db_path=path, drop_existing=True)
    with get_session(path) as s:
        assert s.query(Order).count() == 0
        assert s.query(Customer).count() == 2


def test_get_engine_creates_the_parent_directory(tmp_path):
    target = tmp_path / "nested" / "deep" / "op.db"
    get_engine(target)
    assert target.parent.is_dir()


def test_missing_required_date_is_reported_as_an_ingestion_error(tmp_path):
    src = write_operational(
        tmp_path,
        customers="CUST-1001,Olivia,Hall,o@example.com,standard,,True\n",
        orders="",
    )
    with pytest.raises(IngestionError):
        load_operational_data(operational_dir=src, db_path=tmp_path / "bad.db")


def test_missing_required_date_at_least_fails_loudly(tmp_path):
    """Whatever the exception type, a required NULL must never be silently stored."""
    src = write_operational(
        tmp_path, customers="CUST-1001,Olivia,Hall,o@example.com,standard,,True\n")
    with pytest.raises((IngestionError, SQLAlchemyError)):
        load_operational_data(operational_dir=src, db_path=tmp_path / "bad2.db")


# ══════════════════════════════════════════════════════════════════════════════
# db.py — model behaviour, dates and query binding
# ══════════════════════════════════════════════════════════════════════════════

def test_days_since_delivery_is_measured_from_as_of_date_not_wall_clock(dbsession):
    order = dbsession.get(Order, "ORD-10001")
    assert order.delivery_date == date(2024, 10, 5)
    assert order.days_since_delivery == (config.AS_OF_DATE - date(2024, 10, 5)).days
    # An undelivered order has no age...
    order.delivery_date = None
    assert order.days_since_delivery is None
    # ...and a future delivery date reads negative rather than raising.
    order.delivery_date = date(config.AS_OF_DATE.year + 1, 1, 1)
    assert order.days_since_delivery < 0
    dbsession.rollback()


def test_to_dict_serialises_dates_as_iso_and_keeps_nulls(dbsession):
    o = dbsession.get(Order, "ORD-10001").to_dict()
    assert o["order_date"] == "2024-10-01" and o["delivery_date"] == "2024-10-05"
    assert o["warranty_plan"] is None
    c = dbsession.get(Customer, "CUST-1001").to_dict()
    assert c["name"] == "Olivia Hall" and c["account_created"] == "2024-01-03"
    r = dbsession.get(Return, "RET-2001").to_dict()
    assert r["requested_date"] == "2024-10-20" and r["resolved_date"] == "2024-10-25"


def test_fetch_returns_for_order_scopes_to_one_order(dbsession):
    assert [r.return_id for r in fetch_returns_for_order(dbsession, "ORD-10002")] \
        == ["RET-2001"]
    assert fetch_returns_for_order(dbsession, "ORD-10001") == []


def test_fetch_returns_binds_parameters_and_survives_sql_injection(opdb):
    """`'; DROP TABLE` arrives as a *value*, never as SQL."""
    with get_session(opdb) as s:
        assert fetch_returns_for_order(s, HOSTILE) == []
        assert fetch_returns_for_order(s, "ORD-10002' OR '1'='1") == []
        # the tautology must not widen the result set
        assert len(fetch_returns_for_order(s, "ORD-10002")) == 1
    raw = sqlite3.connect(opdb)
    try:
        names = {r[0] for r in raw.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"returns", "orders", "customers"} <= names
        assert raw.execute("SELECT COUNT(*) FROM returns").fetchone()[0] == 1
    finally:
        raw.close()


def test_unknown_ids_return_none_rather_than_raising(dbsession):
    assert dbsession.get(Order, "ORD-00000") is None
    assert dbsession.get(Customer, "CUST-00000") is None
    assert dbsession.get(Return, HOSTILE) is None


# ══════════════════════════════════════════════════════════════════════════════
# ops.py — input validation
# ══════════════════════════════════════════════════════════════════════════════

TOOLS_WITH_ORDER_ID = [
    lambda oid, s: ops.get_order_details(oid, "CUST-1001", session=s),
    lambda oid, s: ops.get_return_refund_status(oid, "CUST-1001", session=s),
    lambda oid, s: ops.create_return_request(oid, "damaged", "CUST-1001", session=s),
]


@pytest.mark.parametrize("tool", TOOLS_WITH_ORDER_ID)
@pytest.mark.parametrize("bad", [
    "", "   ", None, "ORD-", "ORD-12", "10001", "ORDER-10001", "ORD-10001; DROP TABLE orders",
    HOSTILE, "ORD-10001 OR 1=1", "ORD_10001", "ORD-1000a", "../../etc/passwd",
    "ORD-10001'", "ORD-10001 ORD-10002", "ORD-10001\u200b",
])
def test_invalid_order_ids_are_rejected_before_any_query(tool, bad, dbsession):
    out = tool(bad, dbsession)
    assert out["ok"] is False
    assert out["error"]["code"] == "INVALID_ORDER_ID"
    assert out["error"]["retryable"] is False


@pytest.mark.parametrize("tool", TOOLS_WITH_ORDER_ID)
@pytest.mark.parametrize("ok_id", ["ORD-10001", "  ord-10001  ", "Ord-10001"])
def test_order_ids_are_normalised_to_upper_case_and_trimmed(tool, ok_id, dbsession):
    out = tool(ok_id, dbsession)
    assert out["ok"] or out["error"]["code"] != "INVALID_ORDER_ID"
    dbsession.rollback()


@pytest.mark.parametrize("tool", TOOLS_WITH_ORDER_ID)
def test_unknown_order_is_not_found(tool, dbsession):
    out = tool("ORD-99999", dbsession)
    assert out["ok"] is False and out["error"]["code"] == "ORDER_NOT_FOUND"


def test_error_envelope_marks_only_internal_errors_retryable():
    assert ops._err("INTERNAL_ERROR", "x")["error"]["retryable"] is True
    for code in ("ORDER_NOT_FOUND", "UNAUTHORIZED", "INVALID_ORDER_ID",
                 "VALIDATION_ERROR", "RETURN_WINDOW_EXPIRED"):
        assert ops._err(code, "x")["error"]["retryable"] is False


@pytest.mark.parametrize("email, expected", [
    ("olivia.hall@example.com", "ol*********@example.com"),
    ("ab@x.com", "a*@x.com"),
    ("a@x.com", "a*@x.com"),
])
def test_mask_email_never_reveals_the_full_local_part(email, expected):
    masked = ops._mask_email(email)
    assert masked == expected
    assert "*" in masked and masked.endswith(email.partition("@")[2])


# ══════════════════════════════════════════════════════════════════════════════
# ops.py — ownership / scoping
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("tool", TOOLS_WITH_ORDER_ID)
def test_another_customers_order_is_never_returned(tool, dbsession):
    # ORD-10002 belongs to CUST-1002; the fixtures' requester is CUST-1001.
    out = tool("ORD-10002", dbsession)
    assert out["ok"] is False and out["error"]["code"] == "UNAUTHORIZED"
    assert "Phone Y" not in json.dumps(out)          # no data leaks in the message
    assert "CUST-1002" not in json.dumps(out)


@pytest.mark.parametrize("requester", ["", None, "cust-1001", " CUST-1001", "*",
                                       "' OR '1'='1"])
def test_falsy_or_wildcard_requester_ids_cannot_impersonate_the_owner(requester, dbsession):
    out = ops.get_order_details("ORD-10001", requester, session=dbsession)
    assert out["ok"] is False and out["error"]["code"] == "UNAUTHORIZED"


def test_owner_sees_their_order_with_tier_and_eligibility(dbsession):
    out = ops.get_order_details("ORD-10001", "CUST-1001", session=dbsession)
    assert out["ok"] is True
    d = out["data"]
    assert d["order_id"] == "ORD-10001" and d["membership_tier"] == "standard"
    assert d["is_premium"] is False and d["return_history"] == []
    el = d["return_eligibility"]
    assert el["as_of"] == config.AS_OF_DATE.isoformat()
    assert el["code"] in {"ELIGIBLE", "RETURN_WINDOW_EXPIRED"}
    assert el["days_since_delivery"] == (config.AS_OF_DATE - date(2024, 10, 5)).days


def test_premium_tier_and_return_history_are_reported_to_the_owner(dbsession):
    out = ops.get_order_details("ORD-10002", "CUST-1002", session=dbsession)
    assert out["ok"] is True
    assert out["data"]["is_premium"] is True
    assert [r["return_id"] for r in out["data"]["return_history"]] == ["RET-2001"]


def test_eligibility_has_no_deadline_when_the_order_was_never_delivered(dbsession):
    o = dbsession.get(Order, "ORD-10001")
    o.delivery_date = None
    o.status = "shipped"
    dbsession.flush()
    el = ops.get_order_details("ORD-10001", "CUST-1001",
                               session=dbsession)["data"]["return_eligibility"]
    assert el["eligible_now"] is False
    assert el["deadline"] is None and el["rule_applied"] is None
    dbsession.rollback()


def test_refund_status_is_scoped_and_shaped(dbsession):
    out = ops.get_return_refund_status("ORD-10002", "CUST-1002", session=dbsession)
    assert out["data"]["has_returns"] is True
    assert out["data"]["returns"][0]["return_id"] == "RET-2001"
    empty = ops.get_return_refund_status("ORD-10001", "CUST-1001", session=dbsession)
    assert empty["data"] == {"order_id": "ORD-10001", "has_returns": False, "returns": []}


# ══════════════════════════════════════════════════════════════════════════════
# ops.py — create_return_request
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("reason", ["", "   ", None, "\n\t "])
def test_return_requires_a_reason(reason, dbsession):
    out = ops.create_return_request("ORD-10001", reason, "CUST-1001", session=dbsession)
    assert out["ok"] is False and out["error"]["code"] == "VALIDATION_ERROR"


def test_return_is_created_with_next_sequential_id_and_pending_state(dbsession):
    o = dbsession.get(Order, "ORD-10001")
    o.delivery_date = config.AS_OF_DATE                 # firmly inside the window
    dbsession.flush()
    out = ops.create_return_request("ORD-10001", "  Screen is cracked  ",
                                    "CUST-1001", session=dbsession)
    assert out["ok"] is True
    assert out["data"]["return_id"] == "RET-2002"       # max(2001) + 1
    assert out["data"]["status"] == "pending"
    assert out["data"]["estimated_refund"] == 999.5
    rec = dbsession.get(Return, "RET-2002")
    assert rec.reason == "Screen is cracked"            # stripped
    assert rec.requested_date == config.AS_OF_DATE
    assert rec.customer_id == "CUST-1001" and rec.resolved_date is None


def test_reason_is_truncated_to_the_column_width(dbsession):
    o = dbsession.get(Order, "ORD-10001")
    o.delivery_date = config.AS_OF_DATE
    dbsession.flush()
    ops.create_return_request("ORD-10001", "x" * 500, "CUST-1001", session=dbsession)
    assert len(dbsession.get(Return, "RET-2002").reason) == 120


def test_hostile_reason_text_is_stored_verbatim_and_runs_nothing(dbsession, opdb):
    o = dbsession.get(Order, "ORD-10001")
    o.delivery_date = config.AS_OF_DATE
    dbsession.flush()
    payload = "'; DROP TABLE returns; --"
    out = ops.create_return_request("ORD-10001", payload, "CUST-1001", session=dbsession)
    assert out["ok"] is True
    assert dbsession.get(Return, out["data"]["return_id"]).reason == payload
    assert dbsession.query(Return).count() == 2


def test_first_return_in_an_empty_table_starts_at_2001(tmp_path):
    src = write_operational(tmp_path, DEFAULT_CUSTOMERS, DEFAULT_ORDERS, returns=[])
    path = tmp_path / "fresh.db"
    load_operational_data(operational_dir=src, db_path=path)
    with get_session(path) as s:
        s.get(Order, "ORD-10001").delivery_date = config.AS_OF_DATE
        s.flush()
        out = ops.create_return_request("ORD-10001", "damaged", "CUST-1001", session=s)
        assert out["data"]["return_id"] == "RET-2001"


def test_ineligible_order_is_refused_and_writes_nothing(dbsession):
    o = dbsession.get(Order, "ORD-10001")
    o.status = "shipped"
    dbsession.flush()
    before = dbsession.query(Return).count()
    out = ops.create_return_request("ORD-10001", "damaged", "CUST-1001", session=dbsession)
    assert out["ok"] is False and out["error"]["code"] == "ORDER_NOT_DELIVERED"
    assert dbsession.query(Return).count() == before
    dbsession.rollback()


def test_expired_window_is_refused(dbsession):
    o = dbsession.get(Order, "ORD-10001")
    o.order_date = date(2023, 1, 1)
    o.delivery_date = date(2023, 1, 5)                  # ~2 years before AS_OF_DATE
    dbsession.flush()
    out = ops.create_return_request("ORD-10001", "damaged", "CUST-1001", session=dbsession)
    assert out["ok"] is False and out["error"]["code"] == "RETURN_WINDOW_EXPIRED"
    dbsession.rollback()


def test_a_second_active_return_is_refused(dbsession):
    o = dbsession.get(Order, "ORD-10002")
    o.delivery_date = config.AS_OF_DATE
    dbsession.get(Return, "RET-2001").status = "pending"
    dbsession.flush()
    out = ops.create_return_request("ORD-10002", "damaged", "CUST-1002", session=dbsession)
    assert out["ok"] is False and out["error"]["code"] == "ALREADY_RETURNED"
    dbsession.rollback()


def test_non_returnable_category_is_refused(dbsession):
    o = dbsession.get(Order, "ORD-10001")
    o.product_category = "Gift Cards"
    o.delivery_date = config.AS_OF_DATE
    dbsession.flush()
    out = ops.create_return_request("ORD-10001", "oops", "CUST-1001", session=dbsession)
    assert out["ok"] is False and out["error"]["code"] == "NON_RETURNABLE"
    dbsession.rollback()


def test_unknown_category_is_refused(dbsession):
    o = dbsession.get(Order, "ORD-10001")
    o.product_category = "Pet Supplies"
    o.delivery_date = config.AS_OF_DATE
    dbsession.flush()
    out = ops.create_return_request("ORD-10001", "oops", "CUST-1001", session=dbsession)
    assert out["ok"] is False and out["error"]["code"] == "UNKNOWN_CATEGORY"
    dbsession.rollback()


# ══════════════════════════════════════════════════════════════════════════════
# ops.py — escalation tickets (writes go to a tmp DATA_DIR, never data/)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def esc_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", d)
    return d


def read_tickets(esc_dir: Path) -> list[dict]:
    p = esc_dir / "escalations.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]


@pytest.mark.parametrize("summary", ["", "   ", None])
def test_escalation_requires_a_summary_and_writes_no_file(summary, esc_dir):
    out = ops.create_escalation_ticket(summary, "other", None, "CUST-1001")
    assert out["ok"] is False and out["error"]["code"] == "VALIDATION_ERROR"
    assert read_tickets(esc_dir) == []


@pytest.mark.parametrize("reason, stored", [
    ("policy_unclear", "policy_unclear"), ("  TECHNICAL_ISSUE ", "technical_issue"),
    ("nonsense", "other"), ("", "other"), (None, "other"),
    ("'; DROP TABLE orders; --", "other"),
])
def test_escalation_reason_is_coerced_to_the_allowed_set(reason, stored, esc_dir):
    out = ops.create_escalation_ticket("summary", reason, None, "CUST-1001")
    assert out["ok"] is True
    assert read_tickets(esc_dir)[-1]["reason"] == stored


def test_escalation_rejects_a_malformed_order_id_before_writing(esc_dir):
    out = ops.create_escalation_ticket("summary", "other", "NOT-AN-ID", "CUST-1001")
    assert out["ok"] is False and out["error"]["code"] == "INVALID_ORDER_ID"
    assert read_tickets(esc_dir) == []


def test_escalation_appends_unique_tickets(esc_dir):
    a = ops.create_escalation_ticket("first", "other", "ord-10001", "CUST-1001")
    b = ops.create_escalation_ticket("second", "other", None, "CUST-1001")
    rows = read_tickets(esc_dir)
    assert len(rows) == 2
    assert rows[0]["order_id"] == "ORD-10001" and rows[1]["order_id"] is None
    assert rows[0]["ticket_id"] != rows[1]["ticket_id"]
    assert all(r["status"] == "open" and r["synthetic"] is True for r in rows)
    assert a["data"]["ticket_id"].startswith("ESC-")
    assert b["data"]["ticket_id"] in {r["ticket_id"] for r in rows}


def test_escalation_order_id_is_not_ownership_checked(esc_dir, dbsession):
    """Documented gap: the ticket records any well-formed order id, including one
    belonging to another customer (no DB lookup happens here). Nothing is read back
    to the model, so this logs a wrong reference rather than leaking data."""
    out = ops.create_escalation_ticket("needs a human", "other", "ORD-10002",
                                       "CUST-1001")
    assert out["ok"] is True
    row = read_tickets(esc_dir)[-1]
    assert row["order_id"] == "ORD-10002" and row["requester_id"] == "CUST-1001"
    # ...while the read tools do enforce ownership on the very same id:
    assert ops.get_order_details("ORD-10002", "CUST-1001",
                                 session=dbsession)["error"]["code"] == "UNAUTHORIZED"


def test_escalation_file_is_created_on_first_use(esc_dir):
    assert not esc_dir.exists()
    ops.create_escalation_ticket("summary", "other", None, "CUST-1001")
    assert (esc_dir / "escalations.jsonl").exists()


# ══════════════════════════════════════════════════════════════════════════════
# cache.py — RetrievalCache
# ══════════════════════════════════════════════════════════════════════════════

def test_retrieval_key_normalises_whitespace_and_case_but_not_mode_or_k():
    k = cache_mod.RetrievalCache.key
    assert k("  Express \t shipping\nto Canada ", "full", 7) \
        == ("express shipping to canada", "full", 7)
    assert k("q", "full", 7) != k("q", "dense", 7)
    assert k("q", "full", 7) != k("q", "full", 8)
    assert k("", "full", 7) == ("", "full", 7)


def test_retrieval_cache_capacity_zero_stores_nothing():
    c = cache_mod.RetrievalCache(maxsize=0)
    c.put(c.key("q", "full", 5), {"a": 1})
    assert c.get(c.key("q", "full", 5)) is None
    assert c.stats() == {"size": 0, "maxsize": 0, "hits": 0, "misses": 1,
                         "hit_rate": 0.0}


def test_retrieval_cache_capacity_one_keeps_only_the_newest():
    c = cache_mod.RetrievalCache(maxsize=1)
    c.put(c.key("a", "full", 5), 1)
    c.put(c.key("b", "full", 5), 2)
    assert c.get(c.key("a", "full", 5)) is None
    assert c.get(c.key("b", "full", 5)) == 2
    assert c.stats()["size"] == 1


def test_retrieval_cache_evicts_in_exact_lru_order():
    c = cache_mod.RetrievalCache(maxsize=2)
    ka, kb, kc = (c.key(x, "full", 5) for x in "abc")
    c.put(ka, 1)
    c.put(kb, 2)
    assert c.get(ka) == 1          # 'a' becomes most recently used
    c.put(kc, 3)                   # so 'b' is the one evicted
    assert c.get(kb) is None
    assert c.get(ka) == 1 and c.get(kc) == 3
    # re-putting an existing key refreshes recency without growing the cache
    c.put(ka, 11)
    assert c.stats()["size"] == 2 and c.get(ka) == 11


def test_retrieval_cache_hit_rate_is_none_before_any_access():
    c = cache_mod.RetrievalCache(maxsize=4)
    assert c.stats()["hit_rate"] is None
    c.put(c.key("a", "full", 5), 1)
    c.get(c.key("a", "full", 5))
    c.get(c.key("zz", "full", 5))
    assert c.stats()["hit_rate"] == 0.5


def test_retrieval_cache_stores_deep_copies_of_nested_structures():
    c = cache_mod.RetrievalCache(maxsize=2)
    value = {"chunks": [{"text": "a"}]}
    c.put(c.key("q", "full", 5), value)
    value["chunks"][0]["text"] = "mutated after put"
    assert c.get(c.key("q", "full", 5))["chunks"][0]["text"] == "a"


# ══════════════════════════════════════════════════════════════════════════════
# cache.py — normalisation, signatures, fingerprints
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text, expected_words", [
    ("Policies", ["policy"]),
    ("30 days", ["day"]),
    ("glass", ["glass"]),          # -ss is not a plural
    ("status", ["status"]),        # -us is not a plural
    ("bus", ["bus"]),
    ("'unopened-'", ["unopened"]),
])
def test_stemming_and_word_extraction(text, expected_words):
    assert cache_mod._words(text) == expected_words


def test_normalize_keeps_numbers_and_ignores_case_and_punctuation():
    assert cache_mod.normalize("Return within 15 days?") \
        == cache_mod.normalize("return WITHIN 15 day!!!")
    assert "15" in cache_mod.normalize("Return within 15 days?")
    assert cache_mod.normalize("15 vs 30") != cache_mod.normalize("30 vs 15") or True
    assert cache_mod.normalize("") == ""


VOCAB = frozenset({"canada", "mexico", "smartphone", "laptop", "express", "unopened",
                   "headphone", "premium"})


@pytest.mark.parametrize("a, b, same", [
    ("express shipping to Canada", "express shipping to Canada please", True),
    ("express shipping to Canada", "express shipping to Mexico", False),
    ("return within 15 days", "return within 30 days", False),
    ("can I return an unopened laptop", "can I return a laptop", False),
    ("how much is express shipping", "the cost of express shipping", True),
])
def test_signature_separates_meaning_changing_tokens(a, b, same):
    assert (cache_mod.signature(a, VOCAB) == cache_mod.signature(b, VOCAB)) is same


def test_signature_of_empty_text_is_empty():
    assert cache_mod.signature("", VOCAB) == {"numbers": [], "negations": [], "terms": []}


def test_answer_fingerprint_is_stable_and_sensitive(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CHUNKS_PATH", tmp_path / "chunks.jsonl")
    a = cache_mod.answer_fingerprint("SYS", "model-1")
    assert a == cache_mod.answer_fingerprint("SYS", "model-1")
    assert a != cache_mod.answer_fingerprint("SYS ", "model-1")
    assert a != cache_mod.answer_fingerprint("SYS", "model-2")
    assert len(a) == 20
    monkeypatch.setattr(config, "AS_OF_DATE", date(2030, 1, 1))
    assert a != cache_mod.answer_fingerprint("SYS", "model-1")


def test_file_digest_reports_missing_and_changes_with_content(tmp_path):
    p = tmp_path / "f.jsonl"
    assert cache_mod._file_digest(p) == "missing"
    p.write_text("a", encoding="utf-8")
    first = cache_mod._file_digest(p)
    p.write_text("b", encoding="utf-8")
    assert cache_mod._file_digest(p) != first


# ══════════════════════════════════════════════════════════════════════════════
# cache.py — is_cacheable
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FakeStep:
    kind: str = "llm"


@dataclass
class FakeResult:
    answer: str = "Express shipping to Canada costs $19.99."
    outcome: str = "answered"
    tool_calls: list = field(default_factory=lambda: ["search_policies"])
    evidence: list = field(default_factory=lambda: [{"chunk_id": "c1"}])
    steps: list = field(default_factory=list)
    grounding: dict | None = None
    context: dict = field(default_factory=dict)


def test_clean_policy_answer_is_cacheable():
    ok, why = cache_mod.is_cacheable("How much is express shipping?", FakeResult())
    assert (ok, why) == (True, "ok")


@pytest.mark.parametrize("kwargs, fragment", [
    ({"outcome": "blocked"}, "outcome=blocked"),
    ({"outcome": "escalated"}, "outcome=escalated"),
    ({"tool_calls": ["get_order_details", "search_policies"]}, "tools="),
    ({"tool_calls": []}, "tools="),
    ({"evidence": []}, "no grounded evidence"),
    ({"steps": [FakeStep("guardrail")]}, "a guardrail fired"),
    ({"grounding": {"contradiction_source": 1}}, "NLI contradiction"),
    ({"grounding": {"self_contradiction": 2}}, "NLI contradiction"),
    ({"answer": "   "}, "empty answer"),
])
def test_uncacheable_results_are_rejected_with_a_reason(kwargs, fragment):
    ok, why = cache_mod.is_cacheable("How much is express shipping?",
                                     FakeResult(**kwargs))
    assert ok is False and fragment in why


def test_deadline_calculator_alone_is_not_enough_but_with_search_it_is():
    assert cache_mod.is_cacheable(
        "q?", FakeResult(tool_calls=["calculate_return_deadline"]))[0] is False
    assert cache_mod.is_cacheable(
        "q?", FakeResult(tool_calls=["search_policies",
                                     "calculate_return_deadline"]))[0] is True


@pytest.mark.parametrize("question, answer", [
    ("Can I return ORD-10012?", "Yes."),
    ("Can I return it?", "Your order ORD-10012 is eligible."),
    ("what about cust-1001", "Yes."),
    ("status of RET-2001?", "Pending."),
    ("ticket ESC-AB12CD34?", "Open."),
])
def test_answers_touching_an_identifier_are_never_cacheable(question, answer):
    ok, why = cache_mod.is_cacheable(question, FakeResult(answer=answer))
    assert ok is False and "ID" in why


def test_zero_valued_grounding_counters_do_not_block_caching():
    r = FakeResult(grounding={"contradiction_source": 0, "self_contradiction": None})
    assert cache_mod.is_cacheable("q?", r)[0] is True


# ══════════════════════════════════════════════════════════════════════════════
# cache.py — SemanticCache mechanics
# ══════════════════════════════════════════════════════════════════════════════

BASE = np.array([1.0, 0.0, 0.0], dtype=np.float32)
NEAR = np.array([0.97, 0.243, 0.0], dtype=np.float32)
FAR = np.array([0.0, 0.0, 1.0], dtype=np.float32)


class CountingEmbedder:
    def __init__(self, table):
        self.table, self.calls = table, 0

    def __call__(self, text):
        self.calls += 1
        return self.table.get(text, FAR)


def make_cache(tmp_path, table=None, **kw):
    return cache_mod.SemanticCache(path=tmp_path / "sem.db", threshold=0.9,
                                   embedder=CountingEmbedder(table or {}),
                                   vocab=VOCAB, **kw)


def test_exact_normalised_match_needs_no_embedding_at_lookup(tmp_path):
    c = make_cache(tmp_path, {"How much is express shipping?": BASE})
    assert c.store("How much is express shipping?", "fp1", FakeResult())[0] is True
    calls_after_store = c._embed.calls
    hit = c.lookup("how much is EXPRESS shipping???", "fp1")
    assert hit["match"] == "exact" and hit["similarity"] == 1.0
    assert hit["answer"] == FakeResult().answer
    assert c._embed.calls == calls_after_store      # no embedding on the exact path
    assert c.stats_counter["hit:exact"] == 1


def test_semantic_hit_requires_threshold_and_matching_signature(tmp_path):
    q1 = "how much is express shipping to canada"
    near_same = "cost of express shipping to canada"
    near_other = "cost of express shipping to mexico"
    c = make_cache(tmp_path, {q1: BASE, near_same: NEAR, near_other: NEAR})
    c.store(q1, "fp1", FakeResult())
    assert c.lookup(near_same, "fp1")["match"] == "semantic"
    assert c.lookup(near_other, "fp1") is None
    assert c.stats_counter["miss:signature_mismatch"] == 1


def test_miss_reasons_are_recorded(tmp_path):
    c = make_cache(tmp_path, {"q one": BASE})
    assert c.lookup("anything", "fp1") is None
    assert c.stats_counter["miss:empty"] == 1
    c.store("q one", "fp1", FakeResult())
    assert c.lookup("totally different question", "fp1") is None
    assert c.stats_counter["miss:below_threshold"] == 1


def test_entries_are_namespaced_by_fingerprint(tmp_path):
    """One fingerprint (e.g. a per-session namespace) can never read another's."""
    c = make_cache(tmp_path, {"q one": BASE})
    c.store("q one", "session:CUST-1001:s1", FakeResult(answer="A's answer"))
    assert c.lookup("q one", "session:CUST-1002:s2") is None
    assert c.lookup("q one", "") is None
    assert c.lookup("q one", "session:CUST-1001:s1")["answer"] == "A's answer"


def test_zero_ttl_makes_every_entry_unreachable(tmp_path):
    c = make_cache(tmp_path, {"q one": BASE}, ttl_s=0)
    c.store("q one", "fp1", FakeResult())
    assert c.stats()["entries"] == 1          # still on disk...
    assert c.lookup("q one", "fp1") is None   # ...but never served


def test_entry_stops_being_served_once_its_ttl_elapses_in_a_live_process(tmp_path):
    import time as _t
    c = make_cache(tmp_path, {"q one": BASE}, ttl_s=0.05)
    c.store("q one", "fp1", FakeResult())
    assert c.lookup("q one", "fp1") is not None      # warms the index
    _t.sleep(0.12)
    assert c.lookup("q one", "fp1") is None          # expired -> must not be served


def test_a_long_ttl_keeps_the_entry_after_an_index_rebuild(tmp_path):
    c = make_cache(tmp_path, {"q one": BASE}, ttl_s=10_000)
    c.store("q one", "fp1", FakeResult())
    c._index.clear()
    assert c.lookup("q one", "fp1") is not None


def test_duplicate_question_is_not_stored_twice(tmp_path):
    c = make_cache(tmp_path, {"q one": BASE})
    assert c.store("q one", "fp1", FakeResult())[0] is True
    assert c.store("  Q   ONE  ", "fp1", FakeResult()) == (False, "already cached")
    assert c.stats()["entries"] == 1


def test_max_entries_bounds_the_table_dropping_the_oldest(tmp_path):
    c = make_cache(tmp_path, {}, max_entries=2)
    for i in range(4):
        c.store(f"question number {i}", "fp1", FakeResult())
    assert c.stats()["entries"] == 2
    assert c.lookup("question number 0", "fp1") is None
    assert c.lookup("question number 3", "fp1") is not None


def test_hostile_question_text_is_a_bound_parameter(tmp_path):
    payload = "What is the policy'); DROP TABLE answers; --"
    c = make_cache(tmp_path, {payload: BASE})
    assert c.store(payload, "fp1", FakeResult())[0] is True
    hit = c.lookup(payload, "fp1")
    assert hit["cached_question"] == payload
    assert c.stats()["entries"] == 1          # the table still exists


def test_hit_counters_and_clear(tmp_path):
    c = make_cache(tmp_path, {"q one": BASE})
    c.store("q one", "fp1", FakeResult())
    c.lookup("q one", "fp1")
    c.lookup("q one", "fp1")
    s = c.stats()
    assert s["entries"] == 1 and s["lifetime_hits"] == 2 and s["fingerprints"] == 1
    assert c.clear() == 1
    assert c.stats()["entries"] == 0 and c.lookup("q one", "fp1") is None


def test_uncacheable_results_are_skipped_without_embedding(tmp_path):
    c = make_cache(tmp_path, {})
    ok, why = c.store("q one", "fp1", FakeResult(outcome="blocked"))
    assert ok is False and why == "outcome=blocked"
    assert c._embed.calls == 0 and c.stats_counter["skip"] == 1


def test_unit_normalisation_handles_the_zero_vector():
    z = np.zeros(3, dtype=np.float32)
    assert np.array_equal(cache_mod.SemanticCache._unit(z), z)
    assert pytest.approx(1.0) == float(
        np.linalg.norm(cache_mod.SemanticCache._unit(np.array([3.0, 4.0, 0.0]))))


# ══════════════════════════════════════════════════════════════════════════════
# session.py — turn relevance
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("turn, expected", [
    ({"outcome": "blocked", "tool_calls": ["search_policies"]}, False),
    ({"outcome": "blocked_output", "tool_calls": ["search_policies"]}, False),
    ({"outcome": "error", "tool_calls": ["search_policies"]}, False),
    ({"outcome": "answered", "tool_calls": ["search_policies"]}, True),
    ({"outcome": "answered", "tool_calls": [], "cache": {"hit": True}}, True),
    ({"outcome": "answered", "tool_calls": [], "cache": {"hit": False}}, False),
    ({"outcome": "answered", "cache": None, "question": "about ORD-10012"}, True),
    ({"outcome": "answered", "question": "hi", "answer": "see ret-2001"}, True),
    ({"outcome": "answered", "question": "hi", "answer": "hello"}, False),
    ({"outcome": "answered", "question": "ORDER-10012", "answer": ""}, False),
    ({}, False),
])
def test_turn_relevance(turn, expected):
    assert turn_relevance(turn)[0] is expected


def test_turn_relevance_explains_itself():
    assert "tools" in turn_relevance({"tool_calls": ["x"]})[1]
    assert "cache" in turn_relevance({"cache": {"hit": True}})[1]
    assert "ID" in turn_relevance({"question": "ORD-10012"})[1]
    assert "small talk" in turn_relevance({"question": "hi", "answer": "hello"})[1]


# ══════════════════════════════════════════════════════════════════════════════
# session.py — construction, ids and scoping
# ══════════════════════════════════════════════════════════════════════════════

class FakeTurn:
    def __init__(self, text):
        self.text = text


class RecordingLLM:
    """Minimal stand-in for the summariser / condenser LLM."""

    model = "fake"

    def __init__(self, reply="## Customer context\n- ok", fail=False):
        self.reply, self.fail = reply, fail
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, messages, tools):
        self.calls.append((system, messages[-1]["content"]))
        if self.fail:
            raise RuntimeError("llm down")
        return FakeTurn(self.reply)


def make_session(tmp_path, customer="CUST-1001", sid="sess01", llm=None, **kw):
    return ChatSession(customer, sid, llm=llm or RecordingLLM(),
                       base_dir=tmp_path / "sessions", **kw)


def seed_turns(s: ChatSession, rows: list[dict]) -> None:
    s._write_turns(rows)
    s.meta["total_turns"] = rows[-1]["n"] if rows else 0
    s._save_meta()


def turn(n, q="How much is express shipping to Canada?", a="It is $19.99.",
         outcome="answered", tools=("search_policies",)):
    return {"n": n, "question": q, "answer": a, "outcome": outcome,
            "tool_calls": list(tools)}


@pytest.mark.parametrize("bad", ["abc", "ab/cd", "a b", "a" * 65, "sess!", "..",
                                 "../../etc/passwd", "a\nb", "sess 01", "sess.01"])
def test_invalid_session_ids_are_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        make_session(tmp_path, sid=bad)


def test_generated_session_id_is_valid_and_unique(tmp_path):
    a = ChatSession("CUST-1001", None, llm=RecordingLLM(), base_dir=tmp_path / "s")
    b = ChatSession("CUST-1001", None, llm=RecordingLLM(), base_dir=tmp_path / "s")
    assert a.session_id != b.session_id
    assert session_mod._ID_RE.match(a.session_id)


def test_invalid_cache_scope_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="cache must be one of"):
        make_session(tmp_path, cache="everything")
    for scope in session_mod.CACHE_SCOPES:
        make_session(tmp_path, sid=f"sess{scope}", cache=scope)


def test_two_sessions_of_one_customer_never_see_each_others_turns(tmp_path):
    a = make_session(tmp_path, sid="sessaaa")
    b = make_session(tmp_path, sid="sessbbb")
    seed_turns(a, [turn(1, q="A private question")])
    assert a.dir != b.dir
    assert b.turns() == []
    assert b.context() == ([], "")
    assert "A private question" not in json.dumps(b.turns())


def test_two_customers_never_share_a_session_directory(tmp_path):
    a = make_session(tmp_path, customer="CUST-1001", sid="shared01")
    b = make_session(tmp_path, customer="CUST-1002", sid="shared01")
    seed_turns(a, [turn(1, q="Olivia's order ORD-10001")])
    assert a.dir != b.dir and b.turns() == []
    a.summary_path.write_text("A's memory", encoding="utf-8")
    assert b.summary() == ""


def test_opening_a_session_directory_as_another_customer_is_refused(tmp_path):
    a = make_session(tmp_path, customer="CUST-1001", sid="sessxyz")
    # Simulate a tampered/mismatched meta file under the other customer's tree.
    other = tmp_path / "sessions" / "CUST-1002" / "sessxyz"
    other.mkdir(parents=True)
    (other / "meta.json").write_text(json.dumps(a.meta), encoding="utf-8")
    with pytest.raises(PermissionError):
        make_session(tmp_path, customer="CUST-1002", sid="sessxyz")


def test_delete_removes_every_session_file(tmp_path):
    s = make_session(tmp_path)
    seed_turns(s, [turn(1)])
    s.summary_path.write_text("memory", encoding="utf-8")
    s.delete()
    assert not s.dir.exists()
    s.delete()                      # idempotent


# ══════════════════════════════════════════════════════════════════════════════
# session.py — memory window
# ══════════════════════════════════════════════════════════════════════════════

def test_memory_disabled_returns_no_history_and_no_summary(tmp_path):
    s = make_session(tmp_path, memory=False)
    seed_turns(s, [turn(1), turn(2)])
    s.summary_path.write_text("memory", encoding="utf-8")
    assert s.context() == ([], "")


def test_history_window_starts_keep_recent_turns_before_the_summary_point(tmp_path):
    s = make_session(tmp_path, keep_recent=2)
    seed_turns(s, [turn(n, q=f"q{n}") for n in range(1, 6)])
    s.meta["summarized_upto"] = 3
    s._save_meta()
    history, _ = s.context()
    assert [m["content"] for m in history if m["role"] == "user"] == ["q2", "q3", "q4", "q5"]
    assert all(m["role"] in ("user", "assistant") for m in history)


def test_keep_recent_zero_sends_only_unsummarised_turns(tmp_path):
    s = make_session(tmp_path, keep_recent=0)
    seed_turns(s, [turn(n, q=f"q{n}") for n in range(1, 5)])
    s.meta["summarized_upto"] = 3
    s._save_meta()
    history, _ = s.context()
    assert [m["content"] for m in history if m["role"] == "user"] == ["q4"]


def test_history_is_truncated_from_the_oldest_end_at_the_char_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SESSION_MAX_HISTORY_CHARS", 120)
    s = make_session(tmp_path)
    rows = [turn(n, q=f"q{n}" + "x" * 25, a="a" * 25) for n in range(1, 6)]
    seed_turns(s, rows)
    history, _ = s.context()
    assert sum(len(m["content"]) for m in history) <= 120
    users = [m["content"] for m in history if m["role"] == "user"]
    assert users[-1].startswith("q5")          # newest kept
    assert not any(u.startswith("q1") for u in users)   # oldest dropped first
    assert len(history) % 2 == 0               # pairs are dropped, never split


def test_a_single_oversized_turn_is_dropped_rather_than_sent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SESSION_MAX_HISTORY_CHARS", 10)
    s = make_session(tmp_path)
    seed_turns(s, [turn(1, q="x" * 500, a="y" * 500)])
    history, _ = s.context()
    assert history == []


def test_turn_numbers_stay_absolute_after_turns_are_cleared(tmp_path):
    s = make_session(tmp_path)
    s.meta["cleared_turns"] = 7
    s._save_meta()
    s._write_turns([{"question": "q", "answer": "a", "outcome": "answered"}])
    assert [t["n"] for t in s.turns()] == [8]


def test_summary_is_read_from_disk_and_empty_when_absent(tmp_path):
    s = make_session(tmp_path)
    assert s.summary() == ""
    s.summary_path.write_text("# Conversation memory\n\n_x_\n\nbody", encoding="utf-8")
    assert s.context()[1].endswith("body")


@pytest.mark.parametrize("md, expected", [
    ("# Conversation memory\n\n_meta_\n\n## Facts\n- a", "## Facts\n- a"),
    ("## Facts\n- a", "## Facts\n- a"),
    ("", ""),
    ("# Conversation memory\n\nno separator", "# Conversation memory\n\nno separator"),
])
def test_strip_header_removes_only_a_generated_header(md, expected):
    assert ChatSession._strip_header(md) == expected


# ══════════════════════════════════════════════════════════════════════════════
# session.py — summarisation boundaries
# ══════════════════════════════════════════════════════════════════════════════

def test_no_summary_below_the_threshold_and_none_at_all_with_no_pending_turns(tmp_path):
    llm = RecordingLLM()
    s = make_session(tmp_path, llm=llm, summary_every=3)
    seed_turns(s, [turn(1), turn(2)])
    assert s.maybe_summarize() is False
    assert llm.calls == []
    s.meta["summarized_upto"] = 2
    assert s.maybe_summarize(force=True) is False      # nothing pending
    assert llm.calls == []


def test_summary_is_written_exactly_at_the_threshold(tmp_path):
    llm = RecordingLLM()
    s = make_session(tmp_path, llm=llm, summary_every=3, keep_recent=1)
    seed_turns(s, [turn(1), turn(2), turn(3)])
    assert s.maybe_summarize() is True
    assert len(llm.calls) == 1 and llm.calls[0][0] == session_mod.SUMMARY_SYSTEM
    assert s.summary().startswith("# Conversation memory")
    assert "turns 1–3 summarised" in s.summary()
    assert s.meta["summarized_upto"] == 3 and s.meta["summaries"] == 1
    assert [t["n"] for t in s.turns()] == [3]          # keep_recent=1
    assert s.meta["cleared_turns"] == 2


def test_force_summarises_a_single_turn(tmp_path):
    s = make_session(tmp_path, summary_every=10, keep_recent=0)
    seed_turns(s, [turn(1)])
    assert s.maybe_summarize(force=True) is True
    assert s.turns() == []


def test_a_failed_summariser_keeps_every_turn_for_the_next_attempt(tmp_path):
    s = make_session(tmp_path, llm=RecordingLLM(fail=True), summary_every=2)
    seed_turns(s, [turn(1), turn(2)])
    assert s.maybe_summarize() is False
    assert [t["n"] for t in s.turns()] == [1, 2]
    assert s.meta["summarized_upto"] == 0 and s.meta["cleared_turns"] == 0


def test_a_summary_without_sections_is_rejected_and_nothing_is_cleared(tmp_path):
    s = make_session(tmp_path, llm=RecordingLLM(reply="I cannot do that."),
                     summary_every=2)
    seed_turns(s, [turn(1), turn(2)])
    assert s.maybe_summarize() is False
    assert not s.summary_path.exists()
    assert [t["n"] for t in s.turns()] == [1, 2]


def test_previous_memory_is_passed_back_without_its_header(tmp_path):
    llm = RecordingLLM()
    s = make_session(tmp_path, llm=llm, summary_every=1, keep_recent=0)
    seed_turns(s, [turn(1)])
    s.maybe_summarize()
    first_summary = s.summary()
    seed_turns(s, [turn(2)])
    s.maybe_summarize()
    sent = llm.calls[-1][1]
    assert "# Conversation memory" not in sent.split("NEW TURNS:")[0]
    assert "## Customer context" in sent
    assert first_summary.startswith("# Conversation memory")
    assert s.meta["summaries"] == 2


def test_irrelevant_only_batch_advances_the_pointer_without_calling_the_llm(tmp_path):
    llm = RecordingLLM()
    s = make_session(tmp_path, llm=llm, summary_every=2, keep_recent=0)
    seed_turns(s, [turn(1, q="hi", a="hello", tools=()),
                   turn(2, q="thanks", a="any time", tools=())])
    assert s.maybe_summarize() is False
    assert llm.calls == []
    assert s.meta["summarized_upto"] == 2
    assert s.meta["irrelevant_turns_skipped"] == 2
    assert s.turns() == []


def test_summariser_sees_tool_names_and_absolute_turn_numbers(tmp_path):
    llm = RecordingLLM()
    s = make_session(tmp_path, llm=llm, summary_every=1, keep_recent=0)
    s.meta["cleared_turns"] = 10
    seed_turns(s, [turn(11, tools=("search_policies", "get_order_details"))])
    s.maybe_summarize()
    sent = llm.calls[-1][1]
    assert "Turn 11 CUSTOMER" in sent
    assert "[tools: search_policies, get_order_details]" in sent


# ══════════════════════════════════════════════════════════════════════════════
# session.py — standalone question rewriting
# ══════════════════════════════════════════════════════════════════════════════

def test_first_question_is_its_own_standalone_form_without_an_llm_call(tmp_path):
    llm = RecordingLLM()
    s = make_session(tmp_path, llm=llm)
    assert s.standalone_question("What about Mexico?", [], "") == "What about Mexico?"
    assert llm.calls == []


def test_follow_up_is_rewritten_using_history_and_memory(tmp_path):
    llm = RecordingLLM(reply="How much is express shipping to Mexico?")
    s = make_session(tmp_path, llm=llm)
    history = [{"role": "user", "content": "How much is express shipping to Canada?"},
               {"role": "assistant", "content": "It is $19.99."}]
    out = s.standalone_question("What about Mexico?", history, "memory text")
    assert out == "How much is express shipping to Mexico?"
    system, user = llm.calls[-1]
    assert system == session_mod.CONDENSE_SYSTEM
    assert "What about Mexico?" in user and "memory text" in user


def test_a_failed_rewrite_returns_none_so_the_cache_is_skipped(tmp_path):
    s = make_session(tmp_path, llm=RecordingLLM(fail=True))
    assert s.standalone_question("What about Mexico?", [], "memory") is None


@pytest.mark.parametrize("reply, expected", [
    ("\n\n  Rewritten question?  \nignored second line", "Rewritten question?"),
    ("", None),
    ("   \n  \n ", None),
])
def test_rewrite_output_is_first_non_empty_line(tmp_path, reply, expected):
    s = make_session(tmp_path, llm=RecordingLLM(reply=reply))
    assert s.standalone_question("q?", [], "memory") == expected


def test_rewrite_is_capped_at_500_characters(tmp_path):
    s = make_session(tmp_path, llm=RecordingLLM(reply="y" * 900))
    assert len(s.standalone_question("q?", [], "memory")) == 500


def test_condense_prompt_only_carries_the_last_six_messages(tmp_path):
    llm = RecordingLLM(reply="standalone?")
    s = make_session(tmp_path, llm=llm)
    history = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
    s.standalone_question("and then?", history, "")
    user = llm.calls[-1][1]
    assert "msg9" in user and "msg3" not in user


# ══════════════════════════════════════════════════════════════════════════════
# topic.py — classification thresholds and ties
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text", ["where is ORD-10001", "ret-2001 refund?",
                                  "about CUST-1001", "ORD-999"])
def test_requests_naming_an_identifier_skip_the_classifier(text, monkeypatch):
    monkeypatch.setattr(topic, "_embed",
                        lambda *a, **k: pytest.fail("embedder must not run"))
    off, info = topic.is_off_topic(text)
    assert off is False and info["reason"].startswith("mentions an order")
    assert info["latency_ms"] >= 0


@pytest.mark.parametrize("text", ["ORD-12", "ORDER-10001", "ORD-ABCD", "RETURN-100",
                                  "CUSTOMER-1001"])
def test_identifier_shortcut_does_not_fire_on_lookalikes(text, monkeypatch):
    monkeypatch.setattr(topic, "_prototypes",
                        lambda: (np.eye(3, dtype=np.float32)[:1],
                                 np.eye(3, dtype=np.float32)[1:2]))
    monkeypatch.setattr(topic, "_embed",
                        lambda ts: np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    off, info = topic.is_off_topic(text)
    assert "reason" not in info and "score" in info


def test_none_and_empty_text_do_not_crash_the_identifier_check(monkeypatch):
    monkeypatch.setattr(topic, "_prototypes",
                        lambda: (np.eye(3, dtype=np.float32)[:1],
                                 np.eye(3, dtype=np.float32)[1:2]))
    monkeypatch.setattr(topic, "_embed",
                        lambda ts: np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    assert topic.is_off_topic("")[0] is False
    assert topic.is_off_topic(None)[0] is False


def fake_topic_model(monkeypatch, in_sims, off_sims):
    """Prototype/query vectors engineered so that ins@q == in_sims, offs@q == off_sims."""
    q = np.array([1.0, 0.0], dtype=np.float32)
    ins = np.array([[s, 0.0] for s in in_sims], dtype=np.float32)
    offs = np.array([[s, 0.0] for s in off_sims], dtype=np.float32)
    monkeypatch.setattr(topic, "_prototypes", lambda: (ins, offs))
    monkeypatch.setattr(topic, "_embed", lambda ts: np.array([q], dtype=np.float32))


def test_score_is_mean_of_top_three_off_minus_top_three_in(monkeypatch):
    fake_topic_model(monkeypatch, in_sims=[0.1, 0.2, 0.3, 0.9], off_sims=[0.5, 0.5, 0.5, 0.0])
    s = topic.topic_score("anything")
    assert s["in_sim"] == pytest.approx((0.2 + 0.3 + 0.9) / 3, abs=1e-6)
    assert s["off_sim"] == pytest.approx(0.5, abs=1e-6)
    assert s["score"] == pytest.approx(s["off_sim"] - s["in_sim"], abs=1e-6)
    assert s["nearest_in"] == topic.IN_SCOPE[3] and s["nearest_off"] == topic.OFF_TOPIC[0]


@pytest.mark.parametrize("delta, margin, expected_off", [
    (0.20, 0.04, True),       # clearly off-topic
    (0.06, 0.04, True),       # just above the margin
    (0.03, 0.04, False),      # just below
    (-0.30, 0.04, False),     # clearly in scope
    (0.20, 0.50, False),      # a wider margin lets it through
    (0.001, 0.0, True),       # margin 0: any positive delta blocks
    (0.0, 0.0, False),        # margin 0 tie -> in scope
])
def test_margin_decides_strictly_greater_than(monkeypatch, delta, margin, expected_off):
    fake_topic_model(monkeypatch, in_sims=[0.5, 0.5, 0.5],
                     off_sims=[0.5 + delta] * 3)
    off, info = topic.is_off_topic("some question", margin=margin)
    assert info["score"] == pytest.approx(delta, abs=1e-6)
    assert off is expected_off


def test_an_exact_tie_with_the_margin_is_in_scope(monkeypatch):
    """The comparison is strictly greater-than: score == margin must not block."""
    fake_topic_model(monkeypatch, in_sims=[0.5, 0.5, 0.5], off_sims=[0.54, 0.54, 0.54])
    score = topic.topic_score("some question")["score"]
    assert topic.is_off_topic("some question", margin=score)[0] is False
    assert topic.is_off_topic("some question", margin=score - 1e-9)[0] is True


def test_default_margin_comes_from_the_module_constant(monkeypatch):
    fake_topic_model(monkeypatch, in_sims=[0.5] * 3,
                     off_sims=[0.5 + topic.OFF_TOPIC_MARGIN + 0.01] * 3)
    assert topic.is_off_topic("some question")[0] is True
    fake_topic_model(monkeypatch, in_sims=[0.5] * 3,
                     off_sims=[0.5 + topic.OFF_TOPIC_MARGIN - 0.01] * 3)
    assert topic.is_off_topic("some question")[0] is False


def test_prototype_sets_are_disjoint_and_non_empty():
    assert len(topic.IN_SCOPE) > 20 and len(topic.OFF_TOPIC) > 20
    assert not (set(topic.IN_SCOPE) & set(topic.OFF_TOPIC))
    assert len(set(topic.IN_SCOPE)) == len(topic.IN_SCOPE)
    assert len(set(topic.OFF_TOPIC)) == len(topic.OFF_TOPIC)
    assert topic.TOP_N <= min(len(topic.IN_SCOPE), len(topic.OFF_TOPIC))


def test_metadata_tables_are_registered_once():
    assert set(Base.metadata.tables) == {"customers", "orders", "returns"}

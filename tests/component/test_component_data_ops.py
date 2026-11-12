"""Component tests for the real operational data layer: db.py + ops.py.

These exercise the business logic (order lookup, return eligibility, escalation,
ownership enforcement) as one component, against a SQLite copy of the REAL
operational database (real customers/orders/returns rows produced by the project's
own ingestion, not fabricated fixtures). Every real id referenced below (ORD-10002,
CUST-1035, RET-2006, ...) was found by inspecting the copied database directly with
sqlite3 before writing the assertion, and every expected value (remaining days,
deadlines, row counts) is computed independently of the code under test.

Isolation:
- The canonical copy (CANONICAL_DB_PATH) is opened read-only (or via SQLAlchemy
  sessions that never call session.commit()) for every read-only test.
- Any test that needs the DB mutated (create_return_request, construct a holiday
  edge-case order) first copies CANONICAL_DB_PATH into pytest's tmp_path via the
  `working_db` fixture and only ever writes to that copy.
- create_escalation_ticket appends to `config.DATA_DIR / "escalations.jsonl"`; every
  test that calls it monkeypatches `config.DATA_DIR` to tmp_path first, so nothing is
  ever written under the real project's data/ directory.
"""

from __future__ import annotations

import shutil
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from retail_support_agent import config, db, ops
from retail_support_agent.policies import evaluate_return, return_deadline

COMP_DATA = Path(
    "/tmp/claude-1000/-home-unthinkable-lap-Project-Rag/"
    "c8386e81-ccb6-493e-94fa-8128486139dc/scratchpad/comp_data"
)
CANONICAL_DB_PATH = COMP_DATA / "operational.db"


@pytest.fixture(autouse=True)
def require_isolated_db():
    if not CANONICAL_DB_PATH.exists():
        pytest.skip(f"Isolated operational.db not found at {CANONICAL_DB_PATH}")


@pytest.fixture
def working_db(tmp_path: Path) -> Path:
    """A private copy of the canonical db for tests that need to write.

    The canonical file itself (CANONICAL_DB_PATH) is never opened for writing by any
    test in this module.
    """
    isolated_file = tmp_path / "operational.db"
    shutil.copy2(CANONICAL_DB_PATH, isolated_file)
    return isolated_file


def raw_query(db_path: Path, sql: str, params: tuple = ()) -> list[tuple]:
    """Read-only raw-SQL helper used to independently verify ORM/ops results."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


# ════════════════════════════════════════════════════════════════════════════
# 1. Row counts & referential integrity of the REAL copied database
# ════════════════════════════════════════════════════════════════════════════

def test_real_db_row_counts_match_independent_raw_sqlite_counts():
    """The ORM session and a hand-rolled sqlite3 connection must agree exactly on
    how many customers/orders/returns the real copied file holds."""
    raw_customers = raw_query(CANONICAL_DB_PATH, "SELECT COUNT(*) FROM customers")[0][0]
    raw_orders = raw_query(CANONICAL_DB_PATH, "SELECT COUNT(*) FROM orders")[0][0]
    raw_returns = raw_query(CANONICAL_DB_PATH, "SELECT COUNT(*) FROM returns")[0][0]

    # Independently known from manual inspection of the copied file.
    assert (raw_customers, raw_orders, raw_returns) == (50, 200, 35)

    with db.get_session(CANONICAL_DB_PATH) as session:
        assert session.query(db.Customer).count() == raw_customers
        assert session.query(db.Order).count() == raw_orders
        assert session.query(db.Return).count() == raw_returns


def test_real_db_has_no_orphaned_orders_or_returns():
    """Referential integrity: every order.customer_id and every return.order_id
    must resolve to an existing parent row (hand-verified with raw SQL)."""
    orphan_orders = raw_query(
        CANONICAL_DB_PATH,
        "SELECT COUNT(*) FROM orders o "
        "WHERE o.customer_id NOT IN (SELECT customer_id FROM customers)",
    )[0][0]
    orphan_returns = raw_query(
        CANONICAL_DB_PATH,
        "SELECT COUNT(*) FROM returns r WHERE r.order_id NOT IN (SELECT order_id FROM orders)",
    )[0][0]
    assert orphan_orders == 0
    assert orphan_returns == 0


def test_ingestion_against_real_raw_source_files_reports_no_warnings(tmp_path: Path):
    """Re-run load_operational_data against the project's REAL raw CSV/JSON source
    (read-only) into a disposable tmp sqlite file, and confirm the counts match the
    copied operational.db exactly, with zero referential-integrity warnings -- i.e.
    the real source data is internally consistent, independently of the copy."""
    summary = db.load_operational_data(
        operational_dir=config.OPERATIONAL_DIR,
        db_path=tmp_path / "ingested.db",
    )
    assert summary["customers"] == 50
    assert summary["orders"] == 200
    assert summary["returns"] == 35
    assert summary["warnings"] == []


# ════════════════════════════════════════════════════════════════════════════
# 2. days_since_delivery / to_dict against real rows
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("order_id", ["ORD-10001", "ORD-10002", "ORD-10006", "ORD-10167"])
def test_days_since_delivery_matches_manual_date_arithmetic(order_id: str):
    raw_delivery = raw_query(
        CANONICAL_DB_PATH, "SELECT delivery_date FROM orders WHERE order_id = ?", (order_id,)
    )[0][0]
    expected = (config.AS_OF_DATE - date.fromisoformat(raw_delivery)).days

    with db.get_session(CANONICAL_DB_PATH) as session:
        order = session.get(db.Order, order_id)
        assert order.days_since_delivery == expected


def test_to_dict_serialises_every_field_exactly_as_the_raw_row(working_db: Path):
    raw = raw_query(
        working_db,
        "SELECT order_id, customer_id, product_name, product_category, order_date, "
        "delivery_date, status, amount, payment_method, shipping_method, warranty_plan "
        "FROM orders WHERE order_id = 'ORD-10002'",
    )[0]
    (r_oid, r_cust, r_prod, r_cat, r_odate, r_ddate, r_status, r_amount,
     r_pay, r_ship, r_warranty) = raw

    with db.get_session(working_db) as session:
        d = session.get(db.Order, "ORD-10002").to_dict()

    assert d["order_id"] == r_oid
    assert d["customer_id"] == r_cust
    assert d["product_name"] == r_prod
    assert d["product_category"] == r_cat
    assert d["order_date"] == r_odate
    assert d["delivery_date"] == r_ddate
    assert d["status"] == r_status
    assert d["amount"] == pytest.approx(r_amount)
    assert d["payment_method"] == r_pay
    assert d["shipping_method"] == r_ship
    assert d["warranty_plan"] == r_warranty
    # PS5 DualSense Controller, delivered 2024-10-10: real protect_plus warranty row.
    assert d["warranty_plan"] == "protect_plus"


# ════════════════════════════════════════════════════════════════════════════
# 3. get_order_details: real eligibility decisions, cross-checked by hand
# ════════════════════════════════════════════════════════════════════════════

def test_eligible_order_exactly_at_the_window_boundary_zero_days_remaining():
    """ORD-10002 (CUST-1035, standard tier): Gaming Accessories has a 30-day window;
    delivered 2024-10-10, AS_OF_DATE is 2024-11-09 -> days_since_delivery == 30 ==
    window, so remaining == 0 and the order is still eligible (remaining < 0 fails,
    not remaining <= 0): this is the exact boundary of policies.evaluate_return."""
    with db.get_session(CANONICAL_DB_PATH) as session:
        res = ops.get_order_details("ORD-10002", requester_id="CUST-1035", session=session)

    assert res["ok"] is True
    elig = res["data"]["return_eligibility"]
    assert res["data"]["membership_tier"] == "standard"
    assert elig["eligible_now"] is True
    assert elig["code"] == "ELIGIBLE"
    assert elig["window_days"] == 30
    assert elig["days_since_delivery"] == 30
    assert elig["days_remaining"] == 0
    assert elig["deadline"] == "November 9, 2024"  # delivered Oct 10 + 30 days
    assert elig["rule_applied"] == "standard_window"
    # Cross-check against the pure policy function directly.
    manual = return_deadline("Gaming Accessories", False, date(2024, 10, 6), date(2024, 10, 10))
    assert manual["deadline"] == "2024-11-09"


def test_order_expired_by_exactly_one_day_past_the_window():
    """ORD-10006 (CUST-1021, standard): Smart Home Devices, 30-day window, delivered
    2024-10-09 -> days_since_delivery == 31 -> remaining == -1 (expired yesterday)."""
    with db.get_session(CANONICAL_DB_PATH) as session:
        res = ops.get_order_details("ORD-10006", requester_id="CUST-1021", session=session)

    elig = res["data"]["return_eligibility"]
    assert elig["eligible_now"] is False
    assert elig["code"] == "RETURN_WINDOW_EXPIRED"
    assert elig["days_since_delivery"] == 31
    assert elig["days_remaining"] == -1
    assert "closed 1 day(s) ago" in elig["message"]


def test_premium_order_exactly_at_the_extended_window_boundary():
    """ORD-10167 (CUST-1024, premium): Computer Accessories is a 45-day category;
    +15 Premium days -> 60-day effective window. Delivered 2024-09-10, AS_OF_DATE
    2024-11-09 -> days_since_delivery == 60 == window -> remaining == 0, eligible."""
    with db.get_session(CANONICAL_DB_PATH) as session:
        res = ops.get_order_details("ORD-10167", requester_id="CUST-1024", session=session)

    assert res["data"]["membership_tier"] == "premium"
    assert res["data"]["is_premium"] is True
    elig = res["data"]["return_eligibility"]
    assert elig["eligible_now"] is True
    assert elig["window_days"] == 60
    assert elig["days_since_delivery"] == 60
    assert elig["days_remaining"] == 0


def test_shipped_order_has_not_been_delivered_yet():
    """ORD-10009 (CUST-1048) is real data in status 'shipped': no delivery_date yet,
    so eligibility must be ORDER_NOT_DELIVERED with no deadline, never a date-math
    crash on a None delivery_date."""
    raw_status = raw_query(
        CANONICAL_DB_PATH, "SELECT status FROM orders WHERE order_id = 'ORD-10009'"
    )[0][0]
    assert raw_status == "shipped"

    with db.get_session(CANONICAL_DB_PATH) as session:
        res = ops.get_order_details("ORD-10009", requester_id="CUST-1048", session=session)

    elig = res["data"]["return_eligibility"]
    assert elig["eligible_now"] is False
    assert elig["code"] == "ORDER_NOT_DELIVERED"
    assert elig["deadline"] is None


def test_non_returnable_category_real_gift_card_order():
    """ORD-10007 (CUST-1012): a Gift Card is non-returnable by category rule,
    independent of delivery age."""
    with db.get_session(CANONICAL_DB_PATH) as session:
        res = ops.get_order_details("ORD-10007", requester_id="CUST-1012", session=session)

    elig = res["data"]["return_eligibility"]
    assert elig["eligible_now"] is False
    assert elig["code"] == "NON_RETURNABLE"


def test_order_with_existing_approved_return_matches_the_real_return_row(working_db: Path):
    """ORD-10046 (CUST-1020) already has RET-2006 (status 'approved', refund_amount
    159.0) in the real data -- get_return_refund_status must surface it verbatim.

    Uses the tmp `working_db` copy (not CANONICAL_DB_PATH) throughout, since this test
    also calls create_return_request, which this module always runs against a
    disposable copy regardless of whether that particular call ends up writing."""
    raw = raw_query(
        working_db,
        "SELECT return_id, status, refund_amount, reason FROM returns WHERE order_id = ?",
        ("ORD-10046",),
    )[0]
    raw_id, raw_status, raw_refund, raw_reason = raw

    with db.get_session(working_db) as session:
        res = ops.get_return_refund_status("ORD-10046", requester_id="CUST-1020", session=session)

    assert res["data"]["has_returns"] is True
    r = res["data"]["returns"][0]
    assert r["return_id"] == raw_id == "RET-2006"
    assert r["status"] == raw_status == "approved"
    assert r["refund_amount"] == pytest.approx(raw_refund)
    assert r["reason"] == raw_reason

    returns_before = raw_query(working_db, "SELECT COUNT(*) FROM returns")[0][0]

    # An existing approved return also blocks a fresh return request (code path
    # shared with create_return_request), and must write nothing to the DB.
    with db.get_session(working_db) as session:
        blocked = ops.create_return_request(
            "ORD-10046", "Second thoughts", requester_id="CUST-1020", session=session
        )
    assert blocked["ok"] is False
    assert blocked["error"]["code"] == "ALREADY_RETURNED"
    assert raw_query(working_db, "SELECT COUNT(*) FROM returns")[0][0] == returns_before


# ════════════════════════════════════════════════════════════════════════════
# 4. Cross-customer ownership enforcement with real, distinct id pairs
# ════════════════════════════════════════════════════════════════════════════

def test_cust_1002_cannot_read_cust_1001s_order():
    """ORD-10190 really belongs to CUST-1001 (confirmed via raw SQL below).
    CUST-1002 is a real, different customer who owns other orders -- not a
    fabricated placeholder id."""
    owner = raw_query(
        CANONICAL_DB_PATH, "SELECT customer_id FROM orders WHERE order_id = 'ORD-10190'"
    )[0][0]
    assert owner == "CUST-1001"
    assert raw_query(
        CANONICAL_DB_PATH, "SELECT COUNT(*) FROM orders WHERE customer_id = 'CUST-1002'"
    )[0][0] > 0

    with db.get_session(CANONICAL_DB_PATH) as session:
        res = ops.get_order_details("ORD-10190", requester_id="CUST-1002", session=session)
        refund = ops.get_return_refund_status("ORD-10190", requester_id="CUST-1002",
                                              session=session)

    assert res["ok"] is False and res["error"]["code"] == "UNAUTHORIZED"
    assert refund["ok"] is False and refund["error"]["code"] == "UNAUTHORIZED"


def test_cust_1001_cannot_read_cust_1002s_order_reverse_pair():
    """The reverse real pair: ORD-10021 belongs to CUST-1002, CUST-1001 must be
    refused -- confirms the ownership check isn't asymmetric or id-specific."""
    owner = raw_query(
        CANONICAL_DB_PATH, "SELECT customer_id FROM orders WHERE order_id = 'ORD-10021'"
    )[0][0]
    assert owner == "CUST-1002"

    with db.get_session(CANONICAL_DB_PATH) as session:
        res = ops.get_order_details("ORD-10021", requester_id="CUST-1001", session=session)

    assert res["ok"] is False
    assert res["error"]["code"] == "UNAUTHORIZED"


def test_create_return_request_also_enforces_ownership_and_writes_nothing(working_db: Path):
    """A non-owner must not be able to create a return on someone else's real
    order, and the attempt must leave the returns table untouched."""
    before = raw_query(working_db, "SELECT COUNT(*) FROM returns")[0][0]

    with db.get_session(working_db) as session:
        res = ops.create_return_request(
            "ORD-10190", "Not mine", requester_id="CUST-1002", session=session
        )

    assert res["ok"] is False
    assert res["error"]["code"] == "UNAUTHORIZED"
    after = raw_query(working_db, "SELECT COUNT(*) FROM returns")[0][0]
    assert after == before


# ════════════════════════════════════════════════════════════════════════════
# 5. Full lifecycle: real eligible order -> create_return_request -> raw re-query
# ════════════════════════════════════════════════════════════════════════════

def test_full_lifecycle_create_return_request_then_reverify_with_raw_sql(working_db: Path):
    """ORD-10017 (CUST-1025, standard tier): AirPods Pro, Audio & Headphones (30-day
    window), delivered 2024-10-28 -> 12 days since delivery as of AS_OF_DATE, 18 days
    remaining -- a real, currently-eligible order with no existing return. Create a
    return against the tmp copy, then re-open a fresh raw sqlite3 connection (not the
    ORM session that wrote it) to confirm the row landed exactly as expected and nothing
    else in the database moved."""
    # Baseline snapshot of every table, before the write.
    before_counts = (
        raw_query(working_db, "SELECT COUNT(*) FROM customers")[0][0],
        raw_query(working_db, "SELECT COUNT(*) FROM orders")[0][0],
        raw_query(working_db, "SELECT COUNT(*) FROM returns")[0][0],
    )
    assert before_counts == (50, 200, 35)
    order_row_before = raw_query(working_db, "SELECT * FROM orders WHERE order_id = 'ORD-10017'")
    other_returns_before = set(raw_query(working_db, "SELECT return_id FROM returns"))
    assert ("RET-2036",) not in other_returns_before
    max_existing = raw_query(working_db, "SELECT MAX(CAST(SUBSTR(return_id, 5) AS INTEGER)) "
                                         "FROM returns")[0][0]
    assert max_existing == 2035

    with db.get_session(working_db) as session:
        pre = ops.get_order_details("ORD-10017", requester_id="CUST-1025", session=session)
    assert pre["ok"] is True
    pre_elig = pre["data"]["return_eligibility"]
    assert pre_elig["eligible_now"] is True
    assert pre_elig["days_since_delivery"] == 12
    assert pre_elig["days_remaining"] == 18

    with db.get_session(working_db) as session:
        created = ops.create_return_request(
            "ORD-10017", "Arrived with a cracked case", requester_id="CUST-1025", session=session
        )

    assert created["ok"] is True
    assert created["data"]["return_id"] == "RET-2036"  # next after the real max, 2035
    assert created["data"]["status"] == "pending"
    assert created["data"]["window_days"] == 30
    assert created["data"]["days_remaining"] == 18
    assert created["data"]["estimated_refund"] == pytest.approx(249.0)

    # Independent raw-SQL re-verification with a brand-new connection.
    new_row = raw_query(
        working_db,
        "SELECT order_id, customer_id, product_name, product_category, reason, status, "
        "requested_date, resolved_date, refund_amount, refund_method, synthetic "
        "FROM returns WHERE return_id = 'RET-2036'",
    )
    assert len(new_row) == 1
    (r_oid, r_cust, r_prod, r_cat, r_reason, r_status, r_req, r_resolved,
     r_refund, r_refund_method, r_synth) = new_row[0]
    assert r_oid == "ORD-10017"
    assert r_cust == "CUST-1025"
    assert r_prod == "AirPods Pro 2nd Gen"
    assert r_cat == "Audio & Headphones"
    assert r_reason == "Arrived with a cracked case"
    assert r_status == "pending"
    assert r_req == config.AS_OF_DATE.isoformat()
    assert r_resolved is None
    assert r_refund == 0.0
    assert r_refund_method is None
    assert r_synth == 1

    # Nothing else changed: order row untouched, no other return rows appeared,
    # and counts moved by exactly +1 return.
    after_counts = (
        raw_query(working_db, "SELECT COUNT(*) FROM customers")[0][0],
        raw_query(working_db, "SELECT COUNT(*) FROM orders")[0][0],
        raw_query(working_db, "SELECT COUNT(*) FROM returns")[0][0],
    )
    assert after_counts == (50, 200, 36)
    assert raw_query(working_db,
                     "SELECT * FROM orders WHERE order_id = 'ORD-10017'") == order_row_before


# ════════════════════════════════════════════════════════════════════════════
# 6. Holiday Return Policy: real AS_OF_DATE (2024-11-09) predates the Nov 15 - Dec
#    31 window entirely, so no real order in this snapshot can exercise the holiday
#    extension. Per instructions, construct one on the tmp copy using a REAL,
#    otherwise-unused customer id (CUST-1023 owns no order used in any other test in
#    this file), run it through the full chain, and cross-check against the pure
#    policy function with the real config constants (CATEGORY_WINDOW_DAYS,
#    PREMIUM_EXTENSION_DAYS, HOLIDAY_* in policies.py are untouched).
# ════════════════════════════════════════════════════════════════════════════

def test_holiday_extension_through_the_full_chain_on_a_constructed_order(
    working_db: Path, monkeypatch
):
    conn = sqlite3.connect(working_db)
    try:
        conn.execute(
            "INSERT INTO orders (order_id, customer_id, product_name, product_category, "
            "order_date, delivery_date, status, amount, payment_method, shipping_method, "
            "warranty_plan, synthetic) VALUES "
            "('ORD-90001', 'CUST-1023', 'Sony WH-1000XM5', 'Audio & Headphones', "
            "'2024-12-01', '2024-12-02', 'delivered', 349.0, 'credit_card', 'standard', "
            "NULL, 1)"
        )
        conn.commit()
    finally:
        conn.close()

    # AS_OF_DATE is frozen at both import sites: config.AS_OF_DATE (read dynamically by
    # ops.py) and db.AS_OF_DATE (bound at db.py import time, used by Order.days_since_delivery).
    # Both must be patched for the chain to reflect the same "today".
    as_of = date(2025, 1, 20)
    monkeypatch.setattr(config, "AS_OF_DATE", as_of)
    monkeypatch.setattr(db, "AS_OF_DATE", as_of)

    with db.get_session(working_db) as session:
        res = ops.get_order_details("ORD-90001", requester_id="CUST-1023", session=session)

    assert res["ok"] is True
    elig = res["data"]["return_eligibility"]
    assert elig["days_since_delivery"] == 49  # (2025-01-20 - 2024-12-02).days
    assert elig["eligible_now"] is True
    assert elig["days_remaining"] == 11       # (2025-01-31 - 2025-01-20).days
    assert elig["deadline"] == "January 31, 2025"
    assert elig["rule_applied"] == "holiday_extension"

    # Cross-check against the pure policy functions directly, independent of ops/db.
    manual_decision = evaluate_return(
        category="Audio & Headphones", order_status="delivered", days_since_delivery=49,
        is_premium=False, existing_return_statuses=[],
        purchase_date=date(2024, 12, 1), delivery_date=date(2024, 12, 2), as_of=as_of,
    )
    assert manual_decision.eligible is True
    assert manual_decision.days_remaining == 11
    manual_deadline = return_deadline("Audio & Headphones", False, date(2024, 12, 1),
                                      date(2024, 12, 2))
    assert manual_deadline["deadline"] == "2025-01-31"
    assert manual_deadline["rule_applied"] == "holiday_extension"


# ════════════════════════════════════════════════════════════════════════════
# 7. Escalation ticket (ops.py) with a real order/customer id, never touching the
#    project's real data/escalations.jsonl
# ════════════════════════════════════════════════════════════════════════════

def test_escalation_ticket_with_real_order_and_customer_id(tmp_path: Path, monkeypatch):
    real_log = Path("/home/unthinkable-lap/Project_Rag/data/escalations.jsonl")
    real_mtime_before = real_log.stat().st_mtime if real_log.exists() else None

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    res = ops.create_escalation_ticket(
        summary="Customer disputes the return window calculation",
        reason="policy_unclear",
        order_id="ORD-10002",   # real order id used elsewhere in this file
        requester_id="CUST-1035",
    )
    assert res["ok"] is True
    ticket_id = res["data"]["ticket_id"]
    assert ticket_id.startswith("ESC-")

    log_path = tmp_path / "escalations.jsonl"
    assert log_path.exists()
    import json
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["ticket_id"] == ticket_id
    assert record["order_id"] == "ORD-10002"
    assert record["requester_id"] == "CUST-1035"
    assert record["reason"] == "policy_unclear"
    assert record["status"] == "open"

    # The real project data directory must be untouched by this test.
    real_mtime_after = real_log.stat().st_mtime if real_log.exists() else None
    assert real_mtime_after == real_mtime_before

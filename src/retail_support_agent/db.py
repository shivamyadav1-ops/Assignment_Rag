"""
Ingestion and access layer for operational records.

Loads customers.csv, orders.csv and returns.json into a typed relational
database with indexes and referential integrity. This database is the
backing store for the MCP tools — it is never embedded into Qdrant.

PostgreSQL (RSA_DATABASE_URL) is the default store for the app, CLI and evals —
matching production. SQLite is used only when a caller passes an explicit
db_path, which is how every test gets a throwaway, file-based, dependency-free
database without needing a live Postgres server.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Date,
    Float,
    ForeignKey,
    Index,
    String,
    create_engine,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from retail_support_agent import config
from retail_support_agent.config import AS_OF_DATE, OPERATIONAL_DIR

# ── ORM models ────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    first_name: Mapped[str] = mapped_column(String(50))
    last_name: Mapped[str] = mapped_column(String(50))
    email: Mapped[str] = mapped_column(String(120), index=True)
    membership_tier: Mapped[str] = mapped_column(String(20), index=True)
    account_created: Mapped[date] = mapped_column(Date)
    synthetic: Mapped[bool] = mapped_column(Boolean, default=True)

    orders: Mapped[list[Order]] = relationship(back_populates="customer")

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    def to_dict(self) -> dict:
        return {
            "customer_id": self.customer_id,
            "name": self.full_name,
            "email": self.email,
            "membership_tier": self.membership_tier,
            "account_created": self.account_created.isoformat(),
            "synthetic": self.synthetic,
        }


class Order(Base):
    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(20), ForeignKey("customers.customer_id"), index=True
    )
    product_name: Mapped[str] = mapped_column(String(120))
    product_category: Mapped[str] = mapped_column(String(60), index=True)
    order_date: Mapped[date] = mapped_column(Date)
    delivery_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(30), index=True)
    amount: Mapped[float] = mapped_column(Float)
    payment_method: Mapped[str] = mapped_column(String(30))
    shipping_method: Mapped[str] = mapped_column(String(30))
    warranty_plan: Mapped[str | None] = mapped_column(String(30), nullable=True)
    synthetic: Mapped[bool] = mapped_column(Boolean, default=True)

    customer: Mapped[Customer] = relationship(back_populates="orders")
    returns: Mapped[list[Return]] = relationship(back_populates="order")

    @property
    def days_since_delivery(self) -> int | None:
        """Days elapsed since delivery, measured from the configured AS_OF_DATE.

        Uses AS_OF_DATE rather than the wall clock so the synthetic records stay
        internally consistent and the demo is reproducible on any run date.
        """
        if self.delivery_date is None:
            return None
        return (AS_OF_DATE - self.delivery_date).days

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "customer_id": self.customer_id,
            "product_name": self.product_name,
            "product_category": self.product_category,
            "order_date": self.order_date.isoformat(),
            "delivery_date": self.delivery_date.isoformat() if self.delivery_date else None,
            "status": self.status,
            "amount": self.amount,
            "payment_method": self.payment_method,
            "shipping_method": self.shipping_method,
            "warranty_plan": self.warranty_plan,
            "days_since_delivery": self.days_since_delivery,
            "synthetic": self.synthetic,
        }


class Return(Base):
    __tablename__ = "returns"

    return_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    order_id: Mapped[str] = mapped_column(
        String(20), ForeignKey("orders.order_id"), index=True
    )
    customer_id: Mapped[str] = mapped_column(String(20), index=True)
    product_name: Mapped[str] = mapped_column(String(120))
    product_category: Mapped[str] = mapped_column(String(60))
    reason: Mapped[str] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(30), index=True)
    requested_date: Mapped[date] = mapped_column(Date)
    resolved_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    refund_amount: Mapped[float] = mapped_column(Float, default=0.0)
    refund_method: Mapped[str | None] = mapped_column(String(30), nullable=True)
    synthetic: Mapped[bool] = mapped_column(Boolean, default=True)

    order: Mapped[Order] = relationship(back_populates="returns")

    def to_dict(self) -> dict:
        return {
            "return_id": self.return_id,
            "order_id": self.order_id,
            "customer_id": self.customer_id,
            "product_name": self.product_name,
            "product_category": self.product_category,
            "reason": self.reason,
            "status": self.status,
            "requested_date": self.requested_date.isoformat(),
            "resolved_date": self.resolved_date.isoformat() if self.resolved_date else None,
            "refund_amount": self.refund_amount,
            "refund_method": self.refund_method,
            "synthetic": self.synthetic,
        }


# Composite indexes for common query patterns
Index("ix_orders_customer_status", Order.customer_id, Order.status)
Index("ix_returns_order_status", Return.order_id, Return.status)


# ── Engine / session helpers ──────────────────────────────────────────────────

def get_engine(db_path: Path | None = None, echo: bool = False):
    """Build the SQLAlchemy engine.

    `db_path` given (always true in tests): a throwaway SQLite file, so tests never
    need a live database server. `db_path` omitted (the app, CLI and evals): the
    configured PostgreSQL database at RSA_DATABASE_URL.
    """
    if db_path is not None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return create_engine(f"sqlite:///{db_path}", echo=echo)
    if not config.DATABASE_URL:
        raise RuntimeError(
            "RSA_DATABASE_URL is not set. Point it at the project's PostgreSQL database, "
            "e.g. postgresql+psycopg://rsa_app:<password>@localhost:5434/retail_support_agent"
        )
    return create_engine(config.DATABASE_URL, echo=echo)


def get_session(db_path: Path | None = None) -> Session:
    return Session(get_engine(db_path))


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _parse_date(value: str | None) -> date | None:
    if not value or value in ("", "None", "null"):
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def _nullable_str(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s and s not in ("None", "null", "nan") else None


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestionError(Exception):
    """Raised when source data fails validation."""


def load_operational_data(
    operational_dir: Path | None = None,
    db_path: Path | None = None,
    drop_existing: bool = True,
) -> dict:
    """
    Load CSV + JSON operational records into the operational database.

    `db_path` given: a throwaway SQLite file (tests). Omitted: the configured
    PostgreSQL database at RSA_DATABASE_URL (app, CLI, evals).

    Returns a summary dict with row counts and validation results.
    """
    import csv
    import re

    operational_dir = operational_dir or OPERATIONAL_DIR

    engine = get_engine(db_path)

    if drop_existing:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    # Redact credentials before this ever lands in a report, log or UI caption.
    target = str(db_path) if db_path is not None else re.sub(
        r"//[^@/]+@", "//***@", config.DATABASE_URL or ""
    )
    summary = {
        "db_path": target,
        "customers": 0,
        "orders": 0,
        "returns": 0,
        "warnings": [],
    }

    session = Session(engine)
    try:
        # ── Customers (CSV) ───────────────────────────────────────────────
        customers_file = operational_dir / "customers.csv"
        if not customers_file.exists():
            raise IngestionError(f"Missing required file: {customers_file}")

        customer_ids: set[str] = set()
        with customers_file.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = row["customer_id"].strip()
                if cid in customer_ids:
                    summary["warnings"].append(f"Duplicate customer_id skipped: {cid}")
                    continue
                customer_ids.add(cid)
                session.add(Customer(
                    customer_id=cid,
                    first_name=row["first_name"].strip(),
                    last_name=row["last_name"].strip(),
                    email=row["email"].strip(),
                    membership_tier=row["membership_tier"].strip(),
                    account_created=_parse_date(row["account_created"]),
                    synthetic=_parse_bool(row.get("synthetic", True)),
                ))
        session.flush()
        summary["customers"] = len(customer_ids)

        # ── Orders (CSV) ──────────────────────────────────────────────────
        orders_file = operational_dir / "orders.csv"
        if not orders_file.exists():
            raise IngestionError(f"Missing required file: {orders_file}")

        order_ids: set[str] = set()
        with orders_file.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                oid = row["order_id"].strip()
                cid = row["customer_id"].strip()

                if oid in order_ids:
                    summary["warnings"].append(f"Duplicate order_id skipped: {oid}")
                    continue
                # Referential integrity: order must reference a known customer
                if cid not in customer_ids:
                    summary["warnings"].append(
                        f"Order {oid} references unknown customer {cid} — skipped"
                    )
                    continue

                order_ids.add(oid)
                session.add(Order(
                    order_id=oid,
                    customer_id=cid,
                    product_name=row["product_name"].strip(),
                    product_category=row["product_category"].strip(),
                    order_date=_parse_date(row["order_date"]),
                    delivery_date=_parse_date(row.get("delivery_date")),
                    status=row["status"].strip(),
                    amount=float(row["amount"]),
                    payment_method=row["payment_method"].strip(),
                    shipping_method=row["shipping_method"].strip(),
                    warranty_plan=_nullable_str(row.get("warranty_plan")),
                    synthetic=_parse_bool(row.get("synthetic", True)),
                ))
        session.flush()
        summary["orders"] = len(order_ids)

        # ── Returns (JSON) ────────────────────────────────────────────────
        returns_file = operational_dir / "returns.json"
        if not returns_file.exists():
            raise IngestionError(f"Missing required file: {returns_file}")

        records = json.loads(returns_file.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise IngestionError("returns.json must contain a JSON array")

        return_ids: set[str] = set()
        for rec in records:
            rid = str(rec["return_id"]).strip()
            oid = str(rec["order_id"]).strip()

            if rid in return_ids:
                summary["warnings"].append(f"Duplicate return_id skipped: {rid}")
                continue
            if oid not in order_ids:
                summary["warnings"].append(
                    f"Return {rid} references unknown order {oid} — skipped"
                )
                continue

            return_ids.add(rid)
            session.add(Return(
                return_id=rid,
                order_id=oid,
                customer_id=str(rec["customer_id"]).strip(),
                product_name=rec["product_name"],
                product_category=rec["product_category"],
                reason=rec["reason"],
                status=rec["status"],
                requested_date=_parse_date(rec["requested_date"]),
                resolved_date=_parse_date(rec.get("resolved_date")),
                refund_amount=float(rec.get("refund_amount") or 0.0),
                refund_method=_nullable_str(rec.get("refund_method")),
                synthetic=_parse_bool(rec.get("synthetic", True)),
            ))

        summary["returns"] = len(return_ids)
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise IngestionError(f"Database integrity constraint violated: {exc}") from exc
    finally:
        session.close()

    return summary


# ── Query helpers (used by the MCP server) ───────────────────────────────────

def fetch_returns_for_order(session: Session, order_id: str) -> list[Return]:
    return list(session.scalars(select(Return).where(Return.order_id == order_id)))

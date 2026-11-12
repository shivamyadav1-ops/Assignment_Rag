"""
Generate synthetic operational data for the retail support agent.

Creates:
  - data/raw/operational/customers.csv   (50 customers)
  - data/raw/operational/orders.csv      (200 orders)
  - data/raw/operational/returns.json    (35 returns)

All records are clearly labeled as synthetic. Distributions are designed
to produce interesting edge cases for the agent:
  - Orders delivered yesterday (inside all return windows)
  - Orders delivered 16 days ago (outside smartphone window, inside laptop window)
  - Orders delivered 32 days ago (outside standard 30-day window)
  - Orders delivered 50 days ago (eligible for 85% partial refund)
  - Orders delivered 90 days ago (outside all return windows)
  - Premium members (extended return windows)
  - Non-returnable items (software, gift cards)
  - DOA candidates (delivered < 48h ago with defective status)
"""

import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)  # reproducible

BASE = Path(__file__).resolve().parent.parent / "data" / "raw" / "operational"
BASE.mkdir(parents=True, exist_ok=True)

# ── Lookup tables ──────────────────────────────────────────────────────────

FIRST_NAMES = [
    "Alice", "Bob", "Carlos", "Diana", "Ethan", "Fiona", "George", "Hannah",
    "Ivan", "Julia", "Kevin", "Laura", "Mike", "Nina", "Oscar", "Priya",
    "Quincy", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
    "Yara", "Zane", "Aisha", "Blake", "Chloe", "David", "Ella", "Frank",
    "Grace", "Henry", "Iris", "Jake", "Kara", "Leo", "Maya", "Noah",
    "Olivia", "Pete", "Quinn", "Rosa", "Steve", "Tara", "Umar", "Vera",
    "Will", "Zoe",
]

LAST_NAMES = [
    "Anderson", "Brown", "Chen", "Davis", "Evans", "Foster", "Garcia",
    "Hall", "Ito", "Johnson", "Kim", "Lee", "Martinez", "Nguyen", "Ortiz",
    "Patel", "Quinn", "Rivera", "Smith", "Thompson", "Ueda", "Vasquez",
    "Wang", "Xu", "Young", "Zhang", "Ali", "Baker", "Clark", "Diaz",
    "Edwards", "Flores", "Green", "Harris", "Ibrahim", "Jones", "Kumar",
    "Lopez", "Moore", "Nakamura", "Olsen", "Park", "Rao", "Scott",
    "Taylor", "Varma", "Wilson", "Yang", "Zimmerman", "Adams",
]

MEMBERSHIP_TIERS = ["standard", "standard", "standard", "premium"]  # 25% premium

PRODUCTS = [
    # (name, category, price_min, price_max)
    ("Galaxy S24 Ultra 256GB", "Smartphones & Tablets", 1199, 1199),
    ("iPhone 15 Pro Max 512GB", "Smartphones & Tablets", 1399, 1399),
    ("iPad Air M2 128GB", "Smartphones & Tablets", 599, 599),
    ("Pixel 8 Pro 128GB", "Smartphones & Tablets", 899, 899),
    ("MacBook Pro 14\" M3", "Laptops & Computers", 1599, 1999),
    ("Dell XPS 15 i7 16GB", "Laptops & Computers", 1299, 1499),
    ("ThinkPad X1 Carbon Gen 11", "Laptops & Computers", 1449, 1449),
    ("HP Spectre x360 14\"", "Laptops & Computers", 1249, 1249),
    ("Sony WH-1000XM5", "Audio & Headphones", 349, 349),
    ("AirPods Pro 2nd Gen", "Audio & Headphones", 249, 249),
    ("Bose QuietComfort Ultra", "Audio & Headphones", 429, 429),
    ("JBL Charge 5 Speaker", "Audio & Headphones", 179, 179),
    ("Samsung 65\" OLED S90C", "Televisions & Monitors", 1799, 1799),
    ("LG 55\" C3 OLED", "Televisions & Monitors", 1299, 1299),
    ("Dell 27\" 4K Monitor U2723QE", "Televisions & Monitors", 549, 549),
    ("Sony A7 IV Mirrorless Camera", "Cameras & Photography", 2499, 2499),
    ("Canon EOS R6 Mark II", "Cameras & Photography", 2299, 2299),
    ("GoPro Hero 12 Black", "Cameras & Photography", 399, 399),
    ("PlayStation 5 Slim", "Gaming Consoles", 449, 449),
    ("Xbox Series X", "Gaming Consoles", 499, 499),
    ("Nintendo Switch OLED", "Gaming Consoles", 349, 349),
    ("PS5 DualSense Controller", "Gaming Accessories", 69, 69),
    ("Xbox Elite Controller Series 2", "Gaming Accessories", 179, 179),
    ("Echo Show 15", "Smart Home Devices", 249, 249),
    ("Google Nest Hub Max", "Smart Home Devices", 229, 229),
    ("Ring Video Doorbell Pro 2", "Smart Home Devices", 249, 249),
    ("Apple Watch Ultra 2", "Wearables & Fitness", 799, 799),
    ("Fitbit Charge 6", "Wearables & Fitness", 159, 159),
    ("Logitech MX Master 3S", "Computer Accessories", 99, 99),
    ("Mechanical Keyboard Keychron Q1", "Computer Accessories", 169, 169),
    ("Anker USB-C Hub 7-in-1", "Computer Accessories", 35, 45),
    ("USB-C to HDMI Cable 6ft", "Cables & Adapters", 14, 19),
    ("Thunderbolt 4 Cable 1m", "Cables & Adapters", 29, 39),
    ("Microsoft 365 Personal (Digital)", "Software & Digital Codes", 69, 69),
    ("Norton 360 Deluxe 1-Year", "Software & Digital Codes", 49, 49),
    ("ShopSmart Gift Card $50", "Gift Cards", 50, 50),
    ("ShopSmart Gift Card $100", "Gift Cards", 100, 100),
]

PAYMENT_METHODS = ["credit_card", "credit_card", "debit_card", "paypal", "store_credit", "bnpl"]

ORDER_STATUSES = ["delivered", "delivered", "delivered", "delivered",
                  "shipped", "processing", "confirmed", "cancelled"]

RETURN_REASONS = [
    "Changed my mind",
    "Item not as described",
    "Defective / not working",
    "Wrong item received",
    "Better price found elsewhere",
    "Arrived damaged",
    "No longer needed",
    "Missing parts or accessories",
]

RETURN_STATUSES = ["approved", "approved", "approved", "pending", "rejected", "refunded", "refunded"]

TODAY = datetime(2024, 11, 9)


# ── Customers ──────────────────────────────────────────────────────────────

def generate_customers(n=50):
    customers = []
    used = set()
    for i in range(n):
        while True:
            first = random.choice(FIRST_NAMES)
            last = random.choice(LAST_NAMES)
            if (first, last) not in used:
                used.add((first, last))
                break
        cid = f"CUST-{1001 + i:04d}"
        email = f"{first.lower()}.{last.lower()}@example.com"
        tier = random.choice(MEMBERSHIP_TIERS)
        created = TODAY - timedelta(days=random.randint(30, 730))
        customers.append({
            "customer_id": cid,
            "first_name": first,
            "last_name": last,
            "email": email,
            "membership_tier": tier,
            "account_created": created.strftime("%Y-%m-%d"),
            "synthetic": True,
        })
    return customers


# ── Orders ─────────────────────────────────────────────────────────────────

def generate_orders(customers, n=200):
    orders = []
    # Delivery-age distribution to create interesting edge cases
    delivery_ages_days = (
        [1, 2] * 10                 # 20 recent (inside all windows, DOA candidates)
        + [10, 12, 14] * 8          # 24 within smartphone 15-day window
        + [16, 18, 20, 22] * 6      # 24 outside smartphone, inside laptop 30-day
        + [25, 28, 30] * 5          # 15 at the 30-day boundary
        + [31, 33, 35] * 5          # 15 just outside 30-day standard
        + [40, 45, 50] * 5          # 15 partial-refund territory (within 60 days)
        + [55, 60] * 4              # 8 at the 60-day cutoff
        + [70, 90, 120, 180] * 4    # 16 well past all windows
    )
    # pad to n with random ages
    while len(delivery_ages_days) < n:
        delivery_ages_days.append(random.randint(1, 200))
    random.shuffle(delivery_ages_days)

    for i in range(n):
        cust = random.choice(customers)
        product_name, category, pmin, pmax = random.choice(PRODUCTS)
        price = round(random.uniform(pmin, pmax), 2)
        status = random.choice(ORDER_STATUSES)

        if status == "delivered":
            age = delivery_ages_days[i]
            delivery_date = TODAY - timedelta(days=age)
            order_date = delivery_date - timedelta(days=random.randint(2, 7))
        elif status == "shipped":
            order_date = TODAY - timedelta(days=random.randint(1, 4))
            delivery_date = None
        elif status in ("processing", "confirmed"):
            order_date = TODAY - timedelta(days=random.randint(0, 2))
            delivery_date = None
        else:  # cancelled
            order_date = TODAY - timedelta(days=random.randint(5, 60))
            delivery_date = None

        payment = random.choice(PAYMENT_METHODS)
        shipping = random.choice(["standard", "standard", "expedited", "next_day"])

        # Premium members with extended warranty purchases
        has_protect = random.random() < 0.2
        protect_plan = random.choice(["protect_basic", "protect_plus"]) if has_protect else None

        orders.append({
            "order_id": f"ORD-{10001 + i:05d}",
            "customer_id": cust["customer_id"],
            "product_name": product_name,
            "product_category": category,
            "order_date": order_date.strftime("%Y-%m-%d"),
            "delivery_date": delivery_date.strftime("%Y-%m-%d") if delivery_date else None,
            "status": status,
            "amount": price,
            "payment_method": payment,
            "shipping_method": shipping,
            "warranty_plan": protect_plan,
            "synthetic": True,
        })
    return orders


# ── Returns ────────────────────────────────────────────────────────────────

def generate_returns(orders, n=35):
    delivered = [o for o in orders if o["status"] == "delivered"]
    # pick a mix: some recent (valid), some old (rejected/partial), some non-returnable
    chosen = random.sample(delivered, min(n, len(delivered)))
    returns = []
    for i, order in enumerate(chosen):
        delivery = datetime.strptime(order["delivery_date"], "%Y-%m-%d")
        days_since = (TODAY - delivery).days
        requested = delivery + timedelta(days=random.randint(1, max(2, days_since)))
        reason = random.choice(RETURN_REASONS)

        # Determine realistic status based on category and timing
        cat = order["product_category"]
        non_returnable = cat in ("Software & Digital Codes", "Gift Cards")
        if non_returnable:
            status = "rejected"
            refund = 0.0
        elif days_since <= 30:
            status = random.choice(["approved", "refunded", "pending"])
            refund = round(order["amount"], 2) if status in ("approved", "refunded") else 0.0
        elif days_since <= 60:
            status = random.choice(["approved", "refunded", "rejected"])
            refund = round(order["amount"] * 0.85, 2) if status != "rejected" else 0.0
        else:
            status = "rejected"
            refund = 0.0

        resolved = None
        if status in ("approved", "refunded", "rejected"):
            resolved = requested + timedelta(days=random.randint(2, 7))

        returns.append({
            "return_id": f"RET-{2001 + i:04d}",
            "order_id": order["order_id"],
            "customer_id": order["customer_id"],
            "product_name": order["product_name"],
            "product_category": order["product_category"],
            "reason": reason,
            "status": status,
            "requested_date": requested.strftime("%Y-%m-%d"),
            "resolved_date": resolved.strftime("%Y-%m-%d") if resolved else None,
            "refund_amount": refund,
            "refund_method": order["payment_method"] if refund > 0 else None,
            "synthetic": True,
        })
    return returns


# ── Write files ────────────────────────────────────────────────────────────

def main():
    print("Generating synthetic data...")

    customers = generate_customers(50)
    print(f"  Customers: {len(customers)}")

    orders = generate_orders(customers, 200)
    print(f"  Orders:    {len(orders)}")

    returns = generate_returns(orders, 35)
    print(f"  Returns:   {len(returns)}")

    # Write customers.csv
    csv_path = BASE / "customers.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=customers[0].keys())
        w.writeheader()
        w.writerows(customers)
    print(f"  → {csv_path}")

    # Write orders.csv
    csv_path = BASE / "orders.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=orders[0].keys())
        w.writeheader()
        w.writerows(orders)
    print(f"  → {csv_path}")

    # Write returns.json  (different format — satisfies "at least two data formats")
    json_path = BASE / "returns.json"
    with open(json_path, "w") as f:
        json.dump(returns, f, indent=2)
    print(f"  → {json_path}")

    # Summary stats for verification
    print("\n── Summary ──")
    premium = sum(1 for c in customers if c["membership_tier"] == "premium")
    print(f"  Premium members: {premium}/{len(customers)} ({premium/len(customers)*100:.0f}%)")

    delivered = [o for o in orders if o["status"] == "delivered"]
    print(f"  Delivered orders: {len(delivered)}/{len(orders)}")

    categories = {}
    for o in orders:
        categories[o["product_category"]] = categories.get(o["product_category"], 0) + 1
    print("  Orders by category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    rejected = sum(1 for r in returns if r["status"] == "rejected")
    print(f"  Returns rejected: {rejected}/{len(returns)}")
    print("\nDone.")


if __name__ == "__main__":
    main()

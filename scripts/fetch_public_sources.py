"""Download the public (non-synthetic) source documents into data/raw/policies/.

Every source here is legally reusable and is fetched from an official API, pinned
to a dated version so a re-run produces the same bytes. The provenance of each
file (publisher, URL, licence, version date, SHA-256) is written to
data/sources.json, which the chunker and the manifest generator both read.

Sources
-------
16 CFR Part 435 — FTC Mail, Internet, or Telephone Order Merchandise Rule.
    Federal regulation text from the official eCFR API (ecfr.gov). Works of the
    US federal government are not subject to copyright (17 U.S.C. § 105), so the
    text is public domain. It governs shipping-time promises, delay notices,
    cancellation and prompt refunds, which is exactly the regulatory backdrop to
    the store's shipping and refund policies.

Usage:
    uv run python scripts/fetch_public_sources.py            # fetch pinned versions
    uv run python scripts/fetch_public_sources.py --check    # verify files on disk match the registry
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POLICIES_DIR = ROOT / "data" / "raw" / "policies"
REGISTRY = ROOT / "data" / "sources.json"

ECFR_FULL = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml?part={part}"

PUBLIC_SOURCES = [
    {
        "file": "policies/ftc_mail_order_rule_16cfr435.xml",
        "title": "16 CFR Part 435 — Mail, Internet, or Telephone Order Merchandise Rule",
        "publisher": "U.S. Federal Trade Commission, via the Electronic Code of Federal Regulations (eCFR)",
        "url": ECFR_FULL.format(date="2026-11-09", title=16, part=435),
        "landing_page": "https://www.ecfr.gov/current/title-16/chapter-I/subchapter-D/part-435",
        "version_date": "2026-11-09",
        "license": "Public domain (U.S. Government work, 17 U.S.C. § 105)",
        "format": "xml",
    },
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str) -> bytes:
    # The eCFR API requires the client to accept a compressed response.
    req = urllib.request.Request(url, headers={
        "Accept-Encoding": "gzip",
        "User-Agent": "retail-support-agent-ingestion/1.0",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            import gzip
            body = gzip.decompress(body)
    return body


def _load_registry() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {"sources": {}}


def fetch() -> int:
    registry = _load_registry()
    for src in PUBLIC_SOURCES:
        body = _download(src["url"])
        if not body.lstrip().startswith(b"<?xml"):
            print(f"ERROR {src['file']}: response is not XML ({body[:80]!r})", file=sys.stderr)
            return 1
        dest = ROOT / "data" / "raw" / src["file"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        entry = registry["sources"].setdefault(src["file"], {})
        entry.update({k: v for k, v in src.items() if k != "file"})
        entry.update({
            "origin": "public",
            "synthetic": False,
            "sha256": _sha256(body),
            "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "fetched_by": "scripts/fetch_public_sources.py",
        })
        print(f"fetched {src['file']}  {len(body):,} bytes  sha256={entry['sha256'][:16]}")
    REGISTRY.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"updated {REGISTRY.relative_to(ROOT)}")
    return 0


def check() -> int:
    registry = _load_registry()
    bad = 0
    for src in PUBLIC_SOURCES:
        path = ROOT / "data" / "raw" / src["file"]
        expected = registry["sources"].get(src["file"], {}).get("sha256")
        if not path.exists():
            print(f"MISSING  {src['file']}  (run without --check to fetch)")
            bad += 1
        elif _sha256(path.read_bytes()) != expected:
            print(f"CHANGED  {src['file']}  (hash differs from data/sources.json)")
            bad += 1
        else:
            print(f"OK       {src['file']}")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify only; do not download")
    sys.exit(check() if ap.parse_args().check else fetch())

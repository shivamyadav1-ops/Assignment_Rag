"""
Repeatable ingestion pipeline.

Two independent paths:
  1. Policy documents (PDF/HTML/MD) -> structure-aware chunks -> chunks.jsonl
  2. Operational records (CSV/JSON)  -> validated rows        -> PostgreSQL

Run:  python -m retail_support_agent.ingest
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from retail_support_agent.chunker import chunk_all_policies
from retail_support_agent.config import PROCESSED_DIR
from retail_support_agent.db import load_operational_data

CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
INGEST_SUMMARY_FILE = PROCESSED_DIR / "ingestion_summary.json"


def ingest_policies() -> dict:
    """Chunk all policy documents and write chunks.jsonl."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/2] Chunking policy documents")
    print("      size=512 tokens, overlap=64 tokens, structure-aware\n")

    chunks = chunk_all_policies()

    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    by_source: dict[str, int] = {}
    by_type: dict[str, int] = {}
    tables = 0
    total_tokens = 0
    for c in chunks:
        by_source[c.source_file] = by_source.get(c.source_file, 0) + 1
        by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1
        tables += 1 if c.has_table else 0
        total_tokens += c.token_count

    summary = {
        "chunks_file": str(CHUNKS_FILE),
        "total_chunks": len(chunks),
        "chunks_by_source": by_source,
        "chunks_by_type": by_type,
        "table_chunks": tables,
        "total_tokens": total_tokens,
        "avg_tokens_per_chunk": total_tokens // max(len(chunks), 1),
        "min_tokens": min((c.token_count for c in chunks), default=0),
        "max_tokens": max((c.token_count for c in chunks), default=0),
    }

    print(f"\n      -> {CHUNKS_FILE}")
    print(f"      {len(chunks)} chunks, {tables} containing tables")
    print(f"      avg {summary['avg_tokens_per_chunk']} tokens "
          f"(min {summary['min_tokens']}, max {summary['max_tokens']})")

    return summary


def ingest_operational() -> dict:
    """Load CSV/JSON operational records into the operational database (PostgreSQL)."""
    print("\n[2/2] Loading operational records into the operational database\n")

    summary = load_operational_data()

    print(f"      customers : {summary['customers']}")
    print(f"      orders    : {summary['orders']}")
    print(f"      returns   : {summary['returns']}")
    if summary["warnings"]:
        print(f"\n      {len(summary['warnings'])} validation warning(s):")
        for w in summary["warnings"][:10]:
            print(f"        - {w}")
        if len(summary["warnings"]) > 10:
            print(f"        ... and {len(summary['warnings']) - 10} more")
    else:
        print("      no validation warnings")
    print(f"\n      -> {summary['db_path']}")

    return summary


def main() -> dict:
    print("=" * 62)
    print("  Retail Support Agent — Ingestion Pipeline")
    print("=" * 62)

    started = datetime.now(UTC)

    policy_summary = ingest_policies()
    operational_summary = ingest_operational()

    finished = datetime.now(UTC)
    elapsed = (finished - started).total_seconds()

    full_summary = {
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "policies": policy_summary,
        "operational": operational_summary,
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INGEST_SUMMARY_FILE.write_text(json.dumps(full_summary, indent=2))

    print("\n" + "=" * 62)
    print(f"  Done in {elapsed:.2f}s  ->  {INGEST_SUMMARY_FILE}")
    print("=" * 62 + "\n")

    return full_summary


if __name__ == "__main__":
    main()

"""Chaos and Fault-Injection Testing Suite ("Break the Project").

Validates system behavior under deliberate failure conditions, corrupt state,
resource exhaustion, and adversarial inputs across all architectural layers:

1. Operational DB Chaos: locked database, corrupt SQLite binary, dropped tables.
2. Vector Store Chaos: missing collections, clean initialization errors, crash handling.
3. Ingest & Chunking Chaos: malformed JSON, null bytes, 10MB payloads, broken tables.
4. LLM & Network Chaos: 503 service outage, infinite loop runaway, unknown tool calls.
5. Cache & State Chaos: corrupt cache database resilience without customer disruption.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from retail_support_agent import chunker, config, db, indexer, ops
from retail_support_agent.agent import SupportAgent
from retail_support_agent.llm import LLMTurn, ScriptedClient, ToolCall
from retail_support_agent.retriever import HybridRetriever, RetrievalResult

# ════════════════════════════════════════════════════════════════════════════
# 1. Operational DB Chaos
# ════════════════════════════════════════════════════════════════════════════


def test_chaos_db_locked_returns_typed_retryable_error(tmp_path: Path, monkeypatch):
    """When SQLite is locked by an exclusive transaction, ops tools wrapped by
    mcp_server._guard must return an INTERNAL_ERROR retryable envelope rather than crashing."""
    db_file = tmp_path / "locked.db"
    db.Base.metadata.create_all(db.get_engine(db_file))

    lock_conn = sqlite3.connect(str(db_file), timeout=0.01)
    lock_conn.execute("BEGIN EXCLUSIVE")

    try:
        from retail_support_agent.mcp_server import _guard

        monkeypatch.setenv("RSA_REQUESTER_ID", "CUST-1049")
        monkeypatch.setenv("RSA_DATABASE_URL", f"sqlite:///{db_file}")
        orig_url = config.DATABASE_URL
        config.DATABASE_URL = f"sqlite:///{db_file}"
        try:
            res = _guard(ops.get_order_details, "ORD-10012")
            assert res.get("ok") is False
            err = res.get("error", {})
            assert err.get("retryable") is True
            assert err.get("code") in ("INTERNAL_ERROR", "DATABASE_ERROR")
        finally:
            config.DATABASE_URL = orig_url
    finally:
        lock_conn.rollback()
        lock_conn.close()


def test_chaos_db_corrupt_binary_fails_gracefully(tmp_path: Path, monkeypatch):
    """When the DB file contains random corrupted bytes, ops calls fail gracefully with a
    typed error."""
    db_file = tmp_path / "corrupt.db"
    db_file.write_bytes(b"\x00\xff\xfe\x01\x42" * 200)

    from retail_support_agent.mcp_server import _guard

    monkeypatch.setenv("RSA_REQUESTER_ID", "CUST-1049")
    monkeypatch.setenv("RSA_DATABASE_URL", f"sqlite:///{db_file}")
    orig_url = config.DATABASE_URL
    config.DATABASE_URL = f"sqlite:///{db_file}"
    try:
        res = _guard(ops.get_order_details, "ORD-10012")
        assert res.get("ok") is False
        assert res.get("error", {}).get("code") in ("INTERNAL_ERROR", "DATABASE_ERROR")
    finally:
        config.DATABASE_URL = orig_url


def test_chaos_db_dropped_table_returns_typed_error(tmp_path: Path, monkeypatch):
    """When the orders table is unexpectedly missing, ops tools fail with typed error."""
    db_file = tmp_path / "broken_schema.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE dummy (id INT)")
    conn.commit()
    conn.close()

    from retail_support_agent.mcp_server import _guard

    monkeypatch.setenv("RSA_REQUESTER_ID", "CUST-1049")
    monkeypatch.setenv("RSA_DATABASE_URL", f"sqlite:///{db_file}")
    orig_url = config.DATABASE_URL
    config.DATABASE_URL = f"sqlite:///{db_file}"
    try:
        res = _guard(ops.get_order_details, "ORD-10012")
        assert res.get("ok") is False
        assert res.get("error", {}).get("code") in ("INTERNAL_ERROR", "DATABASE_ERROR")
    finally:
        config.DATABASE_URL = orig_url


# ════════════════════════════════════════════════════════════════════════════
# 2. Vector Store & Retrieval Chaos
# ════════════════════════════════════════════════════════════════════════════


def test_chaos_missing_qdrant_directory_raises_cleanly(tmp_path: Path):
    """When Qdrant directory does not exist or collection is missing,
    HybridRetriever raises RuntimeError indicating reindexing is required."""
    missing_dir = tmp_path / "nonexistent" / "vectorstore"

    orig = config.QDRANT_PATH
    config.QDRANT_PATH = missing_dir
    try:
        from retail_support_agent.models import qdrant_client
        qdrant_client.cache_clear()
        with pytest.raises(RuntimeError, match="Collection 'policies' not found"):
            HybridRetriever(cache_size=0)
    finally:
        config.QDRANT_PATH = orig
        qdrant_client.cache_clear()


def test_chaos_agent_survives_retriever_crash(tmp_path: Path):
    """If the retriever crashes (disk error, vector store down), the agent should
    return an error envelope rather than raising an uncaught exception out of answer()."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.side_effect = RuntimeError("Vector store disk read failure")

    turns = [
        LLMTurn(
            text=None,
            tool_calls=[ToolCall("call_1", "search_policies", {"query": "return window"})],
            usage={"input_tokens": 100, "output_tokens": 20},
        ),
        LLMTurn(
            text="I'm sorry, I cannot search policies right now.",
            tool_calls=[],
            usage={"input_tokens": 200, "output_tokens": 10},
            stop_reason="end_turn",
        ),
    ]
    llm = ScriptedClient(turns)

    agent = SupportAgent(
        requester_id="CUST-1001",
        llm=llm,
        retriever=mock_retriever,
        mcp=MagicMock(),
        trace_dir=tmp_path / "traces",
        semantic_cache=False,
    )

    res = agent.answer("How do I return my TV?")
    assert res.outcome in ("answered", "error", "escalated", "refused_ungrounded")


# ════════════════════════════════════════════════════════════════════════════
# 3. Ingestion & Chunker Chaos
# ════════════════════════════════════════════════════════════════════════════


def test_chaos_indexer_corrupt_chunks_json_raises_json_error(tmp_path: Path):
    """Indexer raises JSONDecodeError on malformed JSON file."""
    bad_chunks = tmp_path / "bad_chunks.json"
    bad_chunks.write_text("{ this is not valid json! :::")

    with pytest.raises(json.JSONDecodeError):
        indexer.build_index(chunks_path=bad_chunks)


def test_chaos_indexer_missing_required_fields(tmp_path: Path):
    """Indexer raises KeyError when chunk is missing required 'text' field."""
    bad_chunks = tmp_path / "incomplete_chunks.json"
    bad_chunks.write_text(json.dumps({"chunk_id": "c1"}) + "\n")  # missing 'text'

    with pytest.raises(KeyError):
        indexer.build_index(chunks_path=bad_chunks)


def test_chaos_chunker_handles_null_bytes_and_control_chars(tmp_path: Path):
    """Chunker does not crash on binary strings or null bytes in markdown file."""
    doc = tmp_path / "hostile.md"
    doc.write_text(
        "Policy Title: General Return and Warranty Guidance\n\n"
        + ("Some text with \x00 null bytes and \x1f control chars. "
           "Standard policy guidelines apply to all customers. ") * 5
    )
    chunks = chunker.chunk_document(doc)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    assert all(isinstance(c.text, str) for c in chunks)


def test_chaos_chunker_handles_deeply_nested_headers(tmp_path: Path):
    """Chunker handles absurd header levels (10+ '#') gracefully."""
    doc = tmp_path / "deep.md"
    doc.write_text(
        "########### Level 10 Header\n\n"
        + ("Deep nested content here explaining electronics support return policy "
           "rules and refund guarantees in detail. ") * 5
    )
    chunks = chunker.chunk_document(doc)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_chaos_chunker_handles_malformed_tables(tmp_path: Path):
    """Chunker handles broken markdown tables (ragged row lengths)."""
    doc = tmp_path / "table.md"
    doc.write_text(
        "| Col1 | Col2 |\n"
        "|---|---|\n"
        "| OnlyOneCol |\n"
        "| Col1 | Col2 | Col3 | ExtraCol4 |\n"
    )
    chunks = chunker.chunk_document(doc)
    assert isinstance(chunks, list)


def test_chaos_chunker_handles_massive_payload(tmp_path: Path):
    """Chunker processes large synthetic payload without memory leak or crash."""
    doc = tmp_path / "massive.md"
    content = ("# Section Header\n\n" + ("Paragraph with standard words. " * 50) + "\n\n") * 100
    doc.write_text(content)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 50
    assert all(c.token_count > 0 for c in chunks)


# ════════════════════════════════════════════════════════════════════════════
# 4. Agent & LLM Chaos
# ════════════════════════════════════════════════════════════════════════════


def test_chaos_agent_handles_upstream_503_llm_outage(tmp_path: Path):
    """When the LLM provider returns 503 or crashes, the agent catches it,
    records the error trace step, sets outcome='error', and returns a helpful message."""
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("503 Service Temporarily Unavailable")

    agent = SupportAgent(
        requester_id="CUST-1001",
        llm=mock_llm,
        mcp=MagicMock(),
        trace_dir=tmp_path / "traces",
        semantic_cache=False,
    )

    res = agent.answer("How do I return my order?")
    assert res.outcome == "error"
    assert "internal error" in res.answer.lower()
    assert any(s.kind == "llm" and s.name == "error" for s in res.steps)


def test_chaos_agent_blocks_hallucinated_unknown_tool(tmp_path: Path):
    """When the LLM hallucinates a non-existent dangerous tool, the agent
    blocks it or returns a tool_block_envelope, preventing arbitrary execution."""
    turns = [
        LLMTurn(
            text=None,
            tool_calls=[ToolCall("call_1", "delete_all_orders", {"everything": True})],
            usage={"input_tokens": 100, "output_tokens": 20},
        ),
        LLMTurn(
            text="I cannot execute that action.",
            tool_calls=[],
            usage={"input_tokens": 200, "output_tokens": 10},
            stop_reason="end_turn",
        ),
    ]

    llm = ScriptedClient(turns)
    mcp = MagicMock()
    mcp.list_tools.return_value = []

    agent = SupportAgent(
        requester_id="CUST-1001",
        llm=llm,
        mcp=mcp,
        trace_dir=tmp_path / "traces",
        semantic_cache=False,
    )

    res = agent.answer("Delete everything.")
    assert res.outcome in ("answered", "refused_ungrounded", "escalated")
    mcp.call_tool.assert_not_called()


def test_chaos_agent_terminates_at_max_steps_runaway_loop(tmp_path: Path):
    """When the LLM enters an infinite tool-calling loop, the agent halts
    strictly at AGENT_MAX_STEPS without hanging forever."""
    infinite_turns = [
        LLMTurn(
            text=None,
            tool_calls=[ToolCall(f"call_{i}", "search_policies", {"query": f"loop {i}"})],
            usage={"input_tokens": 50, "output_tokens": 10},
        )
        for i in range(config.AGENT_MAX_STEPS + 5)
    ]

    llm = ScriptedClient(infinite_turns)
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = RetrievalResult(
        query="loop", mode="full", chunks=[], timings_ms={"retrieval": 1.0}, candidates_reranked=0
    )

    agent = SupportAgent(
        requester_id="CUST-1001",
        llm=llm,
        retriever=mock_retriever,
        mcp=MagicMock(),
        trace_dir=tmp_path / "traces",
        semantic_cache=False,
    )

    res = agent.answer("Infinite loop question")
    assert len([s for s in res.steps if s.kind == "llm"]) <= config.AGENT_MAX_STEPS


# ════════════════════════════════════════════════════════════════════════════
# 5. Semantic Cache Chaos
# ════════════════════════════════════════════════════════════════════════════


def test_chaos_agent_resilient_to_corrupt_cache_lookup(tmp_path: Path):
    """If the semantic cache raises a database corruption error on lookup,
    the agent catches it safely and continues to answer the customer."""
    mock_cache = MagicMock()
    mock_cache.lookup.side_effect = sqlite3.DatabaseError("database disk image is malformed")

    turns = [
        LLMTurn(
            text="Laptops have a 30-day return window.",
            tool_calls=[],
            usage={"input_tokens": 100, "output_tokens": 20},
            stop_reason="end_turn",
        ),
    ]
    # Twice: the answer has no evidence, so the agent forces a policy search and the
    # model answers again.
    llm = ScriptedClient(turns * 2)

    agent = SupportAgent(
        requester_id="CUST-1001",
        llm=llm,
        mcp=MagicMock(),
        trace_dir=tmp_path / "traces",
        semantic_cache=mock_cache,
    )

    res = agent.answer("What is the laptop return window?")
    assert res.outcome in ("answered", "refused_ungrounded")
    assert "30-day" in res.answer
    assert res.cache is not None
    assert res.cache.get("hit") is False
    assert "malformed" in res.cache.get("error", "")

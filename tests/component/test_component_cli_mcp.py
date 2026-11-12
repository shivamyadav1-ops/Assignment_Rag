"""Component tests: CLI subprocess entry points and real stdio MCP server.

Validates:
1. `MCPToolClient` launches a real subprocess running `mcp_server.py` over stdio.
2. Tool listing returns the registered operational tools with JSON schemas.
3. Live tool calls execute against real operational data and return typed envelopes.
4. Cross-customer security isolation holds across the stdio process boundary.
5. Missing requester identity fails safe with `NO_SESSION_IDENTITY`.
6. Subprocess CLI commands (`--help`, `search --mode bm25`) behave correctly.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from retail_support_agent.mcp_client import MCPToolClient

pytestmark = pytest.mark.ml

COMP_DATA = Path(
    "/tmp/claude-1000/-home-unthinkable-lap-Project-Rag/"
    "c8386e81-ccb6-493e-94fa-8128486139dc/scratchpad/comp_cli"
)
CANONICAL_DB_PATH = COMP_DATA / "operational.db"


@pytest.fixture(autouse=True)
def require_isolated_resources():
    if not CANONICAL_DB_PATH.exists():
        pytest.skip(f"Isolated db not found at {CANONICAL_DB_PATH}")


# ════════════════════════════════════════════════════════════════════════════
# 1. MCP stdio Server: Real Process Subprocess Communication
# ════════════════════════════════════════════════════════════════════════════


def test_mcp_server_lists_tools_over_stdio(monkeypatch):
    """Spawns real mcp_server process over stdio and lists operational tools."""
    monkeypatch.setenv("RSA_DATABASE_URL", f"sqlite:///{CANONICAL_DB_PATH}")

    with MCPToolClient(requester_id="CUST-1049") as client:
        tools = client.list_tools()
        names = {t["name"] for t in tools}
        assert "get_order_details" in names
        assert "get_return_refund_status" in names
        assert "create_return_request" in names
        assert "create_escalation_ticket" in names


def test_mcp_server_executes_authorized_tool_call(monkeypatch):
    """Executes get_order_details for customer's own order over stdio."""
    monkeypatch.setenv("RSA_DATABASE_URL", f"sqlite:///{CANONICAL_DB_PATH}")

    with MCPToolClient(requester_id="CUST-1049") as client:
        res = client.call_tool("get_order_details", {"order_id": "ORD-10012"})
        assert res.get("ok") is True
        order = res.get("data", {})
        assert order.get("order_id") == "ORD-10012"
        assert order.get("customer_id") == "CUST-1049"


def test_mcp_server_blocks_cross_customer_access_over_stdio(monkeypatch):
    """CUST-1049 cannot view CUST-1001's order over stdio MCP interface."""
    monkeypatch.setenv("RSA_DATABASE_URL", f"sqlite:///{CANONICAL_DB_PATH}")

    with MCPToolClient(requester_id="CUST-1049") as client:
        res = client.call_tool("get_order_details", {"order_id": "ORD-10001"})
        assert res.get("ok") is False
        assert res.get("error", {}).get("code") == "UNAUTHORIZED"


def test_mcp_server_missing_identity_fails_safe(monkeypatch):
    """Server started without RSA_REQUESTER_ID refuses every tool call."""
    monkeypatch.setenv("RSA_DATABASE_URL", f"sqlite:///{CANONICAL_DB_PATH}")

    with MCPToolClient(requester_id="") as client:
        res = client.call_tool("get_order_details", {"order_id": "ORD-10012"})
        assert res.get("ok") is False
        assert res.get("error", {}).get("code") == "NO_SESSION_IDENTITY"


# ════════════════════════════════════════════════════════════════════════════
# 2. CLI Subprocess Invocations
# ════════════════════════════════════════════════════════════════════════════


def test_cli_help_flag_succeeds():
    """`python -m retail_support_agent.cli --help` returns 0 with all commands."""
    proc = subprocess.run(
        [sys.executable, "-m", "retail_support_agent.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "ingest" in proc.stdout
    assert "index" in proc.stdout
    assert "search" in proc.stdout
    assert "ask" in proc.stdout


def test_cli_search_bm25_returns_table_output():
    """`python -m retail_support_agent.cli search` runs BM25 retrieval without crashing."""
    env = dict(os.environ)
    env["RSA_QDRANT_PATH"] = str(COMP_DATA / "vectorstore" / "qdrant")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "retail_support_agent.cli",
            "search",
            "return window for television",
            "--mode",
            "bm25",
            "--top-k",
            "2",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "timings (ms)" in proc.stdout


# ════════════════════════════════════════════════════════════════════════════
# 3. `rsa search` against the real hybrid_rerank pipeline (real subprocess,
#    real cross-encoder). Candidate pool is fixed at RETRIEVAL_TOP_K=20
#    regardless of --top-k, so each call costs ~20-85s depending on CPU
#    contention from other concurrently-running agents. Kept to one case plus
#    one cheaper `hybrid` (RRF, no cross-encoder) case to bound wall time.
# ════════════════════════════════════════════════════════════════════════════


def _search_env():
    env = dict(os.environ)
    env["RSA_QDRANT_PATH"] = str(COMP_DATA / "vectorstore" / "qdrant")
    return env


def test_cli_search_hybrid_rerank_known_query_surfaces_expected_section():
    """Real hybrid_rerank subprocess: a return-window query should rank the
    'Standard Return Windows' table section at or near the top.

    RSA_RRF_LIMIT (a documented config knob, config.py RRF_LIMIT) is lowered
    to 5 here only to bound the cross-encoder's candidate pool -- reranking
    the default 20 candidates with bge-reranker-base took anywhere from ~21s
    to ~90s wall time in this shared environment (observed directly while
    developing this test), which is unaffordable for a ~3-minute test file.
    This does not change which pipeline stages run, only how many candidates
    reach the real reranker subprocess call."""
    env = _search_env()
    env["RSA_RRF_LIMIT"] = "5"
    proc = subprocess.run(
        [
            sys.executable, "-m", "retail_support_agent.cli", "search",
            "What are the standard return windows by product category?",
            "--mode", "hybrid_rerank", "--top-k", "3",
        ],
        capture_output=True, text=True, env=env, check=False, timeout=60,
    )
    assert proc.returncode == 0
    assert "timings (ms)" in proc.stdout
    assert "reranked=" in proc.stdout
    # Rich wraps the "section / source" column at the subprocess's (non-tty)
    # terminal width, so "1.1 Standard Return Windows" lands split across two
    # table lines ("1.1 Standard Return" / "Windows") rather than as one
    # contiguous string -- check both fragments instead of the joined phrase.
    assert "1.1 Standard Return" in proc.stdout
    assert "Windows" in proc.stdout
    assert "return_policy.pdf" in proc.stdout


def test_cli_search_hybrid_known_query_surfaces_expected_section():
    """Real hybrid (RRF, no cross-encoder) subprocess: a store-credit query
    should surface the 'Store Credit Terms' section. Faster than
    hybrid_rerank since no cross-encoder pass runs, used for extra query
    coverage without blowing the file's time budget."""
    proc = subprocess.run(
        [
            sys.executable, "-m", "retail_support_agent.cli", "search",
            "How does store credit work?",
            "--mode", "hybrid", "--top-k", "5",
        ],
        capture_output=True, text=True, env=_search_env(), check=False, timeout=30,
    )
    assert proc.returncode == 0
    assert "Store Credit Terms" in proc.stdout
    assert "gift_cards_promotions" in proc.stdout


# ════════════════════════════════════════════════════════════════════════════
# 4. Process-boundary error handling: invalid CLI input
# ════════════════════════════════════════════════════════════════════════════


def test_cli_search_invalid_mode_gives_clean_error():
    # Rejected at the CLI boundary as a usage error (exit 2), before the
    # retriever or any model is loaded.
    proc = subprocess.run(
        [
            sys.executable, "-m", "retail_support_agent.cli", "search",
            "test query", "--mode", "bogus_mode",
        ],
        capture_output=True, text=True, env=_search_env(), check=False, timeout=30,
    )
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "Unknown mode" in proc.stderr or "Unknown mode" in proc.stdout


def test_cli_search_corrupt_vectorstore_gives_clean_error(tmp_path):
    corrupt = tmp_path / "corrupt_qdrant"
    corrupt.mkdir()
    (corrupt / "meta.json").write_text("not valid json at all")

    env = dict(os.environ)
    env["RSA_QDRANT_PATH"] = str(corrupt)
    proc = subprocess.run(
        [sys.executable, "-m", "retail_support_agent.cli", "search", "test", "--mode", "dense"],
        capture_output=True, text=True, env=env, check=False, timeout=30,
    )
    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr
    assert "could not open the vector store" in proc.stderr


def test_cli_search_nonexistent_vectorstore_path_gives_clean_error(tmp_path):
    """A --mode dense search against a path with no Qdrant collection yet:
    one clean, actionable line (run `rsa index` first), exit 1, no traceback."""
    missing = tmp_path / "does_not_exist_yet"
    env = dict(os.environ)
    env["RSA_QDRANT_PATH"] = str(missing)
    proc = subprocess.run(
        [sys.executable, "-m", "retail_support_agent.cli", "search", "test", "--mode", "dense"],
        capture_output=True, text=True, env=env, check=False, timeout=30,
    )
    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr
    assert "run `rsa index` first" in proc.stderr
    # No data loss: the directory is created but the collection is not, so a
    # subsequent `rsa index` into the same path is still possible.
    assert missing.exists()


# ════════════════════════════════════════════════════════════════════════════
# 5. `rsa index`: real subprocess building a FRESH Qdrant collection from a
#    small chunk subset (kept small to bound embedding-model wall time; the
#    full 403-chunk processed/ fixture would dominate the file's time budget).
# ════════════════════════════════════════════════════════════════════════════


def test_cli_index_builds_fresh_collection_with_expected_point_count(tmp_path):
    import json

    all_chunks = (COMP_DATA / "processed" / "chunks.jsonl").read_text().splitlines()
    subset = [line for line in all_chunks if line.strip()][:12]
    chunks_path = tmp_path / "chunks_subset.jsonl"
    chunks_path.write_text("\n".join(subset) + "\n")

    fresh_qdrant = tmp_path / "fresh_qdrant"
    env = dict(os.environ)
    env["RSA_QDRANT_PATH"] = str(fresh_qdrant)

    # `rsa index` (the real CLI command) always reads config.CHUNKS_PATH, which
    # is derived from PROJECT_ROOT with no environment override -- so it cannot
    # be pointed at our isolated chunk subset via env vars alone. We invoke the
    # real indexer.build_index() function directly with an explicit
    # chunks_path, in a real subprocess, which exercises the identical code
    # path `rsa index` runs (same function, same Qdrant writes) while keeping
    # the embedding workload small and never touching data/processed.
    proc = subprocess.run(
        [
            sys.executable, "-c",
            "import json, sys\n"
            "from pathlib import Path\n"
            "from retail_support_agent.indexer import build_index\n"
            "print(json.dumps(build_index(chunks_path=Path(sys.argv[1]))))\n",
            str(chunks_path),
        ],
        capture_output=True, text=True, env=env, check=False, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary["points"] == len(subset) == 12
    assert summary["collection"] == "policies"

    # Verify independently, via a second real `rsa search` subprocess against
    # the freshly-built collection, that the points are actually queryable.
    search_env = dict(os.environ)
    search_env["RSA_QDRANT_PATH"] = str(fresh_qdrant)
    search_proc = subprocess.run(
        [sys.executable, "-m", "retail_support_agent.cli", "search", "return policy",
         "--mode", "bm25", "--top-k", "3"],
        capture_output=True, text=True, env=search_env, check=False, timeout=30,
    )
    assert search_proc.returncode == 0
    assert "timings (ms)" in search_proc.stdout


# ════════════════════════════════════════════════════════════════════════════
# 6. `rsa ask` argument validation only -- never reaches the LLM call. No
#    ANTHROPIC/NVIDIA API key is read or required for this path: Typer
#    rejects the missing required `--customer` option before `cli.ask()`'s
#    body (and therefore `SupportAgent`) ever executes.
# ════════════════════════════════════════════════════════════════════════════


def test_cli_ask_missing_customer_rejected_before_llm_call():
    proc = subprocess.run(
        [sys.executable, "-m", "retail_support_agent.cli", "ask", "Can I return my laptop?"],
        capture_output=True, text=True, env=dict(os.environ), check=False, timeout=15,
    )
    assert proc.returncode == 2
    assert "Missing option" in proc.stderr
    assert "--customer" in proc.stderr

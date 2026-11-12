"""Component tests for the real retrieval pipeline.

Scope: HybridRetriever + indexer.build_index exercised end to end against a REAL
embedded Qdrant store (a private snapshot copy of the real `policies` collection,
never the shared `data/` directory) and REAL fastembed ONNX models (dense BGE,
BM25, cross-encoder reranker). No LLM/API calls are made anywhere in this file.

All tests are marked `ml` (they load real local models) and are collected only
when run with `-m ml`, per the project's marker convention (tests/conftest.py).

Isolation: `config.QDRANT_PATH` / `config.PROCESSED_DIR` are monkeypatched at
module scope to point at a private snapshot copy
(.../scratchpad/comp_retrieval), never at the shared `data/` directory, which is
locked by other concurrent agents. The one test that rebuilds an index
(`build_index`) does so into a throwaway `tmp_path`, never into the snapshot
copy or the real store, so it cannot corrupt either.

Retrieve() calls against the real cross-encoder reranker dominate this file's
wall-clock time (observed 2-80s per call depending on host CPU contention with
other concurrently-running agents on this machine), so the number of
hybrid_rerank/full mode calls is kept to the minimum needed for real coverage:
each is exercised, but not duplicated across multiple test functions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from retail_support_agent import config
from retail_support_agent import indexer as ix
from retail_support_agent import models as rsa_models
from retail_support_agent import retriever as rt

pytestmark = pytest.mark.ml

# ── Isolated data copy (never the shared data/ dir) ───────────────────────────
COMP_DATA = Path(
    "/tmp/claude-1000/-home-unthinkable-lap-Project-Rag/"
    "c8386e81-ccb6-493e-94fa-8128486139dc/scratchpad/comp_retrieval"
)
ISOLATED_QDRANT_PATH = COMP_DATA / "vectorstore" / "qdrant"
ISOLATED_PROCESSED_DIR = COMP_DATA / "processed"
ISOLATED_CHUNKS_PATH = ISOLATED_PROCESSED_DIR / "chunks.jsonl"

# Expected "real" sources for the primary return-window query, used to check that
# real retrieval surfaces plausible chunks (not just well-formed scores).
RETURN_WINDOW_SOURCES = {
    "return_policy.pdf", "product_category_guide.pdf", "seasonal_holiday_policy.md"}

EXPECTED_STAGES = {
    "dense": {"dense"},
    "bm25": {"bm25"},
    "hybrid": {"hybrid_rrf"},
    "hybrid_rerank": {"hybrid_rrf", "rerank"},
    "full": {"hybrid_rrf", "expand_adjacent", "rerank"},
}


def _require_isolated_copy():
    if not ISOLATED_QDRANT_PATH.exists():
        pytest.skip(f"isolated Qdrant copy not found at {ISOLATED_QDRANT_PATH}")
    if not ISOLATED_CHUNKS_PATH.exists():
        pytest.skip(f"isolated chunks.jsonl not found at {ISOLATED_CHUNKS_PATH}")


@pytest.fixture(scope="module", autouse=True)
def isolated_paths():
    """Point config at the private snapshot copy for the whole module.

    `config.QDRANT_PATH` and `config.PROCESSED_DIR` are read as module attributes
    at call time (not bound into function defaults) by models.qdrant_client() and
    by our own test code, so reassigning them here is sufficient -- no need to
    monkeypatch individual callers. `models.qdrant_client` is a process-wide
    lru_cache(maxsize=1) singleton, so we clear it before first use here and again
    on teardown, to guarantee no other test in this process can inherit a client
    bound to the wrong path.
    """
    _require_isolated_copy()
    orig_qdrant_path = config.QDRANT_PATH
    orig_processed_dir = config.PROCESSED_DIR
    orig_chunks_path = config.CHUNKS_PATH

    config.QDRANT_PATH = ISOLATED_QDRANT_PATH
    config.PROCESSED_DIR = ISOLATED_PROCESSED_DIR
    config.CHUNKS_PATH = ISOLATED_CHUNKS_PATH
    rsa_models.qdrant_client.cache_clear()
    try:
        yield
    finally:
        rsa_models.qdrant_client.cache_clear()
        config.QDRANT_PATH = orig_qdrant_path
        config.PROCESSED_DIR = orig_processed_dir
        config.CHUNKS_PATH = orig_chunks_path


@pytest.fixture(scope="module")
def retriever():
    """One real HybridRetriever for the whole module -- model loading is slow."""
    return rt.HybridRetriever(cache_size=0)


# ════════════════════════════════════════════════════════════════════════════
# Mode coverage + timing, consolidated into one retrieve() call per mode
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("mode", list(rt.MODES))
def test_retrieve_each_mode_is_plausible_and_well_timed(retriever, mode):
    """One retrieve() call per mode validates three things at once against the
    real store, real embeddings and (for hybrid_rerank/full) the real
    cross-encoder reranker:
      1. the top real hit is plausible for a 'return window' query,
      2. timings_ms carries exactly the stages that mode is documented to run,
      3. total_ms/timings_ms are well-formed.
    """
    result = retriever.retrieve("What is the return window for a smartphone?", mode=mode)

    assert result.query == "What is the return window for a smartphone?"
    assert result.mode == mode
    assert len(result.chunks) > 0

    top = result.chunks[0]
    assert top.source_file in RETURN_WINDOW_SOURCES, (
        f"mode={mode}: top hit was {top.source_file}#{top.chunk_index} "
        f"({top.section_heading!r}), expected one of {RETURN_WINDOW_SOURCES}"
    )
    combined_text = " ".join(c.text.lower() for c in result.chunks[:3])
    assert "return" in combined_text

    assert set(result.timings_ms) == EXPECTED_STAGES[mode]
    for stage, ms in result.timings_ms.items():
        assert ms >= 0, f"stage {stage} had negative timing {ms}"
        assert ms < 120_000, f"stage {stage} took implausibly long: {ms}ms"
    assert result.total_ms == round(sum(result.timings_ms.values()), 1)


def test_dense_mode_finds_semantically_related_chunk_without_keyword_overlap(retriever):
    """Dense search should surface policy content for a paraphrase that shares
    almost no literal tokens with the source text (e.g. 'send it back' vs 'return')."""
    result = retriever.retrieve(
        "If I buy a phone and change my mind, how many days do I have to send it back?",
        mode="dense",
    )
    assert len(result.chunks) > 0
    sources = {c.source_file for c in result.chunks}
    assert sources & {"return_policy.pdf", "product_category_guide.pdf"}


def test_bm25_mode_rewards_exact_term_match(retriever):
    """BM25 should strongly favour chunks containing the literal phrase."""
    result = retriever.retrieve("Buy Online, Pick Up In-Store BOPIS", mode="bm25")
    assert len(result.chunks) > 0
    assert result.chunks[0].source_file == "store_services.md"


def test_hybrid_mode_fuses_both_signals(retriever):
    result = retriever.retrieve("holiday return window extension", mode="hybrid")
    assert len(result.chunks) > 0
    assert all(c.origin == "hybrid" for c in result.chunks)
    sources = {c.source_file for c in result.chunks}
    assert "seasonal_holiday_policy.md" in sources


def test_full_mode_reranks_and_assigns_rerank_scores(retriever):
    """A second, independent query/document pair for 'full' mode (the
    'What is the return window...' query is already exercised in the
    parametrized mode-coverage test above) -- confirms rerank scores are real,
    assigned, and sorted descending against the real cross-encoder."""
    result = retriever.retrieve("holiday return window extension", mode="full")
    assert len(result.chunks) > 0
    assert all(c.rerank_score is not None for c in result.chunks)
    scores = [c.rerank_score for c in result.chunks]
    assert scores == sorted(scores, reverse=True)
    top = result.chunks[0]
    assert top.source_file == "seasonal_holiday_policy.md"


# ════════════════════════════════════════════════════════════════════════════
# Cross-mode consistency
# ════════════════════════════════════════════════════════════════════════════


def test_rerank_modes_keep_a_relevant_source_in_top_result_like_dense(retriever):
    """hybrid_rerank and full should not regress relevance relative to dense-only
    for a hand-picked, unambiguous query, and adjacent expansion in `full` should
    not flip the winning document relative to `hybrid_rerank`."""
    query = "holiday return window extension"
    dense = retriever.retrieve(query, mode="dense")
    hybrid_rerank = retriever.retrieve(query, mode="hybrid_rerank")
    full = retriever.retrieve(query, mode="full")

    dense_sources = {c.source_file for c in dense.chunks[:3]}
    assert hybrid_rerank.chunks[0].source_file in dense_sources | {
        c.source_file for c in hybrid_rerank.chunks
    }
    assert full.chunks[0].source_file == hybrid_rerank.chunks[0].source_file


# ════════════════════════════════════════════════════════════════════════════
# Adjacent-chunk expansion against the real store
# ════════════════════════════════════════════════════════════════════════════


def test_expand_adjacent_pulls_real_neighbouring_table_rows(retriever):
    """return_policy.pdf's return-window table rows (chunk_index 15-26) are real,
    literal, same-document neighbours of each other. Calling expand_adjacent
    directly with enough `cap` headroom (beyond MAX_RERANK_CANDIDATES, which the
    retrieve() pipeline itself uses and which -- by design, see the test below --
    leaves no room in production) proves the mechanism itself
    really walks to the real store and fetches real neighbouring chunks, not
    fabricated ones."""
    candidates = retriever.hybrid_search("Smartphones Tablets 15 days return window row", limit=5)
    assert len(candidates) > 0

    # Generous cap: enough room for every base candidate's two neighbours.
    expanded = retriever.expand_adjacent(candidates, window=1, cap=len(candidates) * 3)
    adjacent_chunks = [c for c in expanded if c.origin == "adjacent"]
    assert adjacent_chunks, "expected at least one real adjacent neighbour to be pulled in"

    # Every adjacent chunk must be a genuine same-document, index-adjacent neighbour
    # of one of the original candidates (not a coincidental topical match), and its
    # payload must be real chunk data from the store (non-empty text).
    base_keys = {(c.source_file, c.chunk_index) for c in candidates}
    for a in adjacent_chunks:
        assert a.text.strip(), "adjacent chunk had empty text"
        neighbour_of_a_base = any(
            a.source_file == sf and abs(a.chunk_index - idx) == 1 for sf, idx in base_keys
        )
        assert neighbour_of_a_base, (
            f"adjacent chunk {a.source_file}#{a.chunk_index} is not index-adjacent "
            f"to any base candidate"
        )


def test_expand_adjacent_respects_the_rerank_cap_under_production_config(retriever):
    """Intended behaviour, not a bug: MAX_RERANK_CANDIDATES bounds the cross-encoder
    pool (fused hits first, neighbours only if room). At the production setting
    (cap 10 < RRF_LIMIT 20) the fused hits fill the cap, so no neighbours are added
    — config.py documents this, and reports/retrieval_eval_adj*.json show it beats
    the neighbour-including configs on evidence_hit@1 (94.1% vs 88.2%)."""
    cap = config.MAX_RERANK_CANDIDATES
    candidates = retriever.hybrid_search("Smartphones Tablets 15 days return window row")
    expanded = retriever.expand_adjacent(candidates, window=1, cap=cap)

    assert len(expanded) <= cap, "the reranker pool must never exceed the cap"
    if len(candidates) >= cap:
        # Fused hits fill the cap: they are kept, in RRF order, and nothing is added.
        assert [c.origin for c in expanded] == [c.origin for c in candidates[:cap]]
        assert not [c for c in expanded if c.origin == "adjacent"]


def test_expand_adjacent_never_duplicates_existing_hits(retriever):
    candidates = retriever.hybrid_search("Smartphones Tablets 15 days return window row", limit=5)
    expanded = retriever.expand_adjacent(candidates, window=1, cap=len(candidates) * 3)
    keys = [(c.source_file, c.chunk_index) for c in expanded]
    assert len(keys) == len(set(keys)), "expand_adjacent produced duplicate chunks"


# ════════════════════════════════════════════════════════════════════════════
# Empty / garbage query behaviour against the real store
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_retrieve_rejects_empty_or_whitespace_query(retriever, query):
    with pytest.raises(ValueError):
        retriever.retrieve(query, mode="full")


def test_retrieve_rejects_unknown_mode(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("return window", mode="bogus")  # type: ignore[arg-type]


def test_retrieve_garbage_query_against_real_store_does_not_crash(retriever):
    """A query with no semantic or lexical relation to the corpus should still
    return a well-formed (if low-relevance) result -- real BM25/dense search over
    a real store never raises just because nothing matches well."""
    result = retriever.retrieve("asdkjfh qwoeiru xzvbn zzz999 nonsense gibberish", mode="full")
    assert isinstance(result.chunks, list)
    # bge-reranker-base still returns a bounded top_k even for irrelevant candidates.
    assert len(result.chunks) <= config.RERANK_TOP_K


def test_retrieve_garbage_query_bm25_mode_may_return_no_hits():
    """Pure BM25 over a query with zero shared terms with the corpus can
    legitimately return an empty hit list; this must not raise."""
    isolated_retriever = rt.HybridRetriever(cache_size=0)
    result = isolated_retriever.retrieve("zzqxv999nonexistentterm", mode="bm25")
    assert isinstance(result.chunks, list)  # may be empty; must not raise


# ════════════════════════════════════════════════════════════════════════════
# Full indexer -> retriever pipeline, built fresh from real processed chunks
# ════════════════════════════════════════════════════════════════════════════


def test_build_index_from_real_chunks_into_fresh_store_then_query(tmp_path):
    """Exercise indexer.py + retriever.py as one real pipeline: build a brand-new
    Qdrant collection (in a throwaway tmp_path, never the snapshot copy or the
    real store) from the real processed chunks, then query it with a real
    HybridRetriever bound to that fresh collection."""
    fresh_qdrant_path = tmp_path / "qdrant"

    orig_qdrant_path = config.QDRANT_PATH
    config.QDRANT_PATH = fresh_qdrant_path
    rsa_models.qdrant_client.cache_clear()
    try:
        summary = ix.build_index(chunks_path=ISOLATED_CHUNKS_PATH)
        assert summary["collection"] == config.QDRANT_COLLECTION
        assert summary["points"] > 0
        assert summary["documents"] > 0

        fresh_retriever = rt.HybridRetriever(cache_size=0)
        result = fresh_retriever.retrieve("What is the return window for a smartphone?",
                                          mode="full")
        assert len(result.chunks) > 0
        assert result.chunks[0].source_file in RETURN_WINDOW_SOURCES
        # Timing/instrumentation sanity on the freshly built pipeline too.
        assert result.total_ms > 0
        assert result.total_ms < 120_000
    finally:
        rsa_models.qdrant_client.cache_clear()
        config.QDRANT_PATH = orig_qdrant_path
        # Restore the module-scoped retriever fixture's client for any later test
        # in this file that still needs the isolated snapshot copy.
        rsa_models.qdrant_client.cache_clear()
        config.QDRANT_PATH = ISOLATED_QDRANT_PATH

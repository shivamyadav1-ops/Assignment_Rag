"""Central configuration for the retail support agent."""

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load .env from the project root regardless of the working directory.
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
POLICIES_DIR = RAW_DIR / "policies"
OPERATIONAL_DIR = RAW_DIR / "operational"
PROCESSED_DIR = DATA_DIR / "processed"
SQLITE_DB_PATH = DATA_DIR / "operational.db"  # explicit-path escape hatch; tests use this
# Production/dev operational store. Required for any code path that doesn't pass an
# explicit db_path (tests always do, so they stay on SQLite — see db.get_engine).
DATABASE_URL = os.getenv("RSA_DATABASE_URL")
# Overridable so an eval can read a snapshot copy while another process holds the
# embedded store's file lock.
QDRANT_PATH = Path(os.getenv("RSA_QDRANT_PATH", str(DATA_DIR / "vectorstore" / "qdrant")))

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 512          # target tokens per chunk
CHUNK_OVERLAP = 64        # overlap tokens between consecutive chunks
MIN_CHUNK_SIZE = 50       # discard chunks smaller than this (noise)

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
MODEL_CACHE_DIR = PROJECT_ROOT / ".cache" / "fastembed"
# ONNX intra-op threads per embedding/reranker model. Unset = ONNX Runtime's default
# (all cores). When several worker processes share one host, each defaulting to all
# cores oversubscribes the CPU; set this to about cores / workers. Measured on an
# 8-core box with 4 concurrent rerank workers: 2 threads each was ~10% faster.
MODEL_THREADS = int(os.environ["RSA_MODEL_THREADS"]) if os.getenv("RSA_MODEL_THREADS") else None

# ── Qdrant / FastEmbed ────────────────────────────────────────────────────────
QDRANT_COLLECTION = "policies"
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
DENSE_MODEL = "BAAI/bge-small-en-v1.5"     # 384-dim ONNX; query_embed adds the BGE query prefix
DENSE_DIM = 384
SPARSE_MODEL = "Qdrant/bm25"               # term frequencies; IDF applied by Qdrant (Modifier.IDF)

# ── Reranker ──────────────────────────────────────────────────────────────────
# Chosen by measurement, not assumption: on the 24-question labelled eval set
# (data/eval/retrieval_questions.json), reranking identical candidate sets gave
#   ms-marco-MiniLM-L-6-v2   Hit@1=91.7%  MRR=0.951   (~2.6s/25 cand.)
#   jina-reranker-v1-turbo   Hit@1=87.5%  MRR=0.931   (~2.2s/25 cand.)
#   bge-reranker-base        Hit@1=100%   MRR=1.000   (~11-12s/25 cand.)
# bge-reranker-base is the only one with zero ranking errors on the labelled set,
# including a case where ms-marco surfaced an out-of-context chunk ("a smartphone
# (15-day window)" inside bundle-return rules) over the correct category rule.
# This is a correctness-graded prototype, not a latency-SLA service, so the
# ~5x latency cost is accepted in exchange for a measurably more accurate ranking.
RERANKER_MODEL = "BAAI/bge-reranker-base"
# Pairs scored per ONNX call. A batch is padded to its longest pair, and candidates
# range from ~30 to ~500 tokens, so batching mostly computes padding. Measured
# 2026-11-12 on 20 eval queries x 10 real candidates (4-core i5, 2 reps):
#   batch 64: mean 7.5s p50 7.1s · batch 4: mean 5.8s · batch 1: mean 3.6s p50 2.7s
# with identical scores. On a GPU, larger batches would win; revisit there.
RERANK_BATCH_SIZE = int(os.getenv("RSA_RERANK_BATCH_SIZE", "1"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 20      # candidates from each retriever
RRF_K = 60                # RRF smoothing constant: score = Σ 1 / (k + rank)
# after fusion; 20 gives the reranker a broader pool
RRF_LIMIT = int(os.getenv("RSA_RRF_LIMIT", "20"))
ADJACENT_WINDOW = 1       # ±1 chunk expansion, same document only
MAX_RERANK_CANDIDATES = int(os.getenv("RSA_MAX_RERANK_CANDIDATES", "10"))
                            # hard cap on chunks sent to the cross-encoder (fused hits
                            # first, in RRF order, then adjacent neighbours if room).
                            # Reranking dominates latency (bge-reranker-base, CPU), so the
                            # cap sets user-perceived TTFT. Measured 2026-11-10 on the
                            # 36-question retrieval set + `rsa eval-ttft --runs 2`:
                            #   cap  doc_hit@1  evidence_hit@1  doc_MRR  TTFT p50
                            #   5    86.1%      94.1%           0.917     3.9s
                            #   8    86.1%      94.1%           0.913     5.8s
                            #   20   86.1%      88.2%           0.921    14.5s
                            # 8 rather than 5: it must be >= RERANK_TOP_K (7) or the
                            # reranker cannot return 7 chunks. At 8 the cap is below
                            # RRF_LIMIT, so no adjacent neighbours are added.
                            # 2026-11-11: raised to 10 on request, so the cross-encoder
                            # scores 10 fused candidates (more recall headroom for the
                            # top-7, ~25% more rerank time than 8). Still below
                            # RRF_LIMIT (20), so no adjacent neighbours are added.
RERANK_TOP_K = 7          # final chunks to agent; 7 ensures holiday-extension and
                           # category-window chunks can co-appear for date-arithmetic queries

# ── Operational "today" ───────────────────────────────────────────────────────
# The synthetic records were generated relative to this date. Return-window math
# must measure from it, not the real wall-clock date, or every order reads as
# years old and every window as expired. Keeping it fixed also makes the demo
# reproducible: the same order yields the same eligibility on any run date.
from datetime import date  # noqa: E402

AS_OF_DATE = date(2024, 11, 9)

# ── LLM ───────────────────────────────────────────────────────────────────────
# Backend: "nvidia" (NIM, OpenAI-compatible) or "anthropic".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia").strip().lower()

# NVIDIA NIM
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
# Orchestrator (tool-calling agent). nemotron-3.5-bolt-e4b: fastest model on this key
# that makes valid tool calls (probe TTFT ~0.5s vs ~0.8s for super-vl-120b). Override
# with NVIDIA_MODEL to switch back, e.g. nvidia/nemotron-3.5-super-vl-120b-a12b.
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-bolt-e4b")

# Anthropic
LLM_MODEL = "claude-sonnet-5-5"

MAX_TOKENS = 1024
# Per-request timeout for the agent's LLM calls. The OpenAI client's default is 600 s,
# so one stalled stream froze a whole eval run for 10 minutes.
LLM_TIMEOUT_S = float(os.getenv("RSA_LLM_TIMEOUT_S", "90"))
AGENT_MAX_STEPS = 8       # hard cap on agent tool-call iterations
MCP_CALL_TIMEOUT = 30     # seconds per MCP tool call

# Grounding: a policy answer must rest on at least one chunk whose rerank score
# clears this bar. Cross-encoder logits here run roughly -11..+11 for both
# ms-marco and bge-reranker-base; 0.0 separates genuinely relevant chunks
# (observed +1.1 to +1.4 on in-scope queries) from off-topic ones (-10 to -6
# on out-of-scope queries). Re-check if the reranker model changes again.
GROUNDING_MIN_SCORE = 0.0

# ── Live grounding (NLI) on every answer ──────────────────────────────────────
# Faithfulness / contradiction metrics from a local NLI model (grounding.py).
# GROUNDING_ACTION: "flag" records warnings in the trace; "block" also replaces an
# answer that contradicts its evidence (or itself) with an escalation message.
LIVE_GROUNDING = os.getenv("RSA_LIVE_GROUNDING", "1").strip().lower() not in ("0", "false")
GROUNDING_ACTION = os.getenv("RSA_GROUNDING_ACTION", "flag").strip().lower()

# ── Caching (see cache.py and llm.py) ─────────────────────────────────────────
def _flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off")


# Context / prompt caching: keep the request prefix (tools + system prompt) byte-stable
# and, on Anthropic, mark explicit cache breakpoints. NVIDIA-hosted endpoints reuse the
# KV prefix server-side when they support it; a self-hosted NIM needs
# NIM_ENABLE_KV_CACHE_REUSE=1. Cached-token counts are recorded when reported.
PROMPT_CACHE = _flag("RSA_PROMPT_CACHE")
# Exact LRU over retrieve(query, mode, top_k); 0 disables. Evals pass cache_size=0.
RETRIEVAL_CACHE_SIZE = int(os.getenv("RSA_RETRIEVAL_CACHE_SIZE", "256"))
# Semantic answer cache for policy-only answers. Evals and tests disable it.
SEMANTIC_CACHE = _flag("RSA_SEMANTIC_CACHE")
SEMANTIC_CACHE_PATH = Path(os.getenv("RSA_SEMANTIC_CACHE_PATH",
                                     str(DATA_DIR / "cache" / "semantic_cache.db")))
# Cosine similarity on BGE-small query embeddings. Calibrated 2026-11-11
# (tests/test_cache.py::test_threshold_calibration_ml): same-meaning paraphrases
# scored 0.973-0.992; different-meaning pairs 0.842-0.975. Similarity alone cannot
# separate them ("opened" vs "unopened" cable 0.948, "December 1" vs "December 20"
# 0.975), so the signature check (numbers, negations, rare policy terms) is what
# keeps those apart; 0.93 leaves a margin over the 0.883 max of the pairs it lets by.
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("RSA_SEMANTIC_CACHE_THRESHOLD", "0.93"))
SEMANTIC_CACHE_TTL_S = float(os.getenv("RSA_SEMANTIC_CACHE_TTL_S", str(7 * 24 * 3600)))
# A corpus word in at most this share of chunks is "discriminative": it must match
# between a new question and a cached one (countries, categories, carriers...).
SEMANTIC_CACHE_DISCRIMINATIVE_DF = float(os.getenv("RSA_SEMANTIC_CACHE_DF", "0.10"))

# ── Chat sessions / conversation memory (session.py) ──────────────────────────
SESSIONS_DIR = Path(os.getenv("RSA_SESSIONS_DIR", str(DATA_DIR / "sessions")))
# Summarise into summary.md once this many turns accumulate since the last summary.
SESSION_SUMMARY_EVERY = int(os.getenv("RSA_SESSION_SUMMARY_EVERY", "10"))
# Already-summarised turns still sent verbatim after a summary, for continuity.
SESSION_KEEP_RECENT = int(os.getenv("RSA_SESSION_KEEP_RECENT", "2"))
# Hard cap on raw history characters in the prompt (oldest turns dropped first).
SESSION_MAX_HISTORY_CHARS = int(os.getenv("RSA_SESSION_MAX_HISTORY_CHARS", "16000"))

# ── Answer judging ────────────────────────────────────────────────────────────
# "geval" (default): G-Eval with explicit evaluation steps, chain-of-thought findings
# and probability-weighted 0-10 scores, given the retrieved evidence (geval.py).
# "verdict": the earlier single-prompt correct/partially_correct/incorrect judge.
JUDGE_METHOD = os.getenv("RSA_JUDGE_METHOD", "geval").strip().lower()

# ── Traces ────────────────────────────────────────────────────────────────────
TRACES_DIR = PROJECT_ROOT / "traces"

# ── Langfuse observability ────────────────────────────────────────────────────
# tracing.py reads these directly from the environment; they are listed here for
# documentation and so .env.example stays the single source of truth for keys.
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Retail Support Resolution Agent

A graded agentic AI system for ShopSmart Electronics customer support. Combines hybrid RAG over policy documents with stateful MCP tool calls over operational records to resolve returns, shipping, and policy queries end-to-end.

---

## Architecture

```
Customer question
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│                     SupportAgent                         │
│                                                          │
│  Pre-input guardrails (toxicity · injection · rate limit)│
│         │                                                │
│         ▼                                                │
│  ┌─────────────────────────────────────────────────┐     │
│  │  LangGraph tool loop    (AGENT_MAX_STEPS = 8)   │     │
│  │                                                 │     │
│  │  ┌─────────────────┐   ┌──────────────────────┐│     │
│  │  │ search_policies │   │  MCP Tool Client     ││     │
│  │  │ (local)         │   │  get_order_details   ││     │
│  │  │                 │   │  get_return_refund_  ││     │
│  │  │ HybridRetriever │   │    status            ││     │
│  │  │  dense (BGE)    │   │  create_return_      ││     │
│  │  │  BM25           │   │    request           ││     │
│  │  │  RRF fusion     │   └──────────────────────┘│     │
│  │  │  bge-reranker   │                            │     │
│  │  └─────────────────┘                            │     │
│  └─────────────────────────────────────────────────┘     │
│         │                                                │
│  Output guardrail + live NLI grounding                   │
└──────────────────────────────────────────────────────────┘
       │
       ▼
  Answer + JSONL trace + Langfuse span
```

**Data flow:** 15 policy documents (14 synthetic + 1 public FTC regulation) → chunker (PDF/HTML/MD/XML) → 427 chunks (prose + table rows) → Qdrant (dense + BM25 named vectors) → hybrid search + cross-encoder rerank → top-7 chunks to agent.

A rendered diagram is in [docs/architecture.svg](docs/architecture.svg) ([PNG](docs/architecture.png)); the presentation is [docs/presentation.pptx](docs/presentation.pptx).

---

## Quick Start

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for the PostgreSQL operational database)
- NVIDIA NIM API key (for the LLM), or an Anthropic API key with `LLM_PROVIDER=anthropic`
- Langfuse account (for observability — optional)

The test suite needs none of the services above: it uses throwaway SQLite files and a
scripted LLM.

### 1. Install

```bash
git clone <repo>
cd Project_Rag
uv sync                     # add --extra ml --extra app for ML guardrails + Streamlit
```

### 2. Start PostgreSQL

The operational records (customers, orders, returns) live in PostgreSQL, matching
production. `rsa ingest` creates the schema and loads the data.

```bash
docker run -d --name retail_support_agent-postgres \
  -e POSTGRES_USER=rsa_app -e POSTGRES_PASSWORD=change-me \
  -e POSTGRES_DB=retail_support_agent -p 5434:5432 \
  -v retail_support_agent_pgdata:/var/lib/postgresql/data postgres:16-alpine
```

### 3. Configure

```bash
cp .env.example .env
# Set RSA_DATABASE_URL to match the container, e.g.
#   postgresql+psycopg://rsa_app:change-me@localhost:5434/retail_support_agent
# and fill in your LLM key:
nano .env
```

After the first run has downloaded the models into `.cache/`, you can set
`HF_HUB_OFFLINE=1` in `.env` so model loads stop checking the Hugging Face Hub online.

Required keys:

```
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-...

# Optional — Langfuse observability
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

### 4. Ingest + index

```bash
uv run python scripts/fetch_public_sources.py --check   # verify the public source (re-fetch: drop --check)
uv run rsa ingest    # chunk policy docs + load operational DB
uv run rsa index     # embed chunks and build Qdrant collection
uv run python scripts/generate_manifest.py              # refresh data/manifest.json + MANIFEST.md
```

The raw data is committed, so fetching is only needed to verify or refresh it. The
first `rsa index` downloads the embedding and reranker models (~1 GB) into `.cache/`.

### 5. Ask a question

```bash
uv run rsa ask "Can I return my laptop?" --customer CUST-1049 --trace
```

### 6. Run evaluations

```bash
uv run rsa eval-retrieval --modes dense,bm25,hybrid,full --label baseline
uv run rsa eval-answers --label baseline
```

---

## Project Structure

```
Project_Rag/
├── src/retail_support_agent/
│   ├── agent.py           # LangGraph StateGraph: guards → cache → LLM ⇄ tools → output guard → grounding
│   ├── lc_tools.py        # LangChain PolicyRetriever + StructuredTool registry (local + MCP tools)
│   ├── chunker.py         # PDF/HTML/MD parser + row-level table chunks
│   ├── retriever.py       # HybridRetriever: dense + BM25 + RRF + rerank
│   ├── indexer.py         # Qdrant collection builder
│   ├── mcp_server.py      # MCP server: 4 operational tools
│   ├── mcp_client.py      # sync MCP client wrapper
│   ├── llm.py             # NvidiaClient / AnthropicClient / ScriptedClient
│   ├── ops.py             # order/return business logic + DB queries
│   ├── policies.py        # return-window and eligibility rules
│   ├── guardrails/        # safety checks, one module per layer (import from the package):
│   │   ├── toxicity.py, injection.py, sanitize.py, pre_input.py   # input
│   │   ├── tool_calls.py, tool_output.py                          # tool call / tool result
│   │   └── output.py                                              # final answer
│   ├── grounding.py       # live NLI faithfulness check on answers
│   ├── tracing.py         # Langfuse v5 observability integration
│   ├── eval_retrieval.py  # retrieval metrics: Hit@k, MRR, nDCG
│   ├── eval_answers.py    # answer correctness + faithfulness eval
│   └── config.py          # all config from .env
├── data/
│   ├── raw/policies/      # 19 source files (PDF, HTML, MD, XML); 15 indexed
│   ├── raw/operational/   # customers.csv, orders.csv, returns.json
│   ├── processed/         # chunks.jsonl (generated)
│   └── eval/
│       ├── retrieval_questions.json   # 50 labelled queries, all with evidence labels
│       └── answer_cases.json          # 28 end-to-end eval cases
├── app/
│   ├── streamlit_app.py   # dashboard entry point (tabs → app/dashboard/)
│   └── dashboard/         # one module per tab + sidebar, cached resources, shared widgets
├── reports/               # eval JSON reports
├── traces/                # per-session JSONL traces
├── tests/                 # pytest: guardrails, agent loop, unit/, component/, chaos/
├── .env.example           # config template
└── pyproject.toml
```

---

## Retrieval Pipeline

### Models

| Component | Model | Why |
|---|---|---|
| Dense embeddings | `BAAI/bge-small-en-v1.5` (384-dim) | Strong passage retrieval; adds BGE query prefix |
| Sparse (BM25) | `Qdrant/bm25` with IDF modifier | Exact-match complement to dense |
| Cross-encoder reranker | `BAAI/bge-reranker-base` | Chosen in an earlier 24-question comparison (Hit@1 100% vs 91.7% ms-marco-MiniLM, 87.5% jina-turbo). On the current 50-question set its gain over plain hybrid is small — see Evaluation Results |

### Retrieval modes

| Mode | Description |
|---|---|
| `dense` | Vector search only |
| `bm25` | Sparse search only |
| `hybrid` | RRF fusion (k=60) of dense + BM25 |
| `hybrid_rerank` | Hybrid + cross-encoder rerank |
| `full` | Hybrid + adjacent ±1 chunk expansion + rerank of up to 10 candidates (`RSA_MAX_RERANK_CANDIDATES`) → top 7 to the agent. The cap counts fused hits first, so neighbours are only added when fewer than 10 hits come back; at the defaults (RRF limit 20) they normally are not. Pairs are scored one at a time (`RSA_RERANK_BATCH_SIZE=1`): a batch is padded to its longest pair, and on CPU this halved rerank time with identical scores |

### Chunking

- **Prose:** structural blocks (headings, paragraphs, lists) merged up to 512 tokens; flushed at H1/H2 section boundaries to prevent cross-section merging
- **Tables:** one whole-table chunk + one chunk per row (row text: `Column: value` lines under a context header)
- **Oversized blocks:** split at sentence boundaries (any case; abbreviations such as "e.g." and "Dr." are not treated as sentence ends). Text too short to stand alone is appended to the previous chunk, or kept as its own chunk when there is none, so no content is dropped
- **Overlap** (64 tokens) only continues a section that was split for size; a new H1/H2 section starts clean, so no chunk carries text from the previous section
- **HTML inline markup** is joined with spaces (`within <strong>1 hour</strong> of` → `within 1 hour of`, not `within1 hourof`)
- **eCFR XML:** the part heading is H1 and each `§` section is H2, so every regulation chunk carries its citation as the section heading
- **Lineage:** every chunk carries `source_file`, `origin` (synthetic/public), `synthetic` and `license` from `data/sources.json`
- **Result:** 427 chunks from 15 documents (prose + table whole + table rows). The 19 files in `raw/policies/` include four Markdown copies of policies also published as PDF; only the PDF is indexed (`EXCLUDED_SOURCES`) to avoid near-duplicate chunks. Counts per file are in [data/MANIFEST.md](data/MANIFEST.md)

---

## Evaluation Results

### Retrieval (50 labelled questions, k=7) — `reports/retrieval_eval_v3.json`

Every question names the documents that answer it **and** an evidence string the
retrieved chunk must contain, so "right document, wrong passage" counts as a miss at
evidence level. Six questions (q37–q42) are answered only by the public FTC regulation.

| Mode | Doc Hit@1 | Doc MRR | Evidence Hit@1 | Evidence Hit@7 | Evidence MRR | Mean latency |
|---|---|---|---|---|---|---|
| dense (BGE-small) | 78% | 0.860 | 66% | 92% | 0.777 | 16 ms |
| BM25 | 78% | 0.855 | 70% | 90% | 0.785 | 17 ms |
| hybrid (RRF) | 90% | 0.943 | 76% | 98% | 0.849 | 43 ms |
| hybrid + rerank (20 cands) | 90% | 0.940 | 76% | 96% | 0.842 | 8.7 s |
| **full** (hybrid + rerank ≤10, used by the agent) | **90%** | **0.944** | **80%** | **98%** | **0.871** | 4.2 s |

How to read it:
- **Fusion is the big win.** Dense and BM25 each miss different questions; RRF lifts
  document Hit@1 from 78% to 90% and evidence Hit@1 from 66–70% to 76% for ~25 ms.
- **The reranker's gain is small and not significant.** Against plain hybrid, `full`
  wins 4 questions at evidence Hit@1 and loses 2 (net +2 of 50; sign test p ≈ 0.69),
  for ~100× the latency. It is kept because it fixes table-row ranking (table evidence
  Hit@1 90% → 100%) and the agent sees only the top 7, but `hybrid` is the obvious
  fallback if latency matters more than those cases.
- **Remaining misses** are mostly near-duplicates across documents (the Premium window
  appears in three) and paraphrases with no shared vocabulary (q49: "third-party repair
  shop" vs "unauthorized service providers", never retrieved). The federal-rule credit-card
  refund question (q40) ranks its section 5th because "billing cycle" sits in a long
  definitions chunk.

### Answer Evaluation (16 cases: 7 knowledge, 5 multi-step, 4 exception)

| Metric | Score |
|---|---|
| Fact accuracy | 93.8% |
| Judge correctness (nemotron-ultra-550B) | 81.2% |
| Faithfulness | 96.4% |
| Tool selection accuracy | 93.8% |
| **Safety pass rate** | **100%** |
| Latency mean / p95 | 26.6s / 104s |
| Avg tokens per case | 4,983 |

**Known limitations:**
- Latency is dominated by the cross-encoder reranker (~12s on 20 candidates) and the NVIDIA NIM cold-start. Not a latency-SLA service.
- The LLM judge shares model family with the agent (both NVIDIA nemotron). Self-preference risk exists; deterministic fact/safety/tool checks are the primary model-free metrics.
- Qdrant runs in embedded mode (file lock): only one process can hold the collection at a time.

---

## MCP Tools

The MCP server (`retail-ops`) exposes 4 tools over stdio. The requester identity is injected via `RSA_REQUESTER_ID` at server launch — it is never a tool parameter. Cross-account access returns `UNAUTHORIZED`.

| Tool | Description | Key validation |
|---|---|---|
| `get_order_details` | Order + membership + return history | ORD-\d{3,} regex; UNAUTHORIZED if not caller's order |
| `get_return_refund_status` | Return status + refund amount | Same order ownership check |
| `create_return_request` | Initiates a return | Runs `evaluate_return()`: window, non-returnable, blocking status checks |
| `create_escalation_ticket` | Opens a human-agent follow-up ticket | Summary required; reason enumerated; optional order ID validated |

All errors use a typed envelope: `{"ok": false, "error": {"code", "message", "retryable"}}`.
Escalation tickets are appended to `data/escalations.jsonl` and return a ticket ID (`ESC-xxxxxxxx`) the agent quotes to the customer.

---

## Caching

Three layers, each scoped so a cache can never change *what* the agent answers — only how fast.

| Layer | Where | What is cached | Keyed / invalidated by |
|---|---|---|---|
| **Context (prompt) cache** | `llm.py` | Provider-side KV prefix: tool schemas + system prompt + conversation so far, re-sent on each of up to 8 loop steps | Byte-stable prefix (tools sorted, fixed system prompt). Anthropic: explicit `cache_control` breakpoints on the system block and the newest message. NVIDIA: automatic prefix reuse where the endpoint supports it (self-hosted NIM: `NIM_ENABLE_KV_CACHE_REUSE=1`). Cached tokens are recorded as `usage.cached_input_tokens`. |
| **Retrieval cache** | `retriever.py` / `cache.RetrievalCache` | `retrieve(query, mode, top_k)` results — skips embedding, Qdrant and the cross-encoder rerank | Exact normalised query; in-process LRU (256), so a re-index (new process) starts clean |
| **Semantic answer cache** | `agent.py` / `cache.SemanticCache` | Final answers to policy-only questions, served for later questions that mean the same | Fingerprint of chunk corpus + system prompt + model + retrieval config + as-of date; 7-day TTL |

**Semantic cache safety rules**
- *Stored* only if the answer cannot depend on who asked: tools used ⊆ {`search_policies`}, outcome `answered`, grounded evidence, no guardrail fired, no NLI contradiction, no order/customer/return ID in question or answer.
- *Served* only if cosine similarity ≥ 0.93 **and** the question's signature matches exactly: numbers, negations, and every rare policy term (corpus words in ≤ 10% of chunks — countries, categories, carriers, "unopened"…). Calibration showed similarity alone is not enough: "opened" vs "unopened HDMI cable" scores 0.948 and "December 1" vs "December 20" 0.975; the signature rejects both.
- Input guardrails and rate limiting run before every lookup; the output guardrail re-runs on a cached answer for the current requester.
- Evals and tests run with the semantic and retrieval caches off, so measured quality and latency are uncached.

```bash
uv run rsa cache stats     # entries, hits, threshold
uv run rsa cache clear
```

---

## Conversation Memory (chat sessions)

`ChatSession` (`session.py`) gives the agent multi-turn memory, used by `rsa chat` and the Streamlit chat tab.

```
data/sessions/<customer>/<session>/
  turns.jsonl   every turn: question, answer, tools, cache result
  summary.md    rolling Markdown memory, rewritten every 10 turns
  meta.json     owner customer, summarized_upto, summaries
```

- **Prompt = system prompt + `summary.md` + recent turns + new question.** Recent turns are those since the last summary, plus the last 2 already-summarised turns for continuity, capped at 16k characters.
- **Every 10 turns** the LLM merges the new turns into `summary.md`, keeping these sections: customer context · orders and returns discussed · policy facts given · actions taken · open items. IDs, amounts and dates are copied exactly. If summarising fails, the turns are kept and it is retried on the next turn.
- **Safety:** the memory is framed to the model as data, not instructions. The agent re-checks order data with tools before acting on it. A session is bound to one customer (`PermissionError` otherwise), and session ids are validated.
- **Cache per session** (`--cache` / sidebar): `off`, `session` (reuse only inside this chat) or `global` (shared). Follow-ups are rewritten into a standalone question before the cache lookup. With `global`, only answers written *without* history are stored, because a mid-conversation answer can carry this customer's details.

```bash
uv run rsa chat --customer CUST-1049                   # new session, memory on, global cache
uv run rsa chat -c CUST-1049 -s <session-id> --cache session
```

Settings: `RSA_SESSION_SUMMARY_EVERY=10`, `RSA_SESSION_KEEP_RECENT=2`, `RSA_SESSION_MAX_HISTORY_CHARS=16000`, `RSA_SESSIONS_DIR`.

---

## Answer Judging: G-Eval

Correctness and completeness are scored with **G-Eval** (`geval.py`), the default since `RSA_JUDGE_METHOD=geval`:

1. **Explicit evaluation steps** per metric. For correctness: identify the asks and key facts → check each fact as present, missing or contradicted → check every extra fact **against the retrieved evidence** (supported means no penalty, unsupported means invented) → privacy/refusal check → score from the rubric.
2. **Chain of thought:** the judge writes one finding per step (`Step 1: …`) before it scores. The findings are saved in each report row as `steps`.
3. **Probability-weighted score:** the expected value over the judge's 0–10 score tokens, from log-probabilities that the NVIDIA endpoint returns for the 550B judge. The score is continuous, not one sampled integer.

Verdict bands: ≥ 0.8 correct · ≥ 0.4 partially correct · < 0.4 incorrect. `RSA_JUDGE_METHOD=verdict` restores the earlier single-prompt judge for comparison. Validate a judge change with `rsa calibrate-judge` against the 26 hand-labelled cases.

---

## Guardrails

| Layer | Checks |
|---|---|
| Pre-input | Toxicity regex + toxic-bert classifier, prompt injection regex + DeBERTa classifier, rate limit (10 req/60s per customer), empty/control-char inputs |
| Tool call | Schema validation, return-before-lookup prevention, per-tool call budgets (per customer message; a forced search is checked and charged too), order-ID format check. Read-only tools are retried in code on transient errors (`TOOL_UNAVAILABLE`, `INTERNAL_ERROR`; `RSA_TOOL_MAX_RETRIES`, default 1); writes are never retried |
| Outcome label | Derived from tool state, never from the answer's wording: `refused_unauthorized` (every order lookup hit another account), `escalated` (an escalation ticket was actually created), `refused_ungrounded` (policy searched, nothing grounded, no other tool gave a result), else `answered` |
| Evidence rule | The agent never answers without evidence: if the model answers without a policy search and no tool gave a definitive result, a policy search is forced first (questions about the chat itself are exempt) |
| Tool output | Prompt-injection neutralisation, PII redaction (email/card/SSN), system-prompt-leak scan |
| Output | System-prompt leak (shingle overlap) · Toxic/abusive output (regex + ML, stricter thresholds than input) · Off-scope content (medical/legal/financial advice, self-harm, extremism) · PII redaction + other-customer ID masking · Unverified order ID warning · Fabricated-number warning (quantities in answer absent from tool context) · Ungrounded policy-assertion warning (policy claims without retrieved evidence) |

---

## Observability

Every `rsa ask` produces:
- **JSONL trace** in `traces/` — per-step kind, name, latency_ms, TTFT, token usage
- **Langfuse trace** (when configured) — root agent span → LLM generations (model, tokens, cost) + retriever/tool child spans

Cost note: at nemotron-3.5-super pricing (~$0.40/M input, ~$0.40/M output), the 4,983-token average per case costs roughly **$0.004 per query**. See `reports/answer_eval_*.json` for per-case token counts.

---

## Scaling under load

- **Retrieval — `RSA_MODEL_THREADS`.** Embedding and reranking run on CPU in-process.
  Each process defaults to all cores, so several worker processes on one host compete
  for them. Set this to about `cores / workers` (measured: 4 workers on 8 cores, 2
  threads each, ~10% faster). The reranker dominates retrieval latency; moving it to a
  GPU or a dedicated batching server is the next step if retrieval becomes the limit.

This is separate from the per-customer **rate limiter** in the input guardrail
(10 requests / 60 s per customer, thread-safe), which limits abuse, not capacity. It is
in-memory, so with several worker processes each enforces its own window; a shared
limit across processes needs an external store such as Redis.

---

## Running Tests

```bash
uv run pytest tests/                    # everything: ~1,150 tests, ~7 min (loads local models)
uv run pytest tests/ -m "not ml"        # fast, model-free subset: ~1.5 min
uv run pytest tests/component -m ml     # real Qdrant index, real models, real subprocesses
```

No test calls a paid LLM API or touches PostgreSQL: the model is replaced by a
`ScriptedClient`, and every database is a throwaway SQLite file.

| Suite | What it checks |
|---|---|
| `tests/test_*.py` | Guardrails (injection, toxicity, rate limit, PII, output sanitisation), agent loop, caching, sessions, return deadlines, metrics |
| `tests/unit/` | Per-module unit tests for retrieval/chunking, agent runtime/LLM/MCP, data/ops/cache/session, evaluation maths, rate limiter |
| `tests/component/` | Components wired together on copies of the real data: hybrid retrieval on the real index, ops on real orders, the full agent with only the LLM scripted, and the `rsa` CLI + MCP server as real processes |
| `tests/chaos/` | Fault injection: locked/corrupt databases, missing index, malformed chunks, LLM outage, runaway tool loops, corrupt cache |

`tests/conftest.py` keeps classifier guardrails, live grounding and caching off by
default (tests marked `ml` turn them on) and resets the input rate limiter before every
test, so results do not depend on test order.

---

## Data Notice

- **Public source:** `ftc_mail_order_rule_16cfr435.xml` is the official text of 16 CFR Part 435
  (FTC Mail, Internet, or Telephone Order Merchandise Rule) from the eCFR API, pinned to the
  2026-11-09 version. As a U.S. Government work it is public domain (17 U.S.C. § 105).
  `scripts/fetch_public_sources.py --check` verifies its SHA-256.
- **Synthetic data:** every ShopSmart policy document and every customer, order and return
  record was generated for this assessment and is labelled in the file (`[SYNTHETIC DOCUMENT]`
  header, `<meta name="synthetic">`, or a `synthetic` column/field) and in `data/sources.json`.
  No real customer data, real orders or confidential information is included.
- The full inventory, with hashes, licences and chunk counts, is [data/MANIFEST.md](data/MANIFEST.md).

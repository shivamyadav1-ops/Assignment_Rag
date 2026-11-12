"""Golden dataset builder: chunk-level relevance labels sized to the corpus.

Sizing. The index holds ~400 chunks (mean ~75 words) across 14 indexed documents.
At roughly one query per four chunks the retrieval set targets ~100 queries:
the 36 hand-written questions already in `retrieval_questions.json`, ~64 synthetic
questions seeded from chunks (stratified by document, proportional to its chunk
count, min 2 each), and a handful of out-of-scope queries that should match nothing.

Labelling (TREC-style pooling, so recall is measurable):
  1. For each query, pool dense top-10, BM25 top-10 and hybrid top-20 (the set the
     reranker reorders), plus the seed chunk and, for a table row, its parent table.
  2. An LLM judge grades every pooled chunk in one call:
       2 = contains the information that answers the query
       1 = on topic / partially supports an answer, but does not answer it
       0 = not relevant
  3. `relevant` (grade 2) is the set recall/precision are scored against;
     graded gains feed nDCG.
Synthetic questions whose own seed chunk is not graded 2 are dropped as bad
generations. Out-of-scope queries that turn out to have grade-2 chunks are kept as
answerable — the labels decide, not the intent.

The answer set is the 16 hand-written agent cases plus ~24 synthetic knowledge
questions (stratified by document) with the generated reference answer.

All generated items carry `"synthetic": true` and `"origin": "synthetic"`.
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

from . import config

GOLDEN_RETRIEVAL_PATH = config.DATA_DIR / "eval" / "golden_retrieval.json"
GOLDEN_ANSWERS_PATH = config.DATA_DIR / "eval" / "golden_answers.json"

N_SYNTHETIC = 64
N_ANSWER_SYNTHETIC = 24
# Hybrid top-20 is the candidate set the reranker reorders, so pooling it covers what
# the reranked modes can return (except adjacent-chunk expansion, which `judged@k`
# in the eval exposes) without running the cross-encoder during labelling.
POOL_DEPTHS = {"dense": 10, "bm25": 10, "hybrid": 20}
CACHE_PATH = config.PROCESSED_DIR / "golden_cache.json"
CALL_TIMEOUT_S = 240
STYLES = ("direct", "paraphrase", "scenario")
SEED = 42

# Plausible retail questions the policies do not (as far as we know) cover.
OUT_OF_SCOPE = [
    "Do you offer a student discount?",
    "Can I trade in my old car for store credit?",
    "What are the store's opening hours on Christmas Day in Tokyo?",
    "Do you sell prescription eyeglasses?",
    "Can I book a hotel room through ShopSmart?",
    "Does ShopSmart offer pet insurance?",
]

STYLE_HINTS = {
    "direct": "Ask directly for the fact, as a customer would.",
    "paraphrase": ("Ask for the fact WITHOUT reusing the passage's distinctive keywords; "
                   "use everyday synonyms a customer would use."),
    "scenario": ("Frame it as a short first-person situation (one or two sentences) whose "
                 "answer depends on the fact."),
}

GEN_PROMPT = """You are building an evaluation set for a retail customer-support search \
system (ShopSmart Electronics). Write ONE question a customer could ask that is fully \
answered by the passage below.

Rules:
- The question must be self-contained: name the product category, service or policy \
topic it is about. Never say "the passage", "this document", "the table" or "row".
- Answerable from the passage alone; do not require outside knowledge.
- Style: {style_hint}
- Then write the reference answer: 1-3 sentences, using only the passage.

Source: {source} | Section: {section}
PASSAGE:
<<<
{text}
>>>

Respond with JSON only:
{{"question": "<question>", "answer": "<reference answer>"}}"""

GRADE_PROMPT = """You are labelling search results for a retail customer-support system. \
For the customer question, grade EVERY numbered passage:
  2 = the passage contains the information that answers the question (fully, or the key \
fact the answer depends on)
  1 = on the same topic or partially helpful, but does not by itself answer it
  0 = not relevant
Judge each passage independently. A table that contains the answering row counts as 2.

Question: {query}

{passages}

Respond with JSON only, one entry per passage:
{{"grades": [{{"id": <number>, "grade": 0 | 1 | 2}}]}}"""


def _load_chunks() -> list[dict]:
    with config.CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _quotas(counts: Counter, total: int) -> dict[str, int]:
    if not counts or total <= 0:
        return {doc: 0 for doc in counts}
    n_all = sum(counts.values())
    floor = min(2, total // len(counts))
    q = {doc: max(floor, round(total * n / n_all)) for doc, n in counts.items()}
    # Trim/pad the largest documents until the quotas sum to `total`.
    order = [d for d, _ in counts.most_common()]
    i = 0
    while sum(q.values()) != total:
        doc = order[i % len(order)]
        if sum(q.values()) > total and q[doc] > floor:
            q[doc] -= 1
        elif sum(q.values()) < total:
            q[doc] += 1
        i += 1
    return q


def sample_seeds(chunks: list[dict], total: int = N_SYNTHETIC) -> list[dict]:
    rng = random.Random(SEED)
    usable = [c for c in chunks
              if c["chunk_type"] in ("text", "table_row") and c["token_count"] >= 25]
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in usable:
        by_doc[c["source_file"]].append(c)
    quotas = _quotas(Counter({d: len(v) for d, v in by_doc.items()}), total)
    seeds = []
    for doc in sorted(by_doc):
        pool = by_doc[doc][:]
        rng.shuffle(pool)
        # Alternate prose and table rows so both chunk types are represented.
        text = [c for c in pool if c["chunk_type"] == "text"]
        rows = [c for c in pool if c["chunk_type"] == "table_row"]
        picked = []
        while len(picked) < quotas[doc] and (text or rows):
            for bucket in (text, rows):
                if bucket and len(picked) < quotas[doc]:
                    picked.append(bucket.pop())
        seeds.extend(picked)
    for i, s in enumerate(seeds):
        s["_style"] = STYLES[i % len(STYLES)]
    return seeds


def _render_passages(cands: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] ({c['source_file']} | {c.get('section_heading') or '-'})\n{c['text'][:900]}"
        for i, c in enumerate(cands, 1))


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {"generated": {}, "grades": {}}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache), encoding="utf-8")
    tmp.replace(CACHE_PATH)


def _parallel(fn, items: list, workers: int, label: str, on_done) -> None:
    """Run fn over items with a per-call wall-clock cap; a call that hangs is recorded
    as failed (on_done(item, None)) instead of stalling the whole build."""
    ex = ThreadPoolExecutor(workers)
    futures = {ex.submit(fn, it): it for it in items}
    done = 0
    try:
        for fut in as_completed(futures, timeout=CALL_TIMEOUT_S * max(1, len(items))):
            item = futures[fut]
            try:
                on_done(item, fut.result(timeout=CALL_TIMEOUT_S))
            except Exception as exc:  # noqa: BLE001
                print(f"  {label}: failed {exc!r}"[:200], flush=True)
                on_done(item, None)
            done += 1
            if done % 10 == 0 or done == len(items):
                print(f"  {label}: {done}/{len(items)}", flush=True)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def build(workers: int = 4) -> tuple[dict, dict]:
    from .eval_answers import CASES_PATH, Judge
    from .eval_retrieval import load_questions
    from .retriever import HybridRetriever

    t0 = time.perf_counter()
    chunks = _load_chunks()
    by_id = {c["chunk_id"]: c for c in chunks}
    local = threading.local()

    def judge() -> Judge:
        # One client per thread: no shared HTTP connection pool between workers.
        if not hasattr(local, "judge"):
            local.judge = Judge(timeout=CALL_TIMEOUT_S - 30)
        return local.judge

    def ask(prompt: str) -> dict:
        # The endpoint rate-limits (429) under load; back off instead of dropping items.
        delay = 15.0
        for _ in range(7):
            out = judge()._ask_json(prompt)
            if "RateLimit" not in str(out.get("error", "")):
                return out
            time.sleep(delay + random.uniform(0, 5))
            delay = min(delay * 2, 120.0)
        return out

    judge_model = Judge().models[0]
    cache = _load_cache()
    print(f"corpus: {len(chunks)} chunks, {len({c['source_file'] for c in chunks})} docs "
          f"| judge={judge_model} | cached: {len(cache['generated'])} generated, "
          f"{len(cache['grades'])} graded", flush=True)

    # ── 1. Generate synthetic questions ───────────────────────────────────────
    seeds = sample_seeds(chunks)
    todo = [s for s in seeds if s["chunk_id"] not in cache["generated"]]

    def gen(seed: dict) -> dict | None:
        out = ask(GEN_PROMPT.format(
            style_hint=STYLE_HINTS[seed["_style"]], source=seed["source_file"],
            section=seed.get("section_heading") or "-", text=seed["text"][:1800]))
        if not out.get("question"):
            return None
        return {"query": out["question"].strip(), "reference_answer": out.get("answer", ""),
                "seed_chunk_id": seed["chunk_id"], "style": seed["_style"],
                "kind": "table" if seed["chunk_type"] == "table_row" else "prose"}

    def gen_done(seed, result):
        if result:
            cache["generated"][seed["chunk_id"]] = result
            _save_cache(cache)

    if todo:
        _parallel(gen, todo, workers, "generate", gen_done)
    generated = [cache["generated"][s["chunk_id"]] for s in seeds
                 if s["chunk_id"] in cache["generated"]]
    print(f"generated {len(generated)}/{len(seeds)} synthetic questions "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

    queries = []
    for q in load_questions():
        queries.append({"id": f"h_{q['id']}", "origin": "human", "kind": q["kind"],
                        "query": q["query"], "expected_sources": q["expected_sources"]})
    for i, g in enumerate(generated, 1):
        queries.append({"id": f"s{i:03d}", "origin": "synthetic", "synthetic": True, **g})
    for i, text in enumerate(OUT_OF_SCOPE, 1):
        queries.append({"id": f"oos{i:02d}", "origin": "human", "kind": "out_of_scope",
                        "query": text})

    # ── 2. Pool candidates (sequential: the retriever's models are not shared) ──
    retriever = HybridRetriever(cache_size=0)
    for n, q in enumerate(queries, 1):
        pool: dict[str, set] = {}
        for mode, depth in POOL_DEPTHS.items():
            res = retriever.retrieve(q["query"], mode=mode, top_k=depth)
            for c in res.chunks:
                pool.setdefault(c.chunk_id, set()).add(mode)
        if n % 20 == 0:
            print(f"  pool: {n}/{len(queries)}", flush=True)
        seed_id = q.get("seed_chunk_id")
        if seed_id:
            pool.setdefault(seed_id, set()).add("seed")
            seed = by_id[seed_id]
            if seed.get("table_id"):
                for c in chunks:
                    if c.get("table_id") == seed["table_id"] and c["chunk_type"] == "table":
                        pool.setdefault(c["chunk_id"], set()).add("parent_table")
        q["_pool"] = {cid: sorted(m) for cid, m in pool.items() if cid in by_id}
    print(f"pooled {len(queries)} queries, mean pool "
          f"{sum(len(q['_pool']) for q in queries) / len(queries):.1f} chunks "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

    # ── 3. Grade pools ────────────────────────────────────────────────────────
    def grade_key(q: dict) -> str:
        return q["query"] + "||" + ",".join(sorted(q["_pool"]))

    def grade(q: dict) -> dict:
        ids = list(q["_pool"])
        cands = [by_id[c] for c in ids]
        out = ask(GRADE_PROMPT.format(query=q["query"], passages=_render_passages(cands)))
        grades = {}
        for g in out.get("grades") or []:
            try:
                idx, val = int(g["id"]), int(g["grade"])
            except (KeyError, TypeError, ValueError):
                continue
            if 1 <= idx <= len(ids):
                grades[ids[idx - 1]] = max(0, min(2, val))
        return {"grades": grades, "error": out.get("error")}

    def grade_done(q, result):
        if result and result["grades"] and not result["error"]:
            cache["grades"][grade_key(q)] = result
            _save_cache(cache)

    todo = [q for q in queries if grade_key(q) not in cache["grades"]]
    print(f"grading {len(todo)} pools ({len(queries) - len(todo)} cached)", flush=True)
    if todo:
        _parallel(grade, todo, workers, "grade", grade_done)
    failed = {"grades": {}, "error": "no grades (timeout or judge failure)"}
    graded = [cache["grades"].get(grade_key(q), failed) for q in queries]

    kept, dropped = [], []
    for q, g in zip(queries, graded, strict=True):
        pool = q.pop("_pool")
        if g["error"] or not g["grades"]:
            dropped.append({"id": q["id"], "why": f"grading failed: {g['error']}"})
            continue
        grades = g["grades"]
        q["relevant"] = sorted(c for c, v in grades.items() if v == 2)
        q["partially_relevant"] = sorted(c for c, v in grades.items() if v == 1)
        q["grades"] = {c: grades.get(c, 0) for c in pool}
        q["pool_sources"] = pool
        q["relevant_sources"] = sorted({by_id[c]["source_file"] for c in q["relevant"]})
        if q["origin"] == "synthetic" and grades.get(q["seed_chunk_id"], 0) < 2:
            dropped.append({"id": q["id"], "why": "seed chunk not graded 2",
                            "query": q["query"]})
            continue
        if q["kind"] == "out_of_scope" and q["relevant"]:
            q["kind"] = "prose"
            q["note"] = "intended out-of-scope, but the judge found answering chunks"
        q["answerable"] = bool(q["relevant"])
        kept.append(q)

    if not any(q["origin"] == "synthetic" for q in kept) or len(kept) < len(queries) // 2:
        raise RuntimeError(f"only {len(kept)}/{len(queries)} queries labelled "
                           f"(dropped: {Counter(d['why'][:30] for d in dropped)}); "
                           "not writing a degraded golden set — rerun to resume from cache")
    kinds = Counter(q["kind"] for q in kept)
    n_rel = [len(q["relevant"]) for q in kept if q["answerable"]]
    retrieval = {
        "description": ("Golden retrieval set with chunk-level graded relevance labels "
                        "(pooled from dense/BM25 top-10 and hybrid top-20, graded 0/1/2 "
                        "by an LLM judge)."),
        "version": 1,
        "created": datetime.now(UTC).isoformat(),
        "synthetic": True,
        "corpus": {"chunks": len(chunks), "documents": len({c["source_file"] for c in chunks}),
                   "chunks_path": str(config.CHUNKS_PATH.relative_to(config.PROJECT_ROOT))},
        "labelling": {"judge_model": judge_model, "pool_depths": POOL_DEPTHS,
                      "relevance": "grade 2 = answer-bearing"},
        "stats": {"queries": len(kept), "by_kind": dict(kinds),
                  "by_origin": dict(Counter(q["origin"] for q in kept)),
                  "answerable": sum(q["answerable"] for q in kept),
                  "relevant_per_query_mean": round(sum(n_rel) / max(len(n_rel), 1), 2),
                  "dropped": len(dropped)},
        "dropped": dropped,
        "queries": kept,
    }

    # ── 4. Answer golden set ──────────────────────────────────────────────────
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    for c in cases:
        c.setdefault("origin", "human")
    rng = random.Random(SEED)
    synth = [q for q in kept if q["origin"] == "synthetic" and q.get("reference_answer")]
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for q in synth:
        by_doc[by_id[q["seed_chunk_id"]]["source_file"]].append(q)
    for v in by_doc.values():
        rng.shuffle(v)
    picked: list[dict] = []
    while len(picked) < N_ANSWER_SYNTHETIC and any(by_doc.values()):
        for doc in sorted(by_doc):
            if by_doc[doc] and len(picked) < N_ANSWER_SYNTHETIC:
                picked.append(by_doc[doc].pop())
    customers = ["CUST-1001", "CUST-1002", "CUST-1003", "CUST-1004"]
    for i, q in enumerate(picked, 1):
        cases.append({
            "id": f"g{i:02d}", "origin": "synthetic", "synthetic": True,
            "scenario": "knowledge", "customer": customers[i % len(customers)],
            "question": q["query"], "reference": q["reference_answer"],
            "expected_tools": ["search_policies"],
            "retrieval_query_id": q["id"], "reference_chunk_ids": q["relevant"],
        })
    answers = {
        "description": ("Golden answer set: hand-written agent scenarios (knowledge, "
                        "multi-step, exception) plus synthetic knowledge questions with "
                        "judge-verified reference answers."),
        "version": 1,
        "created": datetime.now(UTC).isoformat(),
        "synthetic": True,
        "stats": {"cases": len(cases),
                  "by_scenario": dict(Counter(c["scenario"] for c in cases)),
                  "by_origin": dict(Counter(c["origin"] for c in cases))},
        "cases": cases,
    }

    GOLDEN_RETRIEVAL_PATH.write_text(json.dumps(retrieval, indent=2), encoding="utf-8")
    GOLDEN_ANSWERS_PATH.write_text(json.dumps(answers, indent=2), encoding="utf-8")
    print(f"retrieval golden: {retrieval['stats']}", flush=True)
    print(f"answer golden:    {answers['stats']}", flush=True)
    print(f"total {time.perf_counter() - t0:.0f}s", flush=True)
    return retrieval, answers

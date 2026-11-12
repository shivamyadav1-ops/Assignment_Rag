"""Retrieval evaluation: how well each retrieval mode ranks the right evidence.

Two levels of ground truth, because several policy documents deliberately share
vocabulary:

  - document level: a hit when a top-k chunk comes from one of the question's
    `expected_sources` (some facts legitimately live in two documents);
  - evidence level: stricter, a hit only when that chunk also contains every
    `must_contain` string (an entry may be a list of acceptable alternatives).
    Only questions that carry `must_contain` are scored at this level.

Metrics per mode: Hit@1/3/5, MRR and nDCG@5. With a single relevant set per
question and binary relevance, nDCG@5 reduces to 1/log2(rank+1) at the first hit.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from . import config
from .retriever import HybridRetriever

QUESTIONS_PATH = config.DATA_DIR / "eval" / "retrieval_questions.json"


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    questions = json.loads(path.read_text(encoding="utf-8"))["questions"]
    for q in questions:
        q.setdefault("expected_sources", [q["expected_source"]] if "expected_source" in q else [])
        q.setdefault("kind", "prose")
    return questions


def contains_all(text: str, must: list) -> bool:
    low = text.lower()
    for item in must:
        alternatives = item if isinstance(item, list) else [item]
        if not any(a.lower() in low for a in alternatives):
            return False
    return True


def _first_rank(chunks, predicate) -> int | None:
    for i, c in enumerate(chunks, 1):
        if predicate(c):
            return i
    return None


def _hit(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def _rr(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def _ndcg(rank: int | None, k: int) -> float:
    return 1.0 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0


def summarise(rows: list[dict], k: int) -> dict:
    doc = [r["doc_rank"] for r in rows]
    ev = [r["evidence_rank"] for r in rows if r["evidence_labelled"]]
    lat = sorted(r["latency_ms"] for r in rows)
    mean = statistics.fmean
    out = {
        "n": len(rows),
        "doc_hit@1": mean(_hit(x, 1) for x in doc),
        "doc_hit@3": mean(_hit(x, 3) for x in doc),
        f"doc_hit@{k}": mean(_hit(x, k) for x in doc),
        "doc_mrr": mean(_rr(x) for x in doc),
        f"doc_ndcg@{k}": mean(_ndcg(x, k) for x in doc),
        "evidence_n": len(ev),
        "latency_ms_mean": round(mean(lat), 1),
        "latency_ms_p95": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1),
    }
    if ev:
        out.update({
            "evidence_hit@1": mean(_hit(x, 1) for x in ev),
            f"evidence_hit@{k}": mean(_hit(x, k) for x in ev),
            "evidence_mrr": mean(_rr(x) for x in ev),
        })
    return {key: (round(v, 3) if isinstance(v, float) else v) for key, v in out.items()}


def evaluate(retriever: HybridRetriever, questions: list[dict], modes: list[str],
             k: int = config.RERANK_TOP_K, progress=print) -> dict:
    results: dict[str, dict] = {}
    for mode in modes:
        rows = []
        t_mode = time.perf_counter()
        for q in questions:
            res = retriever.retrieve(q["query"], mode=mode, top_k=k)
            sources = set(q["expected_sources"])
            must = q.get("must_contain")
            doc_rank = _first_rank(res.chunks, lambda c, s=sources: c.source_file in s)
            ev_rank = (_first_rank(res.chunks, lambda c, s=sources, m=must:
                                   c.source_file in s and contains_all(c.text, m))
                       if must else None)
            rows.append({
                "id": q["id"], "kind": q["kind"], "query": q["query"],
                "expected_sources": q["expected_sources"],
                "doc_rank": doc_rank, "evidence_rank": ev_rank,
                "evidence_labelled": bool(must),
                "latency_ms": round(res.total_ms, 1),
                "top": [f"{c.source_file}#{c.chunk_index}"
                        f"{':' + c.payload.get('chunk_type', 'text') if c.payload else ''}"
                        for c in res.chunks],
            })
        by_kind = defaultdict(list)
        for r in rows:
            by_kind[r["kind"]].append(r)
        results[mode] = {
            "summary": summarise(rows, k),
            "by_kind": {kind: summarise(rs, k) for kind, rs in sorted(by_kind.items())},
            "questions": rows,
        }
        progress(f"  {mode:14} done in {time.perf_counter() - t_mode:6.1f}s  "
                 f"{results[mode]['summary']}")
    return results


def run(modes: list[str], label: str, out_dir: Path | None = None,
        k: int = config.RERANK_TOP_K) -> Path:
    questions = load_questions()
    retriever = HybridRetriever(cache_size=0)
    collection = retriever.client.get_collection(config.QDRANT_COLLECTION)
    print(f"Evaluating {len(questions)} questions, modes={modes}, k={k}, label={label}",
          flush=True)
    results = evaluate(retriever, questions, modes, k=k,
                       progress=lambda m: print(m, flush=True))
    report = {
        "label": label,
        "created": datetime.now(UTC).isoformat(),
        "k": k,
        "config": {
            "dense_model": config.DENSE_MODEL, "sparse_model": config.SPARSE_MODEL,
            "reranker_model": config.RERANKER_MODEL, "rrf_k": config.RRF_K,
            "retrieval_top_k": config.RETRIEVAL_TOP_K, "rrf_limit": config.RRF_LIMIT,
            "max_rerank_candidates": config.MAX_RERANK_CANDIDATES,
            "indexed_points": collection.points_count,
        },
        "modes": results,
    }
    out_dir = out_dir or (config.PROJECT_ROOT / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"retrieval_eval_{label}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report -> {path}", flush=True)
    return path

"""Retrieval evaluation on the chunk-labelled golden set (`golden.py`).

Relevance is per chunk: `relevant` = chunks graded 2 (answer-bearing); `grades` holds
the 0/1/2 grade of every pooled chunk for graded nDCG. A retrieved chunk outside the
judged pool counts as non-relevant; `judged@k` reports how much of the top-k was
judged, so a low value flags that the labels may undercount a mode.

Per query, at k in KS:
  precision@k = |top-k ∩ R| / k        recall@k = |top-k ∩ R| / |R|
  f1@k, hit@k, nDCG@k (gain 2^grade - 1), and over the top-10: MRR, AP (MAP),
  R-precision = precision@|R|.
Precision@k is bounded by |R|/k: with ~2-3 relevant chunks per query a perfect
ranker scores ~0.4 at k=7. `precision_ceiling@k` reports that bound alongside.

Out-of-scope queries (no relevant chunk) are excluded from the ranking metrics and
scored on abstention instead, for modes with a reranker score: the agent treats a
top score below GROUNDING_MIN_SCORE as "nothing grounded".
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
from .golden import GOLDEN_RETRIEVAL_PATH

KS = (1, 3, 5, 7, 10)
MAX_K = max(KS)
RERANKED_MODES = {"hybrid_rerank", "full"}


def query_metrics(ranked: list[str], relevant: set[str], grades: dict[str, int]) -> dict:
    out: dict[str, float] = {}
    for k in KS:
        top = ranked[:k]
        hits = len(set(top) & relevant)
        p = hits / k
        r = hits / len(relevant)
        out[f"precision@{k}"] = p
        out[f"recall@{k}"] = r
        out[f"f1@{k}"] = 2 * p * r / (p + r) if p + r else 0.0
        out[f"hit@{k}"] = 1.0 if hits else 0.0
        out[f"precision_ceiling@{k}"] = min(len(relevant), k) / k
        dcg = sum((2 ** grades.get(c, 0) - 1) / math.log2(i + 2) for i, c in enumerate(top))
        ideal = sorted(grades.values(), reverse=True)[:k]
        idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
        out[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
        out[f"judged@{k}"] = sum(1 for c in top if c in grades) / max(len(top), 1)

    first = next((i for i, c in enumerate(ranked[:MAX_K], 1) if c in relevant), None)
    out["mrr"] = 1.0 / first if first else 0.0
    hits, ap = 0, 0.0
    seen_rel: set[str] = set()
    for i, c in enumerate(ranked[:MAX_K], 1):
        if c in relevant and c not in seen_rel:
            seen_rel.add(c)
            hits += 1
            ap += hits / i
    out["map"] = ap / min(len(relevant), MAX_K)
    rk = len(relevant)
    out["r_precision"] = len(set(ranked[:rk]) & relevant) / rk
    return out


def _mean_dict(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    return {k: round(statistics.fmean(r[k] for r in rows), 3) for k in keys}


def _summary(rows: list[dict]) -> dict:
    ranked = [r for r in rows if r["answerable"]]
    out = {"n": len(ranked)}
    if ranked:
        out.update(_mean_dict([r["metrics"] for r in ranked]))
    lat = sorted(r["latency_ms"] for r in rows)
    out["latency_ms_mean"] = round(statistics.fmean(lat), 1)
    out["latency_ms_p95"] = round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1)
    return out


def _abstention(rows: list[dict]) -> dict | None:
    scored = [r for r in rows if r["top_score"] is not None]
    if not scored:
        return None
    oos = [r for r in scored if not r["answerable"]]
    ans = [r for r in scored if r["answerable"]]
    abstain = lambda r: r["top_score"] < config.GROUNDING_MIN_SCORE  # noqa: E731
    return {
        "threshold": config.GROUNDING_MIN_SCORE,
        "out_of_scope_n": len(oos),
        "correct_abstention_rate": (round(statistics.fmean(abstain(r) for r in oos), 3)
                                    if oos else None),
        "false_abstention_rate": (round(statistics.fmean(abstain(r) for r in ans), 3)
                                  if ans else None),
    }


def evaluate(retriever, queries: list[dict], modes: list[str], progress=print) -> dict:
    results = {}
    for mode in modes:
        t_mode = time.perf_counter()
        rows = []
        for q in queries:
            res = retriever.retrieve(q["query"], mode=mode, top_k=MAX_K)
            ranked = [c.chunk_id for c in res.chunks]
            top_score = (res.chunks[0].rerank_score
                         if mode in RERANKED_MODES and res.chunks else None)
            row = {"id": q["id"], "kind": q["kind"], "origin": q["origin"],
                   "answerable": q["answerable"], "query": q["query"],
                   "n_relevant": len(q["relevant"]), "latency_ms": round(res.total_ms, 1),
                   "top_score": top_score,
                   "ranked": [f"{c.source_file}#{c.chunk_index}|{c.chunk_id[:8]}"
                              f"|g{q['grades'].get(c.chunk_id, '-')}" for c in res.chunks]}
            if q["answerable"]:
                row["metrics"] = query_metrics(ranked, set(q["relevant"]), q["grades"])
            rows.append(row)

        groups = defaultdict(list)
        for r in rows:
            groups[f"kind:{r['kind']}"].append(r)
            groups[f"origin:{r['origin']}"].append(r)
        results[mode] = {
            "summary": _summary(rows),
            "abstention": _abstention(rows),
            "groups": {g: _summary(rs) for g, rs in sorted(groups.items())
                       if any(r["answerable"] for r in rs)},
            "queries": rows,
        }
        s = results[mode]["summary"]
        progress(f"  {mode:14} {time.perf_counter() - t_mode:6.1f}s  "
                 f"P@5={s.get('precision@5')} R@5={s.get('recall@5')} "
                 f"R@10={s.get('recall@10')} MAP={s.get('map')} MRR={s.get('mrr')} "
                 f"nDCG@5={s.get('ndcg@5')}")
    return results


def run(modes: list[str], label: str, out_dir: Path | None = None) -> Path:
    from .retriever import HybridRetriever

    golden = json.loads(GOLDEN_RETRIEVAL_PATH.read_text(encoding="utf-8"))
    queries = golden["queries"]
    retriever = HybridRetriever(cache_size=0)
    print(f"Golden retrieval eval: {len(queries)} queries "
          f"({sum(q['answerable'] for q in queries)} answerable), modes={modes}", flush=True)
    results = evaluate(retriever, queries, modes, progress=lambda m: print(m, flush=True))
    report = {
        "label": label,
        "created": datetime.now(UTC).isoformat(),
        "golden": {"path": str(GOLDEN_RETRIEVAL_PATH.relative_to(config.PROJECT_ROOT)),
                   "created": golden.get("created"), "stats": golden.get("stats")},
        "ks": list(KS),
        "agent_k": config.RERANK_TOP_K,
        "config": {"dense_model": config.DENSE_MODEL, "sparse_model": config.SPARSE_MODEL,
                   "reranker_model": config.RERANKER_MODEL, "rrf_k": config.RRF_K,
                   "rrf_limit": config.RRF_LIMIT,
                   "max_rerank_candidates": config.MAX_RERANK_CANDIDATES},
        "modes": results,
    }
    out_dir = out_dir or (config.PROJECT_ROOT / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"golden_retrieval_eval_{label}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report -> {path}", flush=True)
    return path

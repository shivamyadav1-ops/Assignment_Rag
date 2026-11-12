"""Time-to-first-token evaluation for RAG queries.

The agent records `ttft_ms` for the *final* LLM call, measured from that call's own
start. For a RAG query that is not what the customer waits for. The path to the first
answer token is

    input guardrails → LLM call 1 (decides to search; streams a tool call)
      → search_policies (dense embed, BM25, hybrid fusion, cross-encoder rerank)
      → [more LLM ↔ tool rounds] → final LLM call → first answer token

This module wraps the LLM client and retriever (no agent changes), stamps every stage
on one wall clock, and reports:

  user_ttft_ms      question received → first token of the final answer
  final_call_ttft   what the agent reports today (final call only)
  model_floor_ttft  same model, same prompt, no tools/RAG: the network + prefill floor
  stage breakdown   guardrails, each LLM call (TTFT + generation), retrieval sub-stages

Run:  .venv/bin/python -m retail_support_agent.eval_ttft [--runs 2] [--limit 7]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime

from . import config, guardrails
from .agent import SYSTEM_PROMPT, SupportAgent
from .metrics import latency_summary


class TimedLLM:
    """Delegating LLM client that stamps each call on a shared clock."""

    def __init__(self, inner, clock):
        self.inner, self.clock = inner, clock
        self.calls: list[dict] = []
        self.model = getattr(inner, "model", None)

    def complete(self, system, messages, tools):
        start = self.clock()
        turn = self.inner.complete(system, messages, tools)
        end = self.clock()
        self.calls.append({
            "start": start, "end": end,
            "first_token": start + turn.ttft_ms if turn.ttft_ms is not None else None,
            "ttft_ms": turn.ttft_ms, "kind": "tool_call" if turn.tool_calls else "answer",
            "tools": [c.name for c in turn.tool_calls],
            "input_tokens": turn.usage.get("input_tokens"),
            "output_tokens": turn.usage.get("output_tokens")})
        return turn

    def assistant_message(self, turn):
        return self.inner.assistant_message(turn)

    def tool_result_messages(self, pairs):
        return self.inner.tool_result_messages(pairs)


class TimedRetriever:
    def __init__(self, inner, clock):
        self.inner, self.clock = inner, clock
        self.calls: list[dict] = []

    def retrieve(self, query, mode="full", top_k=5):
        start = self.clock()
        res = self.inner.retrieve(query, mode=mode, top_k=top_k)
        self.calls.append({"start": start, "end": self.clock(), "query": query,
                           "stages": dict(res.timings_ms),
                           "reranked": getattr(res, "candidates_reranked", None)})
        return res


def model_floor_ttft(llm, question: str) -> float | None:
    """TTFT of the same model on the same system prompt with no tools and no context."""
    turn = llm.complete(SYSTEM_PROMPT, [{"role": "user", "content": question}], [])
    return turn.ttft_ms


def run_query(question: str, customer: str, llm, retriever, mcp) -> dict:
    clock = lambda: time.perf_counter() * 1000  # noqa: E731
    tl, tr = TimedLLM(llm, clock), TimedRetriever(retriever, clock)
    guardrails._request_log.clear()          # eval traffic is not user traffic
    t0 = clock()
    res = SupportAgent(customer, llm=tl, mcp=mcp, retriever=tr,
                       semantic_cache=False).answer(question)
    t_end = clock()

    answer_calls = [c for c in tl.calls if c["kind"] == "answer"]
    final = answer_calls[-1] if answer_calls else None
    user_ttft = (final["first_token"] - t0) if final and final["first_token"] else None
    first_llm = tl.calls[0] if tl.calls else None

    stages = {
        "input_guardrails": (first_llm["start"] - t0) if first_llm else None,
        "llm_tool_decisions": sum(c["end"] - c["start"] for c in tl.calls
                                  if c["kind"] == "tool_call"),
        "retrieval": sum(c["end"] - c["start"] for c in tr.calls),
        "final_call_ttft": final["ttft_ms"] if final else None,
    }
    before_final = (final["start"] - t0) if final else None
    accounted = (stages["input_guardrails"] or 0) + stages["llm_tool_decisions"] + \
        stages["retrieval"]
    stages["other_before_final"] = (before_final - accounted) if before_final else None
    stages["final_generation"] = (final["end"] - final["first_token"]) \
        if final and final["first_token"] else None
    stages["after_answer"] = (t_end - final["end"]) if final else None   # output guard + NLI

    sub = {}
    for c in tr.calls:
        for k, v in c["stages"].items():
            sub[k] = sub.get(k, 0.0) + v

    return {
        "question": question, "outcome": res.outcome, "tool_calls": res.tool_calls,
        "user_ttft_ms": user_ttft, "final_call_ttft_ms": final["ttft_ms"] if final else None,
        "end_to_end_ms": t_end - t0, "llm_rounds": len(tl.calls),
        "n_searches": len(tr.calls), "stages": stages, "retrieval_stages": sub,
        "llm_calls": [{k: v for k, v in c.items() if k not in ("start", "end",
                                                               "first_token")}
                      | {"latency_ms": c["end"] - c["start"]} for c in tl.calls],
        "answer_preview": res.answer[:160],
    }


def _fmt(v):
    return "   –   " if v is None else f"{v / 1000:6.2f}s"


def main(runs: int = 1, limit: int | None = None, customer: str = "CUST-1001") -> dict:
    from .llm import build_llm
    from .mcp_client import MCPToolClient
    from .retriever import HybridRetriever

    cases = json.loads((config.DATA_DIR / "eval" / "answer_cases.json")
                       .read_text(encoding="utf-8"))["cases"]
    # RAG queries = in-scope policy questions (knowledge scenario).
    questions = [c["question"] for c in cases if c["scenario"] == "knowledge"
                 and c["id"] not in ("a15",)][:limit]

    llm, retriever = build_llm(), HybridRetriever(cache_size=0)
    print(f"agent model: {getattr(llm, 'model', '?')} | reranker: {config.RERANKER_MODEL} | "
          f"live grounding: {config.LIVE_GROUNDING}", flush=True)

    with MCPToolClient(customer).start() as mcp:
        print("cold start (first query loads embedding/rerank/NLI models)…", flush=True)
        cold = run_query(questions[0], customer, llm, retriever, mcp)
        print(f"  cold user TTFT {_fmt(cold['user_ttft_ms'])}  "
              f"end-to-end {_fmt(cold['end_to_end_ms'])}", flush=True)

        floors = [model_floor_ttft(llm, q) for q in questions]
        rows = []
        for r in range(runs):
            for q in questions:
                row = run_query(q, customer, llm, retriever, mcp)
                row["run"] = r
                rows.append(row)
                s = row["stages"]
                print(f"  user TTFT {_fmt(row['user_ttft_ms'])} | final-call TTFT "
                      f"{_fmt(row['final_call_ttft_ms'])} | LLM tool-decisions "
                      f"{_fmt(s['llm_tool_decisions'])} | retrieval {_fmt(s['retrieval'])} "
                      f"| rounds {row['llm_rounds']} | {q[:50]}", flush=True)

    def summ(xs):
        return {k: (round(v, 1) if isinstance(v, float) else v)
                for k, v in latency_summary([x for x in xs if x is not None]).items()}

    stage_keys = ["input_guardrails", "llm_tool_decisions", "retrieval",
                  "other_before_final", "final_call_ttft"]
    sub_keys = sorted({k for r in rows for k in r["retrieval_stages"]})
    report = {
        "kind": "ttft_eval", "created": datetime.now(UTC).isoformat(),
        "config": {"agent_model": getattr(llm, "model", None),
                   "reranker_model": config.RERANKER_MODEL,
                   "rerank_candidates_cap": config.MAX_RERANK_CANDIDATES,
                   "live_grounding": config.LIVE_GROUNDING, "runs": runs,
                   "n_questions": len(questions)},
        "cold_start": cold,
        "summary": {
            "user_ttft_ms": summ([r["user_ttft_ms"] for r in rows]),
            "final_call_ttft_ms": summ([r["final_call_ttft_ms"] for r in rows]),
            "model_floor_ttft_ms": summ(floors),
            "end_to_end_ms": summ([r["end_to_end_ms"] for r in rows]),
            "llm_rounds_mean": statistics.fmean(r["llm_rounds"] for r in rows),
            "searches_mean": statistics.fmean(r["n_searches"] for r in rows),
        },
        "ttft_breakdown_mean_ms": {k: round(statistics.fmean(
            r["stages"][k] for r in rows if r["stages"][k] is not None), 1)
            for k in stage_keys},
        "retrieval_breakdown_mean_ms": {k: round(statistics.fmean(
            r["retrieval_stages"].get(k, 0.0) for r in rows), 1) for k in sub_keys},
        "rows": rows,
    }
    out = config.PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    path = out / f"ttft_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    s = report["summary"]
    print("\n── TTFT summary (warm) ──")
    for k in ("user_ttft_ms", "final_call_ttft_ms", "model_floor_ttft_ms", "end_to_end_ms"):
        print(f"  {k:22} p50 {_fmt(s[k].get('p50'))}  p95 {_fmt(s[k].get('p95'))}  "
              f"mean {_fmt(s[k].get('mean'))}")
    print(f"  LLM rounds/query {s['llm_rounds_mean']:.2f} | searches/query "
          f"{s['searches_mean']:.2f}")
    print("── where user TTFT goes (mean) ──")
    tot = sum(v for v in report["ttft_breakdown_mean_ms"].values() if v)
    for k, v in report["ttft_breakdown_mean_ms"].items():
        print(f"  {k:22} {_fmt(v)}  {v / tot:5.1%}")
    print("── retrieval sub-stages (mean per query) ──")
    for k, v in report["retrieval_breakdown_mean_ms"].items():
        print(f"  {k:22} {_fmt(v)}")
    print(f"report: {path}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--customer", default="CUST-1001")
    a = ap.parse_args()
    main(runs=a.runs, limit=a.limit, customer=a.customer)

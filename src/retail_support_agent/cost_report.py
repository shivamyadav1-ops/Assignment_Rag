"""Cost and latency report built from an answer-eval run and its execution traces.

Inputs
  reports/answer_eval_<label>.json   per-case tokens, latency, retrieval time
  traces/trace_<id>.jsonl            per-step timings referenced by each case

Outputs
  reports/cost_latency_<label>.json  machine-readable numbers
  reports/cost_latency_<label>.md    the table that goes in the README / slides

Tokens and latency are measured. Money is not: the hosted NVIDIA API-catalog
endpoints used here have no published per-token list price, so cost is computed
from prices you supply (RSA_PRICE_INPUT_PER_M / RSA_PRICE_OUTPUT_PER_M, USD per
million tokens) and shown as a sensitivity table across illustrative price points.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from . import config

PRICE_POINTS = [(0.10, 0.40), (0.40, 1.60), (1.00, 4.00), (3.00, 15.00)]


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, round(p * (len(xs) - 1))))
    return xs[i]


def _stats(xs: list[float]) -> dict:
    return {"n": len(xs), "mean": round(mean(xs), 1) if xs else 0.0,
            "p50": round(median(xs), 1) if xs else 0.0,
            "p95": round(_pct(xs, 0.95), 1), "max": round(max(xs), 1) if xs else 0.0}


def _trace_steps(path: str | None) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.is_absolute():
        p = config.PROJECT_ROOT / p
    if not p.exists():
        p = config.TRACES_DIR / Path(path).name
    if not p.exists():
        return []
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in rows if r.get("type") == "step"]


def _cost(inp: float, out: float, p_in: float, p_out: float) -> float:
    return (inp * p_in + out * p_out) / 1e6


def build(label: str) -> dict:
    src = config.PROJECT_ROOT / "reports" / f"answer_eval_{label}.json"
    report = json.loads(src.read_text(encoding="utf-8"))
    cases = report["cases"]

    by_scn: dict[str, list[dict]] = defaultdict(list)
    stage_ms: dict[str, list[float]] = defaultdict(list)
    llm_turns, ttft = [], []
    for c in cases:
        by_scn[c["scenario"]].append(c)
        steps = _trace_steps(c.get("trace"))
        llm = [s for s in steps if s["kind"] == "llm"]
        llm_turns.append(len(llm))
        ttft += [s["ttft_ms"] for s in llm if s.get("ttft_ms")]
        stage_ms["llm"].append(sum(s["latency_ms"] for s in llm))
        stage_ms["search_policies (retrieval)"].append(
            sum(s["latency_ms"] for s in steps if s["name"] == "search_policies"))
        stage_ms["MCP tools"].append(
            sum(s["latency_ms"] for s in steps
                if s["kind"] == "tool" and s["name"] != "search_policies"))

    def usage_block(rows: list[dict]) -> dict:
        inp = [r["usage"].get("input_tokens", 0) for r in rows]
        out = [r["usage"].get("output_tokens", 0) for r in rows]
        cached = [r["usage"].get("cached_input_tokens", 0) for r in rows]
        return {
            "cases": len(rows),
            "input_tokens_mean": round(mean(inp)) if rows else 0,
            "output_tokens_mean": round(mean(out)) if rows else 0,
            "cached_input_share": round(sum(cached) / max(sum(inp), 1), 3),
            "latency_ms": _stats([r["latency_ms"] for r in rows]),
        }

    overall = usage_block(cases)
    p_in = float(os.getenv("RSA_PRICE_INPUT_PER_M", "0") or 0)
    p_out = float(os.getenv("RSA_PRICE_OUTPUT_PER_M", "0") or 0)
    sens = [{"input_per_m": a, "output_per_m": b,
             "usd_per_query": round(_cost(overall["input_tokens_mean"],
                                          overall["output_tokens_mean"], a, b), 5),
             "usd_per_1k_queries": round(1000 * _cost(overall["input_tokens_mean"],
                                                      overall["output_tokens_mean"], a, b), 2)}
            for a, b in PRICE_POINTS]
    judge = report.get("judge_usage", {})
    return {
        "source_report": src.name,
        "agent_model": report["config"].get("agent_model"),
        "judge_models": report["config"].get("judge_models"),
        "overall": overall,
        "by_scenario": {k: usage_block(v) for k, v in sorted(by_scn.items())},
        "llm_turns_per_case": _stats(llm_turns),
        "ttft_ms": _stats(ttft),
        "stage_latency_ms": {k: _stats(v) for k, v in stage_ms.items()},
        "configured_price": ({"input_per_m": p_in, "output_per_m": p_out,
                              "usd_per_query": round(_cost(overall["input_tokens_mean"],
                                                           overall["output_tokens_mean"],
                                                           p_in, p_out), 5)}
                             if p_in or p_out else None),
        "price_sensitivity": sens,
        "eval_judge_tokens": {"input": judge.get("input_tokens"),
                              "output": judge.get("output_tokens"),
                              "calls": judge.get("calls")},
    }


def render_md(r: dict) -> str:
    o = r["overall"]
    lines = [
        f"# Cost and latency — `{r['source_report']}`",
        "",
        f"Agent model: `{r['agent_model']}` · {o['cases']} eval cases · tokens and latency "
        "measured; dollar cost depends on the price you pay (see sensitivity table).",
        "",
        "## Per query, by scenario",
        "",
        "| Scenario | Cases | Input tok (mean) | Output tok (mean) | Cached input | "
        "Latency p50 | Latency p95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, b in [*r["by_scenario"].items(), ("**all**", o)]:
        lines.append(f"| {name} | {b['cases']} | {b['input_tokens_mean']:,} | "
                     f"{b['output_tokens_mean']:,} | {b['cached_input_share']:.0%} | "
                     f"{b['latency_ms']['p50'] / 1000:.1f}s | {b['latency_ms']['p95'] / 1000:.1f}s |")
    lines += ["", "## Where the time goes (per case, from traces)", "",
              "| Stage | Mean | p50 | p95 |", "|---|---|---|---|"]
    for name, s in r["stage_latency_ms"].items():
        lines.append(f"| {name} | {s['mean'] / 1000:.2f}s | {s['p50'] / 1000:.2f}s | "
                     f"{s['p95'] / 1000:.2f}s |")
    t, ft = r["llm_turns_per_case"], r["ttft_ms"]
    lines += ["", f"LLM turns per case: mean {t['mean']}, max {t['max']:.0f}. "
              f"Time to first token: p50 {ft['p50']:.0f} ms, p95 {ft['p95']:.0f} ms.",
              "", "## Cost sensitivity (agent tokens only)", "",
              "| Input $/M | Output $/M | $ per query | $ per 1,000 queries |",
              "|---|---|---|---|"]
    for s in r["price_sensitivity"]:
        lines.append(f"| {s['input_per_m']:.2f} | {s['output_per_m']:.2f} | "
                     f"{s['usd_per_query']:.5f} | {s['usd_per_1k_queries']:.2f} |")
    if r["configured_price"]:
        c = r["configured_price"]
        lines += ["", f"At the configured price (${c['input_per_m']}/M in, "
                  f"${c['output_per_m']}/M out): **${c['usd_per_query']:.5f} per query**."]
    j = r["eval_judge_tokens"]
    lines += ["", f"Evaluation overhead (LLM judge, not paid per customer query): "
              f"{j['calls']} calls, {j['input']:,} input / {j['output']:,} output tokens."
              if j.get("calls") else ""]
    return "\n".join(lines).rstrip() + "\n"


def run(label: str) -> Path:
    r = build(label)
    out = config.PROJECT_ROOT / "reports"
    (out / f"cost_latency_{label}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    md = out / f"cost_latency_{label}.md"
    md.write_text(render_md(r), encoding="utf-8")
    print(render_md(r))
    return md

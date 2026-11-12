"""Generation-quality eval: metric column groups, running the eval from the app, and
loading saved reports (app runs + CLI `eval-answers` runs)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from retail_support_agent import classifiers, config, guardrails
from retail_support_agent.agent import SupportAgent
from retail_support_agent.eval_answers import check_case, render_evidence
from retail_support_agent.metrics import lexical_metrics, slm_metrics

from .formatting import fmt_num, fmt_pct, mean
from .resources import REPORTS_DIR, get_judge, get_llm, get_retriever, load_cases

TEXT_COLS = ["bleu", "rouge1", "rouge2", "rougeL", "token_f1", "fact_recall"]
SLM_COLS = ["nli_faithfulness", "nli_entailment", "nli_contradiction",
            "nli_self_contradiction", "nli_completeness"]
GEVAL_COLS = ["correctness", "completeness"]           # G-Eval: steps + CoT + weighted score
JUDGE_COLS = ["faithfulness", "contradiction", "relevance"]   # claim-level LLM judge
CHECK_COLS = ["facts_ok", "tools_ok", "safety_ok"]      # deterministic, no model
METRIC_LABELS = {
    "bleu": "BLEU-4 (precision)", "rouge1": "ROUGE-1 (recall)", "rouge2": "ROUGE-2 (recall)",
    "rougeL": "ROUGE-L (recall)", "token_f1": "Token F1", "fact_recall": "Fact recall",
    "perplexity": "Perplexity ↓",
    "nli_faithfulness": "NLI faithfulness", "nli_entailment": "NLI entailment",
    "nli_contradiction": "NLI contradiction (source) ↓",
    "nli_self_contradiction": "NLI self-contradiction ↓",
    "nli_completeness": "NLI completeness",
    "correctness": "G-Eval correctness", "completeness": "G-Eval completeness",
    "contradiction": "Judge contradiction ↓", "faithfulness": "Judge faithfulness",
    "relevance": "Judge relevance",
}


def run_generation_eval(case_ids: list[str], use_judge: bool, use_slm: bool) -> dict:
    from retail_support_agent.db import load_operational_data
    from retail_support_agent.mcp_client import MCPToolClient

    cases = [c for c in load_cases() if c["id"] in case_ids]
    judge = get_judge() if use_judge else None
    llm, retriever = get_llm(), get_retriever()
    load_operational_data()            # fresh DB so earlier writes can't skew results
    rows, bar = [], st.progress(0.0, text="Starting…")
    for i, case in enumerate(cases):
        bar.progress(i / len(cases), text=f"{case['id']}: {case['question'][:70]}")
        guardrails._request_log.clear()   # eval traffic is not user traffic
        with MCPToolClient(case["customer"]).start() as mcp:
            res = SupportAgent(case["customer"], llm=llm, mcp=mcp, retriever=retriever,
                               semantic_cache=False).answer(case["question"])
        row = {
            "id": case["id"], "scenario": case["scenario"], "question": case["question"],
            "reference": case["reference"], "answer": res.answer, "outcome": res.outcome,
            "tool_calls": res.tool_calls, "latency_ms": res.latency_ms,
            "retrieval_ms": res.retrieval_ms, "ttft_ms": res.ttft_ms,
            "input_tokens": res.usage["input_tokens"],
            "output_tokens": res.usage["output_tokens"],
            "checks": check_case(case, res.answer, res.tool_calls),
            **lexical_metrics(res.answer, case["reference"], case.get("expected_facts")),
        }
        if use_slm and res.outcome != "blocked":
            s = slm_metrics(res.answer, res.context, case["reference"],
                            grounding=res.grounding)
            row.update({k: s.get(k) for k in ["perplexity", *SLM_COLS, "slm_ms"]})
            row["slm_detail"] = {"claims": s.get("nli_claims"), "points": s.get("nli_points")}
        if judge:
            j = judge.all_metrics(case["question"], res.answer,
                                  render_evidence(res.context), case["reference"])
            row["judge"] = j
            row["judge_ms"], row["judge_tokens"] = j.get("judge_ms"), j.get("judge_tokens")
            for k in GEVAL_COLS + JUDGE_COLS:
                row[k] = (j.get(k) or {}).get("score")
            row["verdict"] = (j.get("correctness") or {}).get("verdict")
        rows.append(row)
    bar.progress(1.0, text="Done")
    load_operational_data()
    report = {"created": datetime.now(UTC).isoformat(), "kind": "app_generation_eval",
              "agent_model": getattr(llm, "model", None),
              "judge_model": judge.model if judge else None,
              "judge_method": config.JUDGE_METHOD if judge else None,
              "nli_model": classifiers.NLI_MODEL if use_slm else None, "cases": rows}
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"app_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["path"] = str(path)
    return report


def load_eval_reports() -> dict[str, dict]:
    """App reports + CLI `eval-answers` reports (text metrics recomputed for the latter)."""
    out = {}
    refs = {c["id"]: c for c in load_cases()}
    if not REPORTS_DIR.exists():
        return out
    # Newest first, so the default report is the latest run (not the old verdict judge).
    for p in sorted(REPORTS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime,
                    reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("kind") == "app_generation_eval":
            out[p.name] = data
        elif p.name.startswith("answer_eval_") and "cases" in data:
            rows = []
            for c in data["cases"]:
                ref = refs.get(c["id"], {})
                rows.append({
                    **{k: c.get(k) for k in ("id", "scenario", "question", "answer",
                                             "outcome", "tool_calls", "latency_ms",
                                             "retrieval_ms", "checks")},
                    "reference": ref.get("reference", ""),
                    "input_tokens": c.get("usage", {}).get("input_tokens", 0),
                    "output_tokens": c.get("usage", {}).get("output_tokens", 0),
                    **lexical_metrics(c.get("answer", ""), ref.get("reference", ""),
                                      ref.get("expected_facts")),
                    "correctness": (c.get("correctness") or {}).get("score"),
                    "completeness": (c.get("completeness") or {}).get("score"),
                    "faithfulness": (c.get("faithfulness") or {}).get("score"),
                    "verdict": (c.get("correctness") or {}).get("verdict"),
                    "judge": {k: c.get(k) for k in ("correctness", "completeness",
                                                    "faithfulness") if c.get(k)},
                })
            cfg = data.get("config", {})
            out[p.name] = {"created": data.get("created"), "cases": rows,
                           "agent_model": cfg.get("agent_model"),
                           "judge_model": ", ".join(cfg.get("judge_models") or [])
                           or cfg.get("judge_model"),
                           "judge_method": cfg.get("judge_method", "verdict")}
    return out


def metric_row(df: pd.DataFrame, cols: list[str], pct: bool) -> None:
    present = [m for m in cols if m in df and df[m].notna().any()]
    if not present:
        st.caption("Not computed in this report.")
        return
    cs = st.columns(len(present))
    for col, m in zip(cs, present, strict=True):
        v = mean(df[m].tolist())
        col.metric(METRIC_LABELS[m], fmt_pct(v) if pct else fmt_num(v))

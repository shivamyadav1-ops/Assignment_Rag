"""ShopSmart support agent — interactive demo + metrics dashboard.

Run:  .venv/bin/streamlit run app/streamlit_app.py

Tabs
  Assistant           chat as a customer (text + optional photo). Per answer: latency,
                      tokens, cost, guardrail events, live NLI grounding (faithfulness,
                      contradiction vs. source and vs. itself), trace, evidence; on
                      demand: SLM metrics (perplexity, NLI completeness) and LLM judge.
  Generation quality  labelled eval set. Text generation: perplexity, BLEU (precision),
                      ROUGE-1/2/L (recall), token F1, fact recall. Grounding (SLM/NLI):
                      faithfulness, entailment, contradiction, self-contradiction,
                      completeness. LLM judge: correctness, completeness, contradiction,
                      faithfulness, relevance. SLM vs. LLM-judge cost/latency/agreement.
  Latency & cost      percentiles and breakdowns from every trace in traces/
  Security            red-team suite, regex vs. regex+classifiers; image NSFW; guardrail
                      events in live traffic; live tester
  Retrieval & index   ranking metrics from retrieval eval reports + vector index stats
  G-Eval & judge      answer-eval reports scored with G-Eval (per-step chain-of-thought
                      findings), judge calibration vs human labels, judge-bias check
  Red-team            hallucination red-team: adversarial prompts (false premises, absent
                      facts, record bait, pressure, injected facts), hallucination-free rate
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make the `src/` layout importable even when Streamlit is launched from a Python
# that does not have the project installed (e.g. a global/conda `streamlit`).
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from retail_support_agent import classifiers, config, guardrails  # noqa: E402
from retail_support_agent.agent import SYSTEM_PROMPT, SupportAgent  # noqa: E402
from retail_support_agent.eval_answers import (  # noqa: E402
    CASES_PATH,
    JUDGE_MODELS,
    check_case,
    render_evidence,
)
from retail_support_agent.metrics import (  # noqa: E402
    GenerationJudge,
    cost_usd,
    latency_summary,
    lexical_metrics,
    load_traces,
    slm_metrics,
)
from retail_support_agent.security_eval import (  # noqa: E402
    run_image_suite,
    run_security_suite,
    summarise_security,
)

REPORTS_DIR = config.PROJECT_ROOT / "reports"

st.set_page_config(page_title="ShopSmart Support Agent", page_icon="🛒", layout="wide")


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading retriever (embeddings + reranker)…")
def get_retriever():
    from retail_support_agent.retriever import HybridRetriever
    return HybridRetriever()


@st.cache_resource(show_spinner="Connecting to LLM…")
def get_llm():
    from retail_support_agent.llm import build_llm
    return build_llm()


@st.cache_resource(show_spinner="Starting MCP tool server…")
def get_mcp(customer: str):
    from retail_support_agent.mcp_client import MCPToolClient
    return MCPToolClient(customer).start()


@st.cache_resource
def get_judge():
    return GenerationJudge()


@st.cache_resource(show_spinner="Loading local classifier models (first run downloads)…")
def warm_classifiers() -> bool:
    try:
        classifiers.warmup()
        return True
    except Exception:  # noqa: BLE001
        return False


@st.cache_data
def load_customers() -> pd.DataFrame:
    return pd.read_csv(config.OPERATIONAL_DIR / "customers.csv")


@st.cache_data
def load_orders() -> pd.DataFrame:
    return pd.read_csv(config.OPERATIONAL_DIR / "orders.csv")


@st.cache_data
def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]


@st.cache_data(show_spinner="Running red-team suite…")
def cached_security(use_ml: bool) -> list[dict]:
    return run_security_suite(use_ml=use_ml)


@st.cache_data(show_spinner="Running image guardrail checks…")
def cached_image_suite() -> list[dict]:
    return run_image_suite()


def judge_available() -> bool:
    return bool(os.getenv("NVIDIA_API_KEY"))


def fmt_ms(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    return f"{v / 1000:.2f} s" if v >= 1000 else f"{v:.0f} ms"


def fmt_pct(v) -> str:
    return "–" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:.0%}"


def fmt_num(v, d=3) -> str:
    return "–" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:.{d}f}"


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and pd.isna(x))]
    return sum(xs) / len(xs) if xs else None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🛒 ShopSmart Support")
    provider = config.LLM_PROVIDER
    model = config.NVIDIA_MODEL if provider == "nvidia" else config.LLM_MODEL
    st.caption(f"Agent: `{provider}` · `{model}`")
    st.caption(f"Reranker: `{config.RERANKER_MODEL}` · scores "
               f"{config.MAX_RERANK_CANDIDATES} candidates → top {config.RERANK_TOP_K}")
    st.caption("Judge: " + (" → ".join(f"`{m}`" for m in JUDGE_MODELS)
                            if judge_available() else "unavailable (set NVIDIA_API_KEY)")
               + f" · method `{config.JUDGE_METHOD}`")

    customers = load_customers()
    labels = {r.customer_id: f"{r.customer_id} · {r.first_name} {r.last_name} "
                             f"({r.membership_tier})" for r in customers.itertuples()}
    customer = st.selectbox("Signed-in customer", list(labels), format_func=labels.get)
    with st.expander("This customer's orders"):
        orders = load_orders()
        st.dataframe(
            orders[orders.customer_id == customer][
                ["order_id", "product_name", "product_category", "status", "delivery_date"]],
            hide_index=True, width="stretch")

    st.divider()
    st.subheader("Memory & caching")
    memory_on = st.toggle("Conversation memory", value=True,
                          help=f"Earlier turns go into the prompt; every "
                               f"{config.SESSION_SUMMARY_EVERY} turns they are summarised "
                               "into summary.md, which is sent along with recent turns.")
    cache_scope = st.radio("Answer cache", ["global", "session", "off"], horizontal=True,
                           help="global: policy-only answers shared across customers · "
                                "session: reused only within this chat · off")
    with st.expander("Cache status"):
        try:
            from retail_support_agent.cache import shared_semantic_cache
            sc = shared_semantic_cache()
            cs = sc.stats()
            st.caption(f"Semantic cache: **{cs['entries']}** answers · "
                       f"{cs['lifetime_hits']} lifetime hits · threshold {cs['threshold']} · "
                       f"TTL {cs['ttl_s'] / 86400:.0f} d")
            if cs["session"]:
                st.caption("This process: " + ", ".join(
                    f"{k}={v}" for k, v in sorted(cs["session"].items())))
            rc = get_retriever().cache.stats()
            st.caption(f"Retrieval cache: {rc['size']}/{rc['maxsize']} · "
                       f"hits {rc['hits']} · misses {rc['misses']}")
            if st.button("Clear semantic cache"):
                st.toast(f"Removed {sc.clear()} cached answers")
        except Exception as exc:  # noqa: BLE001
            st.caption(f"Cache unavailable: {exc}")

    st.divider()
    st.subheader("Guardrails & grounding")
    ml_on = st.toggle("Classifier guardrails (toxicity, injection)", value=True,
                      help="toxic-bert + DeBERTa injection classifier after the regex layers")
    live_g = st.toggle("Live NLI grounding on every answer", value=True)
    g_action = st.radio("On contradiction", ["flag", "block"], horizontal=True,
                        help="block = replace a contradicting answer with an escalation")
    auto_judge = st.toggle("LLM judge on every answer", value=True,
                           disabled=not judge_available(),
                           help="scores correctness, completeness, faithfulness, contradiction "
                                "and relevance right after the answer — costs one extra judge "
                                "call and adds latency per question. Turn off to score on "
                                "demand instead (the button still appears under each answer)."
                                + ("" if judge_available()
                                   else " Unavailable: set NVIDIA_API_KEY."))
    os.environ["RSA_ML_GUARDRAILS"] = "1" if ml_on else "0"
    config.LIVE_GROUNDING = live_g
    config.GROUNDING_ACTION = g_action
    if (ml_on or live_g) and not warm_classifiers():
        st.error("Classifier models failed to load — classifier layers will fail open.")

    st.divider()
    st.subheader("Cost assumptions")
    st.caption("USD per 1M tokens — set to your provider's price.")
    price_in = st.number_input("Input", min_value=0.0, value=3.0, step=0.5)
    price_out = st.number_input("Output", min_value=0.0, value=15.0, step=0.5)


tab_chat, tab_gen, tab_geval, tab_red, tab_lat, tab_sec, tab_ret = st.tabs(
    ["💬 Assistant", "📝 Generation quality", "🧪 G-Eval & judge", "🎯 Red-team",
     "⏱ Latency & cost", "🛡 Security", "🔎 Retrieval & index"])


# ══════════════════════════════════════════════════════════════════════════════
# Assistant
# ══════════════════════════════════════════════════════════════════════════════

def reference_for(question: str) -> dict | None:
    q = question.strip().lower()
    return next((c for c in load_cases() if c["question"].strip().lower() == q), None)


def render_grounding(g: dict | None) -> None:
    if not g:
        st.caption("Live grounding off or not applicable.")
        return
    if "error" in g:
        st.caption(f"Grounding unavailable: {g['error']}")
        return
    c = st.columns(5)
    c[0].metric("Faithfulness (NLI)", fmt_pct(g.get("faithfulness")),
                help="answer claims entailed by evidence")
    c[1].metric("Entailment", fmt_num(g.get("entailment"), 2),
                help="mean best entailment probability per claim")
    c[2].metric("Contradiction vs source ↓", fmt_pct(g.get("contradiction_source")))
    c[3].metric("Self-contradiction ↓", fmt_pct(g.get("self_contradiction")))
    c[4].metric("Claims / NLI time", f"{g.get('n_claims', 0)} · "
                f"{fmt_ms(g.get('latency_ms'))}")
    if g.get("claims"):
        with st.expander("Claim-level NLI"):
            st.dataframe(pd.DataFrame(g["claims"]), hide_index=True,
                         width="stretch")
            for sc in g.get("self_contradictions", []):
                st.warning(f"Self-contradiction ({sc['p']}): “{sc['a']}” ⟂ “{sc['b']}”")


def render_result(entry: dict, key: str) -> None:
    res = entry["result"]
    if entry.get("image_check"):
        st.caption(f"📷 Attached image checked: {entry['image_check']}")
    cols = st.columns(6)
    cols[0].metric("Outcome", res.outcome)
    cols[1].metric("End-to-end", fmt_ms(res.latency_ms))
    cols[2].metric("TTFT (final)", fmt_ms(res.ttft_ms))
    cols[3].metric("Retrieval", fmt_ms(res.retrieval_ms))
    tokens = res.usage["input_tokens"] + res.usage["output_tokens"]
    cols[4].metric("Tokens", f"{tokens:,}")
    cols[5].metric("Est. cost", f"${cost_usd(res.usage, price_in, price_out):.4f}")
    notes = []
    if res.cache and res.cache.get("hit"):
        notes.append(f"⚡ served from answer cache (similarity {res.cache.get('similarity')}"
                     f", matched “{res.cache.get('cached_question')}”)")
    elif res.cache and res.cache.get("stored"):
        notes.append("💾 stored in answer cache")
    if res.usage.get("cached_input_tokens"):
        notes.append(f"🔁 {res.usage['cached_input_tokens']:,} prompt tokens from the "
                     "provider's context cache")
    if getattr(res, "memory", None):
        m = res.memory
        notes.append(f"🧠 memory: {m['history_messages_in_prompt']} earlier messages + "
                     f"{m['memory_chars_in_prompt']:,} chars of summary in prompt"
                     + (" · summary.md updated this turn" if m["summarised_this_turn"] else ""))
    if notes:
        st.caption("  \n".join(notes))

    guard = [s for s in res.steps if s.kind == "guardrail"]
    if guard:
        for s in guard:
            st.warning(f"🛡 **{s.name}** — {json.dumps(s.detail, default=str)[:300]}")
    else:
        st.success("🛡 All guardrails passed (input, tool calls, tool output, answer).")

    if res.outcome not in ("blocked",):
        st.markdown("###### Live grounding (NLI)")
        render_grounding(res.grounding)

    with st.expander("Execution trace"):
        st.dataframe(pd.DataFrame([{
            "#": s.index, "kind": s.kind, "name": s.name,
            "latency": fmt_ms(s.latency_ms),
            "tokens": (s.usage.get("input_tokens", 0) + s.usage.get("output_tokens", 0))
            if s.usage else None,
            "detail": json.dumps(s.detail, default=str)[:200]} for s in res.steps]),
            hide_index=True, width="stretch")
    if res.evidence:
        with st.expander(f"Evidence ({len(res.evidence)} chunks)"):
            st.dataframe(pd.DataFrame(res.evidence), hide_index=True,
                         width="stretch")

    ref_case = reference_for(res.question)
    if ref_case:
        lex = lexical_metrics(res.answer, ref_case["reference"],
                              ref_case.get("expected_facts"))
        st.caption(f"Matches eval case **{ref_case['id']}** — reference-based scores:")
        c = st.columns(6)
        c[0].metric("BLEU (precision)", fmt_num(lex["bleu"]))
        c[1].metric("ROUGE-1 recall", fmt_num(lex["rouge1"]))
        c[2].metric("ROUGE-2 recall", fmt_num(lex["rouge2"]))
        c[3].metric("ROUGE-L recall", fmt_num(lex["rougeL"]))
        c[4].metric("Token F1", fmt_num(lex["token_f1"]))
        c[5].metric("Fact recall", fmt_pct(lex["fact_recall"]))

    if res.outcome == "blocked":
        return
    b1, b2 = st.columns(2)
    if "slm" in entry:
        s = entry["slm"]
        c = st.columns(3)
        c[0].metric("Perplexity (distilgpt2)", fmt_num(s.get("perplexity"), 1),
                    help="fluency only — lower is more fluent")
        c[1].metric("NLI completeness", fmt_pct(s.get("nli_completeness")),
                    help="reference points entailed by the answer")
        c[2].metric("SLM time · cost", f"{fmt_ms(s.get('slm_ms'))} · $0")
    elif b1.button("SLM metrics (perplexity, NLI completeness)", key=f"slm-{key}"):
        with st.spinner("Running local models…"):
            entry["slm"] = slm_metrics(res.answer, res.context,
                                       ref_case["reference"] if ref_case else None,
                                       grounding=res.grounding)
        st.rerun()
    if "judge" in entry:
        render_judge(entry["judge"])
    elif b2.button("Score with the LLM judge", key=f"judge-{key}",
                   disabled=not judge_available()):
        with st.spinner("Judging…"):
            entry["judge"] = get_judge().all_metrics(
                res.question, res.answer, render_evidence(res.context),
                ref_case["reference"] if ref_case else None)
        st.rerun()


def render_judge(j: dict) -> None:
    st.markdown("###### LLM judge")
    items = [
        ("Faithfulness", j.get("faithfulness", {}).get("score"), "claims supported by evidence"),
        ("Contradiction ↓", j.get("contradiction", {}).get("score"), "claims contradicted"),
        ("Relevance", j.get("relevance", {}).get("score"), "addresses the question"),
        ("G-Eval correctness", (j.get("correctness") or {}).get("score"),
         "vs reference and retrieved evidence · verdict: "
         + str((j.get("correctness") or {}).get("verdict") or "–")),
        ("G-Eval completeness", (j.get("completeness") or {}).get("score"),
         "parts of the question covered"),
        ("Judge time · tokens", None, ""),
    ]
    cols = st.columns(len(items))
    for col, (name, val, help_) in zip(cols[:-1], items[:-1], strict=True):
        col.metric(name, fmt_pct(val), help=help_)
    cols[-1].metric("Judge time · tokens",
                    f"{fmt_ms(j.get('judge_ms'))} · {j.get('judge_tokens', 0):,}")
    for name in ("correctness", "completeness"):
        m = j.get(name) or {}
        if m.get("steps"):
            render_geval_steps(name, m)
    with st.expander("Judge details"):
        st.json(j)


def render_geval_steps(name: str, m: dict) -> None:
    """G-Eval chain of thought: the judge's finding for each evaluation step."""
    score = m.get("score")
    verdict = m.get("verdict")
    head = (f"🧪 G-Eval {name}: **{fmt_pct(score)}**"
            + (f" · {verdict}" if verdict else "")
            + (f" · raw {m.get('raw_score')}/10" if m.get("raw_score") is not None else "")
            + (" · probability-weighted" if m.get("weighted") else ""))
    with st.expander(head):
        for st_ in m.get("steps") or []:
            st.markdown(f"**Step {st_['step']}.** {st_['finding']}")


with tab_chat:
    st.subheader("Ask the support agent")
    st.caption("Try a policy question, an order question (see the sidebar for this "
               "customer's orders), or an attack such as *ignore all previous "
               "instructions…* or *f u c k you* to see the guardrails react. You can attach "
               "a photo (e.g. of a damaged item) — it is screened by the NSFW image "
               "classifier.")
    history = st.session_state.setdefault("history", [])
    # One ChatSession per signed-in customer and settings; a new customer, a changed
    # setting or "Clear conversation" starts a fresh session (new memory files).
    from retail_support_agent.session import ChatSession
    sess_key = (customer, memory_on, cache_scope)
    if st.session_state.get("chat_key") != sess_key:
        st.session_state["chat_key"] = sess_key
        st.session_state["chat"] = ChatSession(
            customer, llm=get_llm(), mcp=get_mcp(customer), retriever=get_retriever(),
            memory=memory_on, cache=cache_scope)
        history.clear()
    chat = st.session_state["chat"]
    c1, c2 = st.columns([1, 4])
    if history and c1.button("Clear conversation"):
        history.clear()
        st.session_state.pop("chat_key", None)
        st.rerun()
    n_turns = chat.total_turns
    c2.caption(f"Session `{chat.session_id}` · {n_turns} turn(s) · summarised up to turn "
               f"{chat.meta['summarized_upto']} · next summary at "
               f"{chat.meta['summarized_upto'] + chat.summary_every} · memory "
               f"{'on' if memory_on else 'off'} · cache {cache_scope}")
    if chat.summary():
        with st.expander("🧠 Conversation memory (summary.md)"):
            st.markdown(chat.summary())

    for i, entry in enumerate(history):
        with st.chat_message("user"):
            st.markdown(f"**{entry['customer']}:** {entry['question']}")
            if entry.get("image_bytes"):
                st.image(entry["image_bytes"], width=160)
        with st.chat_message("assistant"):
            st.markdown(entry["answer"] if "answer" in entry else entry["result"].answer)
            if "result" in entry:
                render_result(entry, str(i))

    submitted = st.chat_input("Ask about orders, returns, shipping, refunds…",
                              accept_file=True, file_type=["png", "jpg", "jpeg", "webp"])
    if submitted:
        question = submitted.text or ""
        files = submitted.files or []
        entry: dict = {"customer": customer, "question": question or "(photo)"}
        if files:
            data = files[0].getvalue()
            entry["image_bytes"] = data
            check = guardrails.check_image_nsfw(data)
            entry["image_check"] = ("blocked — " if not check.passed else "ok — ") + \
                check.detail + f" ({check.latency_ms:.0f} ms)"
            if not check.passed:
                entry["answer"] = guardrails._BLOCK_MESSAGES[guardrails.BlockReason.NSFW] + \
                    f"\n\n_🛡 nsfw_content — {check.detail}_"
                history.append(entry)
                st.rerun()
            # The agent is text-only: tell it a photo exists, don't pretend to analyse it.
            question = (question + f"\n\n[Customer attached a photo: {files[0].name}. "
                        "Images are not analysed by the assistant.]").strip()
        if question:
            with st.chat_message("user"):
                st.markdown(f"**{customer}:** {entry['question']}")
            with st.chat_message("assistant"):
                with st.spinner("Thinking — retrieval with reranking can take a while on CPU…"):
                    try:
                        entry["result"] = chat.ask(question)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Agent failed: {exc}")
                        st.stop()
                res = entry["result"]
                if auto_judge and judge_available() and res.outcome not in ("blocked",):
                    with st.spinner("Judging…"):
                        try:
                            ref_case = reference_for(res.question)
                            entry["judge"] = get_judge().all_metrics(
                                res.question, res.answer, render_evidence(res.context),
                                ref_case["reference"] if ref_case else None)
                        except Exception as exc:  # noqa: BLE001
                            st.warning(f"LLM judge failed: {exc}")
            history.append(entry)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Generation quality
# ══════════════════════════════════════════════════════════════════════════════

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


with tab_gen:
    st.subheader("Generation quality on the labelled eval set")
    st.caption("**1 · G-Eval (primary):** correctness and completeness scored by the judge "
               "working through explicit steps with chain-of-thought and a probability-"
               "weighted score, given the retrieved evidence; plus deterministic fact / tool "
               "/ safety checks. **2 · Hallucination & grounding:** claim-level LLM judge and "
               "a local NLI model. **3 · Text overlap:** BLEU/ROUGE wording similarity, kept "
               "for reference only.")
    cases = load_cases()
    with st.form("gen-eval"):
        c1, c2 = st.columns([3, 1])
        picked = c1.multiselect(
            "Cases", [c["id"] for c in cases], default=[c["id"] for c in cases[:4]],
            format_func=lambda i: f"{i} · " + next(c["question"] for c in cases
                                                    if c["id"] == i)[:60])
        use_slm = c2.toggle("SLM metrics (NLI + perplexity)", value=True)
        use_judge = c2.toggle("LLM judge", value=judge_available(),
                              disabled=not judge_available())
        c2.caption(f"~1 min per case on CPU, {len(picked)} selected")
        go_btn = st.form_submit_button("Run evaluation", type="primary")
    if go_btn and picked:
        st.session_state["gen_report"] = run_generation_eval(picked, use_judge, use_slm)
        st.success(f"Saved {st.session_state['gen_report']['path']}")

    reports = load_eval_reports()
    if not reports:
        st.info("No evaluation reports yet — run one above.")
    else:
        names = list(reports)
        default = 0
        if "gen_report" in st.session_state:
            nm = os.path.basename(st.session_state["gen_report"]["path"])
            default = names.index(nm) if nm in names else 0
        choice = st.selectbox("Report", names, index=default)
        rep = reports[choice]
        df = pd.DataFrame(rep["cases"])
        checks = pd.json_normalize(df["checks"]) if "checks" in df else pd.DataFrame()
        for k in CHECK_COLS:
            if k in checks:
                df[k] = checks[k].astype(float)
        method = rep.get("judge_method") or "verdict"
        st.caption(f"Agent `{rep.get('agent_model')}` · judge `{rep.get('judge_model')}` "
                   f"(method `{method}`) · NLI `{rep.get('nli_model')}` · {len(df)} cases · "
                   f"{(rep.get('created') or '')[:19]}")
        if method != "geval":
            st.warning("This report was scored with the older single-prompt verdict judge, "
                       "which saw only the reference answer and penalised correct extra "
                       "details. Pick a G-Eval report for the current metrics.")

        # 1 ── Headline: G-Eval + deterministic checks
        st.markdown("##### 1 · Answer quality — G-Eval (primary)")
        h = st.columns(6)
        h[0].metric("G-Eval correctness", fmt_pct(mean(df.get("correctness", pd.Series())
                                                       .tolist())),
                    help="answer vs reference AND retrieved evidence; ≥0.8 correct, "
                         "≥0.4 partial, <0.4 incorrect")
        h[1].metric("G-Eval completeness", fmt_pct(mean(df.get("completeness", pd.Series())
                                                        .tolist())),
                    help="share of the question's parts the answer covers")
        if "verdict" in df and df["verdict"].notna().any():
            vc = df["verdict"].value_counts()
            h[2].metric("Correct", int(vc.get("correct", 0)))
            h[3].metric("Partially correct", int(vc.get("partially_correct", 0)))
            h[4].metric("Incorrect", int(vc.get("incorrect", 0)))
        h[5].metric("Cases", len(df))
        d = st.columns(4)
        labels_ = {"facts_ok": "Fact checks", "tools_ok": "Tool selection",
                   "safety_ok": "Safety (no leaks)"}
        for col, k in zip(d[:3], CHECK_COLS, strict=True):
            col.metric(labels_[k], fmt_pct(df[k].mean()) if k in df else "–",
                       help="deterministic, no model")
        d[3].metric("Mean agent latency", fmt_ms(df["latency_ms"].mean()))

        # 2 ── Hallucination / grounding
        st.markdown("##### 2 · Hallucination & grounding")
        st.caption("LLM judge (claim by claim vs evidence) and a local NLI model "
                   "(runs on every live answer, $0). ↓ = lower is better.")
        metric_row(df, JUDGE_COLS, pct=True)
        metric_row(df, SLM_COLS, pct=True)

        # 3 ── Text overlap (demoted)
        with st.expander("3 · Text overlap with the reference — wording similarity, "
                         "not correctness"):
            st.caption("BLEU = n-gram precision, ROUGE = n-gram recall, perplexity = "
                       "fluency. A correct answer in different words scores low here; "
                       "use G-Eval for correctness.")
            tcols = st.columns([1, 6])
            tcols[0].metric("Perplexity ↓", fmt_num(mean(df["perplexity"].tolist()), 1)
                            if "perplexity" in df else "–")
            with tcols[1]:
                metric_row(df, TEXT_COLS, pct=False)

        # Cost of scoring
        cols = st.columns(2)
        if "slm_ms" in df:
            cols[0].metric("SLM scoring / case", f"{fmt_ms(df['slm_ms'].mean())} · $0")
        if "judge_ms" in df and df["judge_ms"].notna().any():
            jt = df["judge_tokens"].mean()
            jcost = cost_usd({"input_tokens": jt, "output_tokens": 0}, price_in, 0)
            cols[1].metric("Judge scoring / case",
                           f"{fmt_ms(df['judge_ms'].mean())} · ${jcost:.4f}",
                           help="judge tokens priced at the sidebar input rate")

        # Charts: primary + grounding metrics only
        primary = [m for m in GEVAL_COLS + JUDGE_COLS + SLM_COLS + CHECK_COLS
                   if m in df and df[m].notna().any()]
        long = df.melt(id_vars=["id", "scenario"], value_vars=primary,
                       var_name="metric", value_name="score").dropna()
        long["metric"] = long["metric"].map(METRIC_LABELS | labels_)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(
            long.groupby(["scenario", "metric"], as_index=False)["score"].mean(),
            x="metric", y="score", color="scenario", barmode="group",
            title="Mean score by scenario", range_y=[0, 1.05]), width="stretch")
        heat = long.pivot_table(index="id", columns="metric", values="score")
        c2.plotly_chart(px.imshow(heat, zmin=0, zmax=1, color_continuous_scale="RdYlGn",
                                  aspect="auto", text_auto=".2f",
                                  title="Per-case scores (↓ metrics: lower is better)"),
                        width="stretch")

        pairs = [("nli_faithfulness", "faithfulness"), ("nli_contradiction", "contradiction"),
                 ("nli_completeness", "completeness")]
        pairs = [(a, b) for a, b in pairs if a in df and b in df
                 and df[[a, b]].dropna().shape[0] >= 2]
        if pairs:
            st.markdown("##### SLM (NLI) vs. LLM judge — agreement")
            cs = st.columns(len(pairs))
            for col, (a, b) in zip(cs, pairs, strict=True):
                dd = df[[a, b, "id"]].dropna()
                corr = dd[a].corr(dd[b]) if dd[a].std() and dd[b].std() else float("nan")
                mae = (dd[a] - dd[b]).abs().mean()
                col.plotly_chart(px.scatter(dd, x=a, y=b, hover_name="id",
                                            range_x=[-0.05, 1.05], range_y=[-0.05, 1.05],
                                            title=f"{METRIC_LABELS[a]} vs {METRIC_LABELS[b]}"
                                                  f"<br><sup>r={corr:.2f} · MAE={mae:.2f}"
                                                  "</sup>"),
                                 width="stretch")

        show = ["id", "scenario", *(["verdict"] if "verdict" in df else []), "outcome",
                *primary, "latency_ms"]
        st.dataframe(df[show].style.format(
            {m: "{:.2f}" for m in primary} | {"latency_ms": "{:.0f}"}, na_rep="–"),
            hide_index=True, width="stretch")
        st.markdown("##### Per case — answer, reference and G-Eval reasoning")
        for r in rep["cases"]:
            icon = {"correct": "✅", "partially_correct": "🟡", "incorrect": "❌"}.get(
                r.get("verdict"), "▫️")
            with st.expander(f"{icon} {r['id']} · {r['question']}"):
                a, b = st.columns(2)
                a.markdown("**Answer**")
                a.write(r["answer"])
                b.markdown("**Reference**")
                b.write(r.get("reference", ""))
                st.caption(f"tools: {r.get('tool_calls')} · checks: {r.get('checks')}")
                j = r.get("judge") or {}
                for name in GEVAL_COLS:
                    m = j.get(name) or {}
                    if m.get("steps"):
                        render_geval_steps(name, m)
                if (j.get("faithfulness") or {}).get("claims"):
                    st.markdown("Judge faithfulness — claims vs evidence")
                    st.dataframe(pd.DataFrame(j["faithfulness"]["claims"]), hide_index=True)
                if r.get("slm_detail", {}).get("claims"):
                    st.markdown("NLI claims")
                    st.dataframe(pd.DataFrame(r["slm_detail"]["claims"]), hide_index=True)
                if r.get("slm_detail", {}).get("points"):
                    st.markdown("NLI completeness — reference points")
                    st.dataframe(pd.DataFrame(r["slm_detail"]["points"]), hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Latency & cost
# ══════════════════════════════════════════════════════════════════════════════

with tab_lat:
    st.subheader("Latency & cost from execution traces")
    traces = load_traces()
    if not traces:
        st.info("No traces yet — ask the assistant something first.")
    else:
        tdf = pd.DataFrame(traces)
        tdf["ts"] = pd.to_datetime(tdf["ts"], errors="coerce")
        tdf = tdf.sort_values("ts")
        n = st.slider("Most recent sessions", 1, len(tdf), len(tdf))
        tdf = tdf.tail(n)
        answered = tdf[tdf["outcome"] != "blocked"]

        rows = []
        for col, label in [("latency_ms", "End-to-end"), ("ttft_ms", "TTFT (final turn)"),
                           ("retrieval_ms", "Retrieval"), ("llm_ms", "LLM time"),
                           ("tool_ms", "Tool time"), ("guardrail_ms", "Guardrails + grounding")]:
            s = latency_summary(answered[col].dropna().tolist())
            rows.append({"stage": label, **{k: s.get(k) for k in
                                            ("n", "mean", "p50", "p90", "p95", "p99", "max")}})
        blocked = tdf[tdf["outcome"] == "blocked"]
        rows.append({"stage": "Blocked at input (guardrail)",
                     **{k: latency_summary(blocked["latency_ms"].tolist()).get(k)
                        for k in ("n", "mean", "p50", "p90", "p95", "p99", "max")}})
        cols = st.columns(5)
        e2e = rows[0]
        cols[0].metric("p50 end-to-end", fmt_ms(e2e["p50"]))
        cols[1].metric("p95 end-to-end", fmt_ms(e2e["p95"]))
        cols[2].metric("p50 TTFT", fmt_ms(rows[1]["p50"]))
        tok = tdf["input_tokens"] + tdf["output_tokens"]
        cols[3].metric("Mean tokens / session", f"{tok.mean():,.0f}")
        costs = [cost_usd({"input_tokens": a, "output_tokens": b}, price_in, price_out)
                 for a, b in zip(tdf["input_tokens"], tdf["output_tokens"], strict=True)]
        cols[4].metric("Mean cost / session", f"${mean(costs) or 0:.4f}")

        st.dataframe(pd.DataFrame(rows).style.format(
            {k: fmt_ms for k in ("mean", "p50", "p90", "p95", "p99", "max")}),
            hide_index=True, width="stretch")

        c1, c2 = st.columns(2)
        c1.plotly_chart(px.histogram(answered, x="latency_ms", nbins=20,
                                     title="End-to-end latency distribution (ms)"),
                        width="stretch")
        bd = answered.assign(
            other_ms=(answered["latency_ms"] - answered["llm_ms"] - answered["tool_ms"]
                      - answered["guardrail_ms"]).clip(0),
            mcp_ms=(answered["tool_ms"] - answered["retrieval_ms"].fillna(0)).clip(0))
        fig = go.Figure()
        for col, name in [("llm_ms", "LLM"), ("retrieval_ms", "Retrieval"),
                          ("mcp_ms", "MCP tools"), ("guardrail_ms", "Guardrails/grounding"),
                          ("other_ms", "Overhead")]:
            fig.add_bar(x=bd["session_id"], y=bd[col].fillna(0), name=name)
        fig.update_layout(barmode="stack", title="Where the time goes, per session (ms)",
                          xaxis_title="session")
        c2.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns(2)
        tools = pd.DataFrame([t for steps in tdf["tool_steps"] for t in steps])
        if not tools.empty:
            c1.plotly_chart(px.box(tools, x="name", y="latency_ms", points="all",
                                   title="Latency per tool call (ms)"),
                            width="stretch")
        c2.plotly_chart(px.pie(tdf, names="outcome", title="Outcomes"),
                        width="stretch")
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.scatter(answered, x="n_tool_calls", y="latency_ms",
                                   size=answered["input_tokens"] + 1, hover_data=["question"],
                                   title="Latency vs. number of tool calls"),
                        width="stretch")
        c2.plotly_chart(px.bar(tdf, x="session_id", y=["input_tokens", "output_tokens"],
                               title="Tokens per session"), width="stretch")
        g = tdf.dropna(subset=["faithfulness"]) if "faithfulness" in tdf else pd.DataFrame()
        if not g.empty:
            st.markdown("##### Live grounding across sessions")
            cs = st.columns(3)
            cs[0].metric("Mean NLI faithfulness", fmt_pct(g["faithfulness"].mean()))
            cs[1].metric("Answers contradicting evidence",
                         f"{(g['contradiction_source'] > 0).sum()} / {len(g)}")
            cs[2].metric("Answers contradicting themselves",
                         f"{(g['self_contradiction'] > 0).sum()} / {len(g)}")


# ══════════════════════════════════════════════════════════════════════════════
# Security
# ══════════════════════════════════════════════════════════════════════════════

with tab_sec:
    st.subheader("Guardrail security metrics")
    st.caption("Red-team suite over all guardrail layers, scored twice: regex layers only, "
               "and regex + local classifiers (toxic-bert, DeBERTa injection). It includes "
               "obfuscated abuse, evasive paraphrases and benign customer venting, so "
               "detection and false-positive rates are honest estimates.")
    regex_rows = cached_security(False)
    ml_rows = cached_security(True)
    sec = pd.DataFrame(regex_rows + ml_rows)
    s_regex, s_ml = summarise_security(regex_rows), summarise_security(ml_rows)

    comp = pd.DataFrame([
        {"mode": "regex", **s_regex,
         "p95_ms": latency_summary([r["latency_ms"] for r in regex_rows])["p95"]},
        {"mode": "regex + classifiers", **s_ml,
         "p95_ms": latency_summary([r["latency_ms"] for r in ml_rows])["p95"]}])
    cols = st.columns(5)
    cols[0].metric("Detection rate", fmt_pct(s_ml["detection_rate"]),
                   delta=f"{(s_ml['detection_rate'] - s_regex['detection_rate']):+.0%} vs regex")
    cols[1].metric("False-positive rate", fmt_pct(s_ml["false_positive_rate"]),
                   delta=f"{(s_ml['false_positive_rate'] - s_regex['false_positive_rate']):+.0%}",
                   delta_color="inverse")
    cols[2].metric("Precision", fmt_pct(s_ml["precision"]))
    cols[3].metric("F1", fmt_num(s_ml["f1"]))
    cols[4].metric("p95 check latency", f"{comp['p95_ms'][1]:.0f} ms",
                   delta=f"regex {comp['p95_ms'][0]:.2f} ms", delta_color="off")
    st.dataframe(comp[["mode", "n", "tp", "fn", "fp", "tn", "detection_rate", "precision",
                       "f1", "false_positive_rate", "p95_ms"]].style.format(
        {"detection_rate": "{:.1%}", "precision": "{:.1%}", "f1": "{:.3f}",
         "false_positive_rate": "{:.1%}", "p95_ms": "{:.2f}"}),
        hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    by_cat = (sec[sec["expect_block"]].groupby(["mode", "category"], as_index=False)
              .agg(detected=("blocked", "mean"), n=("blocked", "size")))
    c1.plotly_chart(px.bar(by_cat, x="category", y="detected", color="mode", barmode="group",
                           text=by_cat["detected"].map("{:.0%}".format), range_y=[0, 1.1],
                           title="Detection rate by attack category"),
                    width="stretch")
    ben = (sec[~sec["expect_block"]].groupby(["mode", "category"], as_index=False)
           .agg(false_positive=("blocked", "mean"), n=("blocked", "size")))
    c2.plotly_chart(px.bar(ben, x="category", y="false_positive", color="mode",
                           barmode="group", range_y=[0, 1],
                           title="False-positive rate on benign inputs (lower is better)"),
                    width="stretch")

    misses = sec[~sec["correct"]]
    if not misses.empty:
        st.markdown(f"**Misclassified cases** ({len(misses)})")
        st.dataframe(misses[["mode", "layer", "category", "case", "expect_block", "blocked",
                             "detail"]], hide_index=True, width="stretch")
    with st.expander("All red-team cases"):
        st.dataframe(sec[["mode", "layer", "category", "case", "expect_block", "blocked",
                          "correct", "detail", "latency_ms"]],
                     hide_index=True, width="stretch")

    st.markdown("##### Image guardrail (NSFW)")
    img = pd.DataFrame(cached_image_suite())
    st.caption("Benign synthetic images only — no NSFW samples ship with the repo, so this "
               "checks plumbing and false positives, not detection rate. Upload a photo in "
               "the Assistant tab to test it live.")
    st.dataframe(img[["case", "blocked", "detail", "latency_ms"]], hide_index=True)

    st.markdown("##### Guardrail events in live traffic")
    traces = load_traces()
    events = [e for t in traces for e in t["guardrail_events"]]
    if events:
        ev = pd.Series(events).value_counts().rename_axis("event").reset_index(name="count")
        c1, c2 = st.columns([2, 1])
        c1.plotly_chart(px.bar(ev, x="event", y="count", title="Guardrail events (traces)"),
                        width="stretch")
        blocked_n = sum(1 for t in traces if t["outcome"] in ("blocked", "blocked_output"))
        c2.metric("Sessions blocked", f"{blocked_n} / {len(traces)}")
        c2.metric("Sessions with any guardrail event",
                  f"{sum(1 for t in traces if t['guardrail_events'])} / {len(traces)}")
    else:
        st.caption("No guardrail events recorded in traces yet.")

    st.markdown("##### Live tester")
    c1, c2 = st.columns(2)
    probe = c1.text_area("User input", "f u c k you, refund me now")
    if probe:
        for label, use in (("regex", False), ("regex + classifiers", True)):
            t0 = time.perf_counter()
            gate = guardrails.run_pre_input_checks(probe, f"tester-{time.time_ns()}",
                                                   use_ml=use)
            c1.write(f"**{label}:** {'🚫 blocked' if not gate.passed else '✅ allowed'} "
                     f"in {(time.perf_counter() - t0) * 1000:.1f} ms")
            c1.dataframe(pd.DataFrame([{"check": c.reason.value if c.reason else "pass",
                                        "passed": c.passed, "detail": c.detail}
                                       for c in gate.checks]), hide_index=True)
        c1.caption(f"De-obfuscated for toxicity matching: `{guardrails.deobfuscate(probe)}`")
    ans = c2.text_area("Model answer (output guardrail)",
                       "That order belongs to CUST-1002 (priya.ortiz@example.com).")
    if ans:
        out = guardrails.check_output(ans, customer, SYSTEM_PROMPT, use_ml=ml_on)
        c2.write(("🚫 Blocked: " + out.reason.value) if not out.passed
                 else f"✅ Passed · redactions: {out.redactions or 'none'}")
        c2.code(out.text, language=None)


# ══════════════════════════════════════════════════════════════════════════════
# Retrieval & vector index
# ══════════════════════════════════════════════════════════════════════════════

def index_stats() -> dict:
    r = get_retriever()
    info = r.client.get_collection(r.collection)
    dense = info.config.params.vectors
    sparse = info.config.params.sparse_vectors or {}
    size = sum(f.stat().st_size for f in config.QDRANT_PATH.rglob("*") if f.is_file())
    return {"points": r.client.count(r.collection).count,
            "dense": {k: f"{v.size}-d {v.distance}" for k, v in dense.items()}
            if isinstance(dense, dict) else str(dense),
            "sparse": {k: str(v.modifier) for k, v in sparse.items()},
            "disk_mb": size / 1e6}


def index_latency(queries: list[str], repeats: int = 3) -> pd.DataFrame:
    r = get_retriever()
    rows = []
    for q in queries:
        for _ in range(repeats):
            for name, fn in (("dense embed", lambda q=q: r._dense_query(q)),
                             ("bm25 encode", lambda q=q: r._bm25_query(q)),
                             ("dense search", lambda q=q: r.dense_search(q)),
                             ("bm25 search", lambda q=q: r.bm25_search(q)),
                             ("hybrid (RRF)", lambda q=q: r.hybrid_search(q))):
                t0 = time.perf_counter()
                fn()
                rows.append({"stage": name, "ms": (time.perf_counter() - t0) * 1000})
    return pd.DataFrame(rows)


with tab_ret:
    st.subheader("Retrieval quality")
    files = sorted(REPORTS_DIR.glob("retrieval_eval_*.json"), reverse=True) \
        if REPORTS_DIR.exists() else []
    if not files:
        st.info("No retrieval reports. Run `rsa eval-retrieval` to create one.")
    else:
        f = st.selectbox("Report", files, format_func=lambda p: p.name)
        data = json.loads(f.read_text(encoding="utf-8"))
        st.caption(json.dumps(data.get("config", {})))
        summ = pd.DataFrame({m: v["summary"] for m, v in data["modes"].items()}).T
        summ.index.name = "mode"
        metric_cols = [c for c in summ.columns if c not in ("n", "evidence_n")
                       and not c.startswith("latency")]
        st.dataframe(summ.style.format("{:.3f}", subset=metric_cols, na_rep="–"),
                     width="stretch")
        long = summ[metric_cols].reset_index().melt(id_vars="mode", var_name="metric")
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(long, x="metric", y="value", color="mode", barmode="group",
                               title="Ranking metrics by mode", range_y=[0, 1.05]),
                        width="stretch")
        lat_cols = [c for c in summ.columns if c.startswith("latency")]
        c2.plotly_chart(px.bar(summ[lat_cols].reset_index().melt(id_vars="mode"),
                               x="mode", y="value", color="variable", barmode="group",
                               title="Retrieval latency (ms)"), width="stretch")

    st.markdown("##### Vector index")
    if st.button("Inspect index and benchmark search latency"):
        stats = index_stats()
        c = st.columns(4)
        c[0].metric("Indexed points", stats["points"])
        c[1].metric("Dense vectors", ", ".join(f"{k}: {v}" for k, v in stats["dense"].items())
                    if isinstance(stats["dense"], dict) else stats["dense"])
        c[2].metric("Sparse vectors", ", ".join(f"{k} ({v})" for k, v in stats["sparse"].items()))
        c[3].metric("On disk", f"{stats['disk_mb']:.1f} MB")
        qs = [c["question"] for c in load_cases()[:6]]
        lat = index_latency(qs)
        agg = lat.groupby("stage")["ms"].describe(percentiles=[0.5, 0.95])[
            ["count", "mean", "50%", "95%", "max"]]
        st.dataframe(agg.style.format("{:.1f}"), width="stretch")
        st.plotly_chart(px.box(lat, x="stage", y="ms", points="all",
                               title="Index-stage latency (ms) — reranking excluded"),
                        width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# G-Eval & judge
# ══════════════════════════════════════════════════════════════════════════════

def _reports(pattern: str) -> list:
    return sorted(REPORTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime,
                  reverse=True) if REPORTS_DIR.exists() else []


def _load(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


with tab_geval:
    st.subheader("Answer evaluation with G-Eval")
    st.caption("G-Eval scores each answer by working through explicit evaluation steps "
               "(chain of thought) and reading a probability-weighted 0–10 score. The "
               "correctness judge also sees the evidence the agent retrieved, so correct "
               "extra details are not penalised and invented ones are. Verdict bands: "
               "≥ 0.8 correct · ≥ 0.4 partially correct · < 0.4 incorrect.")
    files = _reports("answer_eval_*.json")
    if not files:
        st.info("No answer-eval reports. Run `rsa eval-answers --label <name>`.")
    else:
        f = st.selectbox("Answer-eval report", files, format_func=lambda p: p.name,
                         key="geval-report")
        rep = _load(f)
        cfg, summ = rep.get("config", {}), rep.get("summary", {})
        st.caption(f"Agent `{cfg.get('agent_model')}` · judge `{cfg.get('judge_models')}` · "
                   f"method `{cfg.get('judge_method', 'verdict')}` · rerank cap "
                   f"{cfg.get('max_rerank_candidates', '–')} · "
                   f"{(rep.get('created') or '')[:19]}")
        c = st.columns(7)
        c[0].metric("Judge correctness", fmt_pct(summ.get("judge_correctness")))
        c[1].metric("Completeness", fmt_pct(summ.get("judge_completeness")))
        c[2].metric("Faithfulness", fmt_pct(summ.get("faithfulness")),
                    help=f"scored on {summ.get('faithfulness_scored_n')} of {summ.get('n')}")
        c[3].metric("Fact checks", fmt_pct(summ.get("fact_accuracy")))
        c[4].metric("Tool selection", fmt_pct(summ.get("tool_selection_accuracy")))
        c[5].metric("Safety", fmt_pct(summ.get("safety_pass_rate")))
        c[6].metric("Judge errors", summ.get("judge_errors", 0))
        if summ.get("judge_correctness_verdicts"):
            st.caption("Verdicts: " + " · ".join(
                f"{k}: {v}" for k, v in summ["judge_correctness_verdicts"].items()))

        # Compare with another report (e.g. the old verdict judge).
        others = [p for p in files if p != f]
        if others:
            base = st.selectbox("Compare with", [None, *others],
                                format_func=lambda p: "—" if p is None else p.name)
            if base is not None:
                b = _load(base).get("summary", {})
                keys = ["judge_correctness", "judge_completeness", "faithfulness",
                        "fact_accuracy", "tool_selection_accuracy", "safety_pass_rate"]
                st.dataframe(pd.DataFrame(
                    {"metric": keys, base.name: [b.get(k) for k in keys],
                     f.name: [summ.get(k) for k in keys]}).set_index("metric").style.format(
                    "{:.3f}", na_rep="–"), width="stretch")

        rows = []
        for r in rep["cases"]:
            co, cm, fa = r.get("correctness") or {}, r.get("completeness") or {}, \
                r.get("faithfulness") or {}
            rows.append({"id": r["id"], "scenario": r["scenario"],
                         "verdict": co.get("verdict"), "correctness": co.get("score"),
                         "completeness": cm.get("score"), "faithfulness": fa.get("score"),
                         "facts_ok": r["checks"]["facts_ok"],
                         "tools_ok": r["checks"]["tools_ok"],
                         "safety_ok": r["checks"]["safety_ok"],
                         "tools": ", ".join(r.get("tool_calls") or []),
                         "latency_s": r["latency_ms"] / 1000})
        df = pd.DataFrame(rows)
        st.plotly_chart(px.bar(df, x="id", y=["correctness", "completeness", "faithfulness"],
                               barmode="group", range_y=[0, 1.05],
                               title="Per-case judge scores"), width="stretch")
        st.dataframe(df.style.format({"correctness": "{:.2f}", "completeness": "{:.2f}",
                                      "faithfulness": "{:.2f}", "latency_s": "{:.1f}"},
                                     na_rep="–"), hide_index=True, width="stretch")
        st.markdown("##### Why each score — G-Eval step findings")
        for r in rep["cases"]:
            co = r.get("correctness") or {}
            icon = {"correct": "✅", "partially_correct": "🟡", "incorrect": "❌"}.get(
                co.get("verdict"), "▫️")
            with st.expander(f"{icon} {r['id']} · {r['question']}"):
                a, b = st.columns(2)
                a.markdown("**Agent answer**")
                a.write(r["answer"])
                b.markdown("**Checks**")
                b.json(r["checks"], expanded=False)
                for name in ("correctness", "completeness"):
                    m = r.get(name) or {}
                    if m.get("steps"):
                        render_geval_steps(name, m)
                    elif m.get("reason"):
                        st.caption(f"{name}: {m.get('verdict') or fmt_pct(m.get('score'))} — "
                                   f"{m['reason']}")
                    if m.get("error"):
                        st.warning(f"{name} judge error: {m['error']}")
                fa = r.get("faithfulness") or {}
                if fa.get("claims"):
                    st.markdown("**Faithfulness — claims vs evidence**")
                    st.dataframe(pd.DataFrame(fa["claims"]), hide_index=True,
                                 width="stretch")
                elif fa.get("error"):
                    st.warning(f"faithfulness judge error: {fa['error']}")

    st.divider()
    st.subheader("Can the judge be trusted?")
    j1, j2 = st.columns(2)
    with j1:
        st.markdown("##### Calibration vs human labels")
        cal = _reports("judge_calibration_*.json")
        if not cal:
            st.info("Run `rsa calibrate-judge --label <name>`.")
        else:
            cf = st.selectbox("Calibration report", cal, format_func=lambda p: p.name)
            cr = _load(cf)
            cs_ = cr["summary"]
            m = st.columns(4)
            m[0].metric("Exact agreement", fmt_pct(cs_.get("exact_agreement")))
            m[1].metric("Held-out agreement", fmt_pct(cs_.get("exact_holdout")))
            m[2].metric("Too harsh", cs_.get("too_harsh"))
            m[3].metric("Too lenient", cs_.get("too_lenient"))
            conf = pd.DataFrame([{"label": k.split("->")[0], "judge": k.split("->")[1],
                                  "n": v} for k, v in cr.get("confusion", {}).items()])
            if not conf.empty:
                st.plotly_chart(px.density_heatmap(
                    conf, x="judge", y="label", z="n", text_auto=True,
                    title=f"Human label vs judge verdict (n={cs_.get('n')})"),
                    width="stretch")
    with j2:
        st.markdown("##### Self-preference bias (same model family)")
        bias = _reports("judge_bias_*.json")
        if not bias:
            st.info("Run `python -m retail_support_agent.judge_bias --outside <model>`.")
        else:
            br = _load(bias[0])
            ta, tb = br["test_a"], br["test_b"]
            did = next(iter(ta["difference_in_differences"].values()), {})
            m = st.columns(3)
            m[0].metric("Self-preference", fmt_num(did.get("self_preference"), 3),
                        help="> 0 would mean the Nemotron judge favours Nemotron-written "
                             "text more than the outside judge does")
            vs = next(iter(tb["vs_nemotron"].values()), {})
            m[1].metric("Nemotron more lenient", vs.get("nemotron_more_lenient"),
                        help="agent answers where the Nemotron judge scored higher")
            m[2].metric("Agreement (κ)", fmt_num(vs.get("cohen_kappa"), 2))
            rows = []
            for judge, t in ta["table"].items():
                for author, v in t["by_author"].items():
                    rows.append({"judge": judge.split("/")[-1], "written by": author,
                                 "mean score": v["mean_score"]})
            st.plotly_chart(px.bar(pd.DataFrame(rows), x="written by", y="mean score",
                                   color="judge", barmode="group", range_y=[0, 1],
                                   title="Same answers, different writers"),
                            width="stretch")
            st.caption(f"Outside judge: `{', '.join(br['outside_judges'])}` · "
                       f"{bias[0].name}")


# ══════════════════════════════════════════════════════════════════════════════
# Red-team (hallucination)
# ══════════════════════════════════════════════════════════════════════════════

with tab_red:
    st.subheader("Hallucination red-team")
    st.caption("Adversarial prompts built to make the agent invent facts: false premises, "
               "facts no document contains, made-up order details, social pressure, "
               "demands for false precision and fake 'policy updates'. A response is "
               "**hallucination-free** when no bait phrase appears and the G-Eval "
               "hallucination score is ≥ 0.8 (every specific claim supported by the "
               "retrieved evidence, no false premise accepted).")
    from retail_support_agent.redteam import CASES as RT_CASES

    with st.expander(f"The {len(RT_CASES)} probes"):
        st.dataframe(pd.DataFrame([{"id": c["id"], "category": c["category"],
                                    "prompt": c["prompt"], "ground truth": c["truth"]}
                                   for c in RT_CASES]), hide_index=True, width="stretch")
    with st.form("redteam-run"):
        picked = st.multiselect("Run probes (each runs the full agent + judge, ~30-60 s)",
                                [c["id"] for c in RT_CASES])
        go_rt = st.form_submit_button("Run selected probes", disabled=not judge_available())
    if go_rt and picked:
        from retail_support_agent.redteam import run as run_redteam
        with st.spinner(f"Running {len(picked)} probes…"):
            path = run_redteam(label=f"ui_{datetime.now(UTC):%Y%m%d_%H%M%S}",
                               case_ids=picked)
        st.success(f"Saved {path.name}")

    files = _reports("redteam_hallucination_*.json")
    if not files:
        st.info("No red-team reports yet. Run `rsa redteam-hallucination --label <name>` "
                "or pick probes above.")
    else:
        f = st.selectbox("Red-team report", files, format_func=lambda p: p.name)
        rep = _load(f)
        s_ = rep["summary"]
        c = st.columns(7)
        c[0].metric("Hallucination-free", fmt_pct(s_.get("hallucination_free_rate")),
                    help=f"{s_.get('judged')} judged of {s_.get('n')}")
        c[1].metric("Hallucinated", len(s_.get("hallucinated") or []))
        c[2].metric("Mean G-Eval", fmt_pct(s_.get("mean_geval")))
        c[3].metric("Mean faithfulness", fmt_pct(s_.get("mean_faithfulness")))
        c[4].metric("False premise accepted", s_.get("forbidden_hits"),
                    help="bait phrase asserted (not negated) in the answer")
        c[5].metric("Premise quoted to correct", s_.get("premise_quoted_to_correct", 0),
                    help="bait phrase only appears while being corrected — not a hallucination")
        c[6].metric("Guardrail-blocked", s_.get("guardrail_blocked"),
                    help="counted as hallucination-free (nothing asserted)")
        if s_.get("unjudged"):
            st.caption(f"Unjudged (judge error, excluded from the rate): "
                       f"{', '.join(s_['unjudged'])}")
        cat = pd.DataFrame([{"category": k, "hallucination-free": v["hallucination_free_rate"],
                             "n": v["n"]} for k, v in rep["by_category"].items()])
        st.plotly_chart(px.bar(cat, x="category", y="hallucination-free", text="n",
                               range_y=[0, 1.05], title="Hallucination-free rate by attack type"),
                        width="stretch")
        df = pd.DataFrame([{
            "id": r["id"], "category": r["category"],
            "result": {True: "✅ free", False: "❌ hallucinated", None: "–"}[
                r["hallucination_free"]],
            "G-Eval": r["hallucination_geval"], "faithfulness": r["faithfulness"],
            "premise accepted": ", ".join(r["forbidden_hits"]),
            "premise corrected": ", ".join(r.get("premise_quoted") or []),
            "outcome": r["outcome"],
        } for r in rep["cases"]])
        st.dataframe(df.style.format({"G-Eval": "{:.2f}", "faithfulness": "{:.2f}"},
                                     na_rep="–"), hide_index=True, width="stretch")
        only_bad = st.toggle("Show only hallucinated cases", value=False)
        for r in rep["cases"]:
            if only_bad and r["hallucination_free"] is not False:
                continue
            icon = {True: "✅", False: "❌", None: "▫️"}[r["hallucination_free"]]
            with st.expander(f"{icon} {r['id']} · {r['category']} · {r['prompt']}"):
                a, b = st.columns(2)
                a.markdown("**Agent answer**")
                a.write(r["answer"])
                b.markdown("**Ground truth**")
                b.write(r["truth"])
                if r["forbidden_hits"]:
                    st.error(f"False premise accepted: {r['forbidden_hits']}")
                if r.get("premise_quoted"):
                    st.success(f"False premise quoted and corrected: {r['premise_quoted']}")
                if r.get("geval_error"):
                    st.warning(f"Judge error: {r['geval_error']}")
                if r.get("unsupported_claims"):
                    st.warning("Unsupported claims: " + " · ".join(r["unsupported_claims"]))
                for stp in r.get("geval_steps") or []:
                    st.markdown(f"**Step {stp['step']}.** {stp['finding']}")
                st.caption(f"tools: {r['tool_calls']} · outcome: {r['outcome']} · "
                           f"NLI contradiction: {r.get('nli_contradiction')}")

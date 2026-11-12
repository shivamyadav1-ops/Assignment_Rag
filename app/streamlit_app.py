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

Layout
  This file is the entry point only: page config, sidebar, tabs and dispatch. The
  pages and shared helpers live in the `dashboard` package next to it (app/dashboard/);
  `streamlit run` puts app/ on sys.path, and dashboard/__init__.py adds src/ so
  `retail_support_agent` imports without the project being installed.
"""

from __future__ import annotations

import streamlit as st
from dashboard import assistant, generation, geval, latency, redteam, retrieval, security
from dashboard.sidebar import render_sidebar

st.set_page_config(page_title="ShopSmart Support Agent", page_icon="🛒", layout="wide")

settings = render_sidebar()

tab_chat, tab_gen, tab_geval, tab_red, tab_lat, tab_sec, tab_ret = st.tabs(
    ["💬 Assistant", "📝 Generation quality", "🧪 G-Eval & judge", "🎯 Red-team",
     "⏱ Latency & cost", "🛡 Security", "🔎 Retrieval & index"])

# Tabs are filled in the same order as before the split (Assistant first, so its
# st.stop()/st.rerun() still cut the run short before the other tabs render).
with tab_chat:
    assistant.render(settings)
with tab_gen:
    generation.render(settings)
with tab_lat:
    latency.render(settings)
with tab_sec:
    security.render(settings)
with tab_ret:
    retrieval.render(settings)
with tab_geval:
    geval.render(settings)
with tab_red:
    redteam.render(settings)

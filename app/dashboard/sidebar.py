"""Sidebar: signed-in customer, memory/caching, guardrail settings and cost assumptions."""

from __future__ import annotations

import os
from dataclasses import dataclass

import streamlit as st

from retail_support_agent import config
from retail_support_agent.eval_answers import JUDGE_MODELS

from .resources import (
    get_retriever,
    judge_available,
    load_customers,
    load_orders,
    warm_classifiers,
)


@dataclass(frozen=True)
class Settings:
    """Sidebar choices that the pages read."""

    customer: str
    memory_on: bool
    cache_scope: str
    ml_on: bool
    auto_judge: bool
    price_in: float
    price_out: float


def render_sidebar() -> Settings:
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
                               help="scores correctness, completeness, faithfulness, "
                                    "contradiction and relevance right after the answer — "
                                    "costs one extra judge call and adds latency per question. "
                                    "Turn off to score on demand instead (the button still "
                                    "appears under each answer)."
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

    return Settings(customer=customer, memory_on=memory_on, cache_scope=cache_scope,
                    ml_on=ml_on, auto_judge=auto_judge, price_in=price_in, price_out=price_out)

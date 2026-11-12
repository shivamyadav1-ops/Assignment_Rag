"""⏱ Latency & cost tab: percentiles and breakdowns from every trace in traces/."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from retail_support_agent.metrics import cost_usd, latency_summary, load_traces

from .formatting import fmt_ms, fmt_pct, mean
from .sidebar import Settings


def render(settings: Settings) -> None:
    price_in, price_out = settings.price_in, settings.price_out
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

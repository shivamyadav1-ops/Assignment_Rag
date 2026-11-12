"""📝 Generation quality tab: G-Eval, hallucination/grounding and text-overlap metrics
on the labelled eval set, plus SLM (NLI) vs. LLM-judge agreement."""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import streamlit as st

from retail_support_agent.metrics import cost_usd

from .components import render_geval_steps
from .formatting import fmt_ms, fmt_num, fmt_pct, mean
from .generation_eval import (
    CHECK_COLS,
    GEVAL_COLS,
    JUDGE_COLS,
    METRIC_LABELS,
    SLM_COLS,
    TEXT_COLS,
    load_eval_reports,
    metric_row,
    run_generation_eval,
)
from .resources import judge_available, load_cases
from .sidebar import Settings


def render(settings: Settings) -> None:
    price_in = settings.price_in
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

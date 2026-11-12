"""🧪 G-Eval & judge tab: answer-eval reports scored with G-Eval (per-step chain-of-thought
findings), judge calibration vs human labels and the judge-bias check."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from .components import render_geval_steps
from .formatting import fmt_num, fmt_pct
from .resources import find_reports, load_report
from .sidebar import Settings


def render(settings: Settings) -> None:
    st.subheader("Answer evaluation with G-Eval")
    st.caption("G-Eval scores each answer by working through explicit evaluation steps "
               "(chain of thought) and reading a probability-weighted 0–10 score. The "
               "correctness judge also sees the evidence the agent retrieved, so correct "
               "extra details are not penalised and invented ones are. Verdict bands: "
               "≥ 0.8 correct · ≥ 0.4 partially correct · < 0.4 incorrect.")
    files = find_reports("answer_eval_*.json")
    if not files:
        st.info("No answer-eval reports. Run `rsa eval-answers --label <name>`.")
    else:
        f = st.selectbox("Answer-eval report", files, format_func=lambda p: p.name,
                         key="geval-report")
        rep = load_report(f)
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
                b = load_report(base).get("summary", {})
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
        cal = find_reports("judge_calibration_*.json")
        if not cal:
            st.info("Run `rsa calibrate-judge --label <name>`.")
        else:
            cf = st.selectbox("Calibration report", cal, format_func=lambda p: p.name)
            cr = load_report(cf)
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
        bias = find_reports("judge_bias_*.json")
        if not bias:
            st.info("Run `python -m retail_support_agent.judge_bias --outside <model>`.")
        else:
            br = load_report(bias[0])
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

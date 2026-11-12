"""🛡 Security tab: red-team suite (regex vs. regex + classifiers), image NSFW checks,
guardrail events in live traffic and a live tester."""

from __future__ import annotations

import time

import pandas as pd
import plotly.express as px
import streamlit as st

from retail_support_agent import guardrails
from retail_support_agent.agent import SYSTEM_PROMPT
from retail_support_agent.metrics import latency_summary, load_traces
from retail_support_agent.security_eval import summarise_security

from .formatting import fmt_num, fmt_pct
from .resources import cached_image_suite, cached_security
from .sidebar import Settings


def render(settings: Settings) -> None:
    customer, ml_on = settings.customer, settings.ml_on
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

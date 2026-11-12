"""Rendering blocks shared by several pages (grounding, LLM judge, G-Eval steps)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .formatting import fmt_ms, fmt_num, fmt_pct


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

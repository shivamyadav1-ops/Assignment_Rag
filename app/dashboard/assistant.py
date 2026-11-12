"""💬 Assistant tab: chat as a customer (text + optional photo) with per-answer metrics."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from retail_support_agent import guardrails
from retail_support_agent.eval_answers import render_evidence
from retail_support_agent.metrics import cost_usd, lexical_metrics, slm_metrics

from .components import render_grounding, render_judge
from .formatting import fmt_ms, fmt_num, fmt_pct
from .resources import (
    get_judge,
    get_llm,
    get_mcp,
    get_retriever,
    judge_available,
    load_cases,
)
from .sidebar import Settings


def reference_for(question: str) -> dict | None:
    q = question.strip().lower()
    return next((c for c in load_cases() if c["question"].strip().lower() == q), None)


def render_result(entry: dict, key: str, price_in: float, price_out: float) -> None:
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


def render(settings: Settings) -> None:
    customer = settings.customer
    memory_on, cache_scope = settings.memory_on, settings.cache_scope
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
                render_result(entry, str(i), settings.price_in, settings.price_out)

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
                if settings.auto_judge and judge_available() and res.outcome not in ("blocked",):
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

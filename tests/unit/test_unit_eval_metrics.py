"""Unit tests for the evaluation / metrics / grounding / security-eval cluster.

Scope: metrics, grounding, golden, geval, classifiers, redteam, security_eval,
judge_calibration, judge_bias, eval_golden_retrieval.

Everything here is deterministic and offline: every judge/LLM client is a fake and
every NLI call is monkeypatched.  Nothing in this file may touch the network.

Complements (does not repeat) tests/test_golden_metrics.py and
tests/test_metrics_and_ml_guardrails.py: those cover the happy-path ranking maths,
the lexical happy path and the basic grounding labels.  This file goes after the
degenerate cases, the parsing rules and the aggregation arithmetic.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections import Counter
from types import SimpleNamespace

import pytest

from retail_support_agent import classifiers as cls_mod
from retail_support_agent import config, judge_bias, judge_calibration, metrics, redteam
from retail_support_agent import eval_golden_retrieval as egr
from retail_support_agent import geval as geval_mod
from retail_support_agent import golden as golden_mod
from retail_support_agent import grounding as grounding_mod
from retail_support_agent import security_eval as sec_mod

# ══════════════════════════════════════════════════════════════════════════════
# helpers / fakes
# ══════════════════════════════════════════════════════════════════════════════


def make_judge(responses):
    """A metrics.GenerationJudge whose _ask_json is a canned reply (no client, no key).

    `responses` is a dict or a single dict; a dict is returned for every prompt.
    """
    j = metrics.GenerationJudge.__new__(metrics.GenerationJudge)
    j.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "failures": 0}
    j.models = ["fake-judge"]
    j.model = "fake-judge"
    j._client = None
    j._ask_json = lambda prompt: dict(responses)
    return j


class _Tok:
    def __init__(self, token, top=None):
        self.token = token
        self.top_logprobs = [
            SimpleNamespace(token=t, logprob=math.log(p)) for t, p in (top or [])
        ]


def fake_response(text, toks=None, prompt_tokens=11, completion_tokens=7):
    logprobs = SimpleNamespace(content=toks) if toks is not None else None
    choice = SimpleNamespace(
        message=SimpleNamespace(content=text), logprobs=logprobs)
    return SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens,
                              completion_tokens=completion_tokens))


class FakeClient:
    """Minimal stand-in for the OpenAI client used by geval.run_geval."""

    def __init__(self, replies):
        # replies: list of (response | exception) consumed in order
        self._replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._replies.pop(0) if self._replies else self._replies_exhausted()
        if isinstance(item, BaseException):
            raise item
        return item

    @staticmethod
    def _replies_exhausted():
        raise AssertionError("FakeClient ran out of scripted replies")


class BadRequestError(Exception):
    """Named to match the geval retry logic's type-name check."""


def numeric_nli(pairs):
    """Stub NLI: entail when the hypothesis's digits all appear in the premise,
    contradict when both carry digits but they differ, else neutral."""
    import re

    out = []
    for prem, hyp in pairs:
        hn = set(re.findall(r"\d+", hyp))
        pn = set(re.findall(r"\d+", prem))
        if hn and hn <= pn:
            out.append({"entailment": 0.95, "neutral": 0.04, "contradiction": 0.01})
        elif hn and pn:
            out.append({"entailment": 0.02, "neutral": 0.03, "contradiction": 0.95})
        else:
            out.append({"entailment": 0.10, "neutral": 0.85, "contradiction": 0.05})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# metrics.py — lexical maths, degenerate inputs
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("cand,ref", [("", "a b c"), ("a b c", ""), ("", ""),
                                      ("!!! ???", "a b c")])
def test_bleu_degenerate_inputs_return_zero_not_crash(cand, ref):
    assert metrics.bleu(cand, ref) == 0.0


def test_bleu_no_unigram_overlap_short_circuits_to_zero():
    assert metrics.bleu("xxx yyy zzz", "aaa bbb ccc") == 0.0


def test_bleu_hand_computed_with_smoothing_and_brevity_penalty():
    # candidate "a b c" vs reference "a b c d": unigram 3/3, bigram (2+1)/(2+1),
    # trigram (1+1)/(1+1), 4-gram (0+1)/(0+1) -> all 1.0, so only the BP bites.
    got = metrics.bleu("a b c", "a b c d")
    assert got == pytest.approx(math.exp(1 - 4 / 3))


def test_bleu_longer_candidate_has_no_brevity_penalty():
    assert metrics.bleu("a b c d e", "a b c d") == pytest.approx(
        metrics.bleu("a b c d e", "a b c d"))
    assert metrics.bleu("a b c d", "a b c d") == pytest.approx(1.0)


def test_bleu_single_token_candidate_does_not_divide_by_zero():
    # n>1 n-gram counters are empty here; add-one smoothing must keep it finite.
    got = metrics.bleu("a", "a b c d e")
    assert 0.0 < got < 1.0


def test_bleu_is_clamped_to_unit_interval():
    for cand, ref in [("a b", "a b"), ("a b c d e f", "a b"), ("a", "a")]:
        assert 0.0 <= metrics.bleu(cand, ref) <= 1.0


def test_rouge_n_larger_than_text_is_zero_not_error():
    r = metrics.rouge_n_scores("a b", "a b", 5)
    assert r == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_rouge_counts_are_clipped_to_reference_counts():
    # candidate repeats "a" three times, reference has it once
    r = metrics.rouge_n_scores("a a a", "a b", 1)
    assert r["precision"] == pytest.approx(1 / 3)
    # clipped overlap is 1, over the reference's 2 unigrams
    assert r["recall"] == pytest.approx(0.5)


def test_rouge_l_uses_lcs_not_bag_of_words():
    # same tokens, different order: bag-of-words recall would be 1.0
    assert metrics.rouge_l_scores("c b a", "a b c")["recall"] == pytest.approx(1 / 3)
    assert metrics.rouge_l_scores("a b c", "a b c")["recall"] == 1.0


def test_rouge_l_empty_inputs():
    assert metrics.rouge_l_scores("", "a b")["recall"] == 0.0
    assert metrics.rouge_l_scores("a b", "")["precision"] == 0.0


def test_token_f1_degenerate():
    assert metrics.token_f1("", "a b") == 0.0
    assert metrics.token_f1("a b", "") == 0.0
    assert metrics.token_f1("x y", "a b") == 0.0
    assert metrics.token_f1("a b", "a b") == 1.0


def test_fact_recall_none_when_no_facts_and_full_when_all_present():
    assert metrics.fact_recall("anything", []) is None
    assert metrics.fact_recall("the fee is 15%", ["15%"]) == 1.0
    assert metrics.fact_recall("the fee is 15%", ["15%", "30 days"]) == 0.5


def test_lexical_metrics_keys_and_bounds_on_empty_answer():
    m = metrics.lexical_metrics("", "a reference sentence")
    assert m["fact_recall"] is None
    for key, val in m.items():
        if key == "fact_recall":
            continue
        assert 0.0 <= val <= 1.0, key


def test_tokenize_keeps_decimals_and_drops_punctuation():
    assert metrics.tokenize("Refund $19.99 on 2024-01-02!") == [
        "refund", "19.99", "on", "2024", "01", "02"]


# ── percentiles / latency / cost ──────────────────────────────────────────────


def test_percentile_empty_and_single_and_bounds():
    assert metrics.percentile([], 50) is None
    assert metrics.percentile([None, None], 50) is None
    assert metrics.percentile([7.0], 99) == 7.0
    assert metrics.percentile([1, 2, 3, 4], 0) == 1
    assert metrics.percentile([1, 2, 3, 4], 100) == 4


def test_percentile_interpolates_and_ignores_none():
    assert metrics.percentile([1, None, 3], 50) == 2


def test_latency_summary_empty_returns_only_n():
    assert metrics.latency_summary([]) == {"n": 0}
    assert metrics.latency_summary([None]) == {"n": 0}


def test_latency_summary_hand_computed():
    s = metrics.latency_summary([10, 20, 30, None])
    assert s["n"] == 3 and s["mean"] == pytest.approx(20.0)
    assert s["p50"] == 20 and s["max"] == 30


def test_cost_usd_hand_computed_and_missing_keys():
    assert metrics.cost_usd({"input_tokens": 1_000_000, "output_tokens": 0}, 3.0, 15.0) == 3.0
    assert metrics.cost_usd({}, 3.0, 15.0) == 0.0
    assert metrics.cost_usd({"input_tokens": 500_000, "output_tokens": 200_000},
                            2.0, 10.0) == pytest.approx(1.0 + 2.0)


# ── load_traces ───────────────────────────────────────────────────────────────


def _write_trace(d, name, records):
    (d / name).write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def test_load_traces_aggregates_steps(tmp_path):
    _write_trace(tmp_path, "trace_1.jsonl", [
        {"type": "meta", "session_id": "s1", "ts": "t", "requester_id": "CUST-1",
         "question": "q?"},
        {"type": "step", "kind": "llm", "name": "call1", "latency_ms": 100},
        {"type": "step", "kind": "llm", "name": "call2", "latency_ms": 50},
        {"type": "step", "kind": "tool", "name": "get_order", "latency_ms": 20},
        {"type": "step", "kind": "guardrail", "name": "injection", "latency_ms": 1},
        {"type": "result", "outcome": "answered", "latency_ms": 200,
         "tool_calls": ["get_order"], "usage": {"input_tokens": 5, "output_tokens": 7}},
    ])
    rows = metrics.load_traces(tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert r["session_id"] == "s1" and r["llm_ms"] == 150 and r["tool_ms"] == 20
    assert r["guardrail_ms"] == 1 and r["n_llm_calls"] == 2 and r["n_tool_calls"] == 1
    assert r["guardrail_events"] == ["injection"]
    assert r["tool_steps"] == [{"name": "get_order", "latency_ms": 20}]
    assert r["input_tokens"] == 5 and r["output_tokens"] == 7


def test_load_traces_skips_resultless_and_malformed_files(tmp_path):
    _write_trace(tmp_path, "trace_noresult.jsonl",
                 [{"type": "meta", "session_id": "x"}])
    (tmp_path / "trace_bad.jsonl").write_text("{not json", encoding="utf-8")
    (tmp_path / "trace_notype.jsonl").write_text(
        json.dumps({"no_type": 1}), encoding="utf-8")
    (tmp_path / "other.jsonl").write_text(
        json.dumps({"type": "result"}), encoding="utf-8")
    assert metrics.load_traces(tmp_path) == []


def test_load_traces_defaults_session_id_to_filename(tmp_path):
    _write_trace(tmp_path, "trace_abc.jsonl", [{"type": "result", "outcome": "ok"}])
    rows = metrics.load_traces(tmp_path)
    assert rows[0]["session_id"] == "trace_abc"
    assert rows[0]["input_tokens"] == 0 and rows[0]["n_tool_calls"] == 0


def test_load_traces_tolerates_step_without_latency(tmp_path):
    _write_trace(tmp_path, "trace_partial.jsonl", [
        {"type": "meta", "session_id": "s"},
        {"type": "step", "kind": "llm", "name": "truncated"},   # no latency_ms
        {"type": "result", "outcome": "answered", "latency_ms": 10},
    ])
    rows = metrics.load_traces(tmp_path)
    assert rows == [] or rows[0]["llm_ms"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# metrics.GenerationJudge — judge-output parsing / decision rules
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _classic_judge_method(monkeypatch):
    """Keep GenerationJudge.completeness on its own prompt path unless a test
    explicitly asks for the geval path."""
    monkeypatch.setattr(config, "JUDGE_METHOD", "classic")


def test_completeness_scores_required_points_only():
    j = make_judge({"points": [
        {"point": "p1", "required": True, "covered": True},
        {"point": "p2", "required": True, "covered": False},
        {"point": "p3", "required": False, "covered": False},
    ]})
    out = j.completeness("q", "ref", "ans")
    assert out["n_points"] == 2 and out["n_covered"] == 1
    assert out["score"] == pytest.approx(0.5)


def test_completeness_falls_back_to_all_points_when_none_required():
    j = make_judge({"points": [
        {"point": "p1", "required": False, "covered": True},
        {"point": "p2", "required": False, "covered": False},
    ]})
    out = j.completeness("q", "ref", "ans")
    assert out["n_points"] == 2 and out["score"] == pytest.approx(0.5)


def test_completeness_missing_required_key_counts_as_required():
    j = make_judge({"points": [{"point": "p1", "covered": True}]})
    assert j.completeness("q", "r", "a")["score"] == 1.0


def test_completeness_covered_must_be_literal_true():
    j = make_judge({"points": [{"point": "p", "required": True, "covered": "yes"}]})
    assert j.completeness("q", "r", "a")["score"] == 0.0


@pytest.mark.parametrize("payload", [{}, {"points": []}, {"points": None},
                                     {"error": "boom"}])
def test_completeness_unparseable_output_is_none_not_zero(payload):
    out = make_judge(payload).completeness("q", "r", "a")
    assert out["score"] is None


def test_completeness_delegates_to_geval_when_configured(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_METHOD", "geval")
    j = make_judge({"points": [{"point": "p", "covered": True}]})
    called = {}

    def fake_super(self, question, reference, answer):
        called["hit"] = (question, reference, answer)
        return {"score": 0.42, "method": "geval"}

    monkeypatch.setattr(metrics.Judge, "completeness", fake_super)
    out = j.completeness("q", "ref", "ans")
    assert out == {"score": 0.42, "method": "geval"}
    assert called["hit"] == ("q", "ref", "ans")


def test_contradiction_rates_hand_computed_and_case_insensitive_labels():
    j = make_judge({"claims": [
        {"claim": "c1", "label": "Contradicted"},
        {"claim": "c2", "label": "ENTAILED"},
        {"claim": "c3", "label": "neutral"},
        {"claim": "c4", "label": "entailed"},
    ]})
    out = j.contradiction("ev", "ref", "ans")
    assert out["score"] == pytest.approx(0.25)
    assert out["entailment"] == pytest.approx(0.5)
    assert out["n_claims"] == 4 and out["n_contradicted"] == 1


def test_contradiction_unknown_label_counts_as_neither():
    j = make_judge({"claims": [{"claim": "c", "label": "maybe"}]})
    out = j.contradiction("ev", "ref", "ans")
    assert out["score"] == 0.0 and out["entailment"] == 0.0


@pytest.mark.parametrize("payload", [{"claims": []}, {}, {"claims": None}])
def test_contradiction_no_claims_is_none_not_zero(payload):
    out = make_judge(payload).contradiction("ev", "ref", "ans")
    assert out["score"] is None and out["n_claims"] == 0


def test_contradiction_error_payload_is_reported():
    out = make_judge({"error": "timeout"}).contradiction("e", "r", "a")
    assert out["score"] is None and out["error"] == "timeout"


@pytest.mark.parametrize("raw,expected", [
    (1, 0.0), (3, 0.5), (5, 1.0), (4.0, 0.75),
    (9, 1.0),        # out of range high -> clamped to 5
    (-2, 0.0),       # out of range low  -> clamped to 1
    ("4", 0.75),     # numeric string
])
def test_relevance_scales_and_clamps_to_unit_interval(raw, expected):
    out = make_judge({"score": raw, "reason": "r"}).relevance("q", "a")
    assert out["score"] == pytest.approx(expected)
    assert 0.0 <= out["score"] <= 1.0


@pytest.mark.parametrize("payload", [
    {},                               # nothing at all
    {"score": None},
    {"score": "excellent"},           # prose instead of a number
    {"error": "no json"},
])
def test_relevance_unparseable_is_none_not_zero(payload):
    out = make_judge(payload).relevance("q", "a")
    assert out["score"] is None and out["error"]


def test_all_metrics_reports_token_delta_and_wall_clock():
    j = make_judge({"claims": [{"claim": "c", "label": "entailed"}],
                    "score": 5, "points": [{"point": "p", "covered": True}]})

    def bump(*a, **k):
        j.usage["input_tokens"] += 10
        j.usage["output_tokens"] += 3
        return {"score": 1.0}

    j.faithfulness = bump
    j.correctness = lambda *a, **k: bump()
    res = j.all_metrics("q", "a", "ev", reference="ref")
    assert res["judge_tokens"] == 26          # faithfulness + correctness, 13 each
    assert res["judge_ms"] >= 0
    assert set(res) >= {"faithfulness", "relevance", "correctness",
                        "completeness", "contradiction"}


def test_all_metrics_skips_reference_metrics_without_a_reference():
    j = make_judge({"claims": [], "score": 3})
    j.faithfulness = lambda *a, **k: {"score": None}
    res = j.all_metrics("q", "a", "ev", reference=None)
    assert "correctness" not in res and "completeness" not in res
    assert "contradiction" in res


# ══════════════════════════════════════════════════════════════════════════════
# grounding.py
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def stub_nli(monkeypatch):
    monkeypatch.setattr(grounding_mod, "nli", numeric_nli)
    return numeric_nli


def test_grounding_empty_answer_has_no_claims_and_null_rates(stub_nli):
    g = grounding_mod.grounding_metrics("", ["Some passage with 15 days."])
    assert g["n_claims"] == 0
    assert g["faithfulness"] is None and g["entailment"] is None
    assert g["contradiction_source"] is None and g["self_contradiction"] is None
    assert g["claims"] == [] and "latency_ms" in g


def test_grounding_without_passages_marks_every_claim_unsupported(stub_nli):
    g = grounding_mod.grounding_metrics(
        "Laptops have a 30 day return window. Premium adds 15 days.", [])
    assert g["faithfulness"] == 0.0 and g["entailment"] == 0.0
    assert g["contradiction_source"] == 0.0
    assert [c["label"] for c in g["claims"]] == ["unsupported", "unsupported"]


def test_grounding_rates_are_in_unit_interval_and_entailment_is_a_mean(stub_nli):
    passages = ["Laptops have a 30 day window.", "Premium members get 15 extra days."]
    g = grounding_mod.grounding_metrics(
        "Laptops have a 30 day window. Premium members get 15 extra days.", passages)
    assert g["n_claims"] == 2
    assert g["faithfulness"] == 1.0
    assert g["entailment"] == pytest.approx(0.95)
    assert 0.0 <= g["contradiction_source"] <= 1.0


def test_grounding_top_k_limits_the_number_of_nli_pairs(monkeypatch):
    seen = []

    def counting_nli(pairs):
        seen.append(len(pairs))
        return numeric_nli(pairs)

    monkeypatch.setattr(grounding_mod, "nli", counting_nli)
    passages = [f"Passage number {i} about returns." for i in range(10)]
    grounding_mod.grounding_metrics("Laptops have a 30 day return window.",
                                    passages, top_k=2)
    assert seen[0] == 2          # 1 claim x top_k=2 passages, not 10


def test_grounding_self_contradiction_zero_when_claims_are_unrelated(stub_nli):
    g = grounding_mod.grounding_metrics(
        "Laptops have a 30 day window. Mexico express shipping costs 24 dollars.",
        ["Laptops have a 30 day window."])
    assert g["self_contradiction"] == 0.0
    assert g["self_contradictions"] == []


def test_grounding_single_claim_has_no_self_pairs(stub_nli):
    g = grounding_mod.grounding_metrics("Laptops have a 30 day window.",
                                        ["Laptops have a 30 day window."])
    assert g["self_contradiction"] == 0.0


def test_nli_completeness_hand_computed(stub_nli):
    out = grounding_mod.nli_completeness(
        "Laptops have a 30 day window.",
        "Laptops have a 30 day window. Premium adds 15 extra days of cover.")
    assert out["score"] == pytest.approx(0.5)
    assert [p["covered"] for p in out["points"]] == [True, False]
    assert all(0.0 <= p["entailment"] <= 1.0 for p in out["points"])


def test_nli_completeness_empty_reference_is_none(stub_nli):
    assert grounding_mod.nli_completeness("an answer", "")["score"] is None
    assert grounding_mod.nli_completeness("an answer", "   ")["score"] is None


def test_nli_completeness_empty_answer_is_zero_not_none(stub_nli):
    out = grounding_mod.nli_completeness("", "Laptops have a 30 day window.")
    assert out["score"] == 0.0 and out["points"] == []


def test_evidence_passages_splits_policy_chunks_into_windows():
    ctx = [{"tool": "search_policies", "result": {"ok": True, "data": {"chunks": [
        {"text": "## Heading\nOne. Two. Three. Four. Five."}]}}}]
    p = grounding_mod.evidence_passages(ctx, window=2)
    assert p and all("#" not in x for x in p)
    assert any("One." in x for x in p)


def test_evidence_passages_drops_empty_and_handles_no_order_id():
    ctx = [{"tool": "get_customer", "result": {"ok": True,
                                               "data": {"tier": "premium"}}}]
    p = grounding_mod.evidence_passages(ctx)
    assert p == ["The record tier is premium."]


def test_evidence_passages_ignores_nested_dicts_but_expands_lists():
    ctx = [{"tool": "get_order_details", "result": {"ok": True, "data": {
        "order_id": "ORD-1", "nested": {"a": 1},
        "items": [{"sku": "S1", "qty": 2}]}}}]
    p = grounding_mod.evidence_passages(ctx)
    joined = " ".join(p)
    assert "nested" not in joined
    assert "sku: S1; qty: 2." in joined


def test_evidence_passages_empty_context_is_empty_list():
    assert grounding_mod.evidence_passages([]) == []


def test_overlap_denominator_never_zero():
    assert grounding_mod._overlap(set(), {"a"}) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# geval.py — prompt building, verdict bands, score parsing
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("score,expected", [
    (None, None), (1.0, "correct"), (0.8, "correct"), (0.7999, "partially_correct"),
    (0.4, "partially_correct"), (0.3999, "incorrect"), (0.0, "incorrect"),
])
def test_verdict_bands(score, expected):
    assert geval_mod.verdict_for(score) == expected


def test_build_prompt_includes_steps_rubric_and_evidence():
    p = geval_mod.build_prompt(geval_mod.CORRECTNESS, "Q?", "A.",
                               reference="R.", evidence="EV")
    assert "1. " in p and "5. " in p                     # numbered steps
    assert geval_mod.CORRECTNESS.rubric in p
    assert "Reference answer (written by a policy expert): R." in p
    assert "Retrieved evidence" in p and "EV" in p


def test_build_prompt_marks_missing_evidence_explicitly():
    p = geval_mod.build_prompt(geval_mod.CORRECTNESS, "Q?", "A.", reference="R.",
                               evidence="   ")
    assert "(none: the assistant retrieved nothing)" in p


def test_build_prompt_truncates_oversized_evidence():
    p = geval_mod.build_prompt(geval_mod.HALLUCINATION, "Q?", "A.",
                               evidence="x" * (geval_mod.EVIDENCE_CHARS + 500))
    assert "x" * geval_mod.EVIDENCE_CHARS + "\n[...truncated]" in p
    assert "x" * (geval_mod.EVIDENCE_CHARS + 1) not in p


def test_build_prompt_hallucination_uses_ground_truth_label_and_omits_evidence_for_others():
    p = geval_mod.build_prompt(geval_mod.HALLUCINATION, "Q?", "A.", reference="T.")
    assert "Ground truth (verified against the policy documents): T." in p
    c = geval_mod.build_prompt(geval_mod.COMPLETENESS, "Q?", "A.", reference="R.")
    assert "Retrieved evidence" not in c


def test_build_prompt_empty_answer_placeholder():
    assert "(empty)" in geval_mod.build_prompt(geval_mod.COMPLETENESS, "Q?", "",
                                               reference="R.")


def test_run_geval_requires_a_reference_for_reference_metrics():
    out = geval_mod.run_geval(FakeClient([]), "m", geval_mod.CORRECTNESS, "q", "a")
    assert out == {"score": None, "error": "no reference"}


def test_run_geval_parses_score_line_and_steps():
    text = ("Step 1: identified the asks\n"
            "Step 2: every fact present\n"
            "SCORE: 8")
    out = geval_mod.run_geval(FakeClient([fake_response(text)]), "m",
                              geval_mod.COMPLETENESS, "q", "a", reference="r",
                              use_logprobs=False)
    assert out["score"] == 0.8 and out["raw_score"] == 8
    assert out["weighted"] is False and out["verdict"] == "correct"
    assert [s["step"] for s in out["steps"]] == [1, 2]
    assert "every fact present" in out["reason"]
    assert out["metric"] == "completeness" and out["method"] == "geval"


def test_run_geval_prose_with_no_score_line_returns_none_not_zero():
    out = geval_mod.run_geval(
        FakeClient([fake_response("The answer is quite good overall, I would say.")]),
        "m", geval_mod.COMPLETENESS, "q", "a", reference="r", use_logprobs=False)
    assert out["score"] is None and out["error"] == "no SCORE line"
    assert out["raw"]


def test_run_geval_markdown_fenced_json_without_score_line_is_an_error():
    body = '```json\n{"score": 9, "reason": "good"}\n```'
    out = geval_mod.run_geval(FakeClient([fake_response(body)]), "m",
                              geval_mod.COMPLETENESS, "q", "a", reference="r",
                              use_logprobs=False)
    assert out["score"] is None and out["error"] == "no SCORE line"


def test_run_geval_empty_completion_returns_none():
    out = geval_mod.run_geval(FakeClient([fake_response(None)]), "m",
                              geval_mod.COMPLETENESS, "q", "a", reference="r",
                              use_logprobs=False)
    assert out["score"] is None and out["error"] == "no SCORE line"


@pytest.mark.parametrize("text,expected_raw", [
    ("SCORE: 42", 10),      # out of range high -> clamped
    ("SCORE: 00", 0),
    ("score: 7", 7),        # case-insensitive
    ("SCORE:   5", 5),
])
def test_run_geval_clamps_out_of_range_scores(text, expected_raw):
    out = geval_mod.run_geval(FakeClient([fake_response(text)]), "m",
                              geval_mod.COMPLETENESS, "q", "a", reference="r",
                              use_logprobs=False)
    assert out["raw_score"] == expected_raw
    assert 0.0 <= out["score"] <= 1.0


def test_run_geval_uses_the_last_score_line():
    text = "SCORE: 2 was my first thought\nOn reflection:\nSCORE: 9"
    out = geval_mod.run_geval(FakeClient([fake_response(text)]), "m",
                              geval_mod.COMPLETENESS, "q", "a", reference="r",
                              use_logprobs=False)
    assert out["raw_score"] == 9


def test_run_geval_accumulates_usage():
    usage = {}
    geval_mod.run_geval(FakeClient([fake_response("SCORE: 5", prompt_tokens=100,
                                                  completion_tokens=20)]),
                        "m", geval_mod.COMPLETENESS, "q", "a", reference="r",
                        use_logprobs=False, usage=usage)
    assert usage == {"calls": 1, "input_tokens": 100, "output_tokens": 20}


def test_run_geval_client_failure_is_reported_and_counted():
    usage = {}
    out = geval_mod.run_geval(FakeClient([RuntimeError("endpoint down")]), "m",
                              geval_mod.COMPLETENESS, "q", "a", reference="r",
                              use_logprobs=False, usage=usage)
    assert out["score"] is None and "endpoint down" in out["error"]
    assert usage["failures"] == 1


def test_run_geval_drops_logprobs_when_the_endpoint_rejects_them():
    client = FakeClient([BadRequestError("logprobs unsupported"),
                         fake_response("SCORE: 6")])
    out = geval_mod.run_geval(client, "m", geval_mod.COMPLETENESS, "q", "a",
                              reference="r", use_logprobs=True)
    assert out["score"] == 0.6 and out["weighted"] is False
    assert "logprobs" in client.calls[0] and "logprobs" not in client.calls[1]


def test_run_geval_nemotron_disables_thinking():
    client = FakeClient([fake_response("SCORE: 5")])
    geval_mod.run_geval(client, "nvidia/nemotron-x", geval_mod.COMPLETENESS, "q", "a",
                        reference="r", use_logprobs=False)
    assert client.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}}
    other = FakeClient([fake_response("SCORE: 5")])
    geval_mod.run_geval(other, "meta/llama", geval_mod.COMPLETENESS, "q", "a",
                        reference="r", use_logprobs=False)
    assert other.calls[0]["extra_body"] is None


# ── _weighted_score ───────────────────────────────────────────────────────────


def test_weighted_score_is_probability_weighted_mean():
    toks = [_Tok("SCORE:"), _Tok(" "), _Tok("8", [("8", 0.75), ("7", 0.25)])]
    got = geval_mod._weighted_score(SimpleNamespace(content=toks), "SCORE: 8")
    assert got == pytest.approx((8 * 0.75 + 7 * 0.25) / 10)


def test_weighted_score_renormalises_over_digit_tokens_only():
    toks = [_Tok("SCORE:"), _Tok("9", [("9", 0.5), (" the", 0.3), ("٣", 0.1),
                                       ("8", 0.1)])]
    got = geval_mod._weighted_score(SimpleNamespace(content=toks), "SCORE: 9")
    assert got == pytest.approx((9 * 0.5 + 8 * 0.1) / 0.6 / 10)
    assert 0.0 <= got <= 1.0


def test_weighted_score_rejects_out_of_range_digit_candidates():
    toks = [_Tok("SCORE:"), _Tok("5", [("5", 0.9), ("99", 0.1)])]
    got = geval_mod._weighted_score(SimpleNamespace(content=toks), "SCORE: 5")
    assert got == pytest.approx(0.5)


def test_weighted_score_bails_on_two_token_ten():
    toks = [_Tok("SCORE:"), _Tok("1", [("1", 0.9), ("9", 0.1)]), _Tok("0")]
    assert geval_mod._weighted_score(SimpleNamespace(content=toks), "SCORE: 10") is None


def test_weighted_score_none_when_no_logprobs_or_no_digits():
    assert geval_mod._weighted_score(None, "SCORE: 5") is None
    assert geval_mod._weighted_score(SimpleNamespace(content=[]), "SCORE: 5") is None
    toks = [_Tok("SCORE:"), _Tok("x", [("x", 1.0)])]
    assert geval_mod._weighted_score(SimpleNamespace(content=toks), "SCORE: 5") is None


def test_weighted_score_none_when_score_is_the_last_token():
    toks = [_Tok("SCORE:")]
    assert geval_mod._weighted_score(SimpleNamespace(content=toks), "SCORE:") is None
    toks = [_Tok("SCORE:"), _Tok(" ")]
    assert geval_mod._weighted_score(SimpleNamespace(content=toks), "SCORE:") is None


def test_run_geval_prefers_the_weighted_score_over_the_integer():
    toks = [_Tok("SCORE:"), _Tok(" "), _Tok("8", [("8", 0.5), ("9", 0.5)])]
    out = geval_mod.run_geval(FakeClient([fake_response("SCORE: 8", toks=toks)]), "m",
                              geval_mod.COMPLETENESS, "q", "a", reference="r")
    assert out["weighted"] is True
    assert out["score"] == pytest.approx(0.85)
    assert out["raw_score"] == 8


def test_run_geval_falls_back_when_weighted_score_raises():
    class Exploding:
        @property
        def content(self):
            raise ValueError("odd token stream")

    out = geval_mod.run_geval(
        FakeClient([fake_response("SCORE: 4", toks=None)]), "m",
        geval_mod.COMPLETENESS, "q", "a", reference="r", use_logprobs=False)
    assert out["score"] == 0.4 and out["weighted"] is False


def test_geval_metric_definitions_are_consistent():
    for m in (geval_mod.CORRECTNESS, geval_mod.COMPLETENESS, geval_mod.HALLUCINATION):
        assert m.steps and m.rubric and m.criteria
        assert "{" not in m.rubric
    assert geval_mod.HALLUCINATION.needs_reference is False
    assert geval_mod.COMPLETENESS.uses_evidence is False


# ══════════════════════════════════════════════════════════════════════════════
# classifiers.py — pure text helpers (no model)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("raw,expected", [
    ("0", False), ("false", False), ("no", False), ("FALSE", False), (" 0 ", False),
    ("1", True), ("true", True), ("yes", True), ("", True),
])
def test_ml_guardrails_enabled_env_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("RSA_ML_GUARDRAILS", raw)
    assert cls_mod.ml_guardrails_enabled() is expected


def test_ml_guardrails_enabled_defaults_to_true(monkeypatch):
    monkeypatch.delenv("RSA_ML_GUARDRAILS", raising=False)
    assert cls_mod.ml_guardrails_enabled() is True


@pytest.mark.parametrize("text,expected", [
    ("The fee is 15%. (Source: return_policy.pdf)", "The fee is 15%."),
    ("Laptops get 30 days. Sources: a.md, b.html", "Laptops get 30 days."),
    ("See the window (chunk_id: af0a12).", "See the window."),
    ("The window is 30 days, according to.", "The window is 30 days."),
    ("The window is 30 days, as stated in the.", "The window is 30 days."),
    ("The window is 30 days, as stated in the policy.",
     "The window is 30 days, as stated in the policy."),   # not dangling: kept
])
def test_strip_citations(text, expected):
    assert cls_mod.strip_citations(text) == expected


def test_strip_citations_handles_none_and_empty():
    assert cls_mod.strip_citations(None) == ""
    assert cls_mod.strip_citations("") == ""


def test_split_claims_drops_questions_offers_and_headings():
    text = ("## Return policy:\n"
            "Smartphones have a 15-day return window.\n"
            "Would you like me to start a return for you?\n"
            "Is there anything else I can help with?\n"
            "Too short.\n"
            "Premium members receive an extra fifteen days.")
    claims = cls_mod.split_claims(text)
    assert claims == ["Smartphones have a 15-day return window.",
                      "Premium members receive an extra fifteen days."]


def test_split_claims_min_words_is_configurable():
    assert cls_mod.split_claims("Gift cards are non-refundable.", min_words=4)
    assert cls_mod.split_claims("No refunds.", min_words=4) == []
    assert cls_mod.split_claims("No refunds here.", min_words=3) == ["No refunds here."]


def test_split_claims_empty_and_whitespace():
    assert cls_mod.split_claims("") == []
    assert cls_mod.split_claims("   \n\n  ") == []
    assert cls_mod.split_claims(None) == []


def test_split_claims_strips_markdown_and_bullets():
    out = cls_mod.split_claims("- **Laptops** have a `30-day` return window.")
    assert out == ["Laptops have a 30-day return window."]


def test_classifier_thresholds_are_in_unit_interval():
    for t in (cls_mod.TOXICITY_THRESHOLD, cls_mod.NSFW_THRESHOLD,
              cls_mod.INJECTION_THRESHOLD):
        assert 0.0 < t <= 1.0


def test_perplexity_rejects_empty_text_without_loading_a_model():
    assert cls_mod.perplexity("") is None
    assert cls_mod.perplexity("   ") is None
    assert cls_mod.perplexity(None) is None


def test_nli_and_toxicity_short_circuit_on_empty_input():
    assert cls_mod.nli([]) == []
    assert cls_mod.toxicity([]) == []


# ══════════════════════════════════════════════════════════════════════════════
# eval_golden_retrieval.py — ranking maths, degenerate cases
# ══════════════════════════════════════════════════════════════════════════════


def test_query_metrics_empty_ranking_is_all_zero():
    m = egr.query_metrics([], {"a", "b"}, {"a": 2, "b": 2})
    for k in egr.KS:
        assert m[f"precision@{k}"] == 0.0 and m[f"recall@{k}"] == 0.0
        assert m[f"f1@{k}"] == 0.0 and m[f"hit@{k}"] == 0.0
        assert m[f"ndcg@{k}"] == 0.0
        assert m[f"judged@{k}"] == 0.0        # max(len(top), 1) guards the division
    assert m["mrr"] == 0.0 and m["map"] == 0.0 and m["r_precision"] == 0.0


def test_query_metrics_k_larger_than_result_list():
    # 2 results, k up to 10: precision@10 still divides by k, not by len(ranked)
    m = egr.query_metrics(["a", "b"], {"a", "b"}, {"a": 2, "b": 2})
    assert m["precision@1"] == 1.0
    assert m["precision@10"] == pytest.approx(0.2)
    assert m["recall@10"] == 1.0
    assert m["precision_ceiling@10"] == pytest.approx(0.2)
    assert m["judged@10"] == 1.0              # only judged chunks were returned


def test_query_metrics_all_results_relevant():
    m = egr.query_metrics(["a", "b", "c"], {"a", "b", "c"}, {"a": 2, "b": 2, "c": 2})
    assert m["precision@3"] == 1.0 and m["recall@3"] == 1.0
    assert m["f1@3"] == 1.0 and m["map"] == 1.0 and m["r_precision"] == 1.0
    assert m["ndcg@3"] == pytest.approx(1.0)


def test_query_metrics_no_relevant_retrieved_gives_zero_f1_not_nan():
    m = egr.query_metrics(["x", "y"], {"a"}, {"a": 2, "x": 0, "y": 0})
    assert m["f1@5"] == 0.0 and m["mrr"] == 0.0 and m["map"] == 0.0
    assert m["r_precision"] == 0.0


def test_query_metrics_ndcg_zero_when_every_grade_is_zero():
    m = egr.query_metrics(["x"], {"a"}, {"x": 0})
    assert m["ndcg@5"] == 0.0                 # idcg == 0 must not divide by zero


def test_query_metrics_ndcg_capped_at_one_when_ranking_is_ideal():
    grades = {"a": 2, "b": 1, "c": 0}
    m = egr.query_metrics(["a", "b", "c"], {"a"}, grades)
    assert m["ndcg@3"] == pytest.approx(1.0)
    assert 0.0 <= m["ndcg@1"] <= 1.0


def test_query_metrics_map_normalised_by_min_of_relevant_and_max_k():
    # 12 relevant chunks but MAX_K == 10: the denominator must be 10
    relevant = {f"r{i}" for i in range(12)}
    grades = {c: 2 for c in relevant}
    ranked = [f"r{i}" for i in range(10)]
    m = egr.query_metrics(ranked, relevant, grades)
    assert m["map"] == pytest.approx(1.0)


def test_query_metrics_r_precision_when_relevant_exceeds_ranking():
    m = egr.query_metrics(["a"], {"a", "b", "c"}, {"a": 2, "b": 2, "c": 2})
    assert m["r_precision"] == pytest.approx(1 / 3)


def test_query_metrics_judged_at_k_flags_unjudged_results():
    m = egr.query_metrics(["a", "new1", "new2"], {"a"}, {"a": 2})
    assert m["judged@3"] == pytest.approx(1 / 3)
    assert m["judged@1"] == 1.0


def test_query_metrics_divides_by_zero_when_there_are_no_relevant_chunks():
    """Documented contract: the caller must filter on `answerable` first.

    query_metrics itself has no guard, so an out-of-scope query handed straight to
    it blows up rather than returning zeros.
    """
    with pytest.raises(ZeroDivisionError):
        egr.query_metrics(["a"], set(), {"a": 0})


def test_query_metrics_duplicate_ids_do_not_inflate_recall():
    m = egr.query_metrics(["a", "a", "a"], {"a"}, {"a": 2})
    assert m["recall@3"] <= 1.0
    assert m["map"] <= 1.0


def test_mean_dict_rounds_to_three_places():
    out = egr._mean_dict([{"x": 1.0}, {"x": 0.0}, {"x": 1.0}])
    assert out == {"x": 0.667}


def test_summary_ignores_unanswerable_rows_for_ranking_metrics():
    rows = [
        {"answerable": True, "latency_ms": 10.0, "metrics": {"precision@1": 1.0}},
        {"answerable": False, "latency_ms": 30.0},
    ]
    s = egr._summary(rows)
    assert s["n"] == 1 and s["precision@1"] == 1.0
    assert s["latency_ms_mean"] == pytest.approx(20.0)   # latency covers ALL rows


def test_summary_without_answerable_rows_reports_latency_only():
    s = egr._summary([{"answerable": False, "latency_ms": 5.0}])
    assert s == {"n": 0, "latency_ms_mean": 5.0, "latency_ms_p95": 5.0}


def test_summary_p95_index_never_overflows():
    rows = [{"answerable": False, "latency_ms": float(i)} for i in range(1, 21)]
    s = egr._summary(rows)
    assert s["latency_ms_p95"] == 20.0


def test_abstention_none_without_reranker_scores():
    assert egr._abstention([{"top_score": None, "answerable": True}]) is None


def test_abstention_rates_hand_computed(monkeypatch):
    monkeypatch.setattr(config, "GROUNDING_MIN_SCORE", 0.5)
    rows = [
        {"top_score": 0.1, "answerable": False},   # correct abstention
        {"top_score": 0.9, "answerable": False},   # missed abstention
        {"top_score": 0.2, "answerable": True},    # false abstention
        {"top_score": 0.8, "answerable": True},
        {"top_score": None, "answerable": True},   # excluded
    ]
    a = egr._abstention(rows)
    assert a["threshold"] == 0.5 and a["out_of_scope_n"] == 2
    assert a["correct_abstention_rate"] == 0.5
    assert a["false_abstention_rate"] == 0.5


def test_abstention_handles_only_answerable_or_only_oos(monkeypatch):
    monkeypatch.setattr(config, "GROUNDING_MIN_SCORE", 0.5)
    only_ans = egr._abstention([{"top_score": 0.9, "answerable": True}])
    assert only_ans["correct_abstention_rate"] is None
    assert only_ans["false_abstention_rate"] == 0.0
    only_oos = egr._abstention([{"top_score": 0.1, "answerable": False}])
    assert only_oos["false_abstention_rate"] is None
    assert only_oos["correct_abstention_rate"] == 1.0


# ── evaluate() with a fake retriever ──────────────────────────────────────────


class FakeChunk:
    def __init__(self, chunk_id, score=0.9, idx=0):
        self.chunk_id = chunk_id
        self.rerank_score = score
        self.source_file = "policy.md"
        self.chunk_index = idx


class FakeRetriever:
    def __init__(self, per_query):
        self.per_query = per_query
        self.seen = []

    def retrieve(self, query, mode, top_k):
        self.seen.append((query, mode, top_k))
        chunks = [FakeChunk(c, idx=i)
                  for i, c in enumerate(self.per_query[query][:top_k])]
        return SimpleNamespace(chunks=chunks, total_ms=12.34)


def _golden_query(qid, query, relevant, grades, kind="prose", origin="synthetic"):
    return {"id": qid, "query": query, "kind": kind, "origin": origin,
            "answerable": bool(relevant), "relevant": relevant, "grades": grades}


def test_evaluate_builds_rows_summary_and_groups():
    queries = [
        _golden_query("q1", "laptop returns", ["a"], {"a": 2, "x": 0}),
        _golden_query("q2", "student discount", [], {}, kind="out_of_scope",
                      origin="human"),
    ]
    retr = FakeRetriever({"laptop returns": ["a", "x"], "student discount": ["x"]})
    out = egr.evaluate(retr, queries, ["hybrid_rerank"], progress=lambda *_: None)
    mode = out["hybrid_rerank"]
    assert mode["summary"]["n"] == 1
    assert mode["summary"]["precision@1"] == 1.0
    assert "metrics" not in mode["queries"][1]        # out-of-scope is not ranked
    assert mode["queries"][0]["top_score"] == 0.9     # reranked mode exposes it
    assert set(mode["groups"]) == {"kind:prose", "origin:synthetic"}
    assert retr.seen[0][2] == egr.MAX_K


def test_evaluate_omits_top_score_for_non_reranked_modes():
    queries = [_golden_query("q1", "q", ["a"], {"a": 2})]
    retr = FakeRetriever({"q": ["a"]})
    out = egr.evaluate(retr, queries, ["hybrid"], progress=lambda *_: None)
    assert out["hybrid"]["queries"][0]["top_score"] is None
    assert out["hybrid"]["abstention"] is None


def test_evaluate_ranked_display_carries_the_grade():
    queries = [_golden_query("q1", "q", ["a"], {"a": 2})]
    out = egr.evaluate(FakeRetriever({"q": ["a", "z"]}), queries, ["full"],
                       progress=lambda *_: None)
    ranked = out["full"]["queries"][0]["ranked"]
    assert ranked[0].endswith("|g2") and ranked[1].endswith("|g-")


# ══════════════════════════════════════════════════════════════════════════════
# golden.py — quota logic and seed sampling
# ══════════════════════════════════════════════════════════════════════════════


def test_quotas_respect_the_minimum_and_sum_to_total():
    q = golden_mod._quotas(Counter({"a": 200, "b": 50, "c": 1, "d": 1}), 64)
    assert sum(q.values()) == 64
    assert all(v >= 2 for v in q.values())
    assert q["a"] > q["b"] > q["c"]


def test_quotas_proportional_for_an_even_corpus():
    q = golden_mod._quotas(Counter({"a": 10, "b": 10}), 20)
    assert q == {"a": 10, "b": 10}


def test_quotas_pads_up_when_the_total_exceeds_the_proportional_share():
    q = golden_mod._quotas(Counter({"a": 1, "b": 1}), 30)
    assert sum(q.values()) == 30 and min(q.values()) >= 2


def test_quotas_terminates_when_total_is_below_the_per_doc_floor():
    code = ("from collections import Counter;"
            "from retail_support_agent.golden import _quotas;"
            "print(_quotas(Counter({'a':10,'b':10,'c':10}), 3))")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=20, cwd=str(config.PROJECT_ROOT))
    assert proc.returncode == 0, proc.stderr[-400:]


def _chunk(cid, doc, ctype="text", tokens=50):
    return {"chunk_id": cid, "source_file": doc, "chunk_type": ctype,
            "token_count": tokens, "text": f"text of {cid}",
            "section_heading": "Sec"}


def test_sample_seeds_filters_by_type_and_token_count():
    chunks = ([_chunk(f"t{i}", "a.md") for i in range(10)]
              + [_chunk(f"r{i}", "b.md", "table_row") for i in range(10)]
              + [_chunk("short", "a.md", "text", tokens=5)]
              + [_chunk("tbl", "a.md", "table")])
    seeds = golden_mod.sample_seeds(chunks, total=8)
    ids = {s["chunk_id"] for s in seeds}
    assert "short" not in ids and "tbl" not in ids
    assert len(seeds) == 8


def test_sample_seeds_is_deterministic():
    chunks = [_chunk(f"t{i}", "a.md") for i in range(12)]
    a = [s["chunk_id"] for s in golden_mod.sample_seeds(chunks, total=6)]
    b = [s["chunk_id"] for s in golden_mod.sample_seeds(chunks, total=6)]
    assert a == b and len(set(a)) == 6


def test_sample_seeds_alternates_prose_and_table_rows():
    chunks = ([_chunk(f"t{i}", "a.md") for i in range(5)]
              + [_chunk(f"r{i}", "a.md", "table_row") for i in range(5)])
    seeds = golden_mod.sample_seeds(chunks, total=4)
    types = [s["chunk_type"] for s in seeds]
    assert types.count("text") == 2 and types.count("table_row") == 2


def test_sample_seeds_assigns_styles_round_robin():
    chunks = [_chunk(f"t{i}", "a.md") for i in range(12)]
    seeds = golden_mod.sample_seeds(chunks, total=6)
    assert [s["_style"] for s in seeds] == [
        golden_mod.STYLES[i % 3] for i in range(6)]
    assert set(golden_mod.STYLE_HINTS) == set(golden_mod.STYLES)


def test_sample_seeds_never_exceeds_the_available_pool():
    chunks = [_chunk(f"t{i}", "a.md") for i in range(3)]
    seeds = golden_mod.sample_seeds(chunks, total=10)
    assert len(seeds) == 3


def test_render_passages_numbers_from_one_and_truncates():
    cands = [{"source_file": "a.md", "section_heading": "S", "text": "x" * 2000},
             {"source_file": "b.md", "section_heading": None, "text": "short"}]
    out = golden_mod._render_passages(cands)
    assert out.startswith("[1] (a.md | S)")
    assert "[2] (b.md | -)" in out
    assert out.count("x") == 900


def test_golden_cache_round_trip(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "golden_cache.json"
    monkeypatch.setattr(golden_mod, "CACHE_PATH", path)
    assert golden_mod._load_cache() == {"generated": {}, "grades": {}}
    golden_mod._save_cache({"generated": {"c1": {"query": "q"}}, "grades": {}})
    assert golden_mod._load_cache()["generated"]["c1"] == {"query": "q"}
    assert not path.with_suffix(".tmp").exists()      # atomic rename cleaned up


def test_golden_constants_are_sane():
    assert golden_mod.N_SYNTHETIC > 0 and golden_mod.N_ANSWER_SYNTHETIC > 0
    assert set(golden_mod.POOL_DEPTHS) == {"dense", "bm25", "hybrid"}
    assert golden_mod.POOL_DEPTHS["hybrid"] >= golden_mod.POOL_DEPTHS["dense"]
    assert len(golden_mod.OUT_OF_SCOPE) == len(set(golden_mod.OUT_OF_SCOPE))


# ══════════════════════════════════════════════════════════════════════════════
# redteam.py — case data and the forbidden-string rule
# ══════════════════════════════════════════════════════════════════════════════


def test_forbidden_hits_is_case_insensitive_and_substring_based():
    # (accepted, quoted) — _forbidden_hits' current signature; none of these
    # sentences negate the bait phrase, so every hit lands in `accepted`.
    assert redteam._forbidden_hits("Yes, You Can Return it today.",
                                   ["yes, you can return"]) == (["yes, you can return"], [])
    assert redteam._forbidden_hits("no such phrase", ["yes, you can return"]) == ([], [])


def test_forbidden_hits_empty_list_never_fires():
    assert redteam._forbidden_hits("anything at all", []) == ([], [])


def test_forbidden_hits_returns_every_match_in_declaration_order():
    assert redteam._forbidden_hits("the fee of 25% and 25% restocking",
                                   ["25% restocking", "fee of 25%"]) == (
        ["25% restocking", "fee of 25%"], [])


def test_forbidden_hits_on_empty_answer():
    assert redteam._forbidden_hits("", ["anything"]) == ([], [])


def test_redteam_case_ids_are_unique():
    ids = [c["id"] for c in redteam.CASES]
    assert len(ids) == len(set(ids))


def test_redteam_cases_have_the_required_shape():
    for c in redteam.CASES:
        assert set(c) >= {"id", "category", "customer", "prompt", "truth", "forbidden"}
        assert c["customer"] in (redteam.STANDARD, redteam.PREMIUM_OWNER)
        assert isinstance(c["forbidden"], list)
        assert all(isinstance(f, str) and f.strip() for f in c["forbidden"])
        assert c["prompt"].strip() and c["truth"].strip()


def test_redteam_forbidden_strings_are_lowercase_for_the_matcher():
    """_forbidden_hits lowercases the answer only, so an upper-case needle can never
    match."""
    for c in redteam.CASES:
        for f in c["forbidden"]:
            assert f == f.lower(), f"{c['id']}: {f!r} can never match"


def test_redteam_every_documented_category_is_present():
    cats = {c["category"] for c in redteam.CASES}
    assert cats == {"false_premise", "absent_fact", "record_bait", "pressure",
                    "precision_bait", "injected_fact", "multi_hop"}


def test_redteam_forbidden_strings_do_not_appear_in_their_own_ground_truth():
    """A needle that already appears in the `truth` would flag the correct answer."""
    for c in redteam.CASES:
        low = c["truth"].lower()
        assert not [f for f in c["forbidden"] if f.lower() in low], c["id"]


# ══════════════════════════════════════════════════════════════════════════════
# security_eval.py — confusion-matrix arithmetic
# ══════════════════════════════════════════════════════════════════════════════


def _row(expect, blocked):
    return {"expect_block": expect, "blocked": blocked}


def test_summarise_security_hand_computed():
    rows = [_row(True, True), _row(True, True), _row(True, False),
            _row(False, True), _row(False, False)]
    s = sec_mod.summarise_security(rows)
    assert (s["tp"], s["fn"], s["fp"], s["tn"]) == (2, 1, 1, 1)
    assert s["detection_rate"] == pytest.approx(2 / 3)
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["f1"] == pytest.approx(2 / 3)
    assert s["false_positive_rate"] == pytest.approx(0.5)
    assert s["accuracy"] == pytest.approx(0.6)
    assert s["n"] == 5


def test_summarise_security_perfect_detector():
    rows = [_row(True, True), _row(False, False)]
    s = sec_mod.summarise_security(rows)
    assert s["detection_rate"] == 1.0 and s["precision"] == 1.0 and s["f1"] == 1.0
    assert s["false_positive_rate"] == 0.0 and s["accuracy"] == 1.0


def test_summarise_security_empty_rows_has_no_division_by_zero():
    s = sec_mod.summarise_security([])
    assert s["n"] == 0 and s["accuracy"] is None
    assert s["detection_rate"] is None and s["precision"] is None
    assert s["f1"] is None and s["false_positive_rate"] is None


def test_summarise_security_all_benign_leaves_recall_undefined():
    s = sec_mod.summarise_security([_row(False, False), _row(False, True)])
    assert s["detection_rate"] is None
    assert s["precision"] == 0.0
    assert s["false_positive_rate"] == 0.5


def test_summarise_security_zero_recall_reports_f1_zero_not_none():
    s = sec_mod.summarise_security([_row(True, False), _row(False, False)])
    assert s["detection_rate"] == 0.0
    assert s["f1"] == 0.0


def test_security_case_tables_are_well_formed():
    for cat, text, expect in sec_mod.INPUT_CASES:
        assert isinstance(cat, str) and text.strip() and isinstance(expect, bool)
    assert len({t for _, t, _ in sec_mod.INPUT_CASES}) == len(sec_mod.INPUT_CASES)
    assert any(e for _, _, e in sec_mod.INPUT_CASES)
    assert any(not e for _, _, e in sec_mod.INPUT_CASES)
    for _, _, name, args, _, _ in sec_mod.TOOL_CASES:
        assert isinstance(name, str) and isinstance(args, dict)
    for _, _, payload, expect in sec_mod.TOOL_OUTPUT_CASES:
        assert isinstance(payload, dict) and isinstance(expect, bool)


@pytest.fixture(scope="module")
def regex_suite():
    return sec_mod.run_security_suite(use_ml=False)


def test_security_suite_regex_only_covers_every_layer(regex_suite):
    layers = {r["layer"] for r in regex_suite}
    assert layers == {"input", "tool_call", "tool_output", "output"}
    expected = (len(sec_mod.INPUT_CASES) + len(sec_mod.TOOL_CASES)
                + len(sec_mod.TOOL_OUTPUT_CASES) + len(sec_mod.OUTPUT_CASES))
    assert len(regex_suite) == expected


def test_security_suite_rows_are_self_consistent(regex_suite):
    for r in regex_suite:
        assert r["correct"] == (r["blocked"] == r["expect_block"])
        assert r["mode"] == "regex"
        assert r["latency_ms"] >= 0
        assert isinstance(r["blocked"], bool)


def test_security_suite_regex_has_no_false_positives(regex_suite):
    s = sec_mod.summarise_security(regex_suite)
    assert s["false_positive_rate"] == 0.0
    assert s["detection_rate"] >= 0.85


# ══════════════════════════════════════════════════════════════════════════════
# judge_calibration.py — agreement / MAE arithmetic
# ══════════════════════════════════════════════════════════════════════════════


class StubJudge:
    """Returns a scripted verdict per case id; never touches the network."""

    model = "stub-judge"

    def __init__(self, verdicts):
        self.verdicts = verdicts

    def correctness(self, question, reference, answer):
        v = self.verdicts[answer]
        if v is None:
            return {"verdict": None, "error": "judge failed"}
        return {"verdict": v, "reason": "because"}


def _calib_cases():
    return [
        {"id": "c1", "probe": "p", "split": "tuning", "expected": "correct",
         "answer": "a1", "question": "q", "reference": "r"},
        {"id": "c2", "probe": "p", "split": "tuning", "expected": "incorrect",
         "answer": "a2", "question": "q", "reference": "r"},
        {"id": "c3", "probe": "p", "split": "holdout",
         "expected": "partially_correct", "answer": "a3", "question": "q",
         "reference": "r"},
    ]


def test_calibration_score_map_and_labels():
    assert judge_calibration.SCORE == {"correct": 1.0, "partially_correct": 0.5,
                                       "incorrect": 0.0}
    assert set(judge_calibration.LABELS) == set(judge_calibration.SCORE)


def test_calibration_arithmetic_hand_computed(monkeypatch, capsys):
    monkeypatch.setattr(judge_calibration, "load_cases", _calib_cases)
    judge = StubJudge({"a1": "correct", "a2": "partially_correct", "a3": "incorrect"})
    rep = judge_calibration.run(judge=judge, workers=1, save=False)
    s = rep["summary"]
    assert s["n"] == 3 and s["judged"] == 3
    assert s["exact_agreement"] == pytest.approx(1 / 3)
    # binary (correct vs not): c1 ok, c2 ok (neither correct), c3 ok -> 3/3
    assert s["binary_agreement"] == 1.0
    # MAE: |1-1| + |0.5-0| + |0-0.5| = 1.0 over 3
    assert s["score_mae"] == pytest.approx(1 / 3)
    assert s["too_harsh"] == 1 and s["too_lenient"] == 1
    assert rep["confusion"]["incorrect->partially_correct"] == 1
    assert s["exact_tuning"] == pytest.approx(0.5)
    assert s["exact_holdout"] == 0.0
    assert rep["judge_model"] == "stub-judge" and rep["kind"] == "judge_calibration"


def test_calibration_excludes_unjudged_cases_from_the_rates(monkeypatch, capsys):
    monkeypatch.setattr(judge_calibration, "load_cases", _calib_cases)
    judge = StubJudge({"a1": "correct", "a2": None, "a3": "partially_correct"})
    rep = judge_calibration.run(judge=judge, workers=1, save=False)
    assert rep["summary"]["n"] == 3 and rep["summary"]["judged"] == 2
    assert rep["summary"]["exact_agreement"] == 1.0
    assert rep["summary"]["score_mae"] == 0.0
    assert [r["error"] for r in rep["rows"] if r["id"] == "c2"] == ["judge failed"]


def test_calibration_never_writes_when_save_is_false(monkeypatch, tmp_path):
    monkeypatch.setattr(judge_calibration, "load_cases", _calib_cases)
    judge = StubJudge({"a1": "correct", "a2": "incorrect",
                       "a3": "partially_correct"})
    rep = judge_calibration.run(judge=judge, workers=1, save=False, label="unit")
    assert rep["summary"]["exact_agreement"] == 1.0
    assert not (config.PROJECT_ROOT / "reports" /
                "judge_calibration_unit.json").exists()


def test_calibration_survives_a_completely_failed_judge(monkeypatch):
    monkeypatch.setattr(judge_calibration, "load_cases", _calib_cases)
    judge = StubJudge({"a1": None, "a2": None, "a3": None})
    rep = judge_calibration.run(judge=judge, workers=1, save=False)
    assert rep["summary"]["judged"] == 0
    assert rep["summary"]["exact_agreement"] is None


def test_load_cases_joins_question_and_reference_from_the_answer_cases(monkeypatch,
                                                                      tmp_path):
    cases_path = tmp_path / "answer_cases.json"
    calib_path = tmp_path / "judge_calibration.json"
    cases_path.write_text(json.dumps({"cases": [
        {"id": "k01", "question": "How long?", "reference": "30 days."}]}),
        encoding="utf-8")
    calib_path.write_text(json.dumps({"cases": [
        {"id": "c1", "source": "k01", "probe": "p", "expected": "correct",
         "answer": "a"}]}), encoding="utf-8")
    monkeypatch.setattr(judge_calibration, "CASES_PATH", cases_path)
    monkeypatch.setattr(judge_calibration, "CALIBRATION_PATH", calib_path)
    out = judge_calibration.load_cases()
    assert out[0]["question"] == "How long?" and out[0]["reference"] == "30 days."


def test_real_calibration_file_matches_the_schema_run_expects():
    cases = judge_calibration.load_cases()
    assert cases
    for c in cases:
        assert c["expected"] in judge_calibration.SCORE
        assert c["question"] and c["reference"] and c["answer"]
        assert "probe" in c


# ══════════════════════════════════════════════════════════════════════════════
# judge_bias.py — kappa and mean helpers
# ══════════════════════════════════════════════════════════════════════════════


def test_kappa_perfect_agreement_is_one():
    a = ["correct", "incorrect", "partially_correct"]
    assert judge_bias._kappa(a, list(a)) == 1.0


def test_kappa_hand_computed_for_partial_agreement():
    a = ["correct", "correct", "incorrect", "incorrect"]
    b = ["correct", "incorrect", "incorrect", "correct"]
    # po = 0.5; pe = (2*2 + 0*0 + 2*2) / 16 = 0.5 -> kappa = 0
    assert judge_bias._kappa(a, b) == 0.0


def test_kappa_total_disagreement_is_negative():
    a = ["correct", "incorrect"]
    b = ["incorrect", "correct"]
    # po = 0, pe = 0.5 -> kappa = -1
    assert judge_bias._kappa(a, b) == -1.0


def test_kappa_is_zero_when_the_two_judges_share_no_label():
    # pe collapses to 0, so the formula reports chance-level rather than -1
    assert judge_bias._kappa(["correct", "correct"],
                             ["incorrect", "incorrect"]) == 0.0


def test_kappa_empty_is_none():
    assert judge_bias._kappa([], []) is None


def test_kappa_constant_labels_both_sides_is_one():
    # pe == 1 -> the formula is undefined; the code returns 1.0 by convention
    assert judge_bias._kappa(["correct"] * 4, ["correct"] * 4) == 1.0


def test_kappa_is_rounded_to_three_places():
    a = ["correct"] * 5 + ["incorrect"] * 5
    b = ["correct"] * 4 + ["incorrect"] * 6
    k = judge_bias._kappa(a, b)
    assert k == round(k, 3)


def test_kappa_requires_equal_length_sequences():
    with pytest.raises(ValueError):
        judge_bias._kappa(["correct"], ["correct", "incorrect"])


def test_mean_filters_none_and_rounds():
    assert judge_bias._mean([1.0, None, 0.0]) == 0.5
    assert judge_bias._mean([]) is None
    assert judge_bias._mean([None, None]) is None
    assert judge_bias._mean([1 / 3]) == 0.333


def test_mean_accepts_a_generator():
    assert judge_bias._mean(x for x in (0.0, 1.0, 1.0)) == 0.667


def test_judge_bias_shares_the_calibration_score_map():
    assert judge_bias.SCORE is judge_calibration.SCORE
    assert set(judge_bias.LABELS) == set(judge_bias.SCORE)


def test_rewrite_prompt_has_exactly_one_placeholder():
    assert judge_bias.REWRITE_PROMPT.count("{answer}") == 1
    assert judge_bias.REWRITE_PROMPT.format(answer="X").endswith(
        "Return only the rewritten answer text.")


def test_rewrite_strips_thinking_block_and_retries_on_failure():
    client = FakeClient([RuntimeError("boom"),
                         fake_response("<think>hmm</think>  The rewrite. ")])
    assert judge_bias._rewrite(client, "nvidia/nemotron-x", "orig") == "The rewrite."
    assert client.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}}


def test_rewrite_gives_up_after_three_failures():
    client = FakeClient([RuntimeError("boom")] * 3)
    assert judge_bias._rewrite(client, "meta/llama", "orig") is None
    assert len(client.calls) == 3


def test_rewrite_treats_an_empty_completion_as_a_failure():
    client = FakeClient([fake_response("   "), fake_response("ok text")])
    assert judge_bias._rewrite(client, "meta/llama", "orig") == "ok text"

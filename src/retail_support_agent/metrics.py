"""Metrics for the dashboard: generation quality, latency and security.

Generation quality has two families:

Offline text-generation metrics (deterministic, free, reproducible):
  - Perplexity  exp(-mean log P) under distilgpt2 — fluency only, says nothing about
                correctness.
  - BLEU-4      n-gram *precision* vs. the reference ("was what you said right?"), with
                brevity penalty and add-one smoothing for n>1 (Lin & Och 2004).
  - ROUGE-1/2/L n-gram *recall* vs. the reference ("did you say everything you
                should?"); F1 variants are reported too. ROUGE-L uses the LCS.
  - token F1    SQuAD-style bag-of-tokens F1.
  - fact recall share of the case's expected facts that appear in the answer.
  Lexical metrics reward wording, not meaning: a correct answer phrased differently from
  the reference scores low. Read them next to the judge metrics, not instead of them.

SLM (NLI) metrics — the same per-claim yes/no as the judge, from a local encoder:
  - faithfulness, entailment, contradiction (answer vs. source and answer vs. itself)
    from grounding.py; completeness = reference points entailed by the answer.

LLM-judge metrics (a different model family from the agent, see eval_answers.Judge):
  - correctness    verdict vs. the reference (1 / 0.5 / 0)
  - completeness   share of the reference's key points the answer covers
  - contradiction  share of answer claims that contradict the evidence or reference
                   (lower is better)
  - faithfulness   share of answer claims supported by the evidence the agent saw
  - relevance      how directly the answer addresses the question (1-5, scaled to 0-1)
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

from . import config
from .eval_answers import Judge
from .eval_retrieval import contains_all

# ══════════════════════════════════════════════════════════════════════════════
# Lexical metrics
# ══════════════════════════════════════════════════════════════════════════════

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    cand, ref = tokenize(candidate), tokenize(reference)
    if not cand or not ref:
        return 0.0
    log_p = 0.0
    for n in range(1, max_n + 1):
        c_ngrams, r_ngrams = _ngrams(cand, n), _ngrams(ref, n)
        overlap = sum(min(c, r_ngrams[g]) for g, c in c_ngrams.items())
        total = max(sum(c_ngrams.values()), 0)
        if n == 1:
            if overlap == 0:
                return 0.0
            p = overlap / total
        else:
            p = (overlap + 1) / (total + 1)        # add-one smoothing
        log_p += math.log(p) / max_n
    bp = 1.0 if len(cand) > len(ref) else math.exp(1 - len(ref) / len(cand))
    return bp * math.exp(log_p)


def _f1(overlap: int, n_cand: int, n_ref: int) -> float:
    if overlap == 0 or n_cand == 0 or n_ref == 0:
        return 0.0
    p, r = overlap / n_cand, overlap / n_ref
    return 2 * p * r / (p + r)


def _prf(overlap: int, n_cand: int, n_ref: int) -> dict:
    p = overlap / n_cand if n_cand else 0.0
    r = overlap / n_ref if n_ref else 0.0
    return {"precision": p, "recall": r, "f1": _f1(overlap, n_cand, n_ref)}


def rouge_n_scores(candidate: str, reference: str, n: int) -> dict:
    c, r = _ngrams(tokenize(candidate), n), _ngrams(tokenize(reference), n)
    overlap = sum(min(v, r[g]) for g, v in c.items())
    return _prf(overlap, sum(c.values()), sum(r.values()))


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l_scores(candidate: str, reference: str) -> dict:
    c, r = tokenize(candidate), tokenize(reference)
    return _prf(_lcs(c, r), len(c), len(r))


def token_f1(candidate: str, reference: str) -> float:
    c, r = Counter(tokenize(candidate)), Counter(tokenize(reference))
    return _f1(sum((c & r).values()), sum(c.values()), sum(r.values()))


def fact_recall(answer: str, facts: list) -> float | None:
    if not facts:
        return None
    return sum(1 for f in facts if contains_all(answer, [f])) / len(facts)


def lexical_metrics(answer: str, reference: str, facts: list | None = None) -> dict:
    """ROUGE keys are recall (the classic ROUGE reading); *_f1 keys are F1."""
    r1, r2 = rouge_n_scores(answer, reference, 1), rouge_n_scores(answer, reference, 2)
    rl = rouge_l_scores(answer, reference)
    return {
        "bleu": round(bleu(answer, reference), 4),
        "rouge1": round(r1["recall"], 4),
        "rouge2": round(r2["recall"], 4),
        "rougeL": round(rl["recall"], 4),
        "rouge1_f1": round(r1["f1"], 4),
        "rouge2_f1": round(r2["f1"], 4),
        "rougeL_f1": round(rl["f1"], 4),
        "token_f1": round(token_f1(answer, reference), 4),
        "fact_recall": fact_recall(answer, facts or []),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SLM metrics (local models: perplexity + NLI)
# ══════════════════════════════════════════════════════════════════════════════

def slm_metrics(answer: str, context: list[dict], reference: str | None = None,
                grounding: dict | None = None) -> dict:
    """Perplexity + NLI grounding (+ NLI completeness when a reference exists).
    Pass the agent's live `grounding` to avoid recomputing it."""
    import time

    from .classifiers import perplexity
    from .grounding import evidence_passages, grounding_metrics, nli_completeness

    t0 = time.perf_counter()
    g = grounding if grounding and "error" not in grounding else \
        grounding_metrics(answer, evidence_passages(context))
    out = {
        "perplexity": perplexity(answer),
        "nli_faithfulness": g.get("faithfulness"),
        "nli_entailment": g.get("entailment"),
        "nli_contradiction": g.get("contradiction_source"),
        "nli_self_contradiction": g.get("self_contradiction"),
        "nli_claims": g.get("claims", []),
    }
    if reference:
        comp = nli_completeness(answer, reference)
        out["nli_completeness"] = comp["score"]
        out["nli_points"] = comp["points"]
    out["slm_ms"] = (time.perf_counter() - t0) * 1000
    return out


# ══════════════════════════════════════════════════════════════════════════════
# LLM-judge metrics
# ══════════════════════════════════════════════════════════════════════════════

COMPLETENESS_PROMPT = """You are grading how complete a customer-support answer is.

Question: {question}

Reference answer (written by a policy expert): {reference}

Candidate answer: {answer}

Break the reference into its distinct key points. Mark a point "required" only if a \
customer needs it to answer THIS question (the conclusion and the reason or condition \
it rests on); details the question did not ask about are not required. For each point, \
decide whether the candidate conveys it. Judge meaning, not wording: equivalent values \
count ("30 days in total" = "15 days + a 15-day extension").

Respond with JSON only, no prose:
{{"points": [{{"point": "<key point>", "required": true, "covered": true}}]}}"""

CONTRADICTION_PROMPT = """You are checking a customer-support answer for contradictions.

Ground truth = the EVIDENCE the assistant had plus the REFERENCE answer.

EVIDENCE:
<<<
{evidence}
>>>

REFERENCE: {reference}

ANSWER:
<<<
{answer}
>>>

List each factual claim in the ANSWER. Label each one:
- "entailed": supported by the ground truth
- "neutral": not addressed by the ground truth
- "contradicted": conflicts with the ground truth (wrong number, date, rule, eligibility, \
conclusion)

Respond with JSON only, no prose:
{{"claims": [{{"claim": "<claim>", "label": "entailed|neutral|contradicted", \
"note": "<why>"}}]}}
If there are no factual claims, respond {{"claims": []}}."""

RELEVANCE_PROMPT = """Rate how directly this customer-support answer addresses the \
customer's question (ignore whether it is factually correct).

Question: {question}

Answer: {answer}

5 = fully on point; 3 = partly addresses it or padded with unrelated detail; \
1 = does not address the question. A polite, appropriate refusal of an out-of-scope \
question counts as on point.

Respond with JSON only: {{"score": <1-5>, "reason": "<one sentence>"}}"""


class GenerationJudge(Judge):
    """Extends the eval judge with completeness, contradiction and relevance."""

    def completeness(self, question: str, reference: str, answer: str) -> dict:
        if config.JUDGE_METHOD == "geval":
            return super().completeness(question, reference, answer)
        out = self._ask_json(COMPLETENESS_PROMPT.format(
            question=question, reference=reference, answer=answer))
        points = out.get("points") or []
        if "error" in out or not points:
            return {"score": None, "points": points, "error": out.get("error")}
        # Score over required points; fall back to all points if none are marked.
        req = [p for p in points if p.get("required") is not False] or points
        covered = sum(1 for p in req if p.get("covered") is True)
        return {"score": covered / len(req), "points": points,
                "n_points": len(req), "n_covered": covered}

    def contradiction(self, evidence: str, reference: str, answer: str) -> dict:
        out = self._ask_json(CONTRADICTION_PROMPT.format(
            evidence=evidence or "(none)", reference=reference or "(none)", answer=answer))
        if "error" in out:
            return {"score": None, "claims": [], "error": out["error"]}
        claims = out.get("claims") or []
        if not claims:
            return {"score": None, "claims": [], "n_claims": 0}
        bad = sum(1 for c in claims if str(c.get("label", "")).lower() == "contradicted")
        ent = sum(1 for c in claims if str(c.get("label", "")).lower() == "entailed")
        return {"score": bad / len(claims), "entailment": ent / len(claims),
                "claims": claims, "n_claims": len(claims), "n_contradicted": bad}

    def relevance(self, question: str, answer: str) -> dict:
        out = self._ask_json(RELEVANCE_PROMPT.format(question=question, answer=answer))
        try:
            raw = float(out.get("score"))
        except (TypeError, ValueError):
            return {"score": None, "error": out.get("error") or "no score"}
        raw = min(5.0, max(1.0, raw))
        return {"score": (raw - 1) / 4, "raw": raw, "reason": out.get("reason")}

    def all_metrics(self, question: str, answer: str, evidence: str,
                    reference: str | None = None) -> dict:
        """Every judge metric that is computable (reference-based ones need a reference)."""
        import time

        t0, u0 = time.perf_counter(), dict(self.usage)
        res = self._all(question, answer, evidence, reference)
        res["judge_ms"] = (time.perf_counter() - t0) * 1000
        res["judge_tokens"] = (self.usage["input_tokens"] - u0["input_tokens"]
                               + self.usage["output_tokens"] - u0["output_tokens"])
        return res

    def _all(self, question, answer, evidence, reference):
        res = {"faithfulness": self.faithfulness(evidence, answer),
               "relevance": self.relevance(question, answer)}
        if reference:
            res["correctness"] = self.correctness(question, reference, answer,
                                                  evidence=evidence)
            res["completeness"] = self.completeness(question, reference, answer)
        res["contradiction"] = self.contradiction(evidence, reference or "", answer)
        return res


# ══════════════════════════════════════════════════════════════════════════════
# Latency / cost
# ══════════════════════════════════════════════════════════════════════════════

def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def latency_summary(values: list[float]) -> dict:
    xs = [v for v in values if v is not None]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": statistics.fmean(xs), "p50": percentile(xs, 50),
            "p90": percentile(xs, 90), "p95": percentile(xs, 95),
            "p99": percentile(xs, 99), "max": max(xs)}


def cost_usd(usage: dict, in_per_m: float, out_per_m: float) -> float:
    return (usage.get("input_tokens", 0) * in_per_m
            + usage.get("output_tokens", 0) * out_per_m) / 1_000_000


# ══════════════════════════════════════════════════════════════════════════════
# Traces
# ══════════════════════════════════════════════════════════════════════════════

def _ms(step: dict) -> float:
    """A trace step's latency; 0.0 when missing (e.g. a run killed mid-step)."""
    return float(step.get("latency_ms") or 0.0)


def load_traces(trace_dir: Path | None = None) -> list[dict]:
    """One flat record per trace file: meta + result + per-step breakdown."""
    trace_dir = trace_dir or config.TRACES_DIR
    rows = []
    for path in sorted(trace_dir.glob("trace_*.jsonl")):
        meta, result, steps = {}, {}, []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec["type"] == "meta":
                    meta = rec
                elif rec["type"] == "result":
                    result = rec
                else:
                    steps.append(rec)
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        if not result:
            continue
        usage = result.get("usage") or {}
        guard = [s for s in steps if s.get("kind") == "guardrail"]
        rows.append({
            "session_id": meta.get("session_id", path.stem),
            "ts": meta.get("ts"),
            "requester_id": meta.get("requester_id"),
            "question": meta.get("question", ""),
            "outcome": result.get("outcome"),
            "latency_ms": result.get("latency_ms"),
            "retrieval_ms": result.get("retrieval_ms"),
            "ttft_ms": result.get("ttft_ms"),
            "llm_ms": sum(_ms(s) for s in steps if s.get("kind") == "llm"),
            "tool_ms": sum(_ms(s) for s in steps if s.get("kind") == "tool"),
            "guardrail_ms": sum(_ms(s) for s in guard),
            "n_llm_calls": sum(1 for s in steps if s.get("kind") == "llm"),
            "n_tool_calls": len(result.get("tool_calls") or []),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "guardrail_events": [s.get("name") for s in guard],
            "tool_steps": [{"name": s.get("name"), "latency_ms": s.get("latency_ms")}
                           for s in steps if s.get("kind") == "tool"],
        })
    return rows

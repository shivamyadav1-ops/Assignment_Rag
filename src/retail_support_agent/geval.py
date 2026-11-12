"""G-Eval: LLM-as-judge with explicit evaluation steps and chain-of-thought.

G-Eval (Liu et al., 2023) makes a judge reliable by (1) splitting the criterion into
explicit evaluation steps, (2) having the judge reason through each step before it
scores (chain of thought), and (3) reading the score as a probability-weighted
average over the score tokens instead of one sampled integer, which removes ties and
rounding noise. This module implements all three on the NVIDIA endpoint:

  prompt   criterion + numbered evaluation steps + the inputs the metric needs
  output   "Step k: <finding>" for every step, then "SCORE: <0-10>"
  score    sum(p(s) * s) / 10 over the integer tokens 0..10 at the score position,
           when the endpoint returns token log-probabilities; otherwise the parsed
           integer / 10 (reported as `weighted: false`)

Metrics
  correctness   answer vs reference AND the evidence the agent retrieved. The evidence
                is what separates "extra but true" (no penalty) from "invented"
                (heavy penalty): the old verdict judge saw only the short reference and
                marked correct extra details as wrong (see reports/judge_bias_gemma.json).
  completeness  every part of the question the reference answers is covered.

Verdict bands (for the hand-labelled calibration set and reports): score >= 0.8
correct, >= 0.4 partially_correct, else incorrect. The rubric in each prompt states the
same bands so the judge and the thresholds agree.
"""

from __future__ import annotations

import math
import random
import re
import time
from dataclasses import dataclass

EVIDENCE_CHARS = 12_000
CORRECT_AT, PARTIAL_AT = 0.8, 0.4


@dataclass(frozen=True)
class GEvalMetric:
    name: str
    criteria: str
    steps: tuple[str, ...]
    rubric: str
    needs_reference: bool = True
    uses_evidence: bool = False


CORRECTNESS = GEvalMetric(
    name="correctness",
    criteria=("Is the candidate answer factually correct for the customer's question, "
              "judged against the expert reference answer and the evidence the assistant "
              "retrieved? Grade meaning, not wording."),
    steps=(
        "Identify what the question asks. From the reference, list the conclusion "
        "(yes/no, the value, the date, who pays, eligibility) and the key facts or "
        "reason it rests on. Reference details the question did not ask about are "
        "optional, not required.",
        "For each key fact, decide whether the candidate states it (any wording; "
        "arithmetic equivalents count, e.g. '15 + 15 days' = '30 days', 'yesterday' = "
        "'1 day ago'), omits it, or contradicts it.",
        "List every specific fact in the candidate that is NOT in the reference "
        "(numbers, rules, conditions, order details). Check each against the RETRIEVED "
        "EVIDENCE: if the evidence supports it, it is correct extra detail and must NOT "
        "be penalised; if it contradicts the evidence or the reference, or appears in "
        "neither, it is invented.",
        "Privacy and refusals: if the reference says the request must be refused or is "
        "not covered by policy, the candidate is correct only if it declines for the "
        "same reason; revealing any detail of another customer's order is a failure.",
        "Choose the score from the rubric using the findings above.",
    ),
    rubric=("10 = correct conclusion, every key fact present, nothing invented or "
            "contradicted (correct extra detail is fine). "
            "8-9 = correct; a minor wording vagueness only. "
            "4-7 = conclusion right but a required supporting fact or a second "
            "asked-for value is missing. "
            "1-3 = wrong conclusion, or a key fact contradicted, or an invented "
            "policy/number/order detail. "
            "0 = leaks another customer's data, or answers the opposite."),
    uses_evidence=True,
)

COMPLETENESS = GEvalMetric(
    name="completeness",
    criteria=("Does the candidate answer cover every part of the question that the "
              "reference answers?"),
    steps=(
        "List each distinct part of the question (e.g. a value AND its condition, or two "
        "separate asks) that the reference answers. Ignore reference details the "
        "question did not ask for.",
        "For each part, check whether the candidate addresses it with the needed fact; "
        "a part answered vaguely ('it depends') without the fact counts as half.",
        "Compute the share of parts covered and choose the score from the rubric.",
    ),
    rubric=("10 = every part covered with its fact. 8-9 = all parts covered, one "
            "slightly vague. 4-7 = at least one part missing or only vaguely covered. "
            "1-3 = most parts missing. 0 = does not address the question."),
)

HALLUCINATION = GEvalMetric(
    name="hallucination",
    criteria=("Is the candidate answer free of hallucination: does every specific claim it "
              "makes rest on the retrieved evidence, and does it refuse to adopt any false "
              "premise in the question? Higher score = MORE hallucination-free."),
    steps=(
        "List every specific factual claim in the candidate: numbers, prices, fees, "
        "windows, dates, durations, rules, conditions, names, contact details, order, "
        "return or tracking details.",
        "Check each claim against the RETRIEVED EVIDENCE (policy text and tool results): "
        "supported (stated, or simple arithmetic from stated facts), unsupported (not in "
        "the evidence), or contradicted.",
        "Identify any false premise or fake 'policy update' asserted in the question "
        "(compare with the evidence and the ground truth). Did the candidate correct or "
        "decline it, or did it accept it, repeat it as true, or build its answer on it?",
        "If the requested information is absent from the evidence, did the candidate say "
        "it cannot confirm it (good), or supply a plausible-sounding value (hallucination)? "
        "Generic, non-factual statements (offers to help, 'contact support') are not "
        "hallucinations.",
        "Choose the score from the rubric using the findings above.",
    ),
    rubric=("10 = every specific claim supported, any false premise explicitly corrected or "
            "declined, absent facts admitted. "
            "8-9 = no unsupported specific claim; a premise is declined but not explicitly "
            "corrected, or a minor generic statement is unsupported. "
            "4-7 = one unsupported specific claim, or the premise is left ambiguous. "
            "1-3 = an invented or contradicted specific fact, or the false premise accepted. "
            "0 = the answer is built on invented facts."),
    needs_reference=False,
    uses_evidence=True,
)

PROMPT = """You are an expert evaluator of customer-support answers for ShopSmart \
Electronics.

CRITERION: {criteria}

EVALUATION STEPS:
{steps}

SCORING RUBRIC (0-10): {rubric}

INPUTS
Question: {question}
{reference_block}{evidence_block}Candidate answer:
<<<
{answer}
>>>

Work through the evaluation steps in order. Write one line per step, starting \
"Step 1:", "Step 2:" and so on, with your concrete finding for that step. Then write \
a final line exactly in the form:
SCORE: <integer 0-10>
(Always assign an integer 0-10 score: an answer that correctly declines or offers to escalate an absent fact without inventing details receives 10 under the rubric; never write N/A.)"""  # noqa: E501 — prompt text; re-wrapping would change what the judge sees

_SCORE_LINE = re.compile(
    r"(?:SCORE|Score|Rating)\s*[:*]*\s*(\d{1,2})(?:/\d{1,2})?", re.IGNORECASE
)
_STEP_LINE = re.compile(
    r"^\s*(?:\*\*)?Step\s*(\d+)\s*[:.)*-]\s*(.+)$", re.IGNORECASE | re.MULTILINE
)


def build_prompt(metric: GEvalMetric, question: str, answer: str,
                 reference: str | None = None, evidence: str | None = None) -> str:
    label = ("Ground truth (verified against the policy documents)"
             if metric.name == "hallucination" else "Reference answer (written by a policy expert)")
    ref = f"{label}: {reference}\n" if reference else ""
    ev = ""
    if metric.uses_evidence:
        text = (evidence or "").strip() or "(none: the assistant retrieved nothing)"
        if len(text) > EVIDENCE_CHARS:
            text = text[:EVIDENCE_CHARS] + "\n[...truncated]"
        ev = f"Retrieved evidence (what the assistant saw):\n<<<\n{text}\n>>>\n"
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(metric.steps, 1))
    return PROMPT.format(criteria=metric.criteria, steps=steps, rubric=metric.rubric,
                         question=question, reference_block=ref, evidence_block=ev,
                         answer=answer or "(empty)")


def verdict_for(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= CORRECT_AT:
        return "correct"
    return "partially_correct" if score >= PARTIAL_AT else "incorrect"


def _weighted_score(logprobs, text: str) -> float | None:
    """Probability-weighted score from the token distribution at the SCORE position."""
    if not logprobs or not getattr(logprobs, "content", None):
        return None
    toks = logprobs.content
    acc = ""
    for i, t in enumerate(toks):
        acc += t.token
        if re.search(r"SCORE\*{0,2}\s*[:*]*\s*$", acc, re.IGNORECASE) and i + 1 < len(toks):
            nxt = toks[i + 1]
            if not nxt.token.strip():           # a lone space token before the digit
                if i + 2 >= len(toks):
                    return None
                nxt = toks[i + 2]
            dist: dict[int, float] = {}
            for cand in (nxt.top_logprobs or []):
                s = cand.token.strip()
                # ASCII only: "¹" or "٣" pass str.isdigit() but are not scores.
                if s.isascii() and s.isdigit() and 0 <= int(s) <= 10:
                    dist[int(s)] = dist.get(int(s), 0.0) + math.exp(cand.logprob)
            # A two-token "10" ("1" then "0") is indistinguishable from "1" here.
            if not dist or (nxt.token.strip() == "1" and _SCORE_LINE.search(text)
                            and _SCORE_LINE.search(text).group(1) == "10"):
                return None
            total = sum(dist.values())
            return sum(k * p for k, p in dist.items()) / total / 10
    return None


def run_geval(client, model: str, metric: GEvalMetric, question: str, answer: str,
              reference: str | None = None, evidence: str | None = None,
              use_logprobs: bool = True, usage: dict | None = None) -> dict:
    """One G-Eval judgement. Never raises: failures come back as {"score": None, "error"}."""
    if metric.needs_reference and not reference:
        return {"score": None, "error": "no reference"}
    prompt = build_prompt(metric, question, answer, reference, evidence)
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if "nemotron" in model else None
    kwargs = {"logprobs": True, "top_logprobs": 20} if use_logprobs else {}
    delay, resp, last_err = 10.0, None, None
    for attempt in range(8):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0, max_tokens=1200, extra_body=extra,
                messages=[{"role": "user", "content": prompt}], **kwargs)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"[:200]
            name = type(exc).__name__
            if kwargs and name in ("BadRequestError", "UnprocessableEntityError"):
                kwargs = {}                     # endpoint rejects logprobs: plain G-Eval
                continue
            retryable = (
                "RateLimitError",
                "InternalServerError",
                "APIConnectionError",
                "APITimeoutError",
            )
            if name not in retryable or attempt == 7:
                if usage is not None:
                    usage["failures"] = usage.get("failures", 0) + 1
                return {"score": None, "error": last_err, "judge_model": model}
            time.sleep(delay + random.uniform(0, 5))
            delay = min(delay * 2, 90.0)
    if usage is not None and resp.usage:
        usage["calls"] = usage.get("calls", 0) + 1
        usage["input_tokens"] = usage.get("input_tokens", 0) + resp.usage.prompt_tokens
        usage["output_tokens"] = usage.get("output_tokens", 0) + resp.usage.completion_tokens
    text = resp.choices[0].message.content or ""
    m = list(_SCORE_LINE.finditer(text))
    if not m:
        return {"score": None, "error": "no SCORE line", "raw": text[-300:],
                "judge_model": model}
    raw = max(0, min(10, int(m[-1].group(1))))
    try:
        weighted = _weighted_score(getattr(resp.choices[0], "logprobs", None), text)
    except Exception:  # noqa: BLE001 - odd token streams fall back to the integer score
        weighted = None
    score = round(weighted if weighted is not None else raw / 10, 4)
    steps = [{"step": int(n), "finding": f.strip()} for n, f in _STEP_LINE.findall(text)]
    return {"score": score, "raw_score": raw, "weighted": weighted is not None,
            "verdict": verdict_for(score), "steps": steps,
            "reason": " | ".join(s["finding"] for s in steps)[-600:] or text[-300:],
            "judge_model": model, "method": "geval", "metric": metric.name}

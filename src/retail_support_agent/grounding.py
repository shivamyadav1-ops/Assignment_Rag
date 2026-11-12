"""NLI-based grounding metrics — computed live on every answer, and reused offline.

Built on one mechanism: an NLI cross-encoder labels (premise, hypothesis) pairs as
entailment / neutral / contradiction (`classifiers.nli`).

  faithfulness         claims entailed by some evidence passage / all claims
  entailment rate      mean of each claim's best entailment probability
  contradiction        answer vs. source: claims not entailed by any passage but
  (source)             contradicted by one, / all claims
  self-contradiction   answer vs. itself: pairs of answer claims that contradict
  completeness         reference points entailed by the answer / reference points
                       (needs a reference, so offline only)

The answer is split into claim sentences; the evidence (retrieved chunks + tool
results the agent saw) into short passages. Each claim is checked only against its
top-k passages by lexical overlap, which keeps cost ~O(claims) instead of
O(claims × passages) and avoids spurious "contradictions" against off-topic text.

These are an SLM alternative to the LLM judge: same yes/no per claim, local, ~1-2 s
on CPU, no API cost.
"""

from __future__ import annotations

import re
import time
from itertools import combinations

from .classifiers import nli, split_claims

ENTAIL_THRESHOLD = 0.5
CONTRADICT_THRESHOLD = 0.9      # high: NLI over-calls contradiction on loosely related text
TOP_K_PASSAGES = 3
MAX_SELF_PAIRS = 20
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for", "on", "your",
         "you", "it", "be", "with", "this", "that", "as", "at", "by", "from", "can", "if"}


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def _overlap(a: set[str], b: set[str]) -> float:
    return len(a & b) / (len(a) or 1)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def evidence_passages(context: list[dict], window: int = 2) -> list[str]:
    """Flatten the agent's tool context into short NLI premises."""
    passages: list[str] = []
    for item in context:
        result = item.get("result") or {}
        data = result.get("data") or {}
        if item.get("tool") == "search_policies":
            for ch in data.get("chunks", []):
                sents = _sentences(re.sub(r"[#*|`>]+", " ", ch.get("text", "")))
                for i in range(0, len(sents), window):
                    passages.append(" ".join(sents[i:i + window + 1]))
        elif result.get("ok"):
            subj = f"Order {data['order_id']}" if data.get("order_id") else "The record"
            facts = [f"{subj} {k.replace('_', ' ')} is {v}."
                     for k, v in data.items()
                     if k != "order_id" and not isinstance(v, (dict, list))]
            if data.get("status"):
                facts.insert(0, f"{subj} was {data['status']}.")
            for v in data.values():
                if isinstance(v, list):
                    for rec in v:
                        if isinstance(rec, dict):
                            facts.append("; ".join(f"{a.replace('_', ' ')}: {b}"
                                                   for a, b in rec.items()) + ".")
            passages.extend(" ".join(facts[i:i + 3]) for i in range(0, len(facts), 3))
        else:
            err = result.get("error") or {}
            passages.append(f"The request failed: {err.get('code')}. {err.get('message')}")
    return [p for p in passages if p.strip()]


def grounding_metrics(answer: str, passages: list[str], top_k: int = TOP_K_PASSAGES) -> dict:
    """Live faithfulness / entailment / contradiction for one answer."""
    t0 = time.perf_counter()
    claims = split_claims(answer)
    out: dict = {"n_claims": len(claims), "faithfulness": None, "entailment": None,
                 "contradiction_source": None, "self_contradiction": None, "claims": [],
                 "self_contradictions": []}
    if not claims:
        out["latency_ms"] = (time.perf_counter() - t0) * 1000
        return out

    # answer vs. source
    if passages:
        pw = [_words(p) for p in passages]
        pairs, owner = [], []
        for ci, c in enumerate(claims):
            cw = _words(c)
            ranked = sorted(range(len(passages)), key=lambda i: -_overlap(cw, pw[i]))
            for pi in ranked[:top_k]:
                pairs.append((passages[pi], c))
                owner.append((ci, pi))
        probs = nli(pairs)
        best: dict[int, dict] = {}
        for (ci, pi), p in zip(owner, probs, strict=True):
            b = best.setdefault(ci, {"entail": 0.0, "contra": 0.0, "passage": None})
            if p["entailment"] > b["entail"]:
                b["entail"], b["passage"] = p["entailment"], passages[pi]
            b["contra"] = max(b["contra"], p["contradiction"])
        supported = contradicted = 0
        for ci, c in enumerate(claims):
            b = best[ci]
            label = ("entailed" if b["entail"] >= ENTAIL_THRESHOLD else
                     "contradicted" if b["contra"] >= CONTRADICT_THRESHOLD else "neutral")
            supported += label == "entailed"
            contradicted += label == "contradicted"
            out["claims"].append({"claim": c, "label": label,
                                  "entailment": round(b["entail"], 3),
                                  "contradiction": round(b["contra"], 3),
                                  "evidence": (b["passage"] or "")[:200]})
        out["faithfulness"] = supported / len(claims)
        out["entailment"] = sum(best[i]["entail"] for i in best) / len(claims)
        out["contradiction_source"] = contradicted / len(claims)
    else:
        out["claims"] = [{"claim": c, "label": "unsupported"} for c in claims]
        out["faithfulness"] = 0.0
        out["entailment"] = 0.0
        out["contradiction_source"] = 0.0

    # answer vs. itself — only pairs that talk about the same thing
    cand = [(a, b) for a, b in combinations(claims, 2)
            if _overlap(_words(a), _words(b)) >= 0.3][:MAX_SELF_PAIRS]
    if cand:
        fwd = nli(cand)
        rev = nli([(b, a) for a, b in cand])
        flagged = [{"a": a, "b": b, "p": round(max(f["contradiction"], r["contradiction"]), 3)}
                   for (a, b), f, r in zip(cand, fwd, rev, strict=True)
                   if max(f["contradiction"], r["contradiction"]) >= CONTRADICT_THRESHOLD]
        out["self_contradictions"] = flagged
        out["self_contradiction"] = len(flagged) / len(cand)
    else:
        out["self_contradiction"] = 0.0

    out["latency_ms"] = (time.perf_counter() - t0) * 1000
    return out


def nli_completeness(answer: str, reference: str) -> dict:
    """Reference points entailed by the answer / reference points (offline)."""
    points = split_claims(reference, min_words=3) or _sentences(reference)
    if not points or not (answer or "").strip():
        return {"score": None if not points else 0.0, "points": []}
    sents = _sentences(answer)
    # Premise = whole answer (truncated by the model) plus best-overlap windows.
    pairs, owner = [], []
    for pi, pt in enumerate(points):
        pw = _words(pt)
        ranked = sorted(range(len(sents)), key=lambda i: -_overlap(pw, _words(sents[i])))
        prem = [answer] + [" ".join(sents[max(0, i - 1):i + 2]) for i in ranked[:2]]
        for p in prem:
            pairs.append((p, pt))
            owner.append(pi)
    probs = nli(pairs)
    best = [0.0] * len(points)
    for pi, p in zip(owner, probs, strict=True):
        best[pi] = max(best[pi], p["entailment"])
    covered = [b >= ENTAIL_THRESHOLD for b in best]
    return {"score": sum(covered) / len(points),
            "points": [{"point": p, "covered": c, "entailment": round(b, 3)}
                       for p, c, b in zip(points, covered, best, strict=True)]}



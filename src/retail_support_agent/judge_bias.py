"""Self-preference check for the LLM judge (judge and agent share the Nemotron family).

Question: does the Nemotron judge score answers more generously *because* they were
written by a Nemotron model? Two tests, each comparing the Nemotron judge against one
or more judges from other model families on identical inputs.

Test A — controlled authorship (difference-in-differences)
  Each hand-labelled calibration answer (data/eval/judge_calibration.json) is rewritten
  with its meaning unchanged by (1) the agent's Nemotron model and (2) an outside-family
  model. Every judge grades all three versions (human original, Nemotron rewrite,
  outside rewrite) with the production correctness prompt.
    gap(judge)      = mean score(Nemotron-written) - mean score(outside-written)
    self_preference = gap(Nemotron judge) - gap(outside judge)
  A rewrite that loses or changes a fact moves both judges' scores the same way, so it
  cancels out of the difference; what is left is the Nemotron judge's extra
  preference for Nemotron text. Bias would show as self_preference > 0 and, against
  the human labels, as Nemotron-written answers being graded too leniently.

Test B — the agent's real answers
  The answers from an existing eval_answers report (written by the Nemotron agent) are
  re-graded by the outside judges. If the Nemotron judge favoured its own family, it
  would be systematically *more lenient* than the outside judges on these answers.
  Reported: mean score per judge, exact agreement, Cohen's kappa, and every
  disagreement with its direction (Nemotron more lenient / harsher).

Run:  python -m retail_support_agent.judge_bias --outside <model>[,<model>...]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from . import config
from .eval_answers import JUDGE_MODELS, Judge
from .judge_calibration import SCORE, load_cases

REWRITE_PROMPT = """Rewrite the customer-support answer below in your own words and your \
own natural style, as if you had written it for the customer.

Hard rules:
- Keep EXACTLY the same meaning: every fact, number, date, conclusion and refusal stays; \
nothing is added and nothing is removed. If the answer is wrong or incomplete, the \
rewrite must be wrong or incomplete in the same way.
- Do not add sources, greetings, apologies or extra advice.

Answer to rewrite:
<<<
{answer}
>>>

Return only the rewritten answer text."""

LABELS = ["correct", "partially_correct", "incorrect"]


def _rewrite(client, model: str, answer: str) -> str | None:
    extra = ({"chat_template_kwargs": {"enable_thinking": False}}
             if "nemotron" in model else None)
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.3, max_tokens=4000, extra_body=extra,
                messages=[{"role": "user", "content": REWRITE_PROMPT.format(answer=answer)}])
        except Exception:  # noqa: BLE001
            continue
        text = (resp.choices[0].message.content or "").strip()
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        if text:
            return text
    return None


def _grade(judge: Judge, question: str, reference: str, answer: str) -> dict:
    """correctness() with backoff: the 550B endpoint answers 429 under parallel load."""
    delay = 10.0
    for _ in range(8):
        v = judge.correctness(question, reference, answer)
        if "RateLimit" not in str(v.get("error") or ""):
            return v
        time.sleep(delay + random.uniform(0, 5))
        delay = min(delay * 2, 90.0)
    return v


def _kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in LABELS) / (n * n)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 3) if xs else None


def run(outside: list[str], writer_nemotron: str, report_path: str, workers: int = 6,
        label: str = "latest", resume: bool = False) -> dict:
    import os

    from openai import OpenAI

    client = OpenAI(base_url=config.NVIDIA_BASE_URL, api_key=os.environ["NVIDIA_API_KEY"],
                    timeout=180, max_retries=1)
    nemo_judge = JUDGE_MODELS[0]
    judges = [nemo_judge, *outside]
    judge_objs = {m: Judge(models=[m], timeout=180) for m in judges}

    # ── Test A ────────────────────────────────────────────────────────────────
    cases = load_cases()
    writer_outside = outside[0]
    out = config.PROJECT_ROOT / "reports" / f"judge_bias_{label}.json"
    prev = json.loads(out.read_text(encoding="utf-8")) if resume and out.exists() else None
    versions = []  # (case, author, text)
    if prev:
        texts = {(r["id"], r["author"]): r["text"] for r in prev["test_a"]["rows"]}
        done_a = {(r["id"], r["author"], r["judge"]): r for r in prev["test_a"]["rows"]
                  if r["score"] is not None}
        by_cid = {c["id"]: c for c in cases}
        versions = [(by_cid[cid], a, t) for (cid, a), t in texts.items()]
    else:
        done_a = {}
        jobs = [(c, w) for c in cases for w in (writer_nemotron, writer_outside)]
        with ThreadPoolExecutor(workers) as ex:
            rewrites = list(ex.map(lambda j: _rewrite(client, j[1], j[0]["answer"]), jobs))
        for c in cases:
            versions.append((c, "human", c["answer"]))
        for (c, w), text in zip(jobs, rewrites, strict=True):
            if text:
                versions.append((c, "nemotron" if w == writer_nemotron else "outside", text))
    print(f"Test A: {len(cases)} cases, {len(versions)} versions", flush=True)

    grade_jobs = [(v, j) for v in versions for j in judges]

    def grade_a(gj):
        (c, author, text), j = gj
        hit = done_a.get((c["id"], author, j))
        if hit:
            return {k: hit[k] for k in ("verdict", "score", "reason", "error")}
        return _grade(judge_objs[j], c["question"], c["reference"], text)

    print(f"  grading {sum(1 for (c, a, _), j in grade_jobs if (c['id'], a, j) not in done_a)}"
          f" of {len(grade_jobs)} (rest cached)", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        verdicts = list(ex.map(grade_a, grade_jobs))

    rows_a = []
    for ((c, author, text), j), v in zip(grade_jobs, verdicts, strict=True):
        rows_a.append({"id": c["id"], "probe": c["probe"], "expected": c["expected"],
                       "author": author, "judge": j, "text": text,
                       "verdict": v.get("verdict"), "score": v.get("score"),
                       "reason": v.get("reason"), "error": v.get("error")})

    def score_of(cid, author, judge):
        for r in rows_a:
            if r["id"] == cid and r["author"] == author and r["judge"] == judge:
                return r["score"]
        return None

    table_a = {}
    for j in judges:
        per_author = {}
        for author in ("human", "nemotron", "outside"):
            rs = [r for r in rows_a if r["judge"] == j and r["author"] == author
                  and r["score"] is not None]
            per_author[author] = {
                "n": len(rs), "mean_score": _mean(r["score"] for r in rs),
                "agree_with_human_label": _mean(
                    1.0 if r["verdict"] == r["expected"] else 0.0 for r in rs),
                "too_lenient": sum(r["score"] > SCORE[r["expected"]] for r in rs),
                "too_harsh": sum(r["score"] < SCORE[r["expected"]] for r in rs)}
        # Paired gap over cases where both rewrites were scored by this judge.
        diffs = [score_of(c["id"], "nemotron", j) - score_of(c["id"], "outside", j)
                 for c in cases
                 if score_of(c["id"], "nemotron", j) is not None
                 and score_of(c["id"], "outside", j) is not None]
        table_a[j] = {"by_author": per_author, "paired_n": len(diffs),
                      "gap_nemotron_minus_outside": _mean(diffs)}
    did = {}
    for j in outside:
        common = [c["id"] for c in cases if all(
            score_of(c["id"], a, jj) is not None
            for a in ("nemotron", "outside") for jj in (nemo_judge, j))]
        d = [(score_of(i, "nemotron", nemo_judge) - score_of(i, "outside", nemo_judge))
             - (score_of(i, "nemotron", j) - score_of(i, "outside", j)) for i in common]
        did[j] = {"n": len(d), "self_preference": _mean(d),
                  "cases_favouring_nemotron": sum(x > 0 for x in d),
                  "cases_favouring_outside": sum(x < 0 for x in d)}

    # ── Test B ────────────────────────────────────────────────────────────────
    rep = json.loads(open(report_path, encoding="utf-8").read())
    agent_cases = rep["cases"]
    refs = {c["id"]: c for c in json.loads(
        (config.DATA_DIR / "eval" / "answer_cases.json").read_text())["cases"]}
    b_jobs = [(c, j) for c in agent_cases for j in outside]
    with ThreadPoolExecutor(workers) as ex:
        b_verdicts = list(ex.map(lambda bj: _grade(judge_objs[bj[1]],
            bj[0]["question"], refs[bj[0]["id"]]["reference"], bj[0]["answer"]), b_jobs))
    rows_b = {c["id"]: {"id": c["id"], "scenario": c["scenario"], "question": c["question"],
                        "answer": c["answer"],
                        "facts_ok": c["checks"].get("facts_ok"),
                        "verdicts": {nemo_judge: {"verdict": c["correctness"]["verdict"],
                                                 "reason": c["correctness"].get("reason")}}}
              for c in agent_cases}
    for (c, j), v in zip(b_jobs, b_verdicts, strict=True):
        rows_b[c["id"]]["verdicts"][j] = {"verdict": v.get("verdict"),
                                          "reason": v.get("reason"), "error": v.get("error")}
    table_b = {"agent_model": rep["config"].get("agent_model"), "source_report": report_path,
               "mean_score": {}, "vs_nemotron": {}}
    for j in judges:
        table_b["mean_score"][j] = _mean(SCORE.get(r["verdicts"][j]["verdict"])
                                         for r in rows_b.values())
    for j in outside:
        pairs = [(r["verdicts"][nemo_judge]["verdict"], r["verdicts"][j]["verdict"])
                 for r in rows_b.values()
                 if r["verdicts"][j]["verdict"] in SCORE
                 and r["verdicts"][nemo_judge]["verdict"] in SCORE]
        table_b["vs_nemotron"][j] = {
            "n": len(pairs),
            "exact_agreement": _mean(1.0 if a == b else 0.0 for a, b in pairs),
            "cohen_kappa": _kappa([a for a, _ in pairs], [b for _, b in pairs]),
            "nemotron_more_lenient": sum(SCORE[a] > SCORE[b] for a, b in pairs),
            "nemotron_harsher": sum(SCORE[a] < SCORE[b] for a, b in pairs)}

    report = {"kind": "judge_bias", "label": label, "created": datetime.now(UTC).isoformat(),
              "nemotron_judge": nemo_judge, "outside_judges": outside,
              "writers": {"nemotron": writer_nemotron, "outside": writer_outside},
              "test_a": {"table": table_a, "difference_in_differences": did, "rows": rows_a},
              "test_b": {**table_b, "rows": list(rows_b.values())}}
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"test_a": {"table": table_a, "did": did},
                      "test_b": table_b}, indent=1))
    print(f"report -> {out}")
    return report


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(config.PROJECT_ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--outside", required=True, help="Comma-separated outside-family judges")
    ap.add_argument("--writer", default=config.NVIDIA_MODEL,
                    help="Nemotron model that writes the 'own family' rewrites")
    ap.add_argument("--report", default=str(config.PROJECT_ROOT / "reports"
                                            / "answer_eval_langfuse_eval.json"))
    ap.add_argument("--label", default="latest")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true",
                    help="Reuse rewrites and successful grades from the existing report")
    a = ap.parse_args()
    run([m.strip() for m in a.outside.split(",") if m.strip()], a.writer, a.report,
        a.workers, a.label, a.resume)

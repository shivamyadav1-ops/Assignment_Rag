"""Calibrate the LLM correctness judge against hand-labelled verdicts.

data/eval/judge_calibration.json holds (question, reference, answer) triples with the
verdict a careful human grader would give, built on the real eval questions. Running
the judge over them measures how far its scores can be trusted:

  exact agreement     judge verdict == label (3 classes)
  binary agreement    correct vs. not-correct
  score MAE           |judge score - label score|, score = 1 / 0.5 / 0
  confusion matrix    where it disagrees (too harsh / too lenient)

Run:  rsa calibrate-judge   (or python -m retail_support_agent.judge_calibration)
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from . import config
from .eval_answers import CASES_PATH, Judge

CALIBRATION_PATH = config.DATA_DIR / "eval" / "judge_calibration.json"
SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}
LABELS = ["correct", "partially_correct", "incorrect"]


def load_cases() -> list[dict]:
    refs = {c["id"]: c for c in json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]}
    cases = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))["cases"]
    for c in cases:
        src = refs[c["source"]]
        c["question"], c["reference"] = src["question"], src["reference"]
    return cases


def run(judge: Judge | None = None, label: str = "latest", workers: int = 4,
        save: bool = True) -> dict:
    judge = judge or Judge()
    cases = load_cases()

    def grade(c):
        return c, judge.correctness(c["question"], c["reference"], c["answer"])

    with ThreadPoolExecutor(workers) as ex:
        graded = list(ex.map(grade, cases))

    rows = []
    for c, v in graded:
        rows.append({"id": c["id"], "split": c.get("split", "tuning"), "probe": c["probe"],
                     "expected": c["expected"],
                     "verdict": v.get("verdict"), "reason": v.get("reason"),
                     "error": v.get("error"),
                     "match": v.get("verdict") == c["expected"]})
    judged = [r for r in rows if r["verdict"] in SCORE]
    exact = sum(r["match"] for r in judged) / len(judged) if judged else None
    binary = (sum((r["verdict"] == "correct") == (r["expected"] == "correct")
                  for r in judged) / len(judged)) if judged else None
    mae = (sum(abs(SCORE[r["verdict"]] - SCORE[r["expected"]]) for r in judged)
           / len(judged)) if judged else None
    harsh = sum(SCORE[r["verdict"]] < SCORE[r["expected"]] for r in judged)
    lenient = sum(SCORE[r["verdict"]] > SCORE[r["expected"]] for r in judged)
    confusion = Counter((r["expected"], r["verdict"]) for r in judged)
    report = {
        "kind": "judge_calibration", "label": label,
        "created": datetime.now(UTC).isoformat(), "judge_model": judge.model,
        "summary": {"n": len(rows), "judged": len(judged), "exact_agreement": exact,
                    "binary_agreement": binary, "score_mae": mae,
                    "too_harsh": harsh, "too_lenient": lenient},
        "confusion": {f"{e}->{v}": n for (e, v), n in sorted(confusion.items())},
        "rows": rows,
    }
    exact_str = f"{exact:.0%}" if exact is not None else "n/a"
    binary_str = f"{binary:.0%}" if binary is not None else "n/a"
    mae_str = f"{mae:.3f}" if mae is not None else "n/a"
    print(f"judge={judge.model}  n={len(rows)} judged={len(judged)}  "
          f"exact={exact_str}  binary={binary_str}  MAE={mae_str}  "
          f"too_harsh={harsh}  too_lenient={lenient}")
    for split in sorted({r["split"] for r in rows}):
        sub = [r for r in rows if r["split"] == split and r["verdict"] in SCORE]
        if sub:
            report["summary"][f"exact_{split}"] = sum(r["match"] for r in sub) / len(sub)
            print(f"  {split:8} exact {report['summary'][f'exact_{split}']:.0%} "
                  f"({sum(r['match'] for r in sub)}/{len(sub)})")
    for r in rows:
        if not r["match"]:
            print(f"  {r['id']} {r['probe']:22} expected {r['expected']:17} got "
                  f"{r['verdict']!s:17} | {(r['reason'] or r['error'] or '')[:110]}")
    if save:
        out = config.PROJECT_ROOT / "reports"
        out.mkdir(exist_ok=True)
        path = out / f"judge_calibration_{label}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report -> {path}")
    return report


if __name__ == "__main__":
    import sys
    run(label=sys.argv[1] if len(sys.argv) > 1 else "latest")

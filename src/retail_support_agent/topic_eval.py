"""Calibrate the off-topic guardrail (topic.py) on labelled in-scope / off-topic queries.

In-scope = every question in the retrieval and answer eval sets (except the one
out-of-scope case, a15), the benign red-team inputs, and hand-written borderline
support questions (unanswerable or oddly phrased but on-topic). Off-topic = hand-written
general-knowledge, coding, politics, etc., including "hard" ones that use shopping
words. None of these appear in the classifier's prototypes.

Reports score distributions, the margin sweep, and errors at the configured margin.
Run:  python -m retail_support_agent.topic_eval
"""

from __future__ import annotations

import json

from . import config
from .security_eval import INPUT_CASES
from .topic import OFF_TOPIC_MARGIN, is_off_topic, topic_score

BORDERLINE_IN_SCOPE = [
    "Can you recommend a laptop for programming?",
    "Will the storm delay my package?",
    "Is the PS5 controller compatible with a PC?",
    "My phone gets really hot when charging, what should I do?",
    "hi", "thanks!", "can I talk to a real person?", "what's your customer service phone number?",
    "Do you sell refurbished iPhones?", "Is Black Friday pricing honoured after the sale?",
    "What happens to my data if I delete my account?",
    "The courier left my parcel with a neighbour, what now?",
    "Can I pay with PayPal?", "Do you have a store near Boston?",
    "Is there a warranty on open-box TVs?", "Why was I charged a restocking fee?",
    "Can I gift wrap my order?", "How do I set up my new smart speaker?",
    "My headphones only work on one side", "Do premium members get free shipping?",
]

OFF_TOPIC_CASES = [
    "who is prime minster of india", "What's the capital of France?",
    "write a python function to reverse a list", "how do I bake banana bread",
    "what's 17 times 23", "who won the world cup in 2022",
    "explain quantum entanglement simply", "what's the weather in Delhi tomorrow",
    "should I buy Tesla stock", "write me a poem about love", "tell me a joke",
    "what are the symptoms of diabetes", "who wrote Hamlet",
    "translate good morning to Spanish", "best places to visit in Japan",
    "how do I lose weight fast", "who is the president of the USA",
    "how many planets are in the solar system",
    "help me with my history homework on World War 2", "what is the speed of light",
    "summarise the plot of Harry Potter", "how do I learn machine learning",
    # hard: off-topic but uses shopping / support words
    "How do I return a serve in tennis?", "How does shipping work through the Suez Canal?",
    "What is the refund policy of my gym?", "Explain how the stock market orders are matched",
]


def labelled() -> list[tuple[str, bool, str]]:
    """(text, is_off_topic, source)."""
    rq = json.loads((config.DATA_DIR / "eval" / "retrieval_questions.json")
                    .read_text(encoding="utf-8"))["questions"]
    ac = json.loads((config.DATA_DIR / "eval" / "answer_cases.json")
                    .read_text(encoding="utf-8"))["cases"]
    rows = [(q["query"], False, "retrieval_eval") for q in rq]
    rows += [(c["question"], c["id"] == "a15", "answer_eval") for c in ac]
    rows += [(t, False, "redteam_benign") for cat, t, exp in INPUT_CASES
             if not exp and cat.startswith("benign")]
    rows += [(t, False, "borderline") for t in BORDERLINE_IN_SCOPE]
    rows += [(t, True, "off_topic") for t in OFF_TOPIC_CASES]
    return rows


def run(verbose: bool = True) -> dict:
    rows = []
    for text, off, src in labelled():
        blocked, info = is_off_topic(text)
        score = info.get("score")
        if score is None:            # ID fast path: compute anyway for the sweep
            score = topic_score(text)["score"]
        rows.append({"text": text, "off_topic": off, "source": src, "score": score,
                     "id_fast_path": "reason" in info, "blocked": blocked,
                     "latency_ms": info["latency_ms"]})
    ins = [r for r in rows if not r["off_topic"]]
    offs = [r for r in rows if r["off_topic"]]

    def at(margin):
        fp = sum(1 for r in ins if not r["id_fast_path"] and r["score"] > margin)
        tp = sum(1 for r in offs if r["score"] > margin)
        return fp, tp

    sweep = [{"margin": m / 100, "false_blocks": at(m / 100)[0],
              "detected": at(m / 100)[1]} for m in range(-6, 13, 2)]
    fp, tp = at(OFF_TOPIC_MARGIN)
    lat = sorted(r["latency_ms"] for r in rows)
    out = {"margin": OFF_TOPIC_MARGIN, "n_in": len(ins), "n_off": len(offs),
           "false_blocks": fp, "detected": tp,
           "false_block_rate": fp / len(ins), "detection_rate": tp / len(offs),
           "max_in_scope_score": max(r["score"] for r in ins if not r["id_fast_path"]),
           "min_off_topic_score": min(r["score"] for r in offs),
           "latency_ms_p50": lat[len(lat) // 2], "latency_ms_max": lat[-1],
           "sweep": sweep, "rows": rows}
    if verbose:
        print(f"in-scope n={len(ins)}  off-topic n={len(offs)}  margin={OFF_TOPIC_MARGIN}")
        print(f"  max in-scope score {out['max_in_scope_score']:+.3f} | "
              f"min off-topic score {out['min_off_topic_score']:+.3f}")
        print("  margin  false_blocks  detected")
        for s in sweep:
            print(f"  {s['margin']:+.2f}   {s['false_blocks']:3}/{len(ins)}       "
                  f"{s['detected']:3}/{len(offs)}")
        print(f"at margin {OFF_TOPIC_MARGIN}: false blocks {fp}/{len(ins)}, "
              f"detected {tp}/{len(offs)}, p50 latency {out['latency_ms_p50']:.1f} ms")
        top_in = sorted((r for r in ins if not r["id_fast_path"]), key=lambda r: -r["score"])[:6]
        print("  highest-scoring in-scope:")
        for r in top_in:
            print(f"    {r['score']:+.3f} {r['source']:15} {r['text'][:70]}")
        print("  lowest-scoring off-topic:")
        for r in sorted(offs, key=lambda r: r["score"])[:6]:
            print(f"    {r['score']:+.3f} {r['text'][:70]}")
    return out


if __name__ == "__main__":
    run()

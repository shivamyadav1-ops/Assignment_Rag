"""Response evaluation on the golden answer set with deepeval.

Phase 1 runs the full agent on every case (sequential: one retriever, one MCP
server per case) and caches question, answer, tool calls and the exact context the
agent saw to `reports/deepeval_outputs_<label>.json`. `--reuse` skips this phase.

Phase 2 scores each case (cases in parallel, metrics sequential per case):

  RAG triad / retrieval-in-context (deepeval built-ins)
    faithfulness          claims in the answer supported by retrieval_context
    answer_relevancy      answer statements relevant to the question
    contextual_precision  relevant context ranked above irrelevant (vs expected_output)
    contextual_recall     expected_output sentences attributable to the context
    contextual_relevancy  share of context statements relevant to the question
    contradiction         HallucinationMetric: share of context passages the answer
                          contradicts (lower is better)
  G-Eval (LLM-as-judge with explicit evaluation steps)
    correctness           actual vs expected answer: facts, conclusion, contradictions
    completeness          every part of the question the reference answers is covered
  Deterministic (no model, from eval_answers): facts / tools / safety, where the case
  defines them.

The judge is the same NVIDIA model eval_answers uses (a larger model than the agent),
wrapped as a deepeval custom model. Context metrics are skipped (reported as None)
when the agent retrieved nothing, e.g. a guardrail-blocked input.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import statistics
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_DISABLE_PROGRESS_BAR", "YES")

from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase, SingleTurnParams  # noqa: E402

from . import config  # noqa: E402
from .eval_answers import JUDGE_MODELS, check_case  # noqa: E402
from .golden import GOLDEN_ANSWERS_PATH  # noqa: E402

THRESHOLDS = {
    "faithfulness": 0.7, "answer_relevancy": 0.7, "contextual_precision": 0.7,
    "contextual_recall": 0.7, "contextual_relevancy": 0.5,
    "correctness": 0.7, "completeness": 0.7,
    "contradiction": 0.3,   # lower is better: share of context passages contradicted
}
CONTEXT_METRICS = {"faithfulness", "contextual_precision", "contextual_recall",
                   "contextual_relevancy", "contradiction"}

CORRECTNESS_STEPS = [
    "Identify the conclusion and the key facts (values, windows, dates, fees, who pays, "
    "eligibility) in the expected output that answer the input question.",
    "Check whether the actual output states the same conclusion and each key fact, in any "
    "wording; equivalent values count (e.g. '15 + 15 days' equals '30 days').",
    "Heavily penalise any fact in the actual output that contradicts the expected output, "
    "and any invented policy, number or order detail.",
    "Do not penalise extra details that are correct or harmless, or differences in tone.",
    "If the expected output says the request must be refused or is not covered by policy, "
    "the actual output is correct only if it declines for the same reason and reveals no "
    "other customer's data.",
]

COMPLETENESS_STEPS = [
    "List every distinct part of the input question (e.g. a value AND its condition, or "
    "two separate asks) that the expected output answers.",
    "For each part, check whether the actual output addresses it with the needed fact.",
    "Score by the share of parts covered; a part answered vaguely ('it depends') without "
    "the fact counts as half.",
    "Ignore details in the expected output that the question did not ask for.",
]


# ── Judge model ───────────────────────────────────────────────────────────────

class NvidiaJudge(DeepEvalBaseLLM):
    """deepeval custom model over the NVIDIA OpenAI-compatible endpoint."""

    def __init__(self, model: str | None = None, timeout: float = 120.0):
        from openai import OpenAI

        self.model_name = model or JUDGE_MODELS[0]
        key = os.getenv("NVIDIA_API_KEY")
        if not key:
            raise RuntimeError("NVIDIA_API_KEY is not set (put it in .env).")
        self._client = OpenAI(base_url=config.NVIDIA_BASE_URL, api_key=key,
                              timeout=timeout, max_retries=3)
        self._lock = threading.Lock()
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "failures": 0}
        super().__init__(self.model_name)

    def load_model(self):
        return self._client

    def get_model_name(self) -> str:
        return self.model_name

    def _chat(self, prompt: str) -> str:
        extra = ({"chat_template_kwargs": {"enable_thinking": False}}
                 if "nemotron" in self.model_name else None)
        from openai import RateLimitError

        delay = 15.0
        for attempt in range(7):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_name, temperature=0, max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}], extra_body=extra)
                break
            except RateLimitError:
                if attempt == 6:
                    raise
                time.sleep(delay + random.uniform(0, 5))
                delay = min(delay * 2, 120.0)
            except Exception:
                with self._lock:
                    self.usage["failures"] += 1
                raise
        with self._lock:
            self.usage["calls"] += 1
            if resp.usage:
                self.usage["input_tokens"] += resp.usage.prompt_tokens
                self.usage["output_tokens"] += resp.usage.completion_tokens
        return resp.choices[0].message.content or ""

    def generate(self, prompt: str, schema=None):
        text = self._chat(prompt)
        if schema is None:
            return text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return schema.model_validate(json.loads(match.group(0)))
            except Exception:  # noqa: BLE001 - fall back to deepeval's own JSON repair
                pass
        return text

    async def a_generate(self, prompt: str, schema=None):
        return await asyncio.to_thread(self.generate, prompt, schema)


def build_metrics(judge: NvidiaJudge) -> dict:
    common = {"model": judge, "async_mode": False, "include_reason": True}
    geval_params = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.EXPECTED_OUTPUT]
    return {
        "faithfulness": FaithfulnessMetric(threshold=THRESHOLDS["faithfulness"], **common),
        "answer_relevancy": AnswerRelevancyMetric(
            threshold=THRESHOLDS["answer_relevancy"], **common),
        "contextual_precision": ContextualPrecisionMetric(
            threshold=THRESHOLDS["contextual_precision"], **common),
        "contextual_recall": ContextualRecallMetric(
            threshold=THRESHOLDS["contextual_recall"], **common),
        "contextual_relevancy": ContextualRelevancyMetric(
            threshold=THRESHOLDS["contextual_relevancy"], **common),
        "contradiction": HallucinationMetric(
            threshold=THRESHOLDS["contradiction"], **common),
        "correctness": GEval(name="Correctness", evaluation_steps=CORRECTNESS_STEPS,
                             evaluation_params=geval_params, model=judge,
                             threshold=THRESHOLDS["correctness"], async_mode=False),
        "completeness": GEval(name="Completeness", evaluation_steps=COMPLETENESS_STEPS,
                              evaluation_params=geval_params, model=judge,
                              threshold=THRESHOLDS["completeness"], async_mode=False),
    }


# ── Phase 1: agent outputs ────────────────────────────────────────────────────

def context_strings(context: list[dict]) -> list[str]:
    """The agent's context as deepeval retrieval_context: one string per policy chunk
    (in rank order) and one per operational tool result."""
    out = []
    for item in context:
        result = item["result"]
        data = result.get("data") or {}
        if item["tool"] == "search_policies" and result.get("ok"):
            out.extend(f"[{c['source']} | {c['section']}] {c['text']}"
                       for c in data.get("chunks", []))
        else:
            out.append(f"[{item['tool']} {json.dumps(item['args'])}] "
                       f"{json.dumps(result, default=str)}")
    return out


def run_agent(cases: list[dict]) -> list[dict]:
    from .agent import SupportAgent
    from .db import load_operational_data
    from .llm import build_llm
    from .mcp_client import MCPToolClient
    from .retriever import HybridRetriever

    load_operational_data()
    retriever, llm = HybridRetriever(cache_size=0), build_llm()
    outputs = []
    for case in cases:
        with MCPToolClient(case["customer"]).start() as mcp:
            agent = SupportAgent(case["customer"], llm=llm, mcp=mcp, retriever=retriever,
                                 semantic_cache=False)
            res = agent.answer(case["question"])
        outputs.append({"id": case["id"], "answer": res.answer, "outcome": res.outcome,
                        "tool_calls": res.tool_calls,
                        "retrieval_context": context_strings(res.context),
                        "latency_ms": round(res.latency_ms, 1), "usage": res.usage})
        print(f"  agent {case['id']:5} {res.outcome:18} {res.latency_ms / 1000:5.1f}s "
              f"{res.tool_calls}", flush=True)
    load_operational_data()
    return outputs


# ── Phase 2: scoring ──────────────────────────────────────────────────────────

def score_case(case: dict, out: dict, judge: NvidiaJudge) -> dict:
    tc = LLMTestCase(input=case["question"], actual_output=out["answer"] or "(empty)",
                     expected_output=case["reference"],
                     retrieval_context=out["retrieval_context"] or None,
                     context=out["retrieval_context"] or None)
    scores = {}
    for name, metric in build_metrics(judge).items():
        if name in CONTEXT_METRICS and not out["retrieval_context"]:
            scores[name] = {"score": None, "skipped": "no retrieval context"}
            continue
        t0 = time.perf_counter()
        try:
            metric.measure(tc)
            scores[name] = {"score": round(float(metric.score), 3),
                            "passed": bool(metric.success), "reason": metric.reason,
                            "ms": round((time.perf_counter() - t0) * 1000)}
        except Exception as exc:  # noqa: BLE001 - one failed metric must not sink the run
            scores[name] = {"score": None, "error": f"{type(exc).__name__}: {exc}"[:300]}
    checks = check_case(case, out["answer"], out["tool_calls"])
    return {"id": case["id"], "scenario": case["scenario"],
            "origin": case.get("origin", "human"), "question": case["question"],
            "reference": case["reference"], **out, "deterministic": checks,
            "has_expected_facts": bool(case.get("expected_facts")),
            "metrics": scores}


def _summ(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for name in THRESHOLDS:
        vals = [r["metrics"][name] for r in rows]
        scored = [v for v in vals if v.get("score") is not None]
        out[name] = {
            "mean": round(statistics.fmean(v["score"] for v in scored), 3) if scored else None,
            "pass_rate": (round(statistics.fmean(1.0 if v["passed"] else 0.0
                                                 for v in scored), 3) if scored else None),
            "scored_n": len(scored),
            "errors": sum(1 for v in vals if v.get("error")),
        }
    for key in ("facts_ok", "tools_ok", "safety_ok"):
        applicable = [r for r in rows if key != "facts_ok" or r["has_expected_facts"]]
        out[key.replace("_ok", "_pass_rate")] = (round(statistics.fmean(
            1.0 if r["deterministic"][key] else 0.0 for r in applicable), 3)
            if applicable else None)
    lat = sorted(r["latency_ms"] for r in rows)
    out["agent_latency_ms_mean"] = round(statistics.fmean(lat), 1)
    out["agent_latency_ms_p95"] = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
    return out


def run(label: str = "latest", case_ids: list[str] | None = None, workers: int = 4,
        reuse: bool = False, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or (config.PROJECT_ROOT / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = json.loads(GOLDEN_ANSWERS_PATH.read_text(encoding="utf-8"))["cases"]
    if case_ids:
        cases = [c for c in cases if c["id"] in case_ids]

    cache = out_dir / f"deepeval_outputs_{label}.json"
    t0 = time.perf_counter()
    if reuse and cache.exists():
        outputs = json.loads(cache.read_text(encoding="utf-8"))
        print(f"reusing {len(outputs)} cached agent outputs from {cache}", flush=True)
    else:
        print(f"Phase 1: running agent on {len(cases)} golden cases", flush=True)
        outputs = run_agent(cases)
        cache.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    by_id = {o["id"]: o for o in outputs}
    cases = [c for c in cases if c["id"] in by_id]

    judge = NvidiaJudge()
    print(f"Phase 2: scoring {len(cases)} cases x {len(THRESHOLDS)} metrics "
          f"| judge={judge.model_name} | workers={workers}", flush=True)

    def task(case):
        row = score_case(case, by_id[case["id"]], judge)
        m = row["metrics"]
        fmt = lambda k: "  -  " if m[k].get("score") is None else f"{m[k]['score']:.2f}"  # noqa: E731
        print(f"  {case['id']:5} {case['scenario']:10} faith={fmt('faithfulness')} "
              f"relev={fmt('answer_relevancy')} cPrec={fmt('contextual_precision')} "
              f"cRec={fmt('contextual_recall')} cRel={fmt('contextual_relevancy')} "
              f"correct={fmt('correctness')} complete={fmt('completeness')} "
              f"contra={fmt('contradiction')}", flush=True)
        return row

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(task, cases))

    groups = defaultdict(list)
    for r in rows:
        groups[f"scenario:{r['scenario']}"].append(r)
        groups[f"origin:{r['origin']}"].append(r)
    report = {
        "label": label,
        "created": datetime.now(UTC).isoformat(),
        "framework": "deepeval",
        "config": {"judge_model": judge.model_name, "agent_model": config.NVIDIA_MODEL,
                   "reranker_model": config.RERANKER_MODEL, "rerank_top_k": config.RERANK_TOP_K,
                   "thresholds": THRESHOLDS},
        "summary": _summ(rows),
        "groups": {g: _summ(rs) for g, rs in sorted(groups.items())},
        "judge_usage": judge.usage,
        "wall_s": round(time.perf_counter() - t0, 1),
        "cases": rows,
    }
    path = out_dir / f"deepeval_{label}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    s = report["summary"]
    print("summary: " + " | ".join(
        f"{k}={s[k]['mean']}" for k in THRESHOLDS), flush=True)
    print(f"report -> {path}", flush=True)
    return path

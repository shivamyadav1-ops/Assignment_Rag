# Evaluation reports

**Current results** (quoted in the README and the slides):

| File | What it is |
|---|---|
| `retrieval_eval_v3.json` | Retrieval comparison: dense, BM25, hybrid, hybrid + rerank, full; 50 labelled questions, all with evidence labels, k = 7 |
| `answer_eval_final.json` | End-to-end agent eval: 29 cases (knowledge, multi-step, exception) with deterministic fact/tool/safety checks, G-Eval judge and faithfulness |
| `cost_latency_final.md` / `.json` | Tokens, stage latency, TTFT and cost sensitivity for that run (`rsa cost-report --label final`) |
| `judge_calibration_final.json` | LLM-judge agreement with 26 hand-labelled cases |
| `answer_eval_representative.json` | The six cases behind `traces/representative/` (one per scenario), final code |

**Iteration evidence** (kept to show what each change fixed; not current numbers):

| File | Shows |
|---|---|
| `answer_eval_v3_before_forced_draft_fix.json` | First 29-case run after the evidence rule: 93.1% fact accuracy. Caught a regression (a25): the forced policy search hid the model's correct draft, which was then rewritten. Fixed by keeping the draft in the forced exchange |
| `answer_eval_v2_before_write_guard.json` | First 28-case run: the agent created returns nobody asked for (a08) and retried writes on ineligible orders (a22). Fixed by the write-intent / eligibility guard |
| `answer_eval_v2_before_policy_rule.json` (+ `cost_latency_…`) | After the write guard: safe, but multi-step answers never consulted policy evidence. Fixed by the policy-consultation rule and case a29 |
| `answer_eval_v2_before_leak_fix.json` | After the policy rule: the prompt-leak check blocked a correct federal-rule answer (a17). Fixed by ignoring spans also present in the evidence |
| `retrieval_eval_cap*.json` | Rerank-candidate cap sweep (5/8/10/20) on the older 36-question, 321-chunk index |
| `retrieval_eval_adj*.json`, `…_before/after_rowchunks.json`, `…_after_chunker_fix.json` | Earlier chunking and neighbour-expansion experiments (older index, 36 questions) |
| `answer_eval_geval*.json`, `answer_eval_after_rowchunks.json`, `answer_eval_langfuse_eval.json` | Earlier 16-case answer evals |
| `judge_calibration_*`, `judge_bias_gemma.json` | Judge prompt iterations and a cross-family bias check |
| `ttft_eval_*.json` | Time-to-first-token measurements |

Older reports were produced on earlier indexes and question sets, so their numbers are
not comparable with the current ones.

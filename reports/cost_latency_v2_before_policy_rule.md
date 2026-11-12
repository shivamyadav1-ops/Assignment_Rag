# Cost and latency — `answer_eval_v2.json`

Agent model: `nvidia/nemotron-3.5-bolt-e4b` · 28 eval cases · tokens and latency measured; dollar cost depends on the price you pay (see sensitivity table).

## Per query, by scenario

| Scenario | Cases | Input tok (mean) | Output tok (mean) | Cached input | Latency p50 | Latency p95 |
|---|---|---|---|---|---|---|
| exception | 8 | 2,764 | 64 | 85% | 2.4s | 6.0s |
| knowledge | 10 | 5,497 | 159 | 53% | 5.5s | 45.8s |
| multi_step | 10 | 3,995 | 100 | 91% | 3.7s | 5.4s |
| **all** | 28 | 4,180 | 111 | 72% | 4.1s | 22.9s |

## Where the time goes (per case, from traces)

| Stage | Mean | p50 | p95 |
|---|---|---|---|
| llm | 2.00s | 1.94s | 3.83s |
| search_policies (retrieval) | 1.02s | 0.00s | 7.24s |
| MCP tools | 0.28s | 0.00s | 0.87s |

LLM turns per case: mean 1.9, max 3. Time to first token: p50 497 ms, p95 707 ms.

## Cost sensitivity (agent tokens only)

| Input $/M | Output $/M | $ per query | $ per 1,000 queries |
|---|---|---|---|
| 0.10 | 0.40 | 0.00046 | 0.46 |
| 0.40 | 1.60 | 0.00185 | 1.85 |
| 1.00 | 4.00 | 0.00462 | 4.62 |
| 3.00 | 15.00 | 0.01421 | 14.21 |

Evaluation overhead (LLM judge, not paid per customer query): 84 calls, 77,537 input / 18,956 output tokens.

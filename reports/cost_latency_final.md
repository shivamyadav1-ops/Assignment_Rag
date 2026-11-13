# Cost and latency — `answer_eval_final.json`

Agent model: `nvidia/nemotron-3.5-bolt-e4b` · 29 eval cases · tokens and latency measured; dollar cost depends on the price you pay (see sensitivity table).

## Per query, by scenario

| Scenario | Cases | Input tok (mean) | Output tok (mean) | Cached input | Latency p50 | Latency p95 |
|---|---|---|---|---|---|---|
| exception | 8 | 3,269 | 79 | 99% | 1.9s | 6.6s |
| knowledge | 10 | 5,591 | 170 | 92% | 5.7s | 23.6s |
| multi_step | 11 | 6,866 | 178 | 99% | 6.3s | 8.2s |
| **all** | 29 | 5,434 | 148 | 97% | 5.7s | 11.9s |

## Where the time goes (per case, from traces)

| Stage | Mean | p50 | p95 |
|---|---|---|---|
| llm | 2.79s | 3.10s | 4.90s |
| search_policies (retrieval) | 0.91s | 0.77s | 3.59s |
| MCP tools | 0.09s | 0.00s | 0.22s |

LLM turns per case: mean 2.2, max 4. Time to first token: p50 557 ms, p95 758 ms.

## Cost sensitivity (agent tokens only)

| Input $/M | Output $/M | $ per query | $ per 1,000 queries |
|---|---|---|---|
| 0.10 | 0.40 | 0.00060 | 0.60 |
| 0.40 | 1.60 | 0.00241 | 2.41 |
| 1.00 | 4.00 | 0.00603 | 6.03 |
| 3.00 | 15.00 | 0.01852 | 18.52 |

Evaluation overhead (LLM judge, not paid per customer query): 88 calls, 100,559 input / 21,818 output tokens.

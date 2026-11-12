"""The input guardrail's sliding-window rate limiter: bounded memory and thread safety."""

from __future__ import annotations

import threading
import time

from retail_support_agent import guardrails


def test_idle_requesters_are_pruned_once_the_log_is_large(monkeypatch):
    monkeypatch.setattr(guardrails, "_PRUNE_AT", 5)
    stale = time.monotonic() - 3600          # far outside a 60 s window
    for i in range(10):
        guardrails._request_log[f"IDLE-{i}"] = [stale]

    guardrails.check_rate_limit("ACTIVE")

    assert set(guardrails._request_log) == {"ACTIVE"}


def test_requesters_still_inside_the_window_are_not_pruned(monkeypatch):
    monkeypatch.setattr(guardrails, "_PRUNE_AT", 2)
    for rid in ("A", "B", "C"):
        guardrails.check_rate_limit(rid)
    assert {"A", "B", "C"} <= set(guardrails._request_log)


def test_concurrent_requests_from_one_requester_are_all_counted():
    n_threads, per_thread = 8, 50
    barrier = threading.Barrier(n_threads)

    def hammer():
        barrier.wait()
        for _ in range(per_thread):
            guardrails.check_rate_limit("CUST-RACE", max_requests=10**6)

    threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(guardrails._request_log["CUST-RACE"]) == n_threads * per_thread


def test_limit_still_blocks_after_the_refactor():
    results = [guardrails.check_rate_limit("CUST-LIMIT", max_requests=3) for _ in range(4)]
    assert [r.passed for r in results] == [True, True, True, False]
    assert results[-1].reason == guardrails.BlockReason.RATE_LIMITED

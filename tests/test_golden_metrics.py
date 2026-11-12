"""Precision/recall/MAP/nDCG maths for the golden retrieval eval."""

from __future__ import annotations

import math

import pytest

from retail_support_agent.eval_golden_retrieval import query_metrics
from retail_support_agent.golden import _quotas


def test_perfect_ranking():
    m = query_metrics(["a", "b", "x", "y"], {"a", "b"}, {"a": 2, "b": 2, "x": 0})
    assert m["precision@1"] == 1.0
    assert m["recall@3"] == 1.0
    assert m["precision@5"] == pytest.approx(2 / 5)
    assert m["precision_ceiling@5"] == pytest.approx(2 / 5)
    assert m["map"] == 1.0 and m["mrr"] == 1.0 and m["r_precision"] == 1.0
    assert m["ndcg@5"] == pytest.approx(1.0)


def test_partial_ranking():
    # relevant at ranks 2 and 4, one relevant never retrieved
    m = query_metrics(["x", "a", "y", "b"], {"a", "b", "c"}, {"a": 2, "b": 2, "c": 2})
    assert m["recall@3"] == pytest.approx(1 / 3)
    assert m["recall@10"] == pytest.approx(2 / 3)
    assert m["precision@3"] == pytest.approx(1 / 3)
    assert m["mrr"] == pytest.approx(0.5)
    assert m["map"] == pytest.approx((1 / 2 + 2 / 4) / 3)
    assert m["r_precision"] == pytest.approx(1 / 3)
    assert m["hit@1"] == 0.0 and m["hit@3"] == 1.0


def test_graded_ndcg_rewards_grade2_over_grade1():
    grades = {"a": 2, "p": 1}
    best = query_metrics(["a", "p"], {"a"}, grades)["ndcg@3"]
    worse = query_metrics(["p", "a"], {"a"}, grades)["ndcg@3"]
    assert best == pytest.approx(1.0)
    ideal = 3 + 1 / math.log2(3)
    assert worse == pytest.approx((1 + 3 / math.log2(3)) / ideal)


def test_unjudged_chunks_count_as_irrelevant_and_lower_judged_rate():
    m = query_metrics(["new1", "a"], {"a"}, {"a": 2})
    assert m["precision@1"] == 0.0
    assert m["judged@3"] == pytest.approx(1 / 2)


def test_quotas_sum_to_total_with_minimum():
    from collections import Counter
    q = _quotas(Counter({"big": 100, "mid": 40, "tiny": 3}), 20)
    assert sum(q.values()) == 20 and q["tiny"] >= 2 and q["big"] > q["mid"]

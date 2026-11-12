"""Default test run is model-free: classifier guardrails and live grounding are off.
Tests marked `ml` turn them on explicitly and skip if the models are not cached."""

import os

os.environ.setdefault("RSA_ML_GUARDRAILS", "0")
os.environ.setdefault("RSA_LIVE_GROUNDING", "0")
os.environ.setdefault("RSA_TOPIC_GUARD", "0")
os.environ.setdefault("RSA_SEMANTIC_CACHE", "0")
os.environ.setdefault("RSA_RETRIEVAL_CACHE_SIZE", "0")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_rate_limiter():
    """The input guardrail's rate limiter is a process-global sliding window
    (10 requests / 60 s per requester). Without a reset, tests that share a
    requester id (most use CUST-1001) exhaust it across files and later tests
    are silently blocked as `rate_limited`, so their results depend on run order."""
    from retail_support_agent import guardrails

    guardrails._request_log.clear()
    yield
    guardrails._request_log.clear()

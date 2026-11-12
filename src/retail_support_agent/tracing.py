"""Langfuse v5 observability integration (OpenTelemetry-based API).

Wraps each agent session in a Langfuse trace with one observation per step:
  - root (as_type="agent") — one per agent.answer() call
  - as_type="generation"   — LLM turns (model, tokens, latency, TTFT)
  - as_type="retriever"    — search_policies calls
  - as_type="tool"         — MCP tool calls (get_order_details, etc.)
  - as_type="guardrail"    — guardrail check steps

Silently disabled when LANGFUSE_SECRET_KEY or LANGFUSE_PUBLIC_KEY is missing.

Environment variables (all in .env):
  LANGFUSE_SECRET_KEY   sk-lf-…  (required)
  LANGFUSE_PUBLIC_KEY   pk-lf-…  (required)
  LANGFUSE_BASE_URL     https://us.cloud.langfuse.com  (US region)
  LANGFUSE_HOST         alternative name for LANGFUSE_BASE_URL
"""

from __future__ import annotations

import os
from typing import Any

_lf: Any = None          # Langfuse singleton
_checked = False         # initialisation attempted?


def _client() -> Any | None:
    global _lf, _checked
    if _checked:
        return _lf
    _checked = True
    secret = os.getenv("LANGFUSE_SECRET_KEY", "")
    public = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    # Accept LANGFUSE_BASE_URL (Langfuse v2+ docs) or LANGFUSE_HOST (older convention)
    host = (os.getenv("LANGFUSE_BASE_URL")
            or os.getenv("LANGFUSE_HOST")
            or "https://cloud.langfuse.com")
    if not secret or not public:
        return None
    try:
        from langfuse import Langfuse
        _lf = Langfuse(secret_key=secret, public_key=public, host=host)
    except Exception as exc:
        print(f"[tracing] Langfuse init failed — observability disabled: {exc}")
        _lf = None
    return _lf


# ── Trace lifecycle ────────────────────────────────────────────────────────────

def start_trace(
    session_id: str,
    name: str,
    user_id: str,
    input: str,
    tags: list[str] | None = None,
) -> Any | None:
    """Start a root agent trace. Returns a LangfuseAgent span, or None if disabled."""
    lf = _client()
    if lf is None:
        return None
    try:
        return lf.start_observation(
            name=name,
            as_type="agent",
            input=input,
            metadata={"session_id": session_id, "user_id": user_id,
                      "tags": tags or ["retail-support"]},
        )
    except Exception:
        return None


def end_trace(trace: Any, output: str, metadata: dict) -> None:
    """Update trace output/metadata and mark it complete."""
    if trace is None:
        return
    try:
        trace.update(output=output, metadata=metadata)
        trace.end()
    except Exception:
        pass


# ── Child observations ─────────────────────────────────────────────────────────

def add_generation(
    trace: Any,
    name: str,
    model: str,
    prompt: list[dict],
    completion: str,
    usage: dict,
    latency_ms: float,
    ttft_ms: float | None = None,
) -> None:
    """Record one LLM turn as a Langfuse generation (prompt + completion + usage)."""
    if trace is None:
        return
    try:
        gen = trace.start_observation(
            name=name,
            as_type="generation",
            model=model,
            input=prompt,
            output=completion,
            usage_details={
                "input": usage.get("input_tokens", 0),
                "output": usage.get("output_tokens", 0),
            },
            metadata={"latency_ms": round(latency_ms, 1), "ttft_ms": ttft_ms},
        )
        gen.end()
    except Exception:
        pass


def add_span(
    trace: Any,
    name: str,
    input: dict,
    output: Any,
    latency_ms: float,
    metadata: dict | None = None,
    span_type: str = "tool",
) -> None:
    """Record one retrieval or MCP tool call as a Langfuse span.

    span_type: "retriever" for search_policies, "tool" for MCP calls,
               "guardrail" for guardrail checks.
    """
    if trace is None:
        return
    # Map to valid Langfuse as_type values
    as_type = span_type if span_type in ("retriever", "tool", "guardrail", "span") else "tool"
    try:
        obs = trace.start_observation(
            name=name,
            as_type=as_type,
            input=input,
            output=output,
            metadata={"latency_ms": round(latency_ms, 1), **(metadata or {})},
        )
        obs.end()
    except Exception:
        pass


# ── Flush ──────────────────────────────────────────────────────────────────────

def flush() -> None:
    """Flush buffered events to Langfuse (call at end of each agent.answer())."""
    if _lf is not None:
        try:
            _lf.flush()
        except Exception:
            pass

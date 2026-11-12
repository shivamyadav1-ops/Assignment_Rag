"""Shared result types for every guardrail layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BlockReason(Enum):
    TOXIC = "toxic_content"
    PROMPT_INJECTION = "prompt_injection"
    RATE_LIMITED = "rate_limited"
    EMPTY_INPUT = "empty_input"
    TOOL_NOT_ALLOWED = "tool_not_allowed"
    INVALID_TOOL_ARGS = "invalid_tool_args"
    TOOL_BUDGET_EXCEEDED = "tool_budget_exceeded"
    PRECONDITION_FAILED = "precondition_failed"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    TOXIC_OUTPUT = "toxic_output"
    OFF_TOPIC = "off_topic"
    NSFW = "nsfw_content"
    OFF_SCOPE_CONTENT = "off_scope_content"
    FABRICATED_CLAIM = "fabricated_claim"


@dataclass
class GuardrailResult:
    passed: bool
    reason: BlockReason | None = None
    detail: str = ""
    latency_ms: float = 0.0



def _ml_on() -> bool:
    from ..classifiers import ml_guardrails_enabled
    return ml_guardrails_enabled()

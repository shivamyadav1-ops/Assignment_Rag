"""LLM client abstraction.

The agent depends on the `LLMClient` protocol, not on a vendor SDK, for three reasons:
the orchestration can be tested with a scripted double and no API key, the backend is
swappable, and the two supported backends use *different wire formats*. Anthropic takes
a separate `system` argument with tool results as content blocks inside a user turn;
OpenAI-compatible endpoints (NVIDIA NIM) take system as a message and one `role:"tool"`
message per result. Each client therefore owns its own message formatting, so the agent
loop stays provider-agnostic.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Protocol

from . import config


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMTurn:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)       # input_tokens, output_tokens
    stop_reason: str | None = None
    # Wall-clock ms from request start to the first streamed token (text delta or
    # tool-call delta), measured over the wire. None if the client didn't stream.
    ttft_ms: float | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(Protocol):
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMTurn: ...
    def assistant_message(self, turn: LLMTurn) -> dict: ...
    def tool_result_messages(self, pairs: list[tuple[ToolCall, dict]]) -> list[dict]: ...


# ── Format helpers ─────────────────────────────────────────────────────────────
# Tools are authored once in Anthropic shape (name/description/input_schema); the
# OpenAI-compatible client translates them at call time.

def _anthropic_assistant_message(turn: LLMTurn) -> dict:
    content: list[dict] = []
    if turn.text:
        content.append({"type": "text", "text": turn.text})
    for c in turn.tool_calls:
        content.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.input})
    return {"role": "assistant", "content": content}


def _anthropic_tool_results(pairs: list[tuple[ToolCall, dict]]) -> list[dict]:
    # All results ride in a single user turn.
    return [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": call.id, "content": json.dumps(result)}
        for call, result in pairs
    ]}]


def _openai_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {
        "name": t["name"], "description": t.get("description", ""),
        "parameters": t["input_schema"]}} for t in tools]


def _openai_assistant_message(turn: LLMTurn) -> dict:
    msg: dict = {"role": "assistant", "content": turn.text or ""}
    if turn.tool_calls:
        msg["tool_calls"] = [{
            "id": c.id, "type": "function",
            "function": {"name": c.name, "arguments": json.dumps(c.input)}
        } for c in turn.tool_calls]
    return msg


def _openai_tool_results(pairs: list[tuple[ToolCall, dict]]) -> list[dict]:
    # One message per result, each keyed to its tool_call_id.
    return [{"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
            for call, result in pairs]


# ── NVIDIA NIM (OpenAI-compatible) ─────────────────────────────────────────────

class NvidiaClient:
    def __init__(self, model: str | None = None, max_tokens: int = config.MAX_TOKENS,
                 api_key: str | None = None, base_url: str | None = None,
                 temperature: float = 0.2, enable_thinking: bool = config.ENABLE_THINKING):
        import os

        from openai import OpenAI

        key = api_key or os.getenv("NVIDIA_API_KEY")
        if not key:
            raise RuntimeError("NVIDIA_API_KEY is not set (put it in .env).")
        self._client = OpenAI(base_url=base_url or config.NVIDIA_BASE_URL, api_key=key,
                              timeout=config.LLM_TIMEOUT_S, max_retries=2)
        self.model = model or config.NVIDIA_MODEL
        self.max_tokens = max_tokens
        # Low temperature: tool selection should be near-deterministic, not creative.
        self.temperature = temperature
        self.enable_thinking = enable_thinking

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMTurn:
        extra = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        t0 = time.perf_counter()
        ttft_ms: float | None = None
        content_parts: list[str] = []
        tool_acc: dict[int, dict] = {}
        finish_reason = None
        usage: dict = {}

        stream = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, *messages],
            tools=_openai_tools(tools),
            extra_body=extra,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.usage:
                usage = {"input_tokens": chunk.usage.prompt_tokens,
                         "output_tokens": chunk.usage.completion_tokens}
                # Prefix (KV) cache reuse, when the endpoint reports it.
                details = getattr(chunk.usage, "prompt_tokens_details", None)
                cached = getattr(details, "cached_tokens", None) if details else None
                if cached:
                    usage["cached_input_tokens"] = cached
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            if delta.content:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                content_parts.append(delta.content)
            for tc in (delta.tool_calls or []):
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                acc = tool_acc.setdefault(tc.index, {"id": None, "name": None, "arguments": ""})
                if tc.id:
                    acc["id"] = tc.id
                if tc.function and tc.function.name:
                    acc["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    acc["arguments"] += tc.function.arguments

        calls = []
        for idx in sorted(tool_acc):
            acc = tool_acc[idx]
            try:
                args = json.loads(acc["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=acc["id"] or f"call_{idx}", name=acc["name"], input=args))

        return LLMTurn(text="".join(content_parts) or None, tool_calls=calls, usage=usage,
                       stop_reason=finish_reason, ttft_ms=ttft_ms)

    def assistant_message(self, turn: LLMTurn) -> dict:
        return _openai_assistant_message(turn)

    def tool_result_messages(self, pairs: list[tuple[ToolCall, dict]]) -> list[dict]:
        return _openai_tool_results(pairs)


# ── Anthropic ──────────────────────────────────────────────────────────────────

def _anthropic_cache_breakpoints(system: str, messages: list[dict]) -> tuple[list, list]:
    """Context caching for the agent loop.

    The prefix order is tools -> system -> messages. Breakpoint 1 on the system block
    caches tools + system prompt, identical for every request. Breakpoint 2 on the last
    block of the newest message caches the conversation so far, so step N+1 of the
    tool loop re-reads steps 1..N from cache instead of reprocessing them. Prompts
    below the model's minimum cacheable length are simply not cached (no error).
    Messages are copied: the agent's own history is never mutated.
    """
    system_arg = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    if not messages:
        return system_arg, messages
    msgs = [dict(m) for m in messages]
    last = msgs[-1]
    content = last.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}] if content else []
    else:
        content = [dict(b) for b in (content or [])]
    if content:
        content[-1]["cache_control"] = {"type": "ephemeral"}
        last["content"] = content
    return system_arg, msgs


def _anthropic_usage(u) -> dict:
    usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens}
    read = getattr(u, "cache_read_input_tokens", None)
    write = getattr(u, "cache_creation_input_tokens", None)
    if read:
        usage["cached_input_tokens"] = read
    if write:
        usage["cache_write_input_tokens"] = write
    return usage


class AnthropicClient:
    def __init__(self, model: str | None = None, max_tokens: int = config.MAX_TOKENS,
                 api_key: str | None = None):
        from anthropic import Anthropic
        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self.model = model or config.LLM_MODEL
        self.max_tokens = max_tokens

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMTurn:
        t0 = time.perf_counter()
        ttft_ms: float | None = None
        if config.PROMPT_CACHE:
            system_arg, messages = _anthropic_cache_breakpoints(system, messages)
        else:
            system_arg = system
        with self._client.messages.stream(
            model=self.model, max_tokens=self.max_tokens, system=system_arg,
            messages=messages, tools=tools,
        ) as stream:
            for event in stream:
                # First content_block_start/delta is the first token out of the
                # model, whether it's answer text or a tool-call argument.
                if ttft_ms is None and event.type in ("content_block_start", "content_block_delta"):
                    ttft_ms = (time.perf_counter() - t0) * 1000
            resp = stream.get_final_message()

        text_parts, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))
        return LLMTurn(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=calls,
            usage=_anthropic_usage(resp.usage),
            stop_reason=resp.stop_reason,
            ttft_ms=ttft_ms,
        )

    def assistant_message(self, turn: LLMTurn) -> dict:
        return _anthropic_assistant_message(turn)

    def tool_result_messages(self, pairs: list[tuple[ToolCall, dict]]) -> list[dict]:
        return _anthropic_tool_results(pairs)


# ── Testing: scripted double ───────────────────────────────────────────────────

class ScriptedClient:
    """Replays pre-set turns so the agent's orchestration can be tested without a key.

    Formats messages in the OpenAI shape by default so the tests exercise the same
    wire format as the configured backend.
    """

    def __init__(self, turns: list[LLMTurn], wire: str = "openai"):
        self._turns = list(turns)
        self._i = 0
        self._wire = wire
        self.seen: list[dict] = []   # captured calls, for assertions

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMTurn:
        self.seen.append({"messages": list(messages), "n_tools": len(tools)})
        if self._i >= len(self._turns):
            return LLMTurn(text="(scripted client exhausted)", stop_reason="end_turn")
        turn = self._turns[self._i]
        self._i += 1
        return turn

    def assistant_message(self, turn: LLMTurn) -> dict:
        return (_openai_assistant_message if self._wire == "openai"
                else _anthropic_assistant_message)(turn)

    def tool_result_messages(self, pairs: list[tuple[ToolCall, dict]]) -> list[dict]:
        return (_openai_tool_results if self._wire == "openai"
                else _anthropic_tool_results)(pairs)


# ── Factory ────────────────────────────────────────────────────────────────────

def build_llm(provider: str | None = None) -> LLMClient:
    provider = (provider or config.LLM_PROVIDER).strip().lower()
    if provider == "nvidia":
        return NvidiaClient()
    if provider == "anthropic":
        return AnthropicClient()
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}' (expected 'nvidia' or 'anthropic').")

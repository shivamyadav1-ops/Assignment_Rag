"""Unit tests for the agent runtime cluster: agent loop, LLM clients, config, models,
tracing, MCP server and MCP client.

No network, no real LLM, no real vector store: the model is a `ScriptedClient`
(reusing the pattern in tests/test_agent_guardrails.py), MCP and the retriever are
in-process fakes, and the Langfuse/OpenAI SDK boundaries are stubbed.

These fill the gaps left by tests/test_agent_guardrails.py (which covers the
guardrail decisions end to end); here the subject is the orchestration mechanics:
loop bounds, accounting, serialisation, error envelopes and wire formats.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import threading
import uuid
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from retail_support_agent import config, tracing
from retail_support_agent import llm as llm_mod
from retail_support_agent.agent import (
    DEADLINE_TOOL,
    LOCAL_TOOLS,
    SEARCH_TOOL,
    AgentResult,
    SupportAgent,
    TraceStep,
)
from retail_support_agent.llm import LLMTurn, ScriptedClient, ToolCall

pytestmark = pytest.mark.agent


# ══════════════════════════════════════════════════════════════════════════════
# Fakes
# ══════════════════════════════════════════════════════════════════════════════

ORDER_SCHEMA = {"type": "object", "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"]}


class FakeMCP:
    """Stand-in for MCPToolClient: records calls, replays canned envelopes."""

    def __init__(self, responses: dict[str, dict] | None = None):
        self.responses = responses or {}
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    def list_tools(self):
        return [
            {"name": "get_return_refund_status", "description": "",
             "input_schema": ORDER_SCHEMA},
            {"name": "get_order_details", "description": "", "input_schema": ORDER_SCHEMA},
            {"name": "create_return_request", "description": "", "input_schema": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["order_id", "reason"]}},
        ]

    def call_tool(self, name, args):
        self.calls.append((name, args))
        return self.responses.get(name, {"ok": True, "data": {}})

    def close(self):
        self.closed = True


@dataclass
class FakeChunk:
    chunk_id: str
    source_file: str
    section_heading: str
    text: str
    score: float = 0.5
    rerank_score: float | None = 2.0


@dataclass
class FakeRetrieval:
    chunks: list


class FakeRetriever:
    def __init__(self, chunks=None):
        self._chunks = chunks or []
        self.queries: list[str] = []

    def retrieve(self, query, mode="full", top_k=5):
        self.queries.append(query)
        return FakeRetrieval(list(self._chunks))


def make_agent(turns, tmp_path, mcp=None, chunks=None, semantic_cache=False):
    client = ScriptedClient(turns)
    agent = SupportAgent(
        f"CUST-A{uuid.uuid4().hex[:6]}", llm=client, mcp=mcp if mcp is not None else FakeMCP(),
        retriever=FakeRetriever(chunks), trace_dir=tmp_path, semantic_cache=semantic_cache)
    return agent, client


def tool_turn(*calls, usage=None):
    return LLMTurn(text=None,
                   tool_calls=[ToolCall(f"c{i}", n, a) for i, (n, a) in enumerate(calls)],
                   usage=usage or {})


def final(text, usage=None, ttft=None):
    return LLMTurn(text=text, stop_reason="end_turn", usage=usage or {}, ttft_ms=ttft)


def unevidenced(text, **kw):
    """A model that answers without evidence: the agent forces a policy search, then
    the model answers again (here: the same text)."""
    return [final(text, **kw), final(text, **kw)]


def trace_lines(result):
    with open(result.trace_path, encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def tool_messages(client):
    last = client.seen[-1]["messages"]
    return [json.loads(m["content"]) for m in last if m.get("role") == "tool"]


# ══════════════════════════════════════════════════════════════════════════════
# config.py
# ══════════════════════════════════════════════════════════════════════════════

class TestConfig:
    def test_defaults_have_expected_types_and_values(self):
        assert config.AGENT_MAX_STEPS == 8
        assert isinstance(config.MAX_TOKENS, int) and config.MAX_TOKENS > 0
        assert config.AS_OF_DATE.isoformat() == "2024-11-09"
        assert config.DENSE_DIM == 384
        assert config.MCP_CALL_TIMEOUT > 0
        assert isinstance(config.LLM_TIMEOUT_S, float)
        assert config.RERANK_TOP_K <= config.MAX_RERANK_CANDIDATES
        assert config.GROUNDING_MIN_SCORE == 0.0
        assert config.PROJECT_ROOT.is_dir()
        assert config.QDRANT_COLLECTION == "policies"

    def test_conftest_disables_ml_and_cache_paths(self):
        # The suite must stay model-free and cache-free (tests/conftest.py).
        assert config.LIVE_GROUNDING is False
        assert config.SEMANTIC_CACHE is False
        assert config.RETRIEVAL_CACHE_SIZE == 0

    @pytest.mark.parametrize("raw,expected", [
        ("0", False), ("false", False), ("FALSE", False), ("no", False), ("off", False),
        (" off ", False), ("1", True), ("true", True), ("yes", True), ("", True),
        ("anything", True),
    ])
    def test_flag_parsing(self, monkeypatch, raw, expected):
        monkeypatch.setenv("RSA_UNIT_FLAG", raw)
        assert config._flag("RSA_UNIT_FLAG") is expected

    def test_flag_uses_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("RSA_UNIT_FLAG", raising=False)
        assert config._flag("RSA_UNIT_FLAG") is True
        assert config._flag("RSA_UNIT_FLAG", "0") is False

    def test_env_overrides_applied_on_import(self):
        """Numeric / string env vars are read at import time."""
        env = {"RSA_RRF_LIMIT": "33", "RSA_LLM_TIMEOUT_S": "12.5",
               "LLM_PROVIDER": "  ANTHROPIC  ", "RSA_GROUNDING_ACTION": "BLOCK",
               "RSA_SESSION_KEEP_RECENT": "7"}
        with reloaded_config(env) as cfg:
            assert cfg.RRF_LIMIT == 33
            assert cfg.LLM_TIMEOUT_S == 12.5
            assert cfg.LLM_PROVIDER == "anthropic"     # stripped + lowercased
            assert cfg.GROUNDING_ACTION == "block"
            assert cfg.SESSION_KEEP_RECENT == 7

    def test_invalid_numeric_env_var_crashes_import(self):
        """FRAGILE: a typo'd numeric env var makes `import config` raise, which takes
        down the CLI, the MCP server and the dashboard with a bare ValueError."""
        with pytest.raises(ValueError):
            with reloaded_config({"RSA_RRF_LIMIT": "twenty"}):
                pass

    def test_paths_are_derived_from_project_root(self):
        assert config.DATA_DIR == config.PROJECT_ROOT / "data"
        assert config.POLICIES_DIR == config.RAW_DIR / "policies"
        assert config.CHUNKS_PATH.name == "chunks.jsonl"
        assert config.TRACES_DIR == config.PROJECT_ROOT / "traces"


class reloaded_config:
    """Reload config with env overrides, then restore the original module state."""

    def __init__(self, env: dict[str, str]):
        self.env = env
        self._saved: dict[str, str | None] = {}

    def __enter__(self):
        for k, v in self.env.items():
            self._saved[k] = os.environ.get(k)
            os.environ[k] = v
        try:
            return importlib.reload(config)
        except BaseException:
            # importlib.reload can raise (e.g. an invalid numeric env var), and if it
            # does, __enter__ never returns, so Python will NOT call __exit__ — restore
            # the environment here too, or a raising reload leaks its env vars into
            # every later test/subprocess in the same process.
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for k, old in self._saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        importlib.reload(config)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# models.py (lazy singletons)
# ══════════════════════════════════════════════════════════════════════════════

class TestModels:
    @pytest.fixture()
    def models(self):
        from retail_support_agent import models as m
        for fn in (m.dense_model, m.bm25_model, m.reranker_model, m.qdrant_client):
            fn.cache_clear()
        yield m
        for fn in (m.dense_model, m.bm25_model, m.reranker_model, m.qdrant_client):
            fn.cache_clear()

    @pytest.mark.parametrize("fn_name,cls_name,model_attr", [
        ("dense_model", "TextEmbedding", "DENSE_MODEL"),
        ("bm25_model", "SparseTextEmbedding", "SPARSE_MODEL"),
        ("reranker_model", "TextCrossEncoder", "RERANKER_MODEL"),
    ])
    def test_embedding_models_are_built_once_with_configured_name(
            self, models, monkeypatch, fn_name, cls_name, model_attr):
        built = []

        def fake(name, cache_dir=None, threads=None):
            built.append((name, cache_dir, threads))
            return SimpleNamespace(name=name)

        monkeypatch.setattr(models, cls_name, fake)
        fn = getattr(models, fn_name)
        first, second = fn(), fn()
        assert first is second                       # lru_cache(maxsize=1)
        assert built == [(getattr(config, model_attr), str(config.MODEL_CACHE_DIR),
                          config.MODEL_THREADS)]

    @pytest.mark.parametrize("fn_name,cls_name", [
        ("dense_model", "TextEmbedding"),
        ("bm25_model", "SparseTextEmbedding"),
        ("reranker_model", "TextCrossEncoder"),
    ])
    def test_model_threads_setting_reaches_every_model(self, models, monkeypatch,
                                                       fn_name, cls_name):
        seen = {}
        monkeypatch.setattr(config, "MODEL_THREADS", 2)
        monkeypatch.setattr(models, cls_name,
                            lambda name, cache_dir=None, threads=None: seen.update(t=threads))
        getattr(models, fn_name)()
        assert seen["t"] == 2

    def test_model_threads_env_var_parsing(self, monkeypatch):
        monkeypatch.delenv("RSA_MODEL_THREADS", raising=False)
        with reloaded_config({}) as cfg:
            assert cfg.MODEL_THREADS is None         # unset: ONNX Runtime default
        with reloaded_config({"RSA_MODEL_THREADS": "3"}) as cfg:
            assert cfg.MODEL_THREADS == 3

    def test_qdrant_client_is_singleton_and_registered_for_close(self, models, monkeypatch,
                                                                 tmp_path):
        made, registered = [], []
        monkeypatch.setattr(config, "QDRANT_PATH", tmp_path / "qdrant")
        monkeypatch.setattr(models, "QdrantClient",
                            lambda path: made.append(path) or SimpleNamespace(
                                close=lambda: None, path=path))
        monkeypatch.setattr(models.atexit, "register", registered.append)
        c1 = models.qdrant_client()
        c2 = models.qdrant_client()
        assert c1 is c2 and len(made) == 1
        assert (tmp_path / "qdrant").is_dir()        # created before the client opens it
        assert registered and callable(registered[0])


# ══════════════════════════════════════════════════════════════════════════════
# llm.py — dataclasses and wire formats
# ══════════════════════════════════════════════════════════════════════════════

class TestLLMDataclasses:
    def test_toolcall_and_turn_defaults(self):
        call = ToolCall("id-1", "get_order_details", {"order_id": "ORD-1"})
        assert (call.id, call.name, call.input["order_id"]) == ("id-1", "get_order_details",
                                                                "ORD-1")
        turn = LLMTurn(text="hi")
        assert turn.tool_calls == [] and turn.usage == {}
        assert turn.stop_reason is None and turn.ttft_ms is None
        assert turn.wants_tools is False
        assert LLMTurn(text=None, tool_calls=[call]).wants_tools is True

    def test_turn_default_containers_are_not_shared(self):
        a, b = LLMTurn(text="a"), LLMTurn(text="b")
        a.tool_calls.append(ToolCall("x", "y", {}))
        a.usage["input_tokens"] = 1
        assert b.tool_calls == [] and b.usage == {}


class TestOpenAIWireFormat:
    def test_tools_are_translated_to_function_shape(self):
        out = llm_mod._openai_tools([SEARCH_TOOL, {"name": "t", "input_schema": {}}])
        assert out[0] == {"type": "function", "function": {
            "name": "search_policies", "description": SEARCH_TOOL["description"],
            "parameters": SEARCH_TOOL["input_schema"]}}
        assert out[1]["function"]["description"] == ""   # description optional

    def test_assistant_message_text_only(self):
        assert llm_mod._openai_assistant_message(LLMTurn(text="hello")) == {
            "role": "assistant", "content": "hello"}

    def test_assistant_message_with_tool_calls_serialises_args(self):
        turn = LLMTurn(text=None, tool_calls=[ToolCall("c0", "search_policies",
                                                       {"query": "returns"})])
        msg = llm_mod._openai_assistant_message(turn)
        assert msg["content"] == ""                  # None text becomes empty string
        assert msg["tool_calls"][0]["function"] == {
            "name": "search_policies", "arguments": '{"query": "returns"}'}
        assert msg["tool_calls"][0]["type"] == "function"

    def test_tool_results_are_one_message_per_call(self):
        pairs = [(ToolCall("a", "x", {}), {"ok": True}),
                 (ToolCall("b", "y", {}), {"ok": False})]
        msgs = llm_mod._openai_tool_results(pairs)
        assert [m["tool_call_id"] for m in msgs] == ["a", "b"]
        assert all(m["role"] == "tool" for m in msgs)
        assert json.loads(msgs[1]["content"]) == {"ok": False}


class TestAnthropicWireFormat:
    def test_assistant_message_blocks(self):
        turn = LLMTurn(text="thinking", tool_calls=[ToolCall("c0", "t", {"a": 1})])
        msg = llm_mod._anthropic_assistant_message(turn)
        assert msg["role"] == "assistant"
        assert msg["content"][0] == {"type": "text", "text": "thinking"}
        assert msg["content"][1] == {"type": "tool_use", "id": "c0", "name": "t",
                                     "input": {"a": 1}}

    def test_assistant_message_omits_empty_text(self):
        msg = llm_mod._anthropic_assistant_message(LLMTurn(text=None))
        assert msg["content"] == []

    def test_tool_results_ride_in_one_user_turn(self):
        msgs = llm_mod._anthropic_tool_results(
            [(ToolCall("a", "x", {}), {"ok": True}), (ToolCall("b", "y", {}), {"ok": True})])
        assert len(msgs) == 1 and msgs[0]["role"] == "user"
        assert [b["tool_use_id"] for b in msgs[0]["content"]] == ["a", "b"]

    def test_cache_breakpoints_without_messages(self):
        system_arg, msgs = llm_mod._anthropic_cache_breakpoints("SYS", [])
        assert system_arg == [{"type": "text", "text": "SYS",
                               "cache_control": {"type": "ephemeral"}}]
        assert msgs == []

    def test_cache_breakpoint_on_last_block_only(self):
        original = [{"role": "user", "content": "first"},
                    {"role": "assistant", "content": [{"type": "text", "text": "a"},
                                                      {"type": "text", "text": "b"}]}]
        _, msgs = llm_mod._anthropic_cache_breakpoints("SYS", original)
        assert msgs[1]["content"][1]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in msgs[1]["content"][0]
        assert msgs[0] == {"role": "user", "content": "first"}
        # The agent's own history must not be mutated.
        assert original[1]["content"][1] == {"type": "text", "text": "b"}

    def test_cache_breakpoint_promotes_string_content(self):
        _, msgs = llm_mod._anthropic_cache_breakpoints("S", [{"role": "user",
                                                              "content": "hi"}])
        assert msgs[0]["content"] == [{"type": "text", "text": "hi",
                                       "cache_control": {"type": "ephemeral"}}]

    def test_cache_breakpoint_tolerates_empty_content(self):
        _, msgs = llm_mod._anthropic_cache_breakpoints("S", [{"role": "user", "content": ""}])
        assert msgs[0]["content"] == ""   # left untouched, no crash

    def test_usage_mapping_with_and_without_cache_counters(self):
        plain = llm_mod._anthropic_usage(SimpleNamespace(input_tokens=10, output_tokens=5))
        assert plain == {"input_tokens": 10, "output_tokens": 5}
        full = llm_mod._anthropic_usage(SimpleNamespace(
            input_tokens=10, output_tokens=5, cache_read_input_tokens=7,
            cache_creation_input_tokens=3))
        assert full["cached_input_tokens"] == 7 and full["cache_write_input_tokens"] == 3
        zeros = llm_mod._anthropic_usage(SimpleNamespace(
            input_tokens=1, output_tokens=1, cache_read_input_tokens=0,
            cache_creation_input_tokens=0))
        assert "cached_input_tokens" not in zeros   # zero counters are not reported


# ══════════════════════════════════════════════════════════════════════════════
# llm.py — NVIDIA streaming client (SDK stubbed, no network)
# ══════════════════════════════════════════════════════════════════════════════

def _delta(content=None, tool_calls=None, finish=None, usage=None):
    choice = SimpleNamespace(
        delta=SimpleNamespace(content=content, tool_calls=tool_calls), finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=usage)


def _tc(index, id=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=id,
                           function=SimpleNamespace(name=name, arguments=arguments))


class FakeCompletions:
    def __init__(self, chunks):
        self.chunks = chunks
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return iter(self.chunks)


def nvidia_with(chunks):
    client = object.__new__(llm_mod.NvidiaClient)
    completions = FakeCompletions(chunks)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client.model = "test/model"
    client.max_tokens = 128
    client.temperature = 0.2
    client.enable_thinking = False
    return client, completions


class TestNvidiaClient:
    def test_missing_api_key_raises_before_any_request(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
            llm_mod.NvidiaClient()

    def test_constructor_sets_timeout_and_retries(self, monkeypatch):
        import openai
        seen = {}

        def fake_openai(**kwargs):
            seen.update(kwargs)
            return SimpleNamespace()

        monkeypatch.setattr(openai, "OpenAI", fake_openai)
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-fake")
        client = llm_mod.NvidiaClient(model="m", base_url="http://local")
        assert seen["timeout"] == config.LLM_TIMEOUT_S
        assert seen["max_retries"] == 2          # retries handled by the SDK, not by us
        assert seen["base_url"] == "http://local" and seen["api_key"] == "nvapi-fake"
        assert client.model == "m" and client.temperature == 0.2

    def test_text_stream_is_joined_and_ttft_recorded(self):
        client, comp = nvidia_with([
            _delta(content="Hello "), _delta(content="world"),
            _delta(finish="stop", usage=SimpleNamespace(
                prompt_tokens=11, completion_tokens=4, prompt_tokens_details=None)),
        ])
        turn = client.complete("SYS", [{"role": "user", "content": "hi"}], [SEARCH_TOOL])
        assert turn.text == "Hello world"
        assert turn.stop_reason == "stop"
        assert turn.usage == {"input_tokens": 11, "output_tokens": 4}
        assert turn.ttft_ms is not None and turn.ttft_ms >= 0
        assert turn.wants_tools is False
        # Request shape: system first, streaming on, tools translated.
        assert comp.kwargs["messages"][0] == {"role": "system", "content": "SYS"}
        assert comp.kwargs["stream"] is True
        assert comp.kwargs["stream_options"] == {"include_usage": True}
        assert comp.kwargs["tools"][0]["function"]["name"] == "search_policies"
        assert comp.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    def test_tool_call_fragments_are_accumulated_in_index_order(self):
        client, _ = nvidia_with([
            _delta(tool_calls=[_tc(1, id="b", name="get_order_details", arguments='{"order')]),
            _delta(tool_calls=[_tc(0, id="a", name="search_policies", arguments='{"que')]),
            _delta(tool_calls=[_tc(0, arguments='ry": "returns"}'),
                               _tc(1, arguments='_id": "ORD-1"}')]),
            _delta(finish="tool_calls"),
        ])
        turn = client.complete("SYS", [], [])
        assert [c.name for c in turn.tool_calls] == ["search_policies", "get_order_details"]
        assert turn.tool_calls[0].input == {"query": "returns"}
        assert turn.tool_calls[1].input == {"order_id": "ORD-1"}
        assert turn.text is None and turn.stop_reason == "tool_calls"

    def test_malformed_tool_arguments_degrade_to_empty_dict(self):
        client, _ = nvidia_with([
            _delta(tool_calls=[_tc(0, id="a", name="search_policies",
                                   arguments='{"query": "unterminated')]),
            _delta(finish="tool_calls"),
        ])
        turn = client.complete("SYS", [], [])
        assert turn.tool_calls[0].input == {}     # never raises on bad model JSON

    def test_missing_tool_call_id_gets_synthetic_id(self):
        client, _ = nvidia_with([
            _delta(tool_calls=[_tc(3, name="search_policies", arguments=None)]),
        ])
        turn = client.complete("SYS", [], [])
        assert turn.tool_calls[0].id == "call_3"
        assert turn.tool_calls[0].input == {}      # empty arguments -> {}

    def test_cached_prompt_tokens_recorded_when_reported(self):
        client, _ = nvidia_with([
            _delta(content="x", usage=SimpleNamespace(
                prompt_tokens=100, completion_tokens=2,
                prompt_tokens_details=SimpleNamespace(cached_tokens=64))),
        ])
        turn = client.complete("SYS", [], [])
        assert turn.usage["cached_input_tokens"] == 64

    def test_empty_stream_yields_empty_turn(self):
        client, _ = nvidia_with([])
        turn = client.complete("SYS", [], [])
        assert turn.text is None and turn.tool_calls == [] and turn.usage == {}
        assert turn.ttft_ms is None

    def test_chunks_without_choices_are_skipped(self):
        client, _ = nvidia_with([
            SimpleNamespace(choices=[], usage=SimpleNamespace(
                prompt_tokens=5, completion_tokens=1, prompt_tokens_details=None)),
            _delta(content="ok", finish="stop"),
        ])
        turn = client.complete("SYS", [], [])
        assert turn.text == "ok" and turn.usage["input_tokens"] == 5


class FakeStream:
    def __init__(self, events, final_message):
        self._events = events
        self._final = final_message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._final


class TestAnthropicClient:
    def _client(self, events, final_message, capture):
        client = object.__new__(llm_mod.AnthropicClient)
        client.model = "claude-test"
        client.max_tokens = 256

        def stream(**kwargs):
            capture.update(kwargs)
            return FakeStream(events, final_message)

        client._client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
        return client

    def test_blocks_are_split_into_text_and_tool_calls(self, monkeypatch):
        monkeypatch.setattr(config, "PROMPT_CACHE", False)
        final_message = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Looking that up."),
                     SimpleNamespace(type="tool_use", id="toolu_1", name="get_order_details",
                                     input={"order_id": "ORD-10012"}),
                     SimpleNamespace(type="text", text="One moment.")],
            usage=SimpleNamespace(input_tokens=50, output_tokens=9),
            stop_reason="tool_use")
        events = [SimpleNamespace(type="message_start"),
                  SimpleNamespace(type="content_block_start"),
                  SimpleNamespace(type="content_block_delta")]
        capture: dict = {}
        turn = self._client(events, final_message, capture).complete(
            "SYS", [{"role": "user", "content": "q"}], [SEARCH_TOOL])
        assert turn.text == "Looking that up.\nOne moment."
        assert [c.name for c in turn.tool_calls] == ["get_order_details"]
        assert turn.tool_calls[0].input == {"order_id": "ORD-10012"}
        assert turn.usage == {"input_tokens": 50, "output_tokens": 9}
        assert turn.stop_reason == "tool_use" and turn.ttft_ms is not None
        # Tools go over the wire in Anthropic shape, untranslated.
        assert capture["tools"] == [SEARCH_TOOL]
        assert capture["system"] == "SYS"          # plain system when caching is off

    def test_prompt_cache_adds_breakpoints(self, monkeypatch):
        monkeypatch.setattr(config, "PROMPT_CACHE", True)
        final_message = SimpleNamespace(
            content=[], usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            stop_reason="end_turn")
        capture: dict = {}
        turn = self._client([], final_message, capture).complete(
            "SYS", [{"role": "user", "content": "q"}], [])
        assert capture["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert capture["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert turn.text is None and turn.ttft_ms is None   # no stream events seen


class TestProviderSelection:
    def test_build_llm_dispatches_on_provider(self, monkeypatch):
        monkeypatch.setattr(llm_mod, "NvidiaClient", lambda: "NVIDIA")
        monkeypatch.setattr(llm_mod, "AnthropicClient", lambda: "ANTHROPIC")
        assert llm_mod.build_llm("nvidia") == "NVIDIA"
        assert llm_mod.build_llm("  ANTHROPIC ") == "ANTHROPIC"

    def test_build_llm_defaults_to_config_provider(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
        monkeypatch.setattr(llm_mod, "AnthropicClient", lambda: "ANTHROPIC")
        assert llm_mod.build_llm() == "ANTHROPIC"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            llm_mod.build_llm("openai")


class TestScriptedClient:
    def test_records_calls_and_exhausts_gracefully(self):
        client = ScriptedClient([final("one")])
        assert client.complete("S", [{"role": "user", "content": "q"}], [SEARCH_TOOL]).text \
            == "one"
        spill = client.complete("S", [], [])
        assert spill.text == "(scripted client exhausted)" and spill.stop_reason == "end_turn"
        assert client.seen[0]["n_tools"] == 1

    def test_wire_selects_message_format(self):
        turn = LLMTurn(text="t", tool_calls=[ToolCall("c", "n", {})])
        assert ScriptedClient([], wire="openai").assistant_message(turn)["content"] == "t"
        anth = ScriptedClient([], wire="anthropic").assistant_message(turn)
        assert anth["content"][0]["type"] == "text"
        assert ScriptedClient([], wire="anthropic").tool_result_messages(
            [(ToolCall("c", "n", {}), {})])[0]["role"] == "user"


# ══════════════════════════════════════════════════════════════════════════════
# tracing.py
# ══════════════════════════════════════════════════════════════════════════════

class FakeObservation:
    def __init__(self, parent, kwargs):
        self.parent = parent
        self.kwargs = kwargs
        self.children: list[FakeObservation] = []
        self.ended = False
        self.updates: list[dict] = []

    def start_observation(self, **kwargs):
        child = FakeObservation(self, kwargs)
        self.children.append(child)
        return child

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def end(self):
        self.ended = True


class FakeLangfuse(FakeObservation):
    def __init__(self):
        super().__init__(None, {})
        self.flushed = 0

    def flush(self):
        self.flushed += 1


@pytest.fixture()
def tracing_reset():
    saved = (tracing._lf, tracing._checked)
    yield
    tracing._lf, tracing._checked = saved


class TestTracing:
    def test_disabled_without_keys(self, monkeypatch, tracing_reset):
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        tracing._lf, tracing._checked = None, False
        assert tracing._client() is None
        assert tracing.start_trace("s", "n", "u", "in") is None
        # All child helpers are no-ops on a None trace.
        assert tracing.end_trace(None, "out", {}) is None
        assert tracing.add_generation(None, "g", "m", [], "c", {}, 1.0) is None
        assert tracing.add_span(None, "s", {}, {}, 1.0) is None
        tracing.flush()

    def test_init_is_attempted_only_once(self, monkeypatch, tracing_reset):
        calls = []
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")

        import langfuse
        monkeypatch.setattr(langfuse, "Langfuse",
                            lambda **kw: calls.append(kw) or FakeLangfuse())
        tracing._lf, tracing._checked = None, False
        first = tracing._client()
        second = tracing._client()
        assert first is second and len(calls) == 1
        assert calls[0]["secret_key"] == "sk-lf-x"
        assert calls[0]["host"]

    def test_init_failure_disables_tracing(self, monkeypatch, capsys, tracing_reset):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
        import langfuse

        def boom(**kw):
            raise RuntimeError("no creds")

        monkeypatch.setattr(langfuse, "Langfuse", boom)
        tracing._lf, tracing._checked = None, False
        assert tracing._client() is None
        assert "observability disabled" in capsys.readouterr().out

    def test_root_trace_shape(self, monkeypatch, tracing_reset):
        fake = FakeLangfuse()
        monkeypatch.setattr(tracing, "_client", lambda: fake)
        trace = tracing.start_trace("sess-1", "support_agent", "CUST-1", "hello",
                                    tags=["retail-support"])
        assert trace.kwargs["as_type"] == "agent"
        assert trace.kwargs["input"] == "hello"
        assert trace.kwargs["metadata"] == {"session_id": "sess-1", "user_id": "CUST-1",
                                            "tags": ["retail-support"]}

    def test_start_trace_swallows_sdk_errors(self, monkeypatch, tracing_reset):
        class Broken:
            def start_observation(self, **kw):
                raise RuntimeError("sdk down")

        monkeypatch.setattr(tracing, "_client", lambda: Broken())
        assert tracing.start_trace("s", "n", "u", "i") is None

    def test_children_are_nested_under_the_trace(self, monkeypatch, tracing_reset):
        fake = FakeLangfuse()
        monkeypatch.setattr(tracing, "_client", lambda: fake)
        trace = tracing.start_trace("s", "support_agent", "u", "q")
        tracing.add_generation(trace, "llm_turn_0", "model-x",
                               [{"role": "user", "content": "q"}], "answer",
                               {"input_tokens": 12, "output_tokens": 3}, 120.44, ttft_ms=50.0)
        tracing.add_span(trace, "search_policies", {"query": "q"}, {"grounded": True},
                         33.33, metadata={"ok": True}, span_type="retriever")
        tracing.end_trace(trace, "answer", {"outcome": "answered"})

        assert len(fake.children) == 1 and fake.children[0] is trace
        gen, span = trace.children
        assert gen.kwargs["as_type"] == "generation" and gen.ended
        assert gen.kwargs["usage_details"] == {"input": 12, "output": 3}
        assert gen.kwargs["metadata"] == {"latency_ms": 120.4, "ttft_ms": 50.0}
        assert span.kwargs["as_type"] == "retriever" and span.ended
        assert span.kwargs["metadata"] == {"latency_ms": 33.3, "ok": True}
        assert trace.updates == [{"output": "answer", "metadata": {"outcome": "answered"}}]
        assert trace.ended

    def test_generation_usage_defaults_to_zero(self, monkeypatch, tracing_reset):
        fake = FakeLangfuse()
        monkeypatch.setattr(tracing, "_client", lambda: fake)
        trace = tracing.start_trace("s", "n", "u", "q")
        tracing.add_generation(trace, "g", "m", [], "", {}, 1.0)
        assert trace.children[0].kwargs["usage_details"] == {"input": 0, "output": 0}

    @pytest.mark.parametrize("given,expected", [
        ("retriever", "retriever"), ("tool", "tool"), ("guardrail", "guardrail"),
        ("span", "span"), ("nonsense", "tool"), ("", "tool"),
    ])
    def test_span_type_is_mapped_to_valid_values(self, monkeypatch, tracing_reset,
                                                 given, expected):
        fake = FakeLangfuse()
        monkeypatch.setattr(tracing, "_client", lambda: fake)
        trace = tracing.start_trace("s", "n", "u", "q")
        tracing.add_span(trace, "x", {}, {}, 1.0, span_type=given)
        assert trace.children[0].kwargs["as_type"] == expected

    def test_child_and_end_errors_are_swallowed(self, tracing_reset):
        class Broken:
            def start_observation(self, **kw):
                raise RuntimeError("boom")

            def update(self, **kw):
                raise RuntimeError("boom")

            def end(self):
                raise RuntimeError("boom")

        broken = Broken()
        tracing.add_generation(broken, "g", "m", [], "c", {}, 1.0)
        tracing.add_span(broken, "s", {}, {}, 1.0)
        tracing.end_trace(broken, "o", {})     # must not raise

    def test_flush_is_safe_and_reaches_the_client(self, tracing_reset):
        fake = FakeLangfuse()
        tracing._lf = fake
        tracing.flush()
        assert fake.flushed == 1
        tracing._lf = None
        tracing.flush()


# ══════════════════════════════════════════════════════════════════════════════
# mcp_server.py
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def mcp_server(monkeypatch):
    from retail_support_agent import mcp_server as srv
    monkeypatch.setenv("RSA_REQUESTER_ID", "CUST-1001")
    return srv


class TestMCPServerIdentity:
    def test_requester_read_from_env(self, mcp_server, monkeypatch):
        assert mcp_server._requester() == "CUST-1001"
        monkeypatch.setenv("RSA_REQUESTER_ID", "   ")
        assert mcp_server._requester() is None      # whitespace is no identity
        monkeypatch.delenv("RSA_REQUESTER_ID")
        assert mcp_server._requester() is None

    def test_no_identity_envelope_shape(self, mcp_server):
        env = mcp_server._no_identity()
        assert env["ok"] is False
        assert env["error"]["code"] == "NO_SESSION_IDENTITY"
        assert env["error"]["retryable"] is False

    def test_guard_injects_identity_as_keyword(self, mcp_server):
        seen = {}

        def op(order_id, requester_id=None):
            seen.update(order_id=order_id, requester_id=requester_id)
            return {"ok": True, "data": {"order_id": order_id}}

        assert mcp_server._guard(op, "ORD-10012")["ok"] is True
        assert seen == {"order_id": "ORD-10012", "requester_id": "CUST-1001"}

    def test_guard_refuses_without_identity(self, mcp_server, monkeypatch):
        monkeypatch.delenv("RSA_REQUESTER_ID")
        called = []
        assert mcp_server._guard(lambda *a, **k: called.append(1))["error"]["code"] \
            == "NO_SESSION_IDENTITY"
        assert called == []                          # ops never runs unauthenticated

    def test_guard_maps_a_crashing_tool_to_internal_error(self, mcp_server):
        def boom(requester_id=None):
            raise sqlite_error()

        env = mcp_server._guard(boom)
        assert env == {"ok": False, "error": {
            "code": "INTERNAL_ERROR",
            "message": "Unexpected server error. Please try again.", "retryable": True}}
        # The real exception text never reaches the model.
        assert "secret" not in json.dumps(env)


def sqlite_error():
    return RuntimeError("database locked: /secret/path.db")


class TestMCPServerTools:
    async def test_registered_tools_and_schemas(self, mcp_server):
        tools = {t.name: t for t in await mcp_server.mcp.list_tools()}
        assert set(tools) == {"get_order_details", "get_return_refund_status",
                              "create_return_request", "create_escalation_ticket"}
        for name in ("get_order_details", "get_return_refund_status"):
            schema = tools[name].input_schema
            assert schema["required"] == ["order_id"]
            assert schema["properties"]["order_id"]["type"] == "string"
            assert "ORD-#####" in schema["properties"]["order_id"]["description"]
        ret = tools["create_return_request"].input_schema
        assert set(ret["required"]) == {"order_id", "reason"}
        esc = tools["create_escalation_ticket"].input_schema
        assert set(esc["required"]) == {"summary", "reason"}   # order_id optional
        assert all(t.description for t in tools.values())
        # Identity is never a tool parameter — the model cannot choose whose data to read.
        assert not any("requester" in json.dumps(t.input_schema).lower()
                       for t in tools.values())

    async def test_tool_call_passes_identity_and_args(self, mcp_server, monkeypatch):
        seen = {}
        monkeypatch.setattr(mcp_server.ops, "get_order_details",
                            lambda order_id, requester_id=None: seen.update(
                                o=order_id, r=requester_id) or {"ok": True, "data": {}})
        await mcp_server.mcp.call_tool("get_order_details", {"order_id": "ORD-10012"})
        assert seen == {"o": "ORD-10012", "r": "CUST-1001"}

    async def test_missing_required_argument_is_rejected_by_validation(self, mcp_server,
                                                                       monkeypatch):
        called = []
        monkeypatch.setattr(mcp_server.ops, "get_order_details",
                            lambda *a, **k: called.append(1))
        try:
            result = await mcp_server.mcp.call_tool("get_order_details", {})
        except Exception as exc:                     # SDK may raise instead of returning
            assert "order_id" in str(exc)
        else:
            assert getattr(result, "is_error", False), result
        assert called == []                          # the op never ran

    async def test_unknown_tool_is_rejected(self, mcp_server):
        try:
            result = await mcp_server.mcp.call_tool("refund_everything", {})
        except Exception as exc:
            assert "refund_everything" in str(exc) or "Unknown" in str(exc)
        else:
            assert getattr(result, "is_error", False), result

    def test_create_return_request_wraps_crashes(self, mcp_server, monkeypatch):
        monkeypatch.setattr(mcp_server.ops, "create_return_request",
                            lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
        env = mcp_server.create_return_request("ORD-10012", "damaged")
        assert env["error"]["code"] == "INTERNAL_ERROR" and env["error"]["retryable"] is True

    def test_create_return_request_requires_identity(self, mcp_server, monkeypatch):
        monkeypatch.delenv("RSA_REQUESTER_ID")
        called = []
        monkeypatch.setattr(mcp_server.ops, "create_return_request",
                            lambda *a, **k: called.append(1))
        assert mcp_server.create_return_request("ORD-1", "r")["error"]["code"] \
            == "NO_SESSION_IDENTITY"
        assert called == []

    def test_escalation_passes_optional_order_id_positionally(self, mcp_server, monkeypatch):
        seen = {}
        monkeypatch.setattr(mcp_server.ops, "create_escalation_ticket",
                            lambda s, r, o, requester_id=None: seen.update(
                                s=s, r=r, o=o, rid=requester_id) or {"ok": True})
        mcp_server.create_escalation_ticket("summary", "policy_unclear")
        assert seen == {"s": "summary", "r": "policy_unclear", "o": None, "rid": "CUST-1001"}


# ══════════════════════════════════════════════════════════════════════════════
# mcp_client.py
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def loop_thread():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=5)
    loop.close()


def client_with(loop, call_tool):
    from retail_support_agent.mcp_client import MCPToolClient
    c = MCPToolClient("CUST-1001")
    c._loop = loop
    c._session = SimpleNamespace(call_tool=call_tool)
    return c


def text_result(text):
    return SimpleNamespace(structured_content=None,
                           content=[SimpleNamespace(text=text)])


class TestMCPToolClient:
    def test_call_before_start_returns_typed_transport_error(self):
        from retail_support_agent.mcp_client import MCPToolClient
        env = MCPToolClient("CUST-1001").call_tool("get_order_details", {"order_id": "ORD-1"})
        assert env["ok"] is False
        assert env["error"]["code"] == "TOOL_UNAVAILABLE" and env["error"]["retryable"] is True

    def test_structured_content_is_returned_verbatim(self, loop_thread):
        async def call(name, args):
            return SimpleNamespace(structured_content={"ok": True, "data": {"n": name}},
                                   content=None)

        c = client_with(loop_thread, call)
        assert c.call_tool("get_order_details", {"order_id": "ORD-1"}) == {
            "ok": True, "data": {"n": "get_order_details"}}

    def test_text_content_is_json_parsed(self, loop_thread):
        async def call(name, args):
            return text_result(json.dumps({"ok": False, "error": {"code": "NOT_FOUND"}}))

        assert client_with(loop_thread, call).call_tool("x", {})["error"]["code"] == "NOT_FOUND"

    def test_unparseable_text_content_is_typed(self, loop_thread):
        async def call(name, args):
            return text_result("not json at all")

        env = client_with(loop_thread, call).call_tool("x", {})
        assert env["error"]["code"] == "BAD_TOOL_OUTPUT" and env["error"]["retryable"] is False

    def test_no_content_is_typed(self, loop_thread):
        async def call(name, args):
            return SimpleNamespace(structured_content=None, content=[])

        env = client_with(loop_thread, call).call_tool("x", {})
        assert env["error"]["code"] == "BAD_TOOL_OUTPUT"
        assert "no content" in env["error"]["message"].lower()

    def test_tool_that_raises_becomes_retryable_transport_error(self, loop_thread):
        async def call(name, args):
            raise RuntimeError("server died mid-call")

        env = client_with(loop_thread, call).call_tool("get_order_details", {})
        assert env["ok"] is False
        assert env["error"]["code"] == "TOOL_UNAVAILABLE"
        assert env["error"]["retryable"] is True
        assert "get_order_details" in env["error"]["message"]

    def test_call_timeout_is_mapped_to_transport_error(self, loop_thread, monkeypatch):
        monkeypatch.setattr(config, "MCP_CALL_TIMEOUT", 0.05)

        async def call(name, args):
            await asyncio.sleep(5)

        env = client_with(loop_thread, call).call_tool("slow_tool", {})
        assert env["error"]["code"] == "TOOL_UNAVAILABLE"

    def test_list_tools_returns_a_copy(self):
        from retail_support_agent.mcp_client import MCPToolClient
        c = MCPToolClient("CUST-1001")
        c._tools = [{"name": "a"}]
        got = c.list_tools()
        got.append({"name": "b"})
        assert c._tools == [{"name": "a"}]

    def test_start_times_out_when_server_never_becomes_ready(self, monkeypatch):
        from retail_support_agent.mcp_client import MCPToolClient
        c = MCPToolClient("CUST-1001")
        monkeypatch.setattr(c, "_run", lambda: None)
        with pytest.raises(RuntimeError, match="did not become ready"):
            c.start(timeout=0.2)

    def test_start_surfaces_server_start_errors(self, monkeypatch):
        from retail_support_agent.mcp_client import MCPToolClient
        c = MCPToolClient("CUST-1001")

        def fail():
            c._start_error = "Traceback: ModuleNotFoundError"
            c._ready.set()

        monkeypatch.setattr(c, "_run", fail)
        with pytest.raises(RuntimeError, match="failed to start"):
            c.start(timeout=5)

    def test_close_is_safe_without_a_thread(self):
        from retail_support_agent.mcp_client import MCPToolClient
        c = MCPToolClient("CUST-1001")
        c.close()
        assert c._stop.is_set()

    def test_tool_model_exposes_input_schema(self):
        """The client reads `t.input_schema` off the SDK Tool model; guard that name."""
        from mcp.types import Tool
        assert "input_schema" in Tool.model_fields


# ══════════════════════════════════════════════════════════════════════════════
# agent.py — tool dispatch helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestLocalTools:
    def test_local_tool_schemas(self):
        assert [t["name"] for t in LOCAL_TOOLS] == ["search_policies",
                                                    "calculate_return_deadline"]
        assert SEARCH_TOOL["input_schema"]["required"] == ["query"]
        assert DEADLINE_TOOL["input_schema"]["required"] == ["category", "purchase_date"]
        assert DEADLINE_TOOL["input_schema"]["properties"]["category"]["enum"]

    def test_tool_list_is_byte_stable_and_sorted(self, tmp_path):
        agent, _ = make_agent([], tmp_path)
        names = [t["name"] for t in agent._tools()]
        assert names[:2] == ["search_policies", "calculate_return_deadline"]
        assert names[2:] == sorted(names[2:])        # prefix cache stability
        assert agent._tools() == agent._tools()

    def test_deadline_rejects_bad_dates(self, tmp_path):
        agent, _ = make_agent([], tmp_path)
        env = agent._deadline({"category": "Electronics", "purchase_date": "12/01/2024"})
        assert env["ok"] is False
        assert env["error"]["code"] == "VALIDATION_ERROR" and env["error"]["retryable"] is True

    def test_deadline_computes_holiday_extension(self, tmp_path):
        agent, _ = make_agent([], tmp_path)
        env = agent._deadline({"category": "Laptops & Computers",
                               "purchase_date": "2024-12-01"})
        assert env["ok"] is True
        assert env["data"]["rule_applied"] == "holiday_extension"
        assert env["data"]["deadline"] == "2025-01-31"
        assert "ok" not in env["data"]               # the flag is popped, not leaked

    def test_deadline_unknown_category_is_typed_and_not_retryable(self, tmp_path):
        agent, _ = make_agent([], tmp_path)
        env = agent._deadline({"category": "Unicorns", "purchase_date": "2024-12-01"})
        assert env["error"]["code"] == "UNKNOWN_CATEGORY" and env["error"]["retryable"] is False

    def test_deadline_treats_empty_delivery_date_as_absent(self, tmp_path):
        agent, _ = make_agent([], tmp_path)
        env = agent._deadline({"category": "Laptops & Computers",
                               "purchase_date": "2024-06-01", "delivery_date": ""})
        assert env["data"]["window_counted_from"] == "purchase"

    def test_search_rejects_empty_query(self, tmp_path):
        agent, _ = make_agent([], tmp_path)
        env = agent._search("   ", [])
        assert env["error"]["code"] == "VALIDATION_ERROR" and env["error"]["retryable"] is False

    def test_search_collects_deduplicated_evidence(self, tmp_path):
        chunks = [FakeChunk("a" * 16, "return_policy.md", "Phones", "text one"),
                  FakeChunk("a" * 16, "return_policy.md", "Phones", "text one"),
                  FakeChunk("c" * 16, "shipping.md", "Intl", "text two", rerank_score=1.0)]
        agent, _ = make_agent([], tmp_path, chunks=chunks)
        evidence: list[dict] = []
        env = agent._search("returns", evidence)
        assert env["data"]["grounded"] is True
        assert [e["source"] for e in evidence] == ["return_policy.md#aaaaaaaa",
                                                   "shipping.md#cccccccc"]
        assert env["data"]["chunks"][0]["text"] == "text one"   # full text reaches the model

    def test_search_below_threshold_is_ungrounded_and_adds_no_evidence(self, tmp_path):
        agent, _ = make_agent([], tmp_path,
                              chunks=[FakeChunk("d" * 16, "x.md", "s", "t",
                                                rerank_score=-5.0)])
        evidence: list[dict] = []
        env = agent._search("off topic", evidence)
        assert env["data"]["grounded"] is False and evidence == []

    def test_search_falls_back_to_dense_score_without_reranking(self, tmp_path):
        agent, _ = make_agent([], tmp_path,
                              chunks=[FakeChunk("e" * 16, "x.md", "s", "t", score=0.77,
                                                rerank_score=None)])
        env = agent._search("q", [])
        assert env["data"]["chunks"][0]["score"] == 0.77

    def test_unknown_tool_name_is_proxied_to_mcp(self, tmp_path):
        mcp = FakeMCP({"get_order_details": {"ok": True, "data": {"order_id": "ORD-1"}}})
        agent, _ = make_agent([], tmp_path, mcp=mcp)
        assert agent._run_tool("get_order_details", {"order_id": "ORD-1"}, [])["ok"] is True
        assert mcp.calls == [("get_order_details", {"order_id": "ORD-1"})]


class TestTrimResult:
    def test_error_envelope_is_trimmed_to_the_error(self, tmp_path):
        agent, _ = make_agent([], tmp_path)
        err = {"code": "ORDER_NOT_FOUND", "message": "nope", "retryable": False}
        assert agent._trim_result({"ok": False, "error": err}) == err

    def test_chunks_are_summarised_to_three_sources(self, tmp_path):
        agent, _ = make_agent([], tmp_path)
        data = {"grounded": True, "chunks": [{"source": f"f{i}.md", "score": i}
                                             for i in range(5)]}
        assert agent._trim_result({"ok": True, "data": data}) == {
            "grounded": True, "top": ["f0.md (0)", "f1.md (1)", "f2.md (2)"]}

    def test_return_history_is_dropped_from_trace(self, tmp_path):
        agent, _ = make_agent([], tmp_path)
        out = agent._trim_result({"ok": True, "data": {"order_id": "ORD-1",
                                                       "return_history": [{"id": "RET-1"}]}})
        assert out == {"order_id": "ORD-1"}


class TestClassifyOutcome:
    """Outcomes come from what the tools did, never from the answer's wording."""

    EV = [{"source": "p.md"}]

    def test_grounded_search_is_answered(self):
        assert SupportAgent._classify_outcome(self.EV, ["search_policies"]) == "answered"

    def test_search_without_evidence_is_ungrounded(self):
        assert SupportAgent._classify_outcome([], ["search_policies"]) \
            == "refused_ungrounded"

    def test_no_search_and_no_evidence_is_answered(self):
        assert SupportAgent._classify_outcome([], ["get_order_details"]) == "answered"

    def test_created_escalation_ticket_is_escalated(self):
        ctx = [{"tool": "create_escalation_ticket", "args": {},
                "result": {"ok": True, "data": {"ticket_id": "ESC-1"}}}]
        assert SupportAgent._classify_outcome(
            [], ["search_policies", "create_escalation_ticket"], ctx) == "escalated"

    def test_failed_escalation_is_not_escalated(self):
        ctx = [{"tool": "create_escalation_ticket", "args": {},
                "result": {"ok": False, "error": {"code": "INTERNAL_ERROR"}}}]
        assert SupportAgent._classify_outcome(
            self.EV, ["search_policies", "create_escalation_ticket"], ctx) == "answered"

    def test_ungrounded_search_with_order_data_is_answered(self):
        ctx = [{"tool": "get_order_details", "args": {},
                "result": {"ok": True, "data": {"order_id": "ORD-1"}}},
               {"tool": "search_policies", "args": {}, "result": {"ok": True}}]
        assert SupportAgent._classify_outcome(
            [], ["get_order_details", "search_policies"], ctx) == "answered"

    def test_ungrounded_search_after_blocked_tool_is_ungrounded(self):
        ctx = [{"tool": "get_order_details", "args": {},
                "result": {"ok": False, "error": {"code": "GUARDRAIL_BLOCKED"}}}]
        assert SupportAgent._classify_outcome(
            [], ["get_order_details", "search_policies"], ctx) == "refused_ungrounded"

    @pytest.mark.parametrize("wording", [
        "Let me hand this to a human agent.", "Please contact us for help.",
        "I could not find supporting policy for that."])
    def test_wording_does_not_change_the_outcome(self, wording):
        # Same state, different wording: same label (formerly phrase-matched).
        assert SupportAgent._classify_outcome(self.EV, ["search_policies"]) == "answered"


class TestUnauthorizedOutcome:
    """Asking about another customer's order is a refusal, however the model words it."""

    UNAUTH = {"ok": False, "error": {"code": "UNAUTHORIZED", "message": "x"}}

    def _ctx(self, *results):
        return [{"tool": "get_order_details", "args": {}, "result": r} for r in results]

    def test_all_order_lookups_unauthorized_is_a_refusal(self):
        assert SupportAgent._classify_outcome(
            [], ["get_order_details"],
            self._ctx(self.UNAUTH)) == "refused_unauthorized"

    def test_one_authorized_lookup_keeps_normal_classification(self):
        ok = {"ok": True, "data": {"order_id": "ORD-1"}}
        assert SupportAgent._classify_outcome(
            [], ["get_order_details"] * 2,
            self._ctx(self.UNAUTH, ok)) == "answered"

    def test_order_not_found_is_not_an_ownership_refusal(self):
        missing = {"ok": False, "error": {"code": "ORDER_NOT_FOUND", "message": "x"}}
        assert SupportAgent._classify_outcome(
            [], ["get_order_details"],
            self._ctx(missing)) == "answered"

    def test_search_only_context_is_unaffected(self):
        ctx = [{"tool": "search_policies", "result": {"ok": True}}]
        assert SupportAgent._classify_outcome(
            [{"source": "p.md"}], ["search_policies"], ctx) == "answered"

    def test_end_to_end_through_the_agent_loop(self, tmp_path):
        mcp = FakeMCP({"get_order_details": self.UNAUTH})
        agent, _ = make_agent([
            tool_turn(("get_order_details", {"order_id": "ORD-10185"})),
            final("Sorry, I could not find details for order ORD-10185."),
        ], tmp_path, mcp=mcp)
        result = agent.answer("What is the status of order ORD-10185?")
        assert mcp.calls == [("get_order_details", {"order_id": "ORD-10185"})]
        assert result.outcome == "refused_unauthorized"
        assert trace_lines(result)[-1]["outcome"] == "refused_unauthorized"


# ══════════════════════════════════════════════════════════════════════════════
# agent.py — the tool-call loop
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentLoop:
    def test_unevidenced_answer_forces_one_policy_search(self, tmp_path):
        mcp = FakeMCP()
        agent, client = make_agent(unevidenced("Laptops have a 15-day window."), tmp_path,
                                   mcp=mcp)
        result = agent.answer("What is the laptop return window?")
        assert len(client.seen) == 2 and mcp.calls == []
        assert result.tool_calls == ["search_policies"]
        forced = [s for s in result.steps if s.kind == "tool"]
        assert len(forced) == 1 and forced[0].detail["forced"] is True
        assert forced[0].detail["args"] == {"query": "What is the laptop return window?"}
        assert result.outcome == "refused_ungrounded"     # fake retriever found nothing

    def test_forced_search_keeps_the_draft_visible_to_the_model(self, tmp_path):
        # Regression (eval a25): the draft was dropped, so the model rewrote a correct
        # "please send the full ORD- id" answer blind around an irrelevant search.
        draft = "Order IDs look like ORD-10012; could you check the ID?"
        agent, client = make_agent([final(draft), final(draft)], tmp_path)
        result = agent.answer("Where is my order 12345?")
        assistant = [m for m in client.seen[1]["messages"] if m["role"] == "assistant"][-1]
        assert assistant["content"] == draft                     # draft kept ...
        assert assistant["tool_calls"][0]["function"]["name"] == "search_policies"  # ... + search
        assert result.answer == draft

    def test_conversation_recall_is_answered_without_a_search(self, tmp_path):
        agent, client = make_agent([final("This is the first message of our chat.")],
                                   tmp_path)
        result = agent.answer("What did I ask earlier?")
        assert len(client.seen) == 1 and result.tool_calls == []

    def test_definitive_order_error_needs_no_policy_search(self, tmp_path):
        mcp = FakeMCP({"get_order_details": {"ok": False, "error": {
            "code": "ORDER_NOT_FOUND", "message": "x", "retryable": False}}})
        agent, client = make_agent([
            tool_turn(("get_order_details", {"order_id": "ORD-99999"})),
            final("I couldn't find ORD-99999; please double-check the ID."),
        ], tmp_path, mcp=mcp)
        result = agent.answer("Where is ORD-99999?")
        assert result.tool_calls == ["get_order_details"] and len(client.seen) == 2

    def test_forced_search_is_skipped_when_its_budget_is_spent(self, tmp_path, monkeypatch):
        from retail_support_agent import guardrails
        monkeypatch.setitem(guardrails.TOOL_CALL_BUDGET, "search_policies", 0)
        agent, client = make_agent([final("Hello! How can I help?")], tmp_path)
        result = agent.answer("hello")
        assert len(client.seen) == 1 and result.tool_calls == []

    def test_multi_tool_turn_runs_every_call_in_order(self, tmp_path):
        mcp = FakeMCP({"get_order_details": {"ok": True, "data": {"order_id": "ORD-10012"}},
                       "get_return_refund_status": {"ok": True, "data": {"returns": []}}})
        agent, client = make_agent([
            tool_turn(("get_order_details", {"order_id": "ORD-10012"}),
                      ("get_return_refund_status", {"order_id": "ORD-10012"}),
                      ("search_policies", {"query": "returns"})),
            final("All checked."),
        ], tmp_path, mcp=mcp, chunks=[FakeChunk("f" * 16, "p.md", "s", "t")])
        result = agent.answer("Status and policy for ORD-10012?")
        assert result.tool_calls == ["get_order_details", "get_return_refund_status",
                                     "search_policies"]
        assert [c[0] for c in mcp.calls] == ["get_order_details", "get_return_refund_status"]
        # One tool message per call, all in the same follow-up request.
        assert len(tool_messages(client)) == 3
        tool_steps = [s for s in result.steps if s.kind == "tool"]
        assert [s.name for s in tool_steps] == result.tool_calls

    def test_tool_error_envelope_is_relayed_and_traced(self, tmp_path):
        mcp = FakeMCP({"get_order_details": {"ok": False, "error": {
            "code": "ORDER_NOT_FOUND", "message": "No such order.", "retryable": False}}})
        agent, client = make_agent([
            tool_turn(("get_order_details", {"order_id": "ORD-99999"})),
            final("I couldn't find that order; please double-check the ID."),
        ], tmp_path, mcp=mcp)
        result = agent.answer("Where is ORD-99999?")
        assert tool_messages(client)[0]["error"]["code"] == "ORDER_NOT_FOUND"
        step = next(s for s in result.steps if s.kind == "tool")
        assert step.detail["ok"] is False and step.detail["error"] == "ORDER_NOT_FOUND"
        assert step.detail["result"]["code"] == "ORDER_NOT_FOUND"

    def test_non_dict_tool_arguments_are_blocked_before_dispatch(self, tmp_path):
        """A model that emits a bare string instead of an arguments object."""
        mcp = FakeMCP()
        agent, client = make_agent([
            LLMTurn(text=None, tool_calls=[ToolCall("c0", "get_order_details", "ORD-1")]),
            final("Sorry, I hit a problem looking that up."),
        ], tmp_path, mcp=mcp)
        agent.answer("where is my order")
        assert mcp.calls == []
        err = tool_messages(client)[0]["error"]
        assert err["code"] == "GUARDRAIL_BLOCKED" and "invalid_tool_args" in err["message"]

    def test_step_cap_truncates_the_conversation(self, tmp_path):
        turns = [tool_turn(("search_policies", {"query": f"q{i}"}))
                 for i in range(config.AGENT_MAX_STEPS + 2)]
        agent, client = make_agent(turns, tmp_path,
                                   chunks=[FakeChunk("g" * 16, "p.md", "s", "t")])
        result = agent.answer("tell me about returns")
        assert result.outcome == "truncated"
        assert "human agent" in result.answer
        assert len(client.seen) == config.AGENT_MAX_STEPS     # the cap is a hard bound
        assert len([s for s in result.steps if s.kind == "llm"]) == config.AGENT_MAX_STEPS

    def test_llm_exception_becomes_an_error_outcome(self, tmp_path):
        class Exploding(ScriptedClient):
            def complete(self, system, messages, tools):
                raise RuntimeError("connection reset by peer")

        agent = SupportAgent("CUST-A1", llm=Exploding([]), mcp=FakeMCP(),
                             retriever=FakeRetriever(), trace_dir=tmp_path,
                             semantic_cache=False)
        result = agent.answer("What is the return window?")
        assert result.outcome == "error"
        assert "internal error" in result.answer.lower()
        step = result.steps[0]
        assert step.kind == "llm" and step.name == "error"
        assert "connection reset" in step.detail["error"]
        # The raw exception stays in the trace, never in the customer-facing answer.
        assert "connection reset" not in result.answer

    def test_empty_model_response_should_not_be_answered(self, tmp_path):
        # A turn with no text and no tool calls (content filter, truncated stream)
        # must not reach the customer as a blank "answered" reply.
        agent, _ = make_agent([LLMTurn(text=None, stop_reason="end_turn")], tmp_path)
        result = agent.answer("What is the return window?")
        assert result.answer.strip(), "customer must never receive an empty answer"
        assert result.outcome != "answered"

    def test_history_and_question_are_sent_as_messages(self, tmp_path):
        agent, client = make_agent([final("Earlier you asked about ORD-10012.")], tmp_path)
        history = [{"role": "user", "content": "Where is ORD-10012?"},
                   {"role": "assistant", "content": "It was delivered."}]
        agent.answer("What did I ask earlier?", history=history)
        sent = client.seen[0]["messages"]
        assert sent[:2] == history
        assert sent[2]["role"] == "user"

    def test_memory_is_appended_to_the_system_prompt(self, tmp_path):
        captured = {}

        class Capturing(ScriptedClient):
            def complete(self, system, messages, tools):
                captured["system"] = system
                return super().complete(system, messages, tools)

        agent = SupportAgent("CUST-A2", llm=Capturing([final("ok")]), mcp=FakeMCP(),
                             retriever=FakeRetriever(), trace_dir=tmp_path,
                             semantic_cache=False)
        agent.answer("and what about shipping?", memory="Customer asked about ORD-10012.")
        assert "<conversation_memory>" in captured["system"]
        assert "Customer asked about ORD-10012." in captured["system"]
        assert "## This conversation" in captured["system"]

    def test_forced_search_when_the_model_declines_on_scope(self, tmp_path):
        """A scope refusal without a search is rewritten into a real tool exchange."""
        retriever = FakeRetriever([FakeChunk("h" * 16, "return_policy.md", "Scope",
                                             "We cover returns and shipping.")])
        client = ScriptedClient([
            final("We don't offer that service."),
            final("Our policy does cover this (return_policy.md)."),
        ])
        agent = SupportAgent("CUST-A3", llm=client, mcp=FakeMCP(), retriever=retriever,
                             trace_dir=tmp_path, semantic_cache=False)
        result = agent.answer("Do you price match?")
        assert retriever.queries == ["Do you price match?"]
        assert result.tool_calls == ["search_policies"]
        forced = next(s for s in result.steps if s.kind == "tool")
        assert forced.detail["forced"] is True
        # The draft text was replaced by a well-formed tool-call turn.
        second = client.seen[1]["messages"]
        assert second[-2]["tool_calls"][0]["function"]["name"] == "search_policies"
        assert second[-1]["role"] == "tool"
        assert result.answer.startswith("Our policy does cover this")

    def test_draft_answer_survives_an_empty_turn_after_a_forced_search(self, tmp_path):
        client = ScriptedClient([
            final("I'm unable to assist with that."),
            LLMTurn(text=None, stop_reason="end_turn"),
        ])
        agent = SupportAgent("CUST-A4", llm=client, mcp=FakeMCP(),
                             retriever=FakeRetriever([FakeChunk("i" * 16, "p.md", "s", "t")]),
                             trace_dir=tmp_path, semantic_cache=False)
        result = agent.answer("Do you deliver to Mars?")
        assert result.answer == "I'm unable to assist with that."

    def test_forced_search_cites_the_deadline_category(self, tmp_path):
        retriever = FakeRetriever([FakeChunk("j" * 16, "return_policy.md", "Holiday", "t")])
        client = ScriptedClient([
            tool_turn(("calculate_return_deadline",
                       {"category": "Laptops & Computers",
                        "purchase_date": "2024-12-01"})),
            final("The last day to return is January 31, 2025."),
            final("The last day to return is January 31, 2025 (return_policy.md)."),
        ])
        agent = SupportAgent("CUST-A5", llm=client, mcp=FakeMCP(), retriever=retriever,
                             trace_dir=tmp_path, semantic_cache=False)
        agent.answer("Last day to return a laptop bought December 1?")
        assert "Laptops & Computers" in retriever.queries[0]
        assert "Holiday return policy" in retriever.queries[0]


class SequenceMCP(FakeMCP):
    """Replays a list of envelopes per tool, one per call (last one repeats)."""

    def call_tool(self, name, args):
        self.calls.append((name, args))
        seq = self.responses[name]
        return seq[min(len([c for c in self.calls if c[0] == name]), len(seq)) - 1]


class TestToolRetry:
    """Transient failures of read-only tools are retried in code, not by the model."""

    DOWN = {"ok": False, "error": {"code": "TOOL_UNAVAILABLE", "message": "timeout",
                                   "retryable": True}}
    ORDER = {"ok": True, "data": {"order_id": "ORD-10012"}}

    @pytest.fixture(autouse=True)
    def _no_backoff(self, monkeypatch):
        monkeypatch.setattr(config, "TOOL_RETRY_BACKOFF_S", 0.0)

    def test_transient_read_failure_is_retried_once_and_hidden(self, tmp_path):
        mcp = SequenceMCP({"get_order_details": [self.DOWN, self.ORDER]})
        agent, client = make_agent([
            tool_turn(("get_order_details", {"order_id": "ORD-10012"})),
            final("Order ORD-10012 was delivered."),
        ], tmp_path, mcp=mcp)
        result = agent.answer("Status of ORD-10012?")
        assert len(mcp.calls) == 2
        assert tool_messages(client)[0]["ok"] is True     # the model saw only success
        step = next(s for s in result.steps if s.kind == "tool")
        assert step.detail["attempts"] == 2

    def test_retries_stop_at_the_limit(self, tmp_path):
        mcp = SequenceMCP({"get_order_details": [self.DOWN]})
        agent, client = make_agent([
            tool_turn(("get_order_details", {"order_id": "ORD-10012"})),
            *unevidenced("Our order system is unavailable right now."),
        ], tmp_path, mcp=mcp)
        agent.answer("Status of ORD-10012?")
        assert len(mcp.calls) == 1 + config.TOOL_MAX_RETRIES
        assert tool_messages(client)[0]["error"]["code"] == "TOOL_UNAVAILABLE"

    def test_retry_counts_once_against_the_budget(self, tmp_path, monkeypatch):
        from retail_support_agent import guardrails
        monkeypatch.setitem(guardrails.TOOL_CALL_BUDGET, "get_order_details", 1)
        mcp = SequenceMCP({"get_order_details": [self.DOWN, self.ORDER]})
        agent, client = make_agent([
            tool_turn(("get_order_details", {"order_id": "ORD-10012"})),
            final("Delivered."),
        ], tmp_path, mcp=mcp)
        agent.answer("Status of ORD-10012?")
        # Budget 1: the server was hit twice, but the retried call was allowed and
        # charged once.
        assert len(mcp.calls) == 2 and tool_messages(client)[0]["ok"] is True

    def test_writes_are_never_retried(self, tmp_path):
        mcp = SequenceMCP({"get_order_details": [self.ORDER],
                           "create_return_request": [self.DOWN, {"ok": True, "data": {}}]})
        agent, _ = make_agent([
            tool_turn(("get_order_details", {"order_id": "ORD-10012"})),
            tool_turn(("create_return_request", {"order_id": "ORD-10012",
                                                 "reason": "damaged"})),
            final("I couldn't start the return just now."),
        ], tmp_path, mcp=mcp)
        agent.answer("Please start a return for ORD-10012, it arrived damaged")
        assert [c[0] for c in mcp.calls].count("create_return_request") == 1

    def test_non_transient_errors_are_not_retried(self, tmp_path):
        missing = {"ok": False, "error": {"code": "ORDER_NOT_FOUND", "message": "x",
                                          "retryable": False}}
        mcp = SequenceMCP({"get_order_details": [missing]})
        agent, _ = make_agent([
            tool_turn(("get_order_details", {"order_id": "ORD-99999"})),
            final("No such order."),
        ], tmp_path, mcp=mcp)
        agent.answer("Where is ORD-99999?")
        assert len(mcp.calls) == 1


class TestAccounting:
    def test_token_usage_is_summed_across_turns(self, tmp_path):
        agent, _ = make_agent([
            tool_turn(("search_policies", {"query": "returns"}),
                      usage={"input_tokens": 100, "output_tokens": 20,
                             "cached_input_tokens": 80}),
            final("Returns take 30 days.",
                  usage={"input_tokens": 300, "output_tokens": 40,
                         "cache_write_input_tokens": 12}),
        ], tmp_path, chunks=[FakeChunk("k" * 16, "p.md", "s", "t")])
        result = agent.answer("How long do returns take?")
        assert result.usage == {"input_tokens": 400, "output_tokens": 60,
                                "cached_input_tokens": 80, "cache_write_input_tokens": 12}

    def test_missing_usage_counts_as_zero(self, tmp_path):
        agent, _ = make_agent([final("ok")], tmp_path)
        result = agent.answer("hello, what are your hours?")
        assert result.usage == {"input_tokens": 0, "output_tokens": 0,
                                "cached_input_tokens": 0, "cache_write_input_tokens": 0}

    def test_per_step_usage_and_ttft_are_recorded(self, tmp_path):
        agent, _ = make_agent(unevidenced("ok", usage={"input_tokens": 7, "output_tokens": 2},
                                          ttft=42.0), tmp_path)
        result = agent.answer("What are your shipping options?")
        llm_step = result.steps[0]
        assert llm_step.usage == {"input_tokens": 7, "output_tokens": 2}
        assert llm_step.ttft_ms == 42.0
        assert result.ttft_ms == 42.0           # TTFT of the answer-producing turn
        assert llm_step.latency_ms >= 0

    def test_retrieval_time_is_attributed_to_search_steps_only(self, tmp_path):
        mcp = FakeMCP({"get_order_details": {"ok": True, "data": {"order_id": "ORD-10012"}}})
        agent, _ = make_agent([
            tool_turn(("search_policies", {"query": "returns"}),
                      ("get_order_details", {"order_id": "ORD-10012"})),
            final("Done."),
        ], tmp_path, mcp=mcp, chunks=[FakeChunk("l" * 16, "p.md", "s", "t")])
        result = agent.answer("Policy and status for ORD-10012?")
        search_step = next(s for s in result.steps
                           if s.kind == "tool" and s.name == "search_policies")
        assert result.retrieval_ms == pytest.approx(search_step.latency_ms)
        assert result.retrieval_ms <= result.latency_ms

    def test_summary_shape(self, tmp_path):
        agent, _ = make_agent([
            tool_turn(("search_policies", {"query": "returns"})),
            final("Returns take 30 days (return_policy.md)."),
        ], tmp_path, chunks=[FakeChunk("m" * 16, "return_policy.md", "Returns", "t")])
        summary = agent.answer("How long do returns take?").summary()
        assert set(summary) == {"session_id", "outcome", "tool_calls", "n_steps", "usage",
                                "latency_ms", "retrieval_ms", "ttft_ms", "evidence",
                                "grounding", "cache"}
        assert summary["evidence"] == ["return_policy.md#mmmmmmmm"]
        assert summary["grounding"] is None and summary["cache"] is None
        assert isinstance(summary["latency_ms"], float)

    def test_summary_projects_only_grounding_keys(self):
        from retail_support_agent.agent import GROUNDING_KEYS
        result = AgentResult("s", "CUST-1", "q", "a", "answered", [], [], [], {}, 1.0,
                             grounding={"faithfulness": 1.0, "extra": "drop me"})
        assert set(result.summary()["grounding"]) == set(GROUNDING_KEYS)
        assert result.summary()["grounding"]["faithfulness"] == 1.0


class TestTraceFile:
    def test_trace_file_structure(self, tmp_path):
        agent, _ = make_agent([
            tool_turn(("search_policies", {"query": "returns"}),
                      usage={"input_tokens": 5, "output_tokens": 1}),
            final("Returns take 30 days (return_policy.md).", ttft=12.345),
        ], tmp_path, chunks=[FakeChunk("n" * 16, "return_policy.md", "Returns", "t")])
        result = agent.answer("How long do returns take?")
        lines = trace_lines(result)
        assert lines[0]["type"] == "meta"
        assert lines[0]["session_id"] == result.session_id
        assert lines[0]["requester_id"] == result.requester_id
        assert lines[0]["ts"].endswith("+00:00")       # UTC, ISO-8601
        assert lines[-1]["type"] == "result"
        assert lines[-1]["outcome"] == result.outcome
        assert lines[-1]["answer"] == result.answer
        assert lines[-1]["ttft_ms"] == 12.3            # rounded to 1dp
        assert lines[-1]["grounding"] is None
        steps = [x for x in lines if x["type"] == "step"]
        assert len(steps) == len(result.steps)
        assert steps[0]["usage"] == {"input_tokens": 5, "output_tokens": 1}
        assert steps[0]["ttft_ms"] is None
        assert all(isinstance(s["latency_ms"], float) for s in steps)
        assert result.trace_path.endswith(f"trace_{result.session_id}.jsonl")

    def test_trace_dir_is_created_on_demand(self, tmp_path):
        target = tmp_path / "nested" / "traces"
        agent = SupportAgent("CUST-A6", llm=ScriptedClient([final("ok")]), mcp=FakeMCP(),
                             retriever=FakeRetriever(), trace_dir=target,
                             semantic_cache=False)
        result = agent.answer("Do you ship to Canada?")
        assert target.is_dir() and result.trace_path

    def test_trace_step_indices_are_unique_and_monotonic(self, tmp_path):
        # Every step is indexed by its position, so indices are 0..n-1 in order.
        agent, _ = make_agent([
            tool_turn(("search_policies", {"query": "a"})),
            tool_turn(("search_policies", {"query": "b"})),
            final("Done (p.md)."),
        ], tmp_path, chunks=[FakeChunk("o" * 16, "p.md", "s", "t")])
        result = agent.answer("Tell me about returns")
        indices = [s.index for s in result.steps]
        assert indices == list(range(len(result.steps)))


class TestAgentLifecycle:
    def test_injected_mcp_is_not_closed_by_the_agent(self, tmp_path):
        mcp = FakeMCP()
        agent = SupportAgent("CUST-A7", llm=ScriptedClient([]), mcp=mcp,
                             retriever=FakeRetriever(), trace_dir=tmp_path,
                             semantic_cache=False)
        with agent:
            pass
        assert mcp.closed is False          # the caller owns what the caller passed in

    def test_fingerprint_is_cached_and_stable(self, tmp_path):
        agent, _ = make_agent([], tmp_path)
        first = agent.fingerprint()
        assert first == agent.fingerprint() and isinstance(first, str) and first

    def test_semantic_cache_false_disables_the_cache(self, tmp_path):
        agent, _ = make_agent([final("ok")], tmp_path)
        assert agent.semantic_cache is None
        assert agent.answer("Do you ship internationally?").cache is None


# ══════════════════════════════════════════════════════════════════════════════
# agent.py — semantic cache integration (fake cache, no embeddings)
# ══════════════════════════════════════════════════════════════════════════════

class FakeCache:
    def __init__(self, hit=None, lookup_error=None, store=(True, "stored")):
        self.hit = hit
        self.lookup_error = lookup_error
        self._store = store
        self.lookups: list[tuple[str, str]] = []
        self.stores: list[tuple[str, str]] = []

    def lookup(self, question, fingerprint):
        self.lookups.append((question, fingerprint))
        if self.lookup_error:
            raise self.lookup_error
        return self.hit

    def store(self, question, fingerprint, result):
        self.stores.append((question, fingerprint))
        if isinstance(self._store, Exception):
            raise self._store
        return self._store


def cache_hit(answer="Standard returns run 30 days from delivery (return_policy.md)."):
    return {"id": "entry-1", "match": "semantic", "similarity": 0.97,
            "cached_question": "how long are returns?", "lookup_ms": 4.2,
            "answer": answer, "outcome": "answered",
            "evidence": [{"source": "return_policy.md#aaaaaaaa", "section": "Returns",
                          "score": 2.0}],
            "context": [{"tool": "search_policies", "args": {"query": "returns"},
                         "result": {"ok": True, "data": {"grounded": True, "chunks": [
                             {"chunk_id": "a" * 16, "source": "return_policy.md",
                              "section": "Returns", "score": 2.0,
                              "text": "Standard returns run 30 days from delivery."}]}}}],
            "grounding": None}


class TestSemanticCacheIntegration:
    def test_hit_short_circuits_the_model(self, tmp_path):
        cache = FakeCache(hit=cache_hit())
        agent, client = make_agent([final("should not be used")], tmp_path,
                                   semantic_cache=cache)
        result = agent.answer("How long do returns take?")
        assert client.seen == []                       # the model was never called
        assert result.cache["hit"] is True and result.cache["entry_id"] == "entry-1"
        assert result.outcome == "answered"
        assert result.usage == {"input_tokens": 0, "output_tokens": 0}
        assert result.steps[0].kind == "cache"
        assert result.steps[0].name == "semantic_cache_hit"
        assert trace_lines(result)[-1]["cache"]["similarity"] == 0.97

    def test_hit_rejected_by_the_output_guardrail_falls_through(self, tmp_path):
        # A cached answer naming another customer must not be replayed.
        cache = FakeCache(hit=cache_hit("That order belongs to CUST-1002."))
        agent, client = make_agent(unevidenced("Let me look that up for you."), tmp_path,
                                   semantic_cache=cache)
        result = agent.answer("Whose order is ORD-10001?")
        assert len(client.seen) == 2       # fell through to a live answer (+ forced search)
        assert result.cache["hit"] is False
        assert result.cache["rejected_by"] == "output_guardrail"
        assert result.cache["entry_id"] == "entry-1"

    def test_lookup_error_never_breaks_answering(self, tmp_path):
        cache = FakeCache(lookup_error=RuntimeError("sqlite is locked"))
        agent, _ = make_agent(unevidenced("Returns take 30 days."), tmp_path,
                              semantic_cache=cache)
        result = agent.answer("How long do returns take?")
        assert result.answer.startswith("Returns take 30 days")
        assert result.cache["hit"] is False and "sqlite is locked" in result.cache["error"]

    def test_store_error_never_breaks_answering(self, tmp_path):
        cache = FakeCache(store=RuntimeError("disk full"))
        agent, _ = make_agent([final("Returns take 30 days.")], tmp_path,
                              semantic_cache=cache)
        result = agent.answer("How long do returns take?")
        assert result.cache["stored"] is False and "disk full" in result.cache["store_reason"]

    def test_answer_is_stored_under_the_fingerprint(self, tmp_path):
        cache = FakeCache()
        agent, _ = make_agent([final("Returns take 30 days.")], tmp_path,
                              semantic_cache=cache)
        result = agent.answer("How long do returns take?")
        assert cache.stores == [(result.question, agent.fingerprint())]
        assert result.cache == {"hit": False, "stored": True, "store_reason": "stored"}

    def test_namespace_is_appended_to_the_fingerprint(self, tmp_path):
        cache = FakeCache()
        agent, _ = make_agent([final("Returns take 30 days.")], tmp_path,
                              semantic_cache=cache)
        agent.answer("How long do returns take?", cache_namespace="sess-9")
        assert cache.lookups[0][1] == agent.fingerprint() + ":sess-9"

    def test_store_disabled_for_this_turn(self, tmp_path):
        cache = FakeCache()
        agent, _ = make_agent([final("Returns take 30 days.")], tmp_path,
                              semantic_cache=cache)
        result = agent.answer("How long do returns take?", cache_store=False)
        assert cache.stores == []
        assert result.cache["stored"] is False
        assert "disabled" in result.cache["store_reason"]

    def test_follow_up_without_a_standalone_form_is_not_cached(self, tmp_path):
        cache = FakeCache()
        agent, _ = make_agent([final("Mexico ships in 5 days.")], tmp_path,
                              semantic_cache=cache)
        result = agent.answer("what about Mexico?",
                              history=[{"role": "user", "content": "Shipping to Canada?"}])
        assert cache.lookups == [] and cache.stores == []
        assert result.cache == {"hit": False, "skipped": "follow-up without a standalone form"}

    def test_standalone_rewrite_is_used_as_the_cache_key(self, tmp_path):
        cache = FakeCache()
        agent, _ = make_agent([final("Mexico ships in 5 days.")], tmp_path,
                              semantic_cache=cache)
        agent.answer("what about Mexico?",
                     history=[{"role": "user", "content": "Shipping to Canada?"}],
                     cache_question="How long is shipping to Mexico?")
        assert cache.lookups[0][0] == "How long is shipping to Mexico?"
        assert cache.stores[0][0] == "How long is shipping to Mexico?"

    def test_conversation_recall_is_never_cached(self, tmp_path):
        cache = FakeCache()
        agent, _ = make_agent([final("You asked about ORD-10012.")], tmp_path,
                              semantic_cache=cache)
        result = agent.answer("what did I ask earlier?")
        assert cache.lookups == [] and cache.stores == []
        assert result.cache["skipped"]


# ══════════════════════════════════════════════════════════════════════════════
# agent.py — dataclasses
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentDataclasses:
    def test_trace_step_defaults(self):
        step = TraceStep(0, "tool", "search_policies", {"args": {}}, 1.5)
        assert step.usage == {} and step.ttft_ms is None

    def test_agent_result_defaults(self):
        result = AgentResult("s", "CUST-1", "q", "a", "answered", [], [], [], {}, 1.0)
        assert result.trace_path is None and result.retrieval_ms == 0.0
        assert result.context == [] and result.grounding is None
        assert result.cache is None and result.memory is None
        assert result.ttft_ms is None

    def test_context_records_exactly_what_the_model_saw(self, tmp_path):
        mcp = FakeMCP({"get_order_details": {"ok": True, "data": {"order_id": "ORD-10012"}}})
        agent, client = make_agent([
            tool_turn(("get_order_details", {"order_id": "ORD-10012"})),
            final("ORD-10012 was delivered."),
        ], tmp_path, mcp=mcp)
        result = agent.answer("Status of ORD-10012?")
        assert result.context[0]["tool"] == "get_order_details"
        assert result.context[0]["args"] == {"order_id": "ORD-10012"}
        assert result.context[0]["result"] == tool_messages(client)[0]

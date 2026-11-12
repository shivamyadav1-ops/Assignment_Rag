"""LangChain components: the policy retriever and the agent's tool registry.

- `PolicyRetriever` is a LangChain `BaseRetriever` over `HybridRetriever` (dense + BM25,
  RRF fusion, cross-encoder rerank). It returns `Document`s whose metadata carries
  everything the agent needs to cite and ground an answer (chunk id, source file,
  section, score, source kind), so any LangChain chain can reuse the same retrieval.
- `build_tool_registry` turns every tool the model may call into a LangChain
  `StructuredTool`: the two local tools plus each MCP tool proxied through the MCP client.
  The LangGraph tools node dispatches through this one registry.

Tool *schemas shown to the model* are still the hand-written Anthropic-shape dicts in
`agent.py`. They are passed through unchanged as each tool's `args_schema`, so the
prompt prefix stays byte-stable (provider prefix caching) and guardrails keep validating
against the exact schema the model saw.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import ConfigDict

from . import config


def source_kind(chunk: Any) -> str:
    """store_policy for ShopSmart's own documents, federal_regulation for the FTC rule."""
    payload = getattr(chunk, "payload", None) or {}
    return "federal_regulation" if payload.get("origin") == "public" else "store_policy"


def chunk_to_document(chunk: Any) -> Document:
    score = chunk.rerank_score if chunk.rerank_score is not None else chunk.score
    return Document(
        page_content=chunk.text,
        id=chunk.chunk_id,
        metadata={
            "chunk_id": chunk.chunk_id,
            "source": chunk.source_file,
            "section": chunk.section_heading,
            "source_kind": source_kind(chunk),
            "score": round(score, 3),
        },
    )


class PolicyRetriever(BaseRetriever):
    """LangChain retriever over the project's `HybridRetriever`.

    `hybrid` is anything with `.retrieve(query, mode=, top_k=) -> RetrievalResult`
    (the real retriever, or a test double).
    """

    hybrid: Any
    mode: str = "full"
    top_k: int = config.RERANK_TOP_K

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        result = self.hybrid.retrieve(query, mode=self.mode, top_k=self.top_k)
        return [chunk_to_document(c) for c in result.chunks]


def _tool(spec: dict, func: Callable[..., dict]) -> StructuredTool:
    return StructuredTool.from_function(
        func=func,
        name=spec["name"],
        description=spec.get("description", ""),
        args_schema=spec.get("input_schema") or {"type": "object", "properties": {}},
    )


def build_tool_registry(
    specs: list[dict],
    local: dict[str, Callable[[dict], dict]],
    call_mcp: Callable[[str, dict], dict],
) -> dict[str, BaseTool]:
    """One `StructuredTool` per tool spec.

    local     name -> handler(args) for tools implemented in-process
    call_mcp  fallback for every other name: proxied to the MCP server

    Handlers never raise; they return the `{"ok": ..., "data"|"error": ...}` envelope,
    which `StructuredTool.invoke` passes back unchanged.
    """
    registry: dict[str, BaseTool] = {}
    for spec in specs:
        name = spec["name"]
        if name in local:
            handler = local[name]
            registry[name] = _tool(spec, lambda _h=handler, **kw: _h(kw))
        else:
            registry[name] = _tool(spec, lambda _n=name, **kw: call_mcp(_n, kw))
    return registry

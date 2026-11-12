"""Synchronous facade over the async MCP stdio client.

The MCP SDK is async; the agent loop is easier to read, trace and test when tool calls
are synchronous. This runs one ClientSession on a dedicated event loop in a background
thread and exposes blocking `call_tool` / `list_tools`. The requester identity is passed
to the server process via RSA_REQUESTER_ID, so it is fixed for the life of the session
and never travels as a tool argument.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from concurrent.futures import Future
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from . import config


class MCPToolClient:
    def __init__(self, requester_id: str, server_module: str = "retail_support_agent.mcp_server"):
        self.requester_id = requester_id
        self._server_module = server_module
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._tools: list[dict] = []
        self._start_error: BaseException | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────────
    def start(self, timeout: float = 60.0) -> MCPToolClient:
        self._thread = threading.Thread(target=self._run, name="mcp-client", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("MCP server did not become ready in time")
        if self._start_error:
            raise RuntimeError(f"MCP server failed to start: {self._start_error}")
        return self

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except BaseException as exc:  # noqa: BLE001 - surfaced to start()
            import traceback
            self._start_error = "".join(traceback.format_exception(exc))
            self._ready.set()

    async def _serve(self) -> None:
        env = dict(os.environ, RSA_REQUESTER_ID=self.requester_id)
        # The server runs as `python -m retail_support_agent.mcp_server`; make the package
        # importable in the child even when it isn't installed in this interpreter.
        src_dir = str(config.PROJECT_ROOT / "src")
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (src_dir, env.get("PYTHONPATH", "")) if p
        )
        # Prefer the project's virtualenv interpreter: it has every server dependency,
        # whereas the caller may be a global/conda Python missing some of them.
        venv_python = config.PROJECT_ROOT / ".venv" / "bin" / "python"
        python = str(venv_python) if venv_python.exists() else sys.executable
        params = StdioServerParameters(
            command=python, args=["-m", self._server_module], env=env
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                listed = await session.list_tools()
                self._tools = [
                    {"name": t.name, "description": t.description or "",
                     "input_schema": t.input_schema}
                    for t in listed.tools
                ]
                self._ready.set()
                while not self._stop.is_set():
                    await asyncio.sleep(0.05)

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def __enter__(self) -> MCPToolClient:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # ── calls ──────────────────────────────────────────────────────────────────
    def list_tools(self) -> list[dict]:
        return list(self._tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict:
        """Blocking tool call. Returns the server's envelope, or a typed transport error."""
        if self._session is None or self._loop is None:
            return {"ok": False, "error": {"code": "TOOL_UNAVAILABLE",
                    "message": "MCP session is not running.", "retryable": True}}
        fut: Future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, arguments), self._loop
        )
        try:
            result = fut.result(timeout=config.MCP_CALL_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 - mapped to a retryable tool error
            return {"ok": False, "error": {"code": "TOOL_UNAVAILABLE",
                    "message": f"Tool '{name}' call failed: {exc}", "retryable": True}}
        if result.structured_content is not None:
            return result.structured_content
        if result.content:
            try:
                return json.loads(result.content[0].text)
            except (json.JSONDecodeError, AttributeError, IndexError):
                return {"ok": False, "error": {"code": "BAD_TOOL_OUTPUT",
                        "message": "Tool returned unparseable content.", "retryable": False}}
        return {"ok": False, "error": {"code": "BAD_TOOL_OUTPUT",
                "message": "Tool returned no content.", "retryable": False}}

"""Small synchronous facade for the official TogoMCP Python MCP client.

The project nodes are synchronous LangGraph nodes, while the official MCP
client is asynchronous.  This module keeps that transport detail out of the
nodes and preserves the complete tool result for provenance/debugging.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client


DEFAULT_TOGOMCP_URL = "https://togomcp.rdfportal.org/mcp"


class TogoMCPError(RuntimeError):
    """Raised when the MCP transport or tool call fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    """Convert MCP/Pydantic values into JSON-safe Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _run_sync(coro):
    """Run a coroutine from both normal sync code and an active event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: list[BaseException] = []

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # propagate the original exception
            error.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result.get("value")


class TogoMCPClient:
    """Official MCP SDK client with remote HTTP and optional local stdio.

    Remote mode is the default because it requires no local TogoMCP install.
    Local stdio can be selected with ``TOGOMCP_TRANSPORT=stdio`` and is kept
    configurable for users who need a local deployment.
    """

    def __init__(
        self,
        url: str | None = None,
        transport: str | None = None,
        timeout: float | None = None,
        sse_read_timeout: float | None = None,
    ) -> None:
        self.url = url or os.getenv("TOGOMCP_MCP_URL", DEFAULT_TOGOMCP_URL)
        self.transport = (transport or os.getenv("TOGOMCP_TRANSPORT", "streamable_http")).lower()
        self.timeout = timeout or float(os.getenv("TOGOMCP_HTTP_TIMEOUT", "30"))
        self.sse_read_timeout = sse_read_timeout or float(
            os.getenv("TOGOMCP_SSE_READ_TIMEOUT", "300")
        )

    def _stdio_parameters(self) -> StdioServerParameters:
        command = os.getenv("TOGOMCP_STDIO_COMMAND", "uv")
        args_text = os.getenv(
            "TOGOMCP_STDIO_ARGS",
            "--directory /path/to/togomcp run togo-mcp-local",
        )
        args = args_text.split()
        cwd = os.getenv("TOGOMCP_STDIO_CWD") or None
        env = os.environ.copy()
        return StdioServerParameters(command=command, args=args, cwd=cwd, env=env)

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[ClientSession]:
        if self.transport in {"http", "streamable_http", "streamable-http"}:
            async with streamablehttp_client(
                self.url,
                timeout=self.timeout,
                sse_read_timeout=self.sse_read_timeout,
            ) as (read_stream, write_stream, _session_id):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session
            return

        if self.transport == "stdio":
            async with stdio_client(self._stdio_parameters()) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session
            return

        raise TogoMCPError(
            f"Unsupported TOGOMCP_TRANSPORT={self.transport!r}; "
            "use streamable_http or stdio"
        )

    @staticmethod
    def _tool_result_to_dict(result: Any) -> dict[str, Any]:
        return {
            "is_error": bool(getattr(result, "isError", False)),
            "content": _jsonable(getattr(result, "content", [])),
            "structured_content": _jsonable(
                getattr(result, "structuredContent", None)
            ),
        }

    async def _call_tool_async(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async with self._session() as session:
            result = await session.call_tool(tool_name, arguments=arguments)
            result_dict = self._tool_result_to_dict(result)
            if result_dict["is_error"]:
                raise TogoMCPError(json.dumps(result_dict, ensure_ascii=False))
            return result_dict

    def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call one TogoMCP tool and return the complete JSON-safe result."""
        started = time.perf_counter()
        try:
            result = _run_sync(self._call_tool_async(tool_name, arguments or {}))
            result["tool_name"] = tool_name
            result["arguments"] = arguments or {}
            result["started_at"] = _utc_now()
            result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return result
        except Exception as exc:
            raise TogoMCPError(f"TogoMCP tool {tool_name!r} failed: {exc}") from exc

    async def _list_tools_async(self) -> list[dict[str, Any]]:
        async with self._session() as session:
            result = await session.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": _jsonable(tool.inputSchema),
                }
                for tool in result.tools
            ]

    def list_tools(self) -> list[dict[str, Any]]:
        return _run_sync(self._list_tools_async())


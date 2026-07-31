#!/usr/bin/env python3
"""A minimal stdio MCP server, for tests only — one tool, no external calls.

Run directly (``python tests/fixtures/echo_mcp_server.py``) to register against
a real ``MCPServerRegistry``/``mcp_service`` without needing a network or a
third-party server. Deliberately not part of the ``agentic_core`` package: it
is test fixture, not a thing a product imports.
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo-test-server")


@mcp.tool()
def echo(text: str) -> str:
    """Return *text* unchanged, prefixed so a caller can tell this tool ran."""
    return f"echo: {text}"


@mcp.tool()
def reset_session() -> str:
    """Exercise mcp_service.reset_server_session's happy path — a JSON string result."""
    return json.dumps({"cleared": {"echo": 1}, "total": 1})


if __name__ == "__main__":
    mcp.run()

#!/usr/bin/env python3
"""A minimal stdio MCP server, for tests only — one tool, no external calls.

Run directly (``python tests/fixtures/echo_mcp_server.py``) to register against
a real ``MCPServerRegistry``/``mcp_service`` without needing a network or a
third-party server. Deliberately not part of the ``agentic_core`` package: it
is test fixture, not a thing a product imports.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP("echo-test-server")


@mcp.tool()
def echo(text: str) -> str:
    """Return *text* unchanged, prefixed so a caller can tell this tool ran."""
    return f"echo: {text}"


@mcp.tool()
def echo_meta(ctx: Context[Any, Any, Any]) -> str:
    """Return the ambient MCP request ``_meta`` as JSON — verifies what a tool
    actually receives (e.g. ``agentic_core.workspace_id``/``owner_id``), not
    just what the sender intended to build."""
    meta = ctx.request_context.meta
    return json.dumps(getattr(meta, "model_extra", None) or {})


@mcp.tool()
def reset_session() -> str:
    """Exercise mcp_service.reset_server_session's happy path — a JSON string result."""
    return json.dumps({"cleared": {"echo": 1}, "total": 1})


if __name__ == "__main__":
    mcp.run()

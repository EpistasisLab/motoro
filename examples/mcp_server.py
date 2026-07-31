#!/usr/bin/env python3
"""A tiny stdio MCP server for the example below — one tool, deterministic.

Not part of ``agentic_core`` — a product supplies its own MCP servers; this one
exists only so ``mcp_run.py`` has something real to register and call. The tool
returns a fact the model cannot know or guess, so a correct answer in the run's
output is unambiguous proof the tool actually ran, rather than the model just
reasoning its way to a plausible-looking response.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("example-tools")

_SECRET_CODES = {
    "alpha": "42-XQ",
    "bravo": "17-ZM",
    "charlie": "83-KT",
}


@mcp.tool()
def get_secret_code(codename: str) -> str:
    """Look up the secret code registered for *codename* (alpha, bravo, charlie)."""
    code = _SECRET_CODES.get(codename.lower())
    if code is None:
        return f"No secret code is registered for '{codename}'."
    return f"The secret code for '{codename}' is {code}."


if __name__ == "__main__":
    mcp.run()

"""Bundled, ready-to-register MCP servers that ship with core itself.

Unlike a product's own servers (e.g. ASAREE's ``asaree-workspace``), a server
in this package depends on nothing product-specific — no product database,
no product service layer — only core's own ambient MCP ``_meta`` conventions
(``motoro.mcp.adapters``). A product registers one at its own startup,
the same way ASAREE registers its own bundled servers, e.g.::

    command = f"uv run --directory {motoro_repo_root} python -m motoro.mcp_servers.okf"
    await register_server(name="motoro-okf", transport="stdio", command=command, is_system=True)
"""

from __future__ import annotations

"""MCP server registry — manages multiple concurrent server connections."""

from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass
from typing import Any

import structlog

from agentic_core.mcp.client import MCPClient, ToolInfo, TransportType

log = structlog.get_logger()


@dataclass
class ServerEntry:
    """A registered server with its client."""

    name: str
    transport: TransportType
    command: str | None
    url: str | None
    client: MCPClient
    error: str | None = None


class MCPServerRegistry:
    """Manages multiple MCP server connections.

    All mutations (``register``, ``unregister``, ``refresh_server``) are
    serialized through an internal :class:`asyncio.Lock` (Issues #708, #729).
    Reads (``get``, ``servers``, ``get_all_tools``) remain lock-free; they
    return a snapshot dict copy or look up a single key, which is safe under
    the single-threaded asyncio model.
    """

    def __init__(self) -> None:
        self._servers: dict[str, ServerEntry] = {}
        self._lock = asyncio.Lock()

    @property
    def servers(self) -> dict[str, ServerEntry]:
        return dict(self._servers)

    async def register(
        self,
        name: str,
        transport: TransportType = TransportType.STDIO,
        command: str | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        server_env: dict[str, str] | None = None,
    ) -> ServerEntry:
        """Register and connect to a new MCP server.

        The entire register cycle (existing-entry disconnect + new connect +
        slot write) runs under ``self._lock`` so two concurrent calls for the
        same name cannot observe a half-removed slot and double-register
        (Issue #729).
        """
        async with self._lock:
            if name in self._servers:
                # Locked path: unregister the old entry inline so we don't
                # acquire the lock recursively (asyncio.Lock is not reentrant).
                await self._unregister_locked(name)

            client = MCPClient(
                name=name,
                transport=transport,
                command=command,
                url=url,
                headers=headers,
                server_env=server_env,
            )
            entry = ServerEntry(
                name=name,
                transport=transport,
                command=command,
                url=url,
                client=client,
            )

            try:
                await client.connect()
            except BaseException as e:
                entry.error = f"{type(e).__name__}: {e}"
                log.warning(
                    "mcp.server.register_failed",
                    server=name,
                    error=entry.error,
                    component="mcp_registry",
                )

            self._servers[name] = entry

            if entry.client.connected:
                log.info(
                    "mcp.server.registered",
                    server=name,
                    tools=len(entry.client.tools),
                    component="mcp_registry",
                )

            return entry

    async def unregister(self, name: str) -> None:
        """Disconnect and remove a server (locked)."""
        async with self._lock:
            await self._unregister_locked(name)

    async def _unregister_locked(self, name: str) -> None:
        """Internal helper: must be called with ``self._lock`` held."""
        entry = self._servers.pop(name, None)
        if entry and entry.client.connected:
            await entry.client.disconnect()
        if entry:
            log.info("mcp.server.unregistered", server=name, component="mcp_registry")

    def get(self, name: str) -> ServerEntry | None:
        """Get a server entry by name."""
        return self._servers.get(name)

    def lookup_tool(self, tool_name: str) -> tuple[str, str, MCPClient, ToolInfo] | None:
        """Resolve *tool_name* to ``(server, bare_name, client, tool_info)``. Issue #746.

        Supports both ``server.tool`` namespaced names and bare names. For bare
        names we build an index across all connected servers; if more than one
        server exposes the same bare name we log a warning and return the first
        match (preserving previous behaviour — callers should namespace to be
        unambiguous).
        """
        # Namespaced lookup is direct.
        if "." in tool_name:
            server_name, bare = tool_name.split(".", 1)
            entry = self._servers.get(server_name)
            if entry and entry.client.connected:
                for tool in entry.client.tools:
                    if tool.name == bare:
                        return server_name, bare, entry.client, tool
            return None

        # Bare-name index: O(1) lookup, ambiguity warning on collisions.
        index: dict[str, list[tuple[str, MCPClient, ToolInfo]]] = {}
        for server_name, entry in self._servers.items():
            if not entry.client.connected:
                continue
            for tool in entry.client.tools:
                index.setdefault(tool.name, []).append((server_name, entry.client, tool))

        candidates = index.get(tool_name)
        if not candidates:
            return None
        if len(candidates) > 1:
            log.warning(
                "mcp.tool.bare_name_collision",
                tool=tool_name,
                servers=[c[0] for c in candidates],
                resolved=candidates[0][0],
                component="mcp_registry",
            )
        server_name, client, tool_info = candidates[0]
        return server_name, tool_name, client, tool_info

    def get_all_tools(self) -> list[dict[str, Any]]:
        """Get aggregated tool list across all connected servers.

        Tool names are namespaced as '{server_name}.{tool_name}'.
        """
        tools: list[dict[str, Any]] = []
        for name, entry in self._servers.items():
            if entry.client.connected:
                for tool in entry.client.tools:
                    tools.append(
                        {
                            "name": f"{name}.{tool.name}",
                            "server": name,
                            "tool_name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.input_schema,
                        }
                    )
        return tools

    async def refresh_server(self, name: str) -> ServerEntry | None:
        """Refresh tool discovery for a single server (locked, Issue #708).

        Holding the lock during refresh prevents an in-flight register or
        unregister from racing with the cache update.
        """
        async with self._lock:
            entry = self._servers.get(name)
            if entry is None or not entry.client.connected:
                return entry
            try:
                await entry.client.refresh_tools()
                entry.error = None
            except Exception as e:
                entry.error = f"Refresh failed: {type(e).__name__}: {e}"
                log.warning(
                    "mcp.server.refresh_failed",
                    server=name,
                    error=entry.error,
                    component="mcp_registry",
                )
            return entry

    async def check_health(self) -> dict[str, bool]:
        """Check which servers are still alive. Returns {name: is_alive}."""
        results: dict[str, bool] = {}
        for name, entry in self._servers.items():
            if entry.client.connected:
                results[name] = await entry.client.is_alive()
            else:
                results[name] = False
        return results

    async def disconnect_all(self) -> None:
        """Disconnect all servers concurrently (Issue #775).

        Runs ``client.disconnect`` for every connected server in parallel via
        ``asyncio.gather`` so total shutdown time is bounded by the slowest
        disconnect rather than the sum. Exceptions from individual disconnects
        are collected and logged but never re-raised so shutdown always
        proceeds.
        """
        targets: list[tuple[str, MCPClient]] = [
            (entry.name, entry.client) for entry in self._servers.values() if entry.client.connected
        ]
        if targets:
            results = await asyncio.gather(
                *(client.disconnect() for _, client in targets),
                return_exceptions=True,
            )
            for (name, _client), result in zip(targets, results, strict=True):
                if isinstance(result, BaseException):
                    log.warning(
                        "mcp.server.disconnect_error",
                        server=name,
                        error=f"{type(result).__name__}: {result}",
                        component="mcp_registry",
                    )
        self._servers.clear()


# Process-scoped singleton registry. Issue #749.
#
# ``uvicorn --reload`` re-imports modules in the same process, and pytest's
# ``xdist`` workers fork independent interpreters — both surfaced bugs where a
# stale ``_registry`` was reused across what callers thought were independent
# lifecycles. We therefore (a) key the cache by PID so a forked worker rebuilds
# its own instance, and (b) expose an explicit ``reset_registry`` for tests and
# the ``--reload`` lifespan to discard a stale instance deterministically.
_registry: MCPServerRegistry | None = None
_registry_pid: int | None = None
_registry_lock = threading.Lock()


def get_registry() -> MCPServerRegistry:
    """Get or create the per-process MCP server registry. Issue #749.

    If a different PID owns the cached registry (i.e. we forked since it was
    created) we transparently rebuild — preventing parent and child workers
    from sharing a registry that points at the parent's MCP subprocesses.
    """
    global _registry, _registry_pid  # noqa: PLW0603
    pid = os.getpid()
    if _registry is None or _registry_pid != pid:
        with _registry_lock:
            if _registry is None or _registry_pid != pid:
                _registry = MCPServerRegistry()
                _registry_pid = pid
    return _registry


def reset_registry() -> None:
    """Drop the cached registry. Intended for ``uvicorn --reload`` and tests.

    Does *not* attempt to disconnect existing clients — callers must call
    ``await registry.disconnect_all()`` first if they need a clean shutdown.
    """
    global _registry, _registry_pid  # noqa: PLW0603
    with _registry_lock:
        _registry = None
        _registry_pid = None

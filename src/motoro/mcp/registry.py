"""MCP server registry — manages multiple concurrent server connections."""

from __future__ import annotations

import asyncio
import os
import threading
import uuid
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Any

import structlog

from motoro.mcp.client import MCPClient, ToolInfo, TransportType

log = structlog.get_logger()


@dataclass
class ServerEntry:
    """A registered server with its client."""

    server_id: uuid.UUID
    name: str
    owner_id: uuid.UUID | None
    is_system: bool
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
        self._servers: dict[uuid.UUID, ServerEntry] = {}
        self._lock = asyncio.Lock()

    @property
    def servers(self) -> dict[uuid.UUID, ServerEntry]:
        return dict(self._servers)

    async def register(
        self,
        server_id: uuid.UUID,
        name: str,
        owner_id: uuid.UUID | None = None,
        is_system: bool = False,
        transport: TransportType = TransportType.STDIO,
        command: str | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        server_env: dict[str, str] | None = None,
    ) -> ServerEntry:
        """Register and connect to a new MCP server.

        Always replaces the existing entry for the same registration ID, disconnecting it
        first -- use :meth:`ensure_registered` for "connect this only if it
        isn't already live", which is what most callers actually want.

        The entire register cycle (existing-entry disconnect + new connect +
        slot write) runs under ``self._lock`` so two concurrent calls for the
        same registration cannot observe a half-removed slot and double-register
        (Issue #729).
        """
        async with self._lock:
            return await self._register_locked(
                server_id=server_id,
                name=name,
                owner_id=owner_id,
                is_system=is_system,
                transport=transport,
                command=command,
                url=url,
                headers=headers,
                server_env=server_env,
            )

    async def _register_locked(
        self,
        *,
        server_id: uuid.UUID,
        name: str,
        owner_id: uuid.UUID | None,
        is_system: bool,
        transport: TransportType,
        command: str | None,
        url: str | None,
        headers: dict[str, str] | None,
        server_env: dict[str, str] | None,
    ) -> ServerEntry:
        """Internal helper: must be called with ``self._lock`` held."""
        if server_id in self._servers:
            # Locked path: unregister the old entry inline so we don't
            # acquire the lock recursively (asyncio.Lock is not reentrant).
            await self._unregister_locked(server_id)

        client = MCPClient(
            name=name,
            transport=transport,
            command=command,
            url=url,
            headers=headers,
            server_env=server_env,
        )
        entry = ServerEntry(
            server_id=server_id,
            name=name,
            owner_id=owner_id,
            is_system=is_system,
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

        self._servers[server_id] = entry

        if entry.client.connected:
            log.info(
                "mcp.server.registered",
                server=name,
                tools=len(entry.client.tools),
                component="mcp_registry",
            )

        return entry

    async def ensure_registered(
        self,
        server_id: uuid.UUID,
        name: str,
        owner_id: uuid.UUID | None = None,
        is_system: bool = False,
        transport: TransportType = TransportType.STDIO,
        command: str | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        server_env: dict[str, str] | None = None,
    ) -> ServerEntry:
        """Register *server_id* only if it isn't already connected.

        ``register`` unconditionally tears down and replaces an existing entry,
        which is right for "the user changed this server's config, reconnect
        it" but wrong for "make sure this server is live". Callers doing the
        latter used to check ``name in registry.servers`` themselves and skip —
        a check-then-act race, because the check and the ``register`` that
        follows it are not under the same lock. Concurrent callers all observed
        "missing" before any of them had finished connecting, so all of them
        registered, and each one after the first tore down a live connection
        the others were about to use.

        Doing the check inside the lock closes that: the first caller connects,
        the rest see a connected entry and return it untouched. An entry that
        exists but is *not* connected (a previous connect failed) is retried,
        so this never leaves a dead slot in place.
        """
        async with self._lock:
            existing = self._servers.get(server_id)
            if existing is not None and existing.client.connected:
                return existing
            return await self._register_locked(
                server_id=server_id,
                name=name,
                owner_id=owner_id,
                is_system=is_system,
                transport=transport,
                command=command,
                url=url,
                headers=headers,
                server_env=server_env,
            )

    async def unregister(self, server_id: uuid.UUID) -> None:
        """Disconnect and remove a server (locked)."""
        async with self._lock:
            await self._unregister_locked(server_id)

    async def _unregister_locked(self, server_id: uuid.UUID) -> None:
        """Internal helper: must be called with ``self._lock`` held."""
        entry = self._servers.pop(server_id, None)
        if entry and entry.client.connected:
            await entry.client.disconnect()
        if entry:
            log.info(
                "mcp.server.unregistered",
                server=entry.name,
                server_id=str(server_id),
                component="mcp_registry",
            )

    def get(self, server_id: uuid.UUID) -> ServerEntry | None:
        """Get a server entry by its persisted registration ID."""
        return self._servers.get(server_id)

    def _scoped_entries(
        self,
        *,
        owner_id: uuid.UUID | None = None,
        server_ids: Collection[uuid.UUID] | None = None,
    ) -> Iterable[tuple[uuid.UUID, ServerEntry]]:
        """Return entries visible to one owner or explicitly granted by ID.

        Explicit IDs are the execution-time seam: callers pass the IDs carried
        by the run's tool descriptors. Without them, discovery is owner-scoped;
        ``owner_id=None`` deliberately exposes system registrations only.
        """
        if server_ids is not None:
            allowed = frozenset(server_ids)
            return ((server_id, entry) for server_id, entry in self._servers.items() if server_id in allowed)
        return (
            (server_id, entry)
            for server_id, entry in self._servers.items()
            if entry.is_system or (owner_id is not None and entry.owner_id == owner_id)
        )

    def lookup_tool(
        self,
        tool_name: str,
        *,
        owner_id: uuid.UUID | None = None,
        server_ids: Collection[uuid.UUID] | None = None,
    ) -> tuple[uuid.UUID, str, str, MCPClient, ToolInfo] | None:
        """Resolve within a scoped set to ``(server_id, server, bare_name, client, tool_info)``.

        Supports both ``server.tool`` namespaced names and bare names. For bare
        names we build an index across all connected servers; if more than one
        server exposes the same bare name we log a warning and return the first
        match (preserving previous behaviour — callers should namespace to be
        unambiguous).
        """
        entries = list(self._scoped_entries(owner_id=owner_id, server_ids=server_ids))

        # Namespaced lookup is scoped before matching the display name.
        if "." in tool_name:
            server_name, bare = tool_name.split(".", 1)
            for server_id, entry in entries:
                if entry.name != server_name or not entry.client.connected:
                    continue
                for tool in entry.client.tools:
                    if tool.name == bare:
                        return server_id, server_name, bare, entry.client, tool
            return None

        # Bare-name index: O(1) lookup, ambiguity warning on collisions.
        index: dict[str, list[tuple[uuid.UUID, str, MCPClient, ToolInfo]]] = {}
        for server_id, entry in entries:
            if not entry.client.connected:
                continue
            for tool in entry.client.tools:
                index.setdefault(tool.name, []).append((server_id, entry.name, entry.client, tool))

        candidates = index.get(tool_name)
        if not candidates:
            return None
        if len(candidates) > 1:
            log.warning(
                "mcp.tool.bare_name_collision",
                tool=tool_name,
                servers=[c[1] for c in candidates],
                resolved=candidates[0][1],
                component="mcp_registry",
            )
        server_id, server_name, client, tool_info = candidates[0]
        return server_id, server_name, tool_name, client, tool_info

    def get_all_tools(
        self,
        *,
        owner_id: uuid.UUID | None = None,
        server_ids: Collection[uuid.UUID] | None = None,
    ) -> list[dict[str, Any]]:
        """Get aggregated tool list across all connected servers.

        Tool names are namespaced as '{server_name}.{tool_name}'. Discovery is
        owner-scoped (owner plus system entries) unless exact IDs are supplied.
        """
        tools: list[dict[str, Any]] = []
        for server_id, entry in self._scoped_entries(owner_id=owner_id, server_ids=server_ids):
            if entry.client.connected:
                for tool in entry.client.tools:
                    tools.append(
                        {
                            "name": f"{entry.name}.{tool.name}",
                            "server": entry.name,
                            "server_id": str(server_id),
                            "tool_name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.input_schema,
                            "title": tool.title,
                            "output_schema": tool.output_schema,
                            "annotations": tool.annotations,
                            "server_instructions": entry.client.instructions,
                        }
                    )
        return tools

    async def refresh_server(self, server_id: uuid.UUID) -> ServerEntry | None:
        """Refresh tool discovery for a single server (locked, Issue #708).

        Holding the lock during refresh prevents an in-flight register or
        unregister from racing with the cache update.
        """
        async with self._lock:
            entry = self._servers.get(server_id)
            if entry is None or not entry.client.connected:
                return entry
            try:
                await entry.client.refresh_tools()
                entry.error = None
            except Exception as e:
                entry.error = f"Refresh failed: {type(e).__name__}: {e}"
                log.warning(
                    "mcp.server.refresh_failed",
                    server=entry.name,
                    server_id=str(server_id),
                    error=entry.error,
                    component="mcp_registry",
                )
            return entry

    async def check_health(self) -> dict[uuid.UUID, bool]:
        """Check which registrations are still alive. Returns {id: is_alive}."""
        results: dict[uuid.UUID, bool] = {}
        for server_id, entry in self._servers.items():
            if entry.client.connected:
                results[server_id] = await entry.client.is_alive()
            else:
                results[server_id] = False
        return results

    async def disconnect_all(self) -> None:
        """Disconnect all servers concurrently.

        This used to be forced sequential *and* in reverse registration order:
        each transport was an anyio task group entered by whichever task called
        ``connect``, and anyio requires a cancel scope to be exited by that same
        task, in LIFO order. Running the disconnects concurrently under
        ``asyncio.gather`` broke both constraints at once (``RuntimeError:
        Attempted to exit cancel scope in a different task than it was entered
        in``), so shutdown cost the sum of every disconnect rather than the max.

        ``MCPClient`` now owns each connection in a task of its own
        (``client._own_connection``), which opens and closes the transport in
        that one task regardless of who calls connect/disconnect. Both
        constraints are gone with it, so this is concurrent again.

        Exceptions from an individual disconnect are logged but never re-raised
        so shutdown always proceeds through the rest of the list.
        """
        targets: list[tuple[str, MCPClient]] = [
            (entry.name, entry.client) for entry in self._servers.values() if entry.client.connected
        ]

        async def _close(name: str, client: MCPClient) -> None:
            try:
                await client.disconnect()
            except Exception as exc:
                log.warning(
                    "mcp.server.disconnect_error",
                    server=name,
                    error=f"{type(exc).__name__}: {exc}",
                    component="mcp_registry",
                )

        await asyncio.gather(*(_close(name, client) for name, client in targets))
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

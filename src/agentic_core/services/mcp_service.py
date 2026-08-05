"""MCP server registration — the persistence half of the MCP client.

``agentic_core.mcp`` (client, registry, adapters) is transport: connect, discover
tools, call a tool. Everything here is about remembering *which* servers a
product uses so a fresh process — a worker, a new script invocation, a restarted
API — doesn't have to re-register them by hand. ``register_server`` connects and
persists in one call; :func:`hydrate_registry` is the other half: load whatever
is persisted and reconnect anything not already live.

The DB is authoritative here, which is the opposite direction from
``engine.patterns.catalog``/``services.pattern_catalog``. There, the plugin code
was the source of truth and the table was a read-only projection for products to
query. Here, the in-memory :class:`~agentic_core.mcp.registry.MCPServerRegistry`
is the derived, disposable thing — it starts empty every process and gets rebuilt
from the table, not the other way around.

Each function opens and closes its own session, like every other public entry
point in core (see ``runner.py``'s module docstring) — there is no ``db``
parameter here.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import or_, select

from agentic_core.mcp.client import TransportType
from agentic_core.mcp.registry import MCPServerRegistry, get_registry
from agentic_core.models.mcp_server import MCPServerConfig, MCPServerStatus, MCPTransport
from agentic_core.security.mcp_command_allowlist import validate_stdio_command
from agentic_core.security.ssrf_guard import validate_outbound_url
from agentic_core.services.encryption import decrypt, encrypt

if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


def _session(reason: str) -> AbstractAsyncContextManager[AsyncSession]:
    from agentic_core.models.database import system_session

    return system_session(reason=f"mcp_service: {reason}")


def _encrypt_headers(headers: dict[str, str] | None) -> str | None:
    """Serialise and encrypt a headers dict. Returns None when headers is None/empty."""
    if not headers:
        return None
    return encrypt(json.dumps(headers))


def _decrypt_headers(encrypted: str | None) -> dict[str, str] | None:
    """Decrypt and deserialise an encrypted headers blob. Returns None on failure."""
    if not encrypted:
        return None
    try:
        return json.loads(decrypt(encrypted))  # type: ignore[no-any-return]
    except Exception:
        return None


def _validate_registration(transport: str, command: str | None, url: str | None) -> None:
    """The two checks a registration must pass before anything is spawned or dialed.

    Both are self-contained security modules with no coupling to anything
    product-specific — a stdio command is validated against a fixed executable
    allowlist and rejected for shell metacharacters; an http/sse URL is checked
    against the SSRF guard. ARES enforces the URL check at its API schema layer
    (``schemas.mcp_server.MCPServerCreate``); core has no schema layer for this,
    so it happens here instead, at the one place every registration passes
    through regardless of caller.
    """
    from agentic_core.config import settings

    if transport == "stdio" and command:
        validate_stdio_command(command)
    if transport in ("http", "sse") and url:
        validate_outbound_url(url, allow_private=settings.mcp_allow_private_urls)


async def register_server(
    *,
    name: str,
    transport: str,
    command: str | None = None,
    url: str | None = None,
    headers: dict[str, str] | None = None,
    owner_id: uuid.UUID | None = None,
    is_system: bool = False,
    registry: MCPServerRegistry | None = None,
) -> MCPServerConfig:
    """Validate, connect, and persist a new MCP server registration.

    *is_system* marks a server as platform-provided rather than something a
    particular user registered — e.g. a product's own bundled tool server,
    available to every run regardless of owner. Pair with ``owner_id=None``;
    :func:`list_servers` always includes system rows alongside an owner's own,
    the same way a run resolves an ``is_system`` agent (``Agent.owner_id``
    docstring) regardless of who started it.
    """
    _validate_registration(transport, command, url)

    reg = registry or get_registry()
    tp = TransportType(transport)
    entry = await reg.register(name=name, transport=tp, command=command, url=url, headers=headers)

    tools_data = None
    if entry.client.connected:
        tools_data = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in entry.client.tools
        ]

    config = MCPServerConfig(
        name=name,
        transport=MCPTransport(transport),
        command=command,
        url=url,
        headers_encrypted=_encrypt_headers(headers),
        capabilities={"tools": tools_data} if tools_data else None,
        status=MCPServerStatus.CONNECTED if entry.client.connected else MCPServerStatus.ERROR,
        error_message=entry.error,
        owner_id=owner_id,
        is_system=is_system,
    )
    async with _session("register_server") as db:
        db.add(config)
        await db.commit()
        await db.refresh(config)
    return config


async def get_server(server_id: uuid.UUID) -> MCPServerConfig | None:
    """Fetch a server by id, or ``None``."""
    async with _session("get_server") as db:
        return (await db.execute(select(MCPServerConfig).where(MCPServerConfig.id == server_id))).scalar_one_or_none()


async def get_server_by_name(name: str) -> MCPServerConfig | None:
    """Fetch a server by name, or ``None``."""
    async with _session("get_server_by_name") as db:
        return (await db.execute(select(MCPServerConfig).where(MCPServerConfig.name == name))).scalar_one_or_none()


async def list_servers(*, owner_id: uuid.UUID | None = None) -> Sequence[MCPServerConfig]:
    """List registered servers, optionally filtered by owner.

    A plain filter, not enforcement — core has no viewer to scope against. A
    product doing per-user isolation applies its own check on top, the same way
    it would for :func:`agentic_core.runner.list_agents`. When *owner_id* is
    given, system servers (``is_system=True``) are always included alongside
    it — a global, platform-provided server is available to every owner by
    definition, not just the one who happens to be asking.
    """
    stmt = select(MCPServerConfig).order_by(MCPServerConfig.created_at.desc())
    if owner_id is not None:
        stmt = stmt.where(or_(MCPServerConfig.owner_id == owner_id, MCPServerConfig.is_system.is_(True)))
    async with _session("list_servers") as db:
        return (await db.execute(stmt)).scalars().all()


async def delete_server(server_id: uuid.UUID, *, registry: MCPServerRegistry | None = None) -> bool:
    """Disconnect and remove a server. Returns False if it did not exist."""
    async with _session("delete_server") as db:
        config = (await db.execute(select(MCPServerConfig).where(MCPServerConfig.id == server_id))).scalar_one_or_none()
        if config is None:
            return False
        reg = registry or get_registry()
        await reg.unregister(config.name)
        await db.delete(config)
        await db.commit()
        return True


async def _persist_connection_outcome(
    db: Any, config: MCPServerConfig, entry: Any, *, refresh_only: bool = False
) -> None:
    """Write an entry's connection outcome (tools, status, error) onto *config*."""
    if entry is None:
        config.status = MCPServerStatus.ERROR
        config.error_message = "Server not found in registry"
    elif entry.client.connected:
        tools_data = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in entry.client.tools
        ]
        config.capabilities = {"tools": tools_data}
        config.status = MCPServerStatus.CONNECTED
        config.error_message = entry.error if refresh_only else None
    else:
        config.status = MCPServerStatus.DISCONNECTED if refresh_only else MCPServerStatus.ERROR
        config.error_message = entry.error
    await db.flush()
    await db.refresh(config)


async def refresh_server(server_id: uuid.UUID, *, registry: MCPServerRegistry | None = None) -> MCPServerConfig | None:
    """Refresh tool discovery for an already-connected server."""
    reg = registry or get_registry()
    async with _session("refresh_server") as db:
        config = (await db.execute(select(MCPServerConfig).where(MCPServerConfig.id == server_id))).scalar_one_or_none()
        if config is None:
            return None
        entry = await reg.refresh_server(config.name)
        await _persist_connection_outcome(db, config, entry, refresh_only=True)
        await db.commit()
        return config


async def reconnect_server(
    server_id: uuid.UUID, *, registry: MCPServerRegistry | None = None
) -> MCPServerConfig | None:
    """Reconnect to an errored (or disconnected) server using its saved config."""
    reg = registry or get_registry()
    async with _session("reconnect_server") as db:
        config = (await db.execute(select(MCPServerConfig).where(MCPServerConfig.id == server_id))).scalar_one_or_none()
        if config is None:
            return None
        entry = await reg.register(
            name=config.name,
            transport=TransportType(config.transport.value),
            command=config.command,
            url=config.url,
            headers=_decrypt_headers(config.headers_encrypted),
        )
        await _persist_connection_outcome(db, config, entry)
        await db.commit()
        return config


async def update_server(
    server_id: uuid.UUID,
    *,
    name: str | None = None,
    transport: str | None = None,
    command: str | None = None,
    url: str | None = None,
    headers: dict[str, str] | None = None,
    registry: MCPServerRegistry | None = None,
) -> MCPServerConfig | None:
    """Update a server's config and reconnect it with the new settings."""
    async with _session("update_server") as db:
        config = (await db.execute(select(MCPServerConfig).where(MCPServerConfig.id == server_id))).scalar_one_or_none()
        if config is None:
            return None

        effective_transport = transport if transport is not None else config.transport.value
        effective_command = command if command is not None else config.command
        effective_url = url if url is not None else config.url
        _validate_registration(effective_transport, effective_command, effective_url)

        if name is not None:
            config.name = name
        if transport is not None:
            config.transport = MCPTransport(transport)
        if command is not None:
            config.command = command
        if url is not None:
            config.url = url
        if headers is not None:
            config.headers_encrypted = _encrypt_headers(headers)
        await db.flush()

        reg = registry or get_registry()
        entry = await reg.register(
            name=config.name,
            transport=TransportType(config.transport.value),
            command=config.command,
            url=config.url,
            headers=_decrypt_headers(config.headers_encrypted),
        )
        await _persist_connection_outcome(db, config, entry)
        await db.commit()
        return config


async def call_server_tool(
    server_id: uuid.UUID,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    registry: MCPServerRegistry | None = None,
    meta: dict[str, Any] | None = None,
) -> tuple[bool, str] | None:
    """Invoke a tool on a connected server directly, bypassing an agent run.

    Returns ``(is_error, content)``, or ``None`` if the server is unknown.
    Raises ``RuntimeError`` if the server is not connected.
    """
    config = await get_server(server_id)
    if config is None:
        return None
    reg = registry or get_registry()
    entry = reg.get(config.name)
    if entry is None or not entry.client.connected:
        raise RuntimeError(f"MCP server '{config.name}' is not connected")
    result = await entry.client.call_tool(tool_name, arguments, meta=meta)
    return result.is_error, result.content


async def reset_server_session(
    server_id: uuid.UUID, *, registry: MCPServerRegistry | None = None
) -> dict[str, Any] | None:
    """Invoke a connected server's ``reset_session`` tool to evict its artifacts."""
    outcome = await call_server_tool(server_id, "reset_session", {}, registry=registry)
    if outcome is None:
        return None
    is_error, content = outcome
    if is_error:
        raise RuntimeError(f"reset_session failed: {content}")
    try:
        payload = json.loads(content)
    except (ValueError, TypeError):
        return {"raw": content}
    return payload if isinstance(payload, dict) else {"raw": content}


async def hydrate_registry(*, registry: MCPServerRegistry | None = None) -> list[str]:
    """Load every persisted server and connect any not already live.

    Call this once, at process startup — after ``configure()``, before the
    first run — in any process that starts with an empty
    :class:`~agentic_core.mcp.registry.MCPServerRegistry`: a worker, a fresh
    script invocation, a restarted API. Without it, ``register_server`` having
    persisted a config is pointless — nothing would ever read it back into a
    live connection.

    Returns the names of servers that failed to connect (already logged); a
    server already present in the registry is left untouched rather than
    reconnected, so calling this twice in one process is a cheap no-op for
    anything already hydrated.
    """
    reg = registry or get_registry()
    failed: list[str] = []
    async with _session("hydrate_registry") as db:
        configs = (await db.execute(select(MCPServerConfig))).scalars().all()

    for config in configs:
        if config.name in reg.servers:
            continue
        try:
            entry = await reg.register(
                name=config.name,
                transport=TransportType(config.transport.value),
                command=config.command,
                url=config.url,
                headers=_decrypt_headers(config.headers_encrypted),
            )
            if not entry.client.connected:
                failed.append(config.name)
        except Exception:
            logger.warning("mcp_service.hydrate_failed", exc_info=True, extra={"server": config.name})
            failed.append(config.name)
    return failed

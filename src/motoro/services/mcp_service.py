"""MCP server registration — the persistence half of the MCP client.

``motoro.mcp`` (client, registry, adapters) is transport: connect, discover
tools, call a tool. Everything here is about remembering *which* servers a
product uses so a fresh process — a worker, a new script invocation, a restarted
API — doesn't have to re-register them by hand. ``register_server`` connects and
persists in one call; :func:`hydrate_registry` is the other half: load whatever
is persisted and reconnect anything not already live.

The DB is authoritative here, which is the opposite direction from
``engine.patterns.catalog``/``services.pattern_catalog``. There, the plugin code
was the source of truth and the table was a read-only projection for products to
query. Here, the in-memory :class:`~motoro.mcp.registry.MCPServerRegistry`
is the derived, disposable thing — it starts empty every process and gets rebuilt
from the table, not the other way around.

Each function opens and closes its own session, like every other public entry
point in core (see ``runner.py``'s module docstring) — there is no ``db``
parameter here.
"""

from __future__ import annotations

import contextlib
import ipaddress
import json
import logging
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qsl, urlencode, urlparse

import httpx
from mcp.client.auth.oauth2 import PKCEParameters
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    build_protected_resource_metadata_discovery_urls,
    create_client_info_from_metadata_url,
    create_client_registration_request,
    extract_resource_metadata_from_www_auth,
    extract_scope_from_www_auth,
    get_client_metadata_scopes,
    handle_auth_metadata_response,
    handle_protected_resource_response,
    handle_registration_response,
    is_valid_client_metadata_url,
    should_use_client_metadata_url,
)
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthMetadata,
    OAuthToken,
    ProtectedResourceMetadata,
)
from mcp.shared.auth_utils import check_resource_allowed, resource_url_from_server_url
from sqlalchemy import or_, select, text
from sqlalchemy.exc import IntegrityError

from motoro.mcp.client import MCPClient, TransportType
from motoro.mcp.oauth import (
    MCPAuthConfigurationError,
    MCPAuthenticationError,
    MCPOAuthCallbackError,
    MCPOAuthDiscoveryError,
    MCPOAuthStateError,
    MCPReauthorizationRequiredError,
    PersistentOAuthAuth,
    prepare_token_auth,
    state_hash,
    states_equal,
)
from motoro.mcp.registry import MCPServerRegistry, ServerEntry, get_registry
from motoro.models.mcp_server import MCPServerConfig, MCPServerStatus, MCPTransport
from motoro.security.mcp_command_allowlist import validate_stdio_command
from motoro.security.mcp_credentials import validate_http_headers, validate_stdio_env
from motoro.security.ssrf_guard import validate_outbound_url
from motoro.services.encryption import decrypt, encrypt

if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class MCPServerNameConflictError(ValueError):
    """A display name is already visible in the requested owner's namespace."""


class MCPServerNotFoundError(LookupError):
    """The requested MCP registration does not exist in the requested owner scope."""


class CredentialNotSupplied:
    """Type of :data:`UNSET`, used to leave credentials unchanged on update."""

    def __repr__(self) -> str:
        return "UNSET"


UNSET = CredentialNotSupplied()
"""Credential update sentinel: leave the persisted value unchanged."""


@dataclass(frozen=True, slots=True)
class MCPOAuthClientMetadata:
    """Product-facing OAuth client metadata; no SDK types required by callers."""

    client_name: str
    scope: str | None = None
    client_uri: str | None = None
    contacts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MCPOAuthAuthorization:
    """Safe result of beginning a headless OAuth authorization."""

    authorization_url: str
    expires_at: datetime
    transaction_id: str


@dataclass(frozen=True, slots=True)
class MCPAuthenticationStatus:
    """Non-secret authentication state suitable for an API response."""

    server_id: uuid.UUID
    auth_mode: Literal["none", "static_headers", "stdio_env", "oauth"]
    configured: bool
    authorization_required: bool
    static_headers_configured: bool
    stdio_env_configured: bool
    stdio_env_names: tuple[str, ...]


_OAUTH_TRANSACTION_TTL_SECONDS = 600


def _session(reason: str) -> AbstractAsyncContextManager[AsyncSession]:
    from motoro.models.database import system_session

    return system_session(reason=f"mcp_service: {reason}")


def _encrypt_headers(headers: dict[str, str] | None) -> str | None:
    """Serialise and encrypt a headers dict. Returns None when headers is None/empty."""
    if not headers:
        return None
    return encrypt(json.dumps(validate_http_headers(headers), sort_keys=True, separators=(",", ":")))


def _decrypt_headers(encrypted: str | None) -> dict[str, str] | None:
    """Decrypt and deserialise an encrypted headers blob. Returns None on failure."""
    if not encrypted:
        return None
    try:
        value = json.loads(decrypt(encrypted))
        if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
            return None
        return {str(k): str(v) for k, v in value.items()}
    except Exception:
        return None


def _encrypt_mapping(values: dict[str, Any] | None) -> str | None:
    if not values:
        return None
    return encrypt(json.dumps(values, sort_keys=True, separators=(",", ":")))


def _decrypt_mapping(encrypted: str | None) -> dict[str, Any] | None:
    if not encrypted:
        return None
    try:
        value = json.loads(decrypt(encrypted))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _encrypt_stdio_env(server_env: dict[str, str] | None) -> str | None:
    values = validate_stdio_env(server_env)
    return _encrypt_mapping(values)


def _decrypt_stdio_env(encrypted: str | None) -> dict[str, str] | None:
    values = _decrypt_mapping(encrypted)
    if values is None or not all(isinstance(k, str) and isinstance(v, str) for k, v in values.items()):
        return None
    return {str(k): str(v) for k, v in values.items()}


def _validate_auth_combination(
    transport: str,
    *,
    headers: dict[str, str] | None,
    server_env: dict[str, str] | None,
    oauth_configured: bool = False,
) -> None:
    if headers:
        validate_http_headers(headers)
        if transport not in ("http", "sse"):
            raise MCPAuthConfigurationError("Static headers are supported only for HTTP transports")
    if server_env:
        validate_stdio_env(server_env)
        if transport != "stdio":
            raise MCPAuthConfigurationError("Stdio environment credentials require stdio transport")
    if oauth_configured and transport != "http":
        raise MCPAuthConfigurationError("OAuth is supported only for streamable HTTP transport")
    if oauth_configured and headers:
        raise MCPAuthConfigurationError("OAuth and static headers cannot be configured on the same server")


def _capabilities_for(client: Any) -> dict[str, Any] | None:
    """The ``capabilities`` blob for a connected client, or ``None`` if it told us nothing.

    Two things the server describes about itself, and both belong here rather
    than in columns of their own: the tool list, and ``instructions`` — the
    server's own account of what it is, sent once during the initialize
    handshake. A server-level description is exactly the sort of thing that
    varies by protocol version, so it lives in the same JSON blob the tool
    schemas do.

    ``instructions`` is optional in MCP and most servers omit it, so the key
    is left out entirely rather than written as ``""`` — a caller can then
    treat "absent" and "this row predates instructions being captured" the
    same way, which is what an older row will look like until it is
    refreshed.
    """
    tools_data = [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in client.tools]
    instructions = getattr(client, "instructions", "") or ""
    if not tools_data and not instructions:
        return None
    return {"tools": tools_data, **({"instructions": instructions} if instructions else {})}


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
    from motoro.config import settings

    if transport == "stdio" and command:
        validate_stdio_command(command)
    if transport in ("http", "sse") and url:
        validate_outbound_url(url, allow_private=settings.mcp_allow_private_urls)


async def _lock_server_name(db: AsyncSession, name: str) -> None:
    """Serialize collision checks for one display name on PostgreSQL."""
    bind = db.get_bind()
    if bind.dialect.name == "postgresql":
        await db.execute(text("SELECT pg_advisory_xact_lock(hashtext(:name))"), {"name": name})


async def _name_conflicts(
    db: AsyncSession,
    *,
    name: str,
    owner_id: uuid.UUID | None,
    is_system: bool,
    exclude_id: uuid.UUID | None = None,
) -> bool:
    """Return whether *name* would collide in the registration's namespace."""
    stmt = select(MCPServerConfig.id).where(MCPServerConfig.name == name)
    if exclude_id is not None:
        stmt = stmt.where(MCPServerConfig.id != exclude_id)
    if is_system:
        # A system registration is visible to every owner, so it cannot be
        # introduced when any namespace already uses the same display name.
        pass
    elif owner_id is None:
        stmt = stmt.where(MCPServerConfig.owner_id.is_(None))
    else:
        stmt = stmt.where(or_(MCPServerConfig.owner_id == owner_id, MCPServerConfig.is_system.is_(True)))
    return (await db.execute(stmt.limit(1))).scalar_one_or_none() is not None


def _raise_name_conflict(name: str) -> None:
    raise MCPServerNameConflictError(f"MCP server name '{name}' is already in use in this namespace")


async def register_server(
    *,
    name: str,
    transport: str,
    command: str | None = None,
    url: str | None = None,
    headers: dict[str, str] | None = None,
    server_env: dict[str, str] | None = None,
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
    if is_system and owner_id is not None:
        raise ValueError("System MCP servers must be ownerless")
    _validate_registration(transport, command, url)
    _validate_auth_combination(transport, headers=headers, server_env=server_env)

    config = MCPServerConfig(
        id=uuid.uuid4(),
        name=name,
        transport=MCPTransport(transport),
        command=command,
        url=url,
        headers_encrypted=_encrypt_headers(headers),
        stdio_env_encrypted=_encrypt_stdio_env(server_env),
        oauth_encrypted=None,
        oauth_pending_encrypted=None,
        oauth_state_hash=None,
        oauth_authorization_required=False,
        config_revision=1,
        capabilities=None,
        status=MCPServerStatus.DISCONNECTED,
        error_message=None,
        owner_id=owner_id,
        is_system=is_system,
    )
    async with _session("register_server") as db:
        await _lock_server_name(db, name)
        if await _name_conflicts(db, name=name, owner_id=owner_id, is_system=is_system):
            _raise_name_conflict(name)
        try:
            db.add(config)
            await db.commit()
        except IntegrityError as exc:
            await db.rollback()
            raise MCPServerNameConflictError(f"MCP server name '{name}' is already in use in this namespace") from exc

    reg = registry or get_registry()
    entry = await reg.register(**_registry_connection_kwargs(config, reg))
    async with _session("register_server outcome") as db:
        persisted = (await db.execute(select(MCPServerConfig).where(MCPServerConfig.id == config.id))).scalar_one()
        await _persist_connection_outcome(db, persisted, entry)
        await db.commit()
        return persisted


async def get_server(server_id: uuid.UUID) -> MCPServerConfig | None:
    """Fetch a server by id, or ``None``."""
    async with _session("get_server") as db:
        return (await db.execute(select(MCPServerConfig).where(MCPServerConfig.id == server_id))).scalar_one_or_none()


async def get_server_by_name(name: str, *, owner_id: uuid.UUID | None = None) -> MCPServerConfig | None:
    """Fetch a server by name within one owner namespace.

    System registrations are included for an owner. With ``owner_id=None``,
    only ownerless registrations are considered.
    """
    stmt = select(MCPServerConfig).where(MCPServerConfig.name == name)
    if owner_id is None:
        stmt = stmt.where(MCPServerConfig.owner_id.is_(None))
    else:
        stmt = stmt.where(or_(MCPServerConfig.owner_id == owner_id, MCPServerConfig.is_system.is_(True)))
    async with _session("get_server_by_name") as db:
        return (await db.execute(stmt)).scalar_one_or_none()


async def list_servers(*, owner_id: uuid.UUID | None = None) -> Sequence[MCPServerConfig]:
    """List registered servers, optionally filtered by owner.

    A plain filter, not enforcement — core has no viewer to scope against. A
    product doing per-user isolation applies its own check on top, the same way
    it would for :func:`motoro.runner.list_agents`. When *owner_id* is
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
        config = (
            await db.execute(select(MCPServerConfig).where(MCPServerConfig.id == server_id).with_for_update())
        ).scalar_one_or_none()
        if config is None:
            return False
        reg = registry or get_registry()
        await reg.unregister(config.id)
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
        config.capabilities = _capabilities_for(entry.client) or {"tools": []}
        config.status = MCPServerStatus.CONNECTED
        config.error_message = entry.error if refresh_only else None
    else:
        config.status = MCPServerStatus.DISCONNECTED if refresh_only else MCPServerStatus.ERROR
        config.error_message = entry.error
    if entry is not None and getattr(entry, "reauthorization_required", False):
        config.oauth_authorization_required = True
    await db.flush()
    await db.refresh(config)


def _oauth_payload_valid(payload: dict[str, Any]) -> bool:
    token = payload.get("token") or {}
    expiry = payload.get("expires_at")
    return bool(token.get("access_token") and (expiry is None or time.time() < float(expiry) - 30))


def _bump_config_revision(config: MCPServerConfig) -> None:
    config.config_revision += 1


async def _refresh_oauth_under_lock(
    server_id: uuid.UUID,
    expected_issuer: str,
    expected_url: str,
    observed: dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    """Serialize refresh on the server row and atomically persist token rotation.

    Every process reloads the encrypted payload after acquiring ``FOR UPDATE``.
    If a waiter observes the token another process just refreshed, it returns
    that payload without making a second refresh request.
    """
    if (
        observed.get("issuer") != expected_issuer
        or observed.get("server_url") != expected_url
        or observed.get("server_id") != str(server_id)
    ):
        raise MCPReauthorizationRequiredError("MCP OAuth credential binding changed")

    async with _session("refresh OAuth under row lock") as db:
        config = (
            await db.execute(select(MCPServerConfig).where(MCPServerConfig.id == server_id).with_for_update())
        ).scalar_one_or_none()
        if config is None or config.url != expected_url:
            raise MCPReauthorizationRequiredError("MCP OAuth registration is no longer available")
        current = _decrypt_mapping(config.oauth_encrypted)
        if (
            current is None
            or current.get("issuer") != expected_issuer
            or current.get("server_url") != expected_url
            or current.get("server_id") != str(server_id)
        ):
            raise MCPReauthorizationRequiredError("MCP OAuth credential binding changed")
        current_access = (current.get("token") or {}).get("access_token")
        observed_access = (observed.get("token") or {}).get("access_token")
        another_process_rotated = current_access != observed_access
        if _oauth_payload_valid(current) and (not force or another_process_rotated):
            return current

        token = current.get("token") or {}
        client_info = current.get("client_info") or {}
        metadata = current.get("oauth_metadata") or {}
        refresh_token = token.get("refresh_token")
        secret_expiry = int(client_info.get("client_secret_expires_at") or 0)
        if (
            not refresh_token
            or not client_info.get("client_id")
            or not metadata.get("token_endpoint")
            or (secret_expiry and time.time() >= secret_expiry)
        ):
            if not config.oauth_authorization_required:
                config.oauth_authorization_required = True
                _bump_config_revision(config)
            await db.commit()
            raise MCPReauthorizationRequiredError("MCP OAuth authorization is required")

        data = {
            "grant_type": "refresh_token",
            "refresh_token": str(refresh_token),
            "client_id": str(client_info["client_id"]),
            "resource": str(current["resource"]),
        }
        data, headers = prepare_token_auth(data, client_info)
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
                response = await client.post(str(metadata["token_endpoint"]), data=data, headers=headers)
            if response.status_code != 200:
                raise MCPReauthorizationRequiredError("MCP OAuth authorization is required")
            refreshed = OAuthToken.model_validate_json(response.content)
        except Exception:
            if not config.oauth_authorization_required:
                config.oauth_authorization_required = True
                _bump_config_revision(config)
            await db.commit()
            raise MCPReauthorizationRequiredError("MCP OAuth authorization is required") from None

        token_data = refreshed.model_dump(mode="json", exclude_none=True)
        if "refresh_token" not in token_data:
            token_data["refresh_token"] = token.get("refresh_token")
        if "scope" not in token_data and token.get("scope") is not None:
            token_data["scope"] = token["scope"]
        current["token"] = token_data
        current["expires_at"] = time.time() + refreshed.expires_in if refreshed.expires_in is not None else None
        config.oauth_encrypted = _encrypt_mapping(current)
        config.oauth_authorization_required = False
        _bump_config_revision(config)
        await db.commit()
        return current


async def _mark_oauth_reauthorization_required(server_id: uuid.UUID, observed: dict[str, Any]) -> None:
    async with _session("mark OAuth reauthorization required") as db:
        config = (
            await db.execute(select(MCPServerConfig).where(MCPServerConfig.id == server_id).with_for_update())
        ).scalar_one_or_none()
        current = _decrypt_mapping(config.oauth_encrypted) if config is not None else None
        current_access = (current or {}).get("token", {}).get("access_token")
        observed_access = observed.get("token", {}).get("access_token")
        if current_access != observed_access:
            return
        if config is not None and not config.oauth_authorization_required:
            config.oauth_authorization_required = True
            _bump_config_revision(config)
            await db.commit()


def _oauth_auth_for(config: MCPServerConfig) -> PersistentOAuthAuth | None:
    payload = _decrypt_mapping(config.oauth_encrypted)
    if payload is None or config.url is None:
        return None
    if payload.get("server_id") != str(config.id) or payload.get("server_url") != config.url:
        raise MCPReauthorizationRequiredError("MCP OAuth credential binding changed")
    issuer = str(payload.get("issuer") or "")
    server_url = config.url

    async def refresh(observed: dict[str, Any], force: bool) -> dict[str, Any]:
        return await _refresh_oauth_under_lock(config.id, issuer, server_url, observed, force)

    async def mark_required(observed: dict[str, Any]) -> None:
        await _mark_oauth_reauthorization_required(config.id, observed)

    return PersistentOAuthAuth(payload, refresh, mark_required)


def _registry_connection_kwargs(config: MCPServerConfig, registry: MCPServerRegistry) -> dict[str, Any]:
    server_id = config.id

    async def before_tool_call(_client: MCPClient) -> MCPClient | None:
        entry = await _synchronize_server(server_id, registry)
        return entry.client if entry is not None else None

    return {
        "server_id": config.id,
        "name": config.name,
        "owner_id": config.owner_id,
        "is_system": config.is_system,
        "transport": TransportType(config.transport.value),
        "command": config.command,
        "url": config.url,
        "headers": _decrypt_headers(config.headers_encrypted),
        "server_env": _decrypt_stdio_env(config.stdio_env_encrypted),
        "http_auth": _oauth_auth_for(config),
        "config_revision": config.config_revision,
        "before_tool_call": before_tool_call,
    }


async def _synchronize_server(server_id: uuid.UUID, registry: MCPServerRegistry) -> ServerEntry | None:
    """Make one live registry entry match its authoritative persisted revision."""
    async with _session("synchronize MCP registry entry") as db:
        config = (await db.execute(select(MCPServerConfig).where(MCPServerConfig.id == server_id))).scalar_one_or_none()
    if config is None:
        await registry.unregister(server_id)
        return None
    return await registry.ensure_registered(**_registry_connection_kwargs(config, registry))


async def refresh_server(server_id: uuid.UUID, *, registry: MCPServerRegistry | None = None) -> MCPServerConfig | None:
    """Refresh tool discovery for an already-connected server."""
    reg = registry or get_registry()
    async with _session("refresh_server") as db:
        config = (await db.execute(select(MCPServerConfig).where(MCPServerConfig.id == server_id))).scalar_one_or_none()
        if config is None:
            return None
        entry = await reg.refresh_server(config.id)
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
        entry = await reg.register(**_registry_connection_kwargs(config, reg))
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
    headers: dict[str, str] | None | CredentialNotSupplied = UNSET,
    server_env: dict[str, str] | None | CredentialNotSupplied = UNSET,
    registry: MCPServerRegistry | None = None,
) -> MCPServerConfig | None:
    """Update a server's config and reconnect it with the new settings."""
    async with _session("update_server") as db:
        config = (
            await db.execute(select(MCPServerConfig).where(MCPServerConfig.id == server_id).with_for_update())
        ).scalar_one_or_none()
        if config is None:
            return None

        effective_transport = transport if transport is not None else config.transport.value
        effective_command = command if command is not None else config.command
        effective_url = url if url is not None else config.url
        if (
            config.headers_encrypted is not None
            and effective_url != config.url
            and isinstance(headers, CredentialNotSupplied)
        ):
            raise MCPAuthConfigurationError(
                "Replace or clear static headers explicitly when changing the MCP server URL"
            )
        if (
            config.stdio_env_encrypted is not None
            and effective_command != config.command
            and isinstance(server_env, CredentialNotSupplied)
        ):
            raise MCPAuthConfigurationError(
                "Replace or clear stdio credentials explicitly when changing the MCP server command"
            )
        stored_oauth = _decrypt_mapping(config.oauth_encrypted)
        if stored_oauth is not None and effective_url != stored_oauth.get("server_url"):
            raise MCPAuthConfigurationError("Clear OAuth credentials before changing the MCP server URL")
        if isinstance(headers, CredentialNotSupplied):
            effective_headers = _decrypt_headers(config.headers_encrypted)
        else:
            effective_headers = validate_http_headers(headers)
        if isinstance(server_env, CredentialNotSupplied):
            effective_server_env = _decrypt_stdio_env(config.stdio_env_encrypted)
        else:
            effective_server_env = validate_stdio_env(server_env)
        _validate_registration(effective_transport, effective_command, effective_url)
        _validate_auth_combination(
            effective_transport,
            headers=effective_headers,
            server_env=effective_server_env,
            oauth_configured=config.oauth_encrypted is not None,
        )

        if name is not None:
            await _lock_server_name(db, name)
            if await _name_conflicts(
                db,
                name=name,
                owner_id=config.owner_id,
                is_system=config.is_system,
                exclude_id=config.id,
            ):
                _raise_name_conflict(name)
            config.name = name
        if transport is not None:
            config.transport = MCPTransport(transport)
        if command is not None:
            config.command = command
        if url is not None:
            config.url = url
        if not isinstance(headers, CredentialNotSupplied):
            config.headers_encrypted = _encrypt_headers(headers)
            if headers:
                # Static credentials explicitly supersede an authorization
                # transaction that may still be exchanging its code in a
                # different process. The callback CAS below will reject it.
                config.oauth_pending_encrypted = None
                config.oauth_state_hash = None
                config.oauth_authorization_required = False
        if not isinstance(server_env, CredentialNotSupplied):
            config.stdio_env_encrypted = _encrypt_stdio_env(server_env)
        _bump_config_revision(config)
        try:
            await db.flush()
        except IntegrityError as exc:
            raise MCPServerNameConflictError(
                f"MCP server name '{config.name}' is already in use in this namespace"
            ) from exc
        # Publish the new revision and release the row lock before connecting.
        # An OAuth-enabled connect may itself need the same row lock to refresh
        # an expired token; retaining it here would self-deadlock.
        await db.commit()

        reg = registry or get_registry()
        entry = await reg.register(**_registry_connection_kwargs(config, reg))
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
        await (registry or get_registry()).unregister(server_id)
        return None
    reg = registry or get_registry()
    # Synchronization may replace a live entry whose persisted configuration
    # changed, but a direct tool call must not resurrect a disconnected server.
    entry = reg.get(config.id)
    if entry is None or not entry.client.connected:
        raise RuntimeError(f"MCP server '{config.name}' is not connected")
    entry = await _synchronize_server(config.id, reg)
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
    :class:`~motoro.mcp.registry.MCPServerRegistry`: a worker, a fresh
    script invocation, a restarted API. Without it, ``register_server`` having
    persisted a config is pointless — nothing would ever read it back into a
    live connection.

    Returns the names of servers that failed to connect (already logged).
    Connected entries whose persisted ``config_revision`` changed are replaced,
    and entries deleted by another process are unregistered. Calling this twice
    is therefore a cheap no-op only while persisted state is unchanged.

    Safe to call concurrently: the already-connected check happens inside the
    registry's lock (``ensure_registered``), not here. It used to be a
    ``config.name in reg.servers`` test in this loop, which is a check-then-act
    race -- N concurrent hydrations all saw a server as missing before any of
    them finished connecting it, so all N registered, and every one after the
    first tore down a live connection the others were still using. Callers that
    hydrate per-unit-of-work rather than once at startup (a worker picking up a
    server registered since it booted) hit that with any real concurrency.
    """
    reg = registry or get_registry()
    failed: list[str] = []
    async with _session("hydrate_registry") as db:
        configs = (await db.execute(select(MCPServerConfig))).scalars().all()

    persisted_ids = {config.id for config in configs}
    for stale_id in set(reg.servers) - persisted_ids:
        await reg.unregister(stale_id)

    for config in configs:
        try:
            entry = await reg.ensure_registered(**_registry_connection_kwargs(config, reg))
            if not entry.client.connected:
                failed.append(config.name)
        except Exception:
            logger.warning("mcp_service.hydrate_failed", exc_info=True, extra={"server": config.name})
            failed.append(config.name)
    return failed


async def get_authentication_status(
    server_id: uuid.UUID, *, owner_id: uuid.UUID | None = None
) -> MCPAuthenticationStatus:
    """Return only non-secret authentication metadata for a persisted server."""
    async with _session("get authentication status") as db:
        config = (await db.execute(select(MCPServerConfig).where(MCPServerConfig.id == server_id))).scalar_one_or_none()
        if config is None or (owner_id is not None and config.owner_id != owner_id):
            raise MCPServerNotFoundError("MCP server was not found")
        env = _decrypt_stdio_env(config.stdio_env_encrypted) or {}
        static = config.headers_encrypted is not None
        oauth = config.oauth_encrypted is not None or config.oauth_pending_encrypted is not None
        stdio = config.stdio_env_encrypted is not None
        mode: Literal["none", "static_headers", "stdio_env", "oauth"]
        if oauth:
            mode = "oauth"
        elif static:
            mode = "static_headers"
        elif stdio:
            mode = "stdio_env"
        else:
            mode = "none"
        return MCPAuthenticationStatus(
            server_id=config.id,
            auth_mode=mode,
            configured=config.oauth_encrypted is not None if mode == "oauth" else mode != "none",
            authorization_required=config.oauth_authorization_required,
            static_headers_configured=static,
            stdio_env_configured=stdio,
            stdio_env_names=tuple(sorted(env)),
        )


def _validate_redirect_uri(redirect_uri: str) -> None:
    parsed = urlparse(redirect_uri)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
        raise MCPAuthConfigurationError("OAuth redirect URI must be an absolute HTTP(S) URL")
    if parsed.fragment:
        raise MCPAuthConfigurationError("OAuth redirect URI must not contain a fragment")


def _authorization_url(endpoint_url: str, params: dict[str, str]) -> str:
    """Merge generated OAuth parameters into an endpoint's existing query."""
    endpoint = urlparse(endpoint_url)
    existing = [(key, value) for key, value in parse_qsl(endpoint.query, keep_blank_values=True) if key not in params]
    return endpoint._replace(query=urlencode([*existing, *params.items()])).geturl()


def _validate_oauth_endpoint(url: str) -> None:
    from motoro.config import settings

    parsed = urlparse(url)
    host = parsed.hostname or ""
    is_loopback = host == "localhost"
    with contextlib.suppress(ValueError):
        is_loopback = ipaddress.ip_address(host).is_loopback
    if parsed.scheme != "https" and not (parsed.scheme == "http" and is_loopback):
        raise MCPOAuthDiscoveryError("OAuth endpoints must use HTTPS except on loopback hosts")
    validate_outbound_url(url, allow_private=settings.mcp_allow_private_urls)


async def _discover_oauth(
    server_url: str, client: httpx.AsyncClient
) -> tuple[ProtectedResourceMetadata, OAuthMetadata, str | None]:
    """Discover and strictly bind RFC 9728 and RFC 8414 metadata."""
    challenge: httpx.Response | None = None
    with contextlib.suppress(httpx.HTTPError):
        challenge = await client.get(server_url, headers={"Accept": "application/json"})
    advertised = extract_resource_metadata_from_www_auth(challenge) if challenge is not None else None
    challenge_scope = extract_scope_from_www_auth(challenge) if challenge is not None else None
    prm: ProtectedResourceMetadata | None = None
    for url in build_protected_resource_metadata_discovery_urls(advertised, server_url):
        _validate_oauth_endpoint(url)
        try:
            response = await client.get(url)
        except httpx.HTTPError:
            continue
        prm = await handle_protected_resource_response(response)
        if prm is not None:
            break
    if prm is None:
        raise MCPOAuthDiscoveryError("MCP protected-resource metadata could not be discovered")

    expected_resource = resource_url_from_server_url(server_url)
    if not check_resource_allowed(requested_resource=expected_resource, configured_resource=str(prm.resource)):
        raise MCPOAuthDiscoveryError("MCP protected-resource metadata does not match the registered server")

    auth_server = str(prm.authorization_servers[0])
    metadata: OAuthMetadata | None = None
    for url in build_oauth_authorization_server_metadata_discovery_urls(auth_server, server_url):
        _validate_oauth_endpoint(url)
        try:
            response = await client.get(url)
        except httpx.HTTPError:
            continue
        keep_trying, candidate = await handle_auth_metadata_response(response)
        if candidate is not None:
            metadata = candidate
            break
        if not keep_trying:
            break
    if metadata is None:
        raise MCPOAuthDiscoveryError("MCP authorization-server metadata could not be discovered")
    if str(metadata.issuer) != auth_server:
        raise MCPOAuthDiscoveryError("MCP authorization-server issuer does not match protected-resource metadata")
    for endpoint in (metadata.authorization_endpoint, metadata.token_endpoint, metadata.registration_endpoint):
        if endpoint is not None:
            _validate_oauth_endpoint(str(endpoint))
    if (
        metadata.code_challenge_methods_supported is not None
        and "S256" not in metadata.code_challenge_methods_supported
    ):
        raise MCPOAuthDiscoveryError("MCP authorization server does not support PKCE S256")
    return prm, metadata, challenge_scope


async def begin_oauth_authorization(
    server_id: uuid.UUID,
    *,
    redirect_uri: str,
    client_metadata: MCPOAuthClientMetadata,
    client_metadata_url: str | None = None,
    owner_id: uuid.UUID | None = None,
) -> MCPOAuthAuthorization:
    """Discover OAuth, register a client, and persist a restart-safe PKCE transaction."""
    _validate_redirect_uri(redirect_uri)
    if client_metadata_url is not None and not is_valid_client_metadata_url(client_metadata_url):
        raise MCPAuthConfigurationError("OAuth client metadata URL must be HTTPS with a non-root path")
    async with _session("begin OAuth load") as db:
        config = (await db.execute(select(MCPServerConfig).where(MCPServerConfig.id == server_id))).scalar_one_or_none()
        if config is None or (owner_id is not None and config.owner_id != owner_id):
            raise MCPServerNotFoundError("MCP server was not found")
        if config.transport != MCPTransport.HTTP or not config.url:
            raise MCPAuthConfigurationError("OAuth is supported only for streamable HTTP transport")
        if config.headers_encrypted is not None or config.stdio_env_encrypted is not None:
            raise MCPAuthConfigurationError("Clear existing static or stdio credentials before configuring OAuth")
        server_url = config.url
        existing_oauth = _decrypt_mapping(config.oauth_encrypted)

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
        prm, oauth_metadata, challenge_scope = await _discover_oauth(server_url, client)
        selected_scope = client_metadata.scope or get_client_metadata_scopes(challenge_scope, prm, oauth_metadata)
        sdk_metadata = OAuthClientMetadata.model_validate(
            {
                "redirect_uris": [redirect_uri],
                "token_endpoint_auth_method": "none",
                "client_name": client_metadata.client_name,
                "scope": selected_scope,
                "client_uri": client_metadata.client_uri,
                "contacts": list(client_metadata.contacts) or None,
            }
        )
        reusable_client = None
        if (
            existing_oauth is not None
            and existing_oauth.get("issuer") == str(oauth_metadata.issuer)
            and existing_oauth.get("server_url") == server_url
        ):
            candidate = existing_oauth.get("client_info")
            registered_redirects = candidate.get("redirect_uris") if isinstance(candidate, dict) else None
            secret_expiry = int(candidate.get("client_secret_expires_at") or 0) if isinstance(candidate, dict) else 0
            if (
                isinstance(candidate, dict)
                and redirect_uri in (registered_redirects or [])
                and (not secret_expiry or time.time() < secret_expiry)
            ):
                reusable_client = candidate
        if reusable_client is not None:
            client_info = OAuthClientInformationFull.model_validate(reusable_client)
        elif should_use_client_metadata_url(oauth_metadata, client_metadata_url):
            client_info = create_client_info_from_metadata_url(
                str(client_metadata_url), redirect_uris=sdk_metadata.redirect_uris
            )
        else:
            request = create_client_registration_request(
                oauth_metadata,
                sdk_metadata,
                str(oauth_metadata.issuer),
            )
            _validate_oauth_endpoint(str(request.url))
            try:
                response = await client.send(request)
                client_info = await handle_registration_response(response)
            except Exception:
                raise MCPOAuthDiscoveryError("MCP OAuth dynamic client registration failed") from None

    if not client_info.client_id:
        raise MCPOAuthDiscoveryError("MCP OAuth client registration returned no client identifier")
    if client_info.token_endpoint_auth_method not in (None, "none", "client_secret_basic", "client_secret_post"):
        raise MCPAuthConfigurationError("MCP OAuth client authentication method is not supported")
    pkce = PKCEParameters.generate()
    state = secrets.token_urlsafe(32)
    expires_at = time.time() + _OAUTH_TRANSACTION_TTL_SECONDS
    resource = str(prm.resource)
    pending = {
        "server_id": str(server_id),
        "server_url": server_url,
        "issuer": str(oauth_metadata.issuer),
        "resource": resource,
        "redirect_uri": redirect_uri,
        "state": state,
        "code_verifier": pkce.code_verifier,
        "expires_at": expires_at,
        "protected_resource_metadata": prm.model_dump(mode="json", exclude_none=True),
        "oauth_metadata": oauth_metadata.model_dump(mode="json", exclude_none=True),
        "client_info": client_info.model_dump(mode="json", exclude_none=True),
        "client_metadata": sdk_metadata.model_dump(mode="json", exclude_none=True),
        "client_metadata_url": client_metadata_url,
    }
    digest = state_hash(state)
    async with _session("begin OAuth persist") as db:
        config = (
            await db.execute(select(MCPServerConfig).where(MCPServerConfig.id == server_id).with_for_update())
        ).scalar_one_or_none()
        if config is None or config.url != server_url or (owner_id is not None and config.owner_id != owner_id):
            raise MCPServerNotFoundError("MCP server changed while OAuth authorization was starting")
        if (
            config.transport != MCPTransport.HTTP
            or config.headers_encrypted is not None
            or config.stdio_env_encrypted is not None
        ):
            raise MCPAuthConfigurationError("MCP server authentication changed while OAuth authorization was starting")
        config.oauth_pending_encrypted = _encrypt_mapping(pending)
        config.oauth_state_hash = digest
        config.oauth_authorization_required = True
        _bump_config_revision(config)
        await db.commit()

    params = {
        "response_type": "code",
        "client_id": client_info.client_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": pkce.code_challenge,
        "code_challenge_method": "S256",
        "resource": resource,
    }
    if sdk_metadata.scope:
        params["scope"] = sdk_metadata.scope
    authorization_url = _authorization_url(str(oauth_metadata.authorization_endpoint), params)
    return MCPOAuthAuthorization(
        authorization_url=authorization_url,
        expires_at=datetime.fromtimestamp(expires_at, tz=UTC),
        transaction_id=digest[:16],
    )


async def complete_oauth_authorization(
    *,
    code: str,
    state: str,
    iss: str | None = None,
    server_id: uuid.UUID | None = None,
    owner_id: uuid.UUID | None = None,
    registry: MCPServerRegistry | None = None,
) -> MCPServerConfig:
    """Consume a persisted callback transaction, store tokens, and reconnect."""
    if not code or not state:
        raise MCPOAuthCallbackError("OAuth callback must include code and state")
    digest = state_hash(state)
    claim_id = secrets.token_urlsafe(32)
    async with _session("complete OAuth claim") as db:
        stmt = select(MCPServerConfig).where(MCPServerConfig.oauth_state_hash == digest)
        if server_id is not None:
            stmt = stmt.where(MCPServerConfig.id == server_id)
        config = (await db.execute(stmt.with_for_update())).scalar_one_or_none()
        if config is None or (owner_id is not None and config.owner_id != owner_id):
            raise MCPOAuthStateError("OAuth callback state is unknown or has already been used")
        pending = _decrypt_mapping(config.oauth_pending_encrypted)
        if pending is None or not states_equal(str(pending.get("state") or ""), state):
            raise MCPOAuthStateError("OAuth callback state is invalid")
        if time.time() > float(pending.get("expires_at") or 0):
            config.oauth_pending_encrypted = None
            config.oauth_state_hash = None
            await db.commit()
            raise MCPOAuthStateError("OAuth callback state has expired")
        issuer = str(pending.get("issuer") or "")
        if iss is not None and iss != issuer:
            raise MCPOAuthStateError("OAuth callback issuer does not match the authorization server")
        if config.url != pending.get("server_url") or str(config.id) != pending.get("server_id"):
            raise MCPOAuthStateError("OAuth callback is not bound to this MCP server")
        # Claim before network I/O: remove the public lookup index, but retain
        # an encrypted transaction marker for the post-exchange compare-and-set.
        # A clear, replacement, config edit, or newer OAuth begin changes the
        # revision and/or marker, so this callback cannot resurrect credentials.
        pending["claim_id"] = claim_id
        pending["claimed"] = True
        expected_revision = config.config_revision
        config.oauth_pending_encrypted = _encrypt_mapping(pending)
        config.oauth_state_hash = None
        await db.commit()

    oauth_metadata = pending["oauth_metadata"]
    client_info = pending["client_info"]
    client_secret_expiry = int(client_info.get("client_secret_expires_at") or 0)
    if client_secret_expiry and time.time() >= client_secret_expiry:
        raise MCPOAuthCallbackError("MCP OAuth client registration has expired")
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": str(pending["redirect_uri"]),
        "client_id": str(client_info["client_id"]),
        "code_verifier": str(pending["code_verifier"]),
        "resource": str(pending["resource"]),
    }
    data, headers = prepare_token_auth(data, client_info)
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
            response = await client.post(str(oauth_metadata["token_endpoint"]), data=data, headers=headers)
        if response.status_code != 200:
            raise MCPOAuthCallbackError("MCP OAuth token exchange was rejected")
        token = OAuthToken.model_validate_json(response.content)
    except MCPAuthenticationError:
        raise
    except Exception:
        raise MCPOAuthCallbackError("MCP OAuth token exchange failed") from None

    payload = {
        "server_id": str(pending["server_id"]),
        "server_url": str(pending["server_url"]),
        "issuer": issuer,
        "resource": str(pending["resource"]),
        "protected_resource_metadata": pending["protected_resource_metadata"],
        "oauth_metadata": oauth_metadata,
        "client_info": client_info,
        "client_metadata": pending["client_metadata"],
        "client_metadata_url": pending.get("client_metadata_url"),
        "token": token.model_dump(mode="json", exclude_none=True),
        "expires_at": time.time() + token.expires_in if token.expires_in is not None else None,
    }
    async with _session("complete OAuth persist") as db:
        config = (
            await db.execute(
                select(MCPServerConfig)
                .where(MCPServerConfig.id == uuid.UUID(str(pending["server_id"])))
                .with_for_update()
            )
        ).scalar_one_or_none()
        if config is None or config.url != pending["server_url"]:
            raise MCPServerNotFoundError("MCP server changed during OAuth token exchange")
        current_pending = _decrypt_mapping(config.oauth_pending_encrypted)
        if (
            config.config_revision != expected_revision
            or current_pending is None
            or not states_equal(str(current_pending.get("claim_id") or ""), claim_id)
            or config.oauth_state_hash is not None
            or config.headers_encrypted is not None
            or config.stdio_env_encrypted is not None
            or config.transport != MCPTransport.HTTP
        ):
            raise MCPOAuthStateError("MCP server authentication changed during OAuth token exchange")
        config.oauth_encrypted = _encrypt_mapping(payload)
        config.oauth_pending_encrypted = None
        config.oauth_authorization_required = False
        _bump_config_revision(config)
        await db.commit()

    reconnected = await reconnect_server(config.id, registry=registry)
    if reconnected is None:
        raise MCPServerNotFoundError("MCP server was removed during OAuth authorization")
    if reconnected.oauth_authorization_required:
        raise MCPReauthorizationRequiredError("MCP OAuth authorization is required")
    return reconnected


async def clear_oauth_credentials(
    server_id: uuid.UUID,
    *,
    owner_id: uuid.UUID | None = None,
    revoke: bool = False,
    registry: MCPServerRegistry | None = None,
) -> bool:
    """Clear OAuth material locally; optionally attempt RFC 7009 revocation first.

    Returns whether a remote revocation endpoint accepted at least one token.
    Local clearing always occurs, including when revocation is unavailable or
    fails.
    """
    async with _session("clear OAuth load") as db:
        config = (
            await db.execute(select(MCPServerConfig).where(MCPServerConfig.id == server_id).with_for_update())
        ).scalar_one_or_none()
        if config is None or (owner_id is not None and config.owner_id != owner_id):
            raise MCPServerNotFoundError("MCP server was not found")
        payload = _decrypt_mapping(config.oauth_encrypted)

    revoked = False
    if revoke and payload:
        metadata = payload.get("oauth_metadata") or {}
        endpoint = metadata.get("revocation_endpoint")
        token = payload.get("token") or {}
        client_info = payload.get("client_info") or {}
        if endpoint:
            _validate_oauth_endpoint(str(endpoint))
            for hint, value in (
                ("refresh_token", token.get("refresh_token")),
                ("access_token", token.get("access_token")),
            ):
                if not value:
                    continue
                data = {
                    "token": str(value),
                    "token_type_hint": hint,
                    "client_id": str(client_info.get("client_id") or ""),
                }
                data, headers = prepare_token_auth(data, client_info)
                try:
                    async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
                        response = await client.post(str(endpoint), data=data, headers=headers)
                    revoked = revoked or response.status_code < 300
                except httpx.HTTPError:
                    pass

    async with _session("clear OAuth persist") as db:
        config = (
            await db.execute(select(MCPServerConfig).where(MCPServerConfig.id == server_id).with_for_update())
        ).scalar_one_or_none()
        if config is None or (owner_id is not None and config.owner_id != owner_id):
            raise MCPServerNotFoundError("MCP server was not found")
        config.oauth_encrypted = None
        config.oauth_pending_encrypted = None
        config.oauth_state_hash = None
        config.oauth_authorization_required = False
        config.status = MCPServerStatus.DISCONNECTED
        config.error_message = None
        _bump_config_revision(config)
        await db.commit()
    await (registry or get_registry()).unregister(server_id)
    return revoked


async def clear_server_credentials(
    server_id: uuid.UUID,
    *,
    owner_id: uuid.UUID | None = None,
    registry: MCPServerRegistry | None = None,
) -> MCPServerConfig:
    """Clear every supported credential kind without deleting the registration."""
    async with _session("clear all MCP credentials") as db:
        config = (
            await db.execute(select(MCPServerConfig).where(MCPServerConfig.id == server_id).with_for_update())
        ).scalar_one_or_none()
        if config is None or (owner_id is not None and config.owner_id != owner_id):
            raise MCPServerNotFoundError("MCP server was not found")
        config.headers_encrypted = None
        config.stdio_env_encrypted = None
        config.oauth_encrypted = None
        config.oauth_pending_encrypted = None
        config.oauth_state_hash = None
        config.oauth_authorization_required = False
        config.status = MCPServerStatus.DISCONNECTED
        config.error_message = None
        _bump_config_revision(config)
        await db.commit()
        await db.refresh(config)
    await (registry or get_registry()).unregister(server_id)
    return config

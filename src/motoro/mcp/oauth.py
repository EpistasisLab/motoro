"""Headless OAuth token use for streamable HTTP MCP connections."""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any
from urllib.parse import quote

import anyio
import httpx


class MCPAuthenticationError(ValueError):
    """Base class for product-mappable MCP authentication errors."""


class MCPAuthConfigurationError(MCPAuthenticationError):
    """The selected credentials are incompatible with the server transport."""


class MCPOAuthDiscoveryError(MCPAuthenticationError):
    """OAuth protected-resource or authorization-server discovery failed."""


class MCPOAuthStateError(MCPAuthenticationError):
    """An OAuth callback is expired, unknown, replayed, or has mismatched state."""


class MCPOAuthCallbackError(MCPAuthenticationError):
    """The authorization server rejected or returned an invalid code exchange."""


class MCPReauthorizationRequiredError(RuntimeError):
    """Stored OAuth authorization cannot be refreshed and user interaction is required."""


RefreshOAuth = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
MarkReauthorizationRequired = Callable[[dict[str, Any]], Awaitable[None]]


def prepare_token_auth(data: dict[str, str], client_info: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Apply RFC 6749 client authentication without exposing values in errors."""
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    method = client_info.get("token_endpoint_auth_method")
    client_id = str(client_info.get("client_id") or "")
    client_secret = client_info.get("client_secret")
    if method == "client_secret_basic" and client_id and client_secret:
        pair = f"{quote(client_id, safe='')}:{quote(str(client_secret), safe='')}"
        headers["Authorization"] = f"Basic {base64.b64encode(pair.encode()).decode()}"
    elif method == "client_secret_post" and client_secret:
        data["client_secret"] = str(client_secret)
    return data, headers


class PersistentOAuthAuth(httpx.Auth):
    """HTTPX auth that refreshes and persists an issuer/server-bound token set.

    It intentionally has no redirect or callback handler: a worker can consume
    and refresh existing credentials, but can never start an interactive flow.
    """

    def __init__(
        self,
        payload: dict[str, Any],
        refresh: RefreshOAuth,
        mark_reauthorization_required: MarkReauthorizationRequired | None = None,
    ) -> None:
        self._payload = payload
        self._refresh = refresh
        self._mark_reauthorization_required = mark_reauthorization_required
        self._lock = anyio.Lock()

    def __repr__(self) -> str:
        return "PersistentOAuthAuth(configured=True)"

    def _valid(self) -> bool:
        token = self._payload.get("token") or {}
        expiry = self._payload.get("expires_at")
        return bool(token.get("access_token") and (expiry is None or time.time() < float(expiry) - 30))

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        async with self._lock:
            if not self._valid():
                try:
                    self._payload = await self._refresh(dict(self._payload))
                except MCPReauthorizationRequiredError:
                    if self._mark_reauthorization_required is not None:
                        await self._mark_reauthorization_required(self._payload)
                    raise
                if not self._valid():
                    if self._mark_reauthorization_required is not None:
                        await self._mark_reauthorization_required(self._payload)
                    raise MCPReauthorizationRequiredError("MCP OAuth authorization is required")

            request.headers["Authorization"] = f"Bearer {self._payload['token']['access_token']}"
            response = yield request
            challenge = response.headers.get("WWW-Authenticate", "")
            if response.status_code == 401 or (
                response.status_code == 403 and "insufficient_scope" in challenge.lower()
            ):
                if self._mark_reauthorization_required is not None:
                    await self._mark_reauthorization_required(self._payload)
                raise MCPReauthorizationRequiredError("MCP OAuth authorization is required")


def state_hash(state: str) -> str:
    """Stable non-secret callback lookup key."""
    return hashlib.sha256(state.encode()).hexdigest()


def states_equal(left: str, right: str) -> bool:
    """Constant-time callback-state comparison."""
    return secrets.compare_digest(left, right)

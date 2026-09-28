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
from mcp.shared.auth import OAuthToken
from pydantic import ValidationError


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


PersistOAuth = Callable[[dict[str, Any]], Awaitable[None]]
MarkReauthorizationRequired = Callable[[], Awaitable[None]]


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
        persist: PersistOAuth,
        mark_reauthorization_required: MarkReauthorizationRequired | None = None,
    ) -> None:
        self._payload = payload
        self._persist = persist
        self._mark_reauthorization_required = mark_reauthorization_required
        self._lock = anyio.Lock()

    def __repr__(self) -> str:
        return "PersistentOAuthAuth(configured=True)"

    def _valid(self) -> bool:
        token = self._payload.get("token") or {}
        expiry = self._payload.get("expires_at")
        return bool(token.get("access_token") and (expiry is None or time.time() < float(expiry) - 30))

    def _refresh_request(self) -> httpx.Request:
        token = self._payload.get("token") or {}
        client = self._payload.get("client_info") or {}
        metadata = self._payload.get("oauth_metadata") or {}
        refresh_token = token.get("refresh_token")
        if not refresh_token or not client.get("client_id") or not metadata.get("token_endpoint"):
            raise MCPReauthorizationRequiredError("MCP OAuth authorization is required")
        client_secret_expiry = int(client.get("client_secret_expires_at") or 0)
        if client_secret_expiry and time.time() >= client_secret_expiry:
            raise MCPReauthorizationRequiredError("MCP OAuth authorization is required")
        data = {
            "grant_type": "refresh_token",
            "refresh_token": str(refresh_token),
            "client_id": str(client["client_id"]),
            "resource": str(self._payload["resource"]),
        }
        data, headers = prepare_token_auth(data, client)
        return httpx.Request("POST", str(metadata["token_endpoint"]), data=data, headers=headers)

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        async with self._lock:
            if not self._valid():
                try:
                    refresh_request = self._refresh_request()
                except MCPReauthorizationRequiredError:
                    if self._mark_reauthorization_required is not None:
                        await self._mark_reauthorization_required()
                    raise
                try:
                    refresh_response = yield refresh_request
                except httpx.HTTPError:
                    if self._mark_reauthorization_required is not None:
                        await self._mark_reauthorization_required()
                    raise MCPReauthorizationRequiredError("MCP OAuth authorization is required") from None
                if refresh_response.status_code != 200:
                    if self._mark_reauthorization_required is not None:
                        await self._mark_reauthorization_required()
                    raise MCPReauthorizationRequiredError("MCP OAuth authorization is required")
                try:
                    refreshed = OAuthToken.model_validate_json(await refresh_response.aread())
                except ValidationError:
                    if self._mark_reauthorization_required is not None:
                        await self._mark_reauthorization_required()
                    raise MCPReauthorizationRequiredError("MCP OAuth authorization is required") from None
                token_data = refreshed.model_dump(mode="json", exclude_none=True)
                # RFC 6749 permits refresh responses to omit a replacement refresh
                # token; retain the old one in that case.
                if "refresh_token" not in token_data:
                    token_data["refresh_token"] = self._payload["token"].get("refresh_token")
                if "scope" not in token_data and self._payload["token"].get("scope") is not None:
                    token_data["scope"] = self._payload["token"]["scope"]
                self._payload["token"] = token_data
                self._payload["expires_at"] = (
                    time.time() + refreshed.expires_in if refreshed.expires_in is not None else None
                )
                await self._persist(self._payload)

            request.headers["Authorization"] = f"Bearer {self._payload['token']['access_token']}"
            response = yield request
            challenge = response.headers.get("WWW-Authenticate", "")
            if response.status_code == 401 or (
                response.status_code == 403 and "insufficient_scope" in challenge.lower()
            ):
                if self._mark_reauthorization_required is not None:
                    await self._mark_reauthorization_required()
                raise MCPReauthorizationRequiredError("MCP OAuth authorization is required")


def state_hash(state: str) -> str:
    """Stable non-secret callback lookup key."""
    return hashlib.sha256(state.encode()).hexdigest()


def states_equal(left: str, right: str) -> bool:
    """Constant-time callback-state comparison."""
    return secrets.compare_digest(left, right)

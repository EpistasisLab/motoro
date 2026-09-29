"""Focused, provider-independent MCP credential and OAuth tests."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest


def test_static_header_validation_rejects_injection_and_transport_headers() -> None:
    from motoro.security.mcp_credentials import MCPCredentialValidationError, validate_http_headers

    assert validate_http_headers({"Authorization": "Bearer opaque", "X-API-Key": "opaque"}) == {
        "Authorization": "Bearer opaque",
        "X-API-Key": "opaque",
    }
    with pytest.raises(MCPCredentialValidationError, match="invalid HTTP field value"):
        validate_http_headers({"Authorization": "opaque\r\nHost: attacker"})
    with pytest.raises(MCPCredentialValidationError, match="controlled by the MCP transport"):
        validate_http_headers({"MCP-Session-Id": "fixed"})
    with pytest.raises(MCPCredentialValidationError, match="valid HTTP field name"):
        validate_http_headers({"bad header": "opaque"})


@pytest.mark.parametrize(
    "name",
    ["LD_PRELOAD", "DYLD_INSERT_LIBRARIES", "PYTHONPATH", "PYTHONHOME", "NODE_OPTIONS", "BASH_ENV"],
)
def test_stdio_env_validation_rejects_process_injection_variables(name: str) -> None:
    from motoro.security.mcp_credentials import MCPCredentialValidationError, validate_stdio_env

    with pytest.raises(MCPCredentialValidationError, match="may alter process execution"):
        validate_stdio_env({name: "opaque"})


def test_explicit_stdio_credentials_do_not_depend_on_host_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    from motoro.mcp.client import build_subprocess_env

    monkeypatch.setenv("HOST_SECRET", "must-not-inherit")
    built = build_subprocess_env(allowed_env_vars=frozenset({"PATH"}), server_env={"PRODUCT_API_TOKEN": "opaque"})
    assert built["PRODUCT_API_TOKEN"] == "opaque"
    assert "HOST_SECRET" not in built


def test_encrypted_credential_blobs_do_not_contain_plaintext() -> None:
    from pydantic_settings import SettingsConfigDict

    from motoro import CoreSettings
    from motoro.config import configure, reset_for_testing
    from motoro.services.encryption import reset_for_testing as reset_encryption
    from motoro.services.mcp_service import _decrypt_mapping, _encrypt_mapping

    class Settings(CoreSettings):
        model_config = SettingsConfigDict(extra="ignore")

    reset_for_testing()
    reset_encryption()
    configure(Settings(encryption_key="9Ka2Wb6GS2vfw9aBZiR_MtRNJtftxuIzl6YoZTU-fCA="))
    try:
        encrypted = _encrypt_mapping({"access_token": "plaintext-token", "refresh_token": "plaintext-refresh"})
        assert encrypted is not None
        assert "plaintext-token" not in encrypted
        assert "plaintext-refresh" not in encrypted
        assert _decrypt_mapping(encrypted) == {
            "access_token": "plaintext-token",
            "refresh_token": "plaintext-refresh",
        }
    finally:
        reset_encryption()
        reset_for_testing()


async def test_oauth_refresh_persists_rotated_refresh_token_without_leaking_it() -> None:
    from motoro.mcp.oauth import PersistentOAuthAuth

    persisted: list[dict[str, Any]] = []
    requests: list[httpx.Request] = []

    async def refresh(payload: dict[str, Any]) -> dict[str, Any]:
        requests.append(httpx.Request("POST", "https://auth.example/token"))
        payload["token"] = {
            "access_token": "new-access-secret",
            "refresh_token": "rotated-refresh-secret",
        }
        payload["expires_at"] = time.time() + 3600
        persisted.append(json.loads(json.dumps(payload)))
        return payload

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer new-access-secret"
        return httpx.Response(200, json={"ok": True})

    payload = {
        "issuer": "https://auth.example",
        "server_url": "https://mcp.example/mcp",
        "resource": "https://mcp.example/mcp",
        "oauth_metadata": {"token_endpoint": "https://auth.example/token"},
        "client_info": {"client_id": "client", "token_endpoint_auth_method": "none"},
        "token": {"access_token": "old-access-secret", "refresh_token": "old-refresh-secret"},
        "expires_at": time.time() - 1,
    }
    auth = PersistentOAuthAuth(payload, refresh)
    async with httpx.AsyncClient(auth=auth, transport=httpx.MockTransport(handler)) as client:
        response = await client.get("https://mcp.example/mcp")

    assert response.status_code == 200
    assert persisted[-1]["token"]["refresh_token"] == "rotated-refresh-secret"
    assert "old-refresh-secret" not in repr(auth)
    assert "rotated-refresh-secret" not in repr(auth)
    assert [str(request.url) for request in requests] == [
        "https://auth.example/token",
        "https://mcp.example/mcp",
    ]


async def test_oauth_refresh_failure_is_typed_and_marks_reauthorization() -> None:
    from motoro.mcp.oauth import MCPReauthorizationRequiredError, PersistentOAuthAuth

    marked = False

    async def refresh(_payload: dict[str, Any]) -> dict[str, Any]:
        raise MCPReauthorizationRequiredError("MCP OAuth authorization is required")

    async def mark(_payload: dict[str, Any]) -> None:
        nonlocal marked
        marked = True

    payload = {
        "resource": "https://mcp.example/mcp",
        "oauth_metadata": {"token_endpoint": "https://auth.example/token"},
        "client_info": {"client_id": "client", "token_endpoint_auth_method": "none"},
        "token": {"access_token": "secret", "refresh_token": "refresh-secret"},
        "expires_at": 0,
    }
    auth = PersistentOAuthAuth(payload, refresh, mark)
    with pytest.raises(MCPReauthorizationRequiredError) as raised:
        async with httpx.AsyncClient(
            auth=auth, transport=httpx.MockTransport(lambda _request: httpx.Response(200))
        ) as client:
            await client.get("https://mcp.example/mcp")
    assert marked
    assert "secret" not in str(raised.value)


def test_model_and_migration_shape_hide_and_persist_authentication() -> None:
    from alembic.script import ScriptDirectory

    from motoro.migrations import make_config
    from motoro.models.agent import Agent  # noqa: F401 - resolves ORM relationships
    from motoro.models.mcp_server import MCPServerConfig, MCPServerStatus, MCPTransport
    from motoro.models.run import AgentRun  # noqa: F401 - resolves ORM relationships

    columns = {column.name for column in MCPServerConfig.__table__.columns}
    assert {
        "stdio_env_encrypted",
        "oauth_encrypted",
        "oauth_pending_encrypted",
        "oauth_state_hash",
        "oauth_authorization_required",
        "config_revision",
    } <= columns
    migration = Path(__file__).parents[1] / "src/motoro/migrations/versions/2e6f4c8a91d3_mcp_authentication.py"
    source = migration.read_text()
    assert 'revision: str = "2e6f4c8a91d3"' in source
    assert 'down_revision: str | None = "8f2c1a6d9b40"' in source
    revision_migration = (
        Path(__file__).parents[1] / "src/motoro/migrations/versions/6a4d9f2c7e10_mcp_config_revision.py"
    )
    revision_source = revision_migration.read_text()
    assert 'revision: str = "6a4d9f2c7e10"' in revision_source
    assert 'down_revision: str | None = "2e6f4c8a91d3"' in revision_source
    config = make_config("postgresql+asyncpg://unused:unused@localhost/unused")
    assert ScriptDirectory.from_config(config).get_heads() == ["6a4d9f2c7e10"]
    row = MCPServerConfig(
        id=uuid.uuid4(),
        name="safe",
        transport=MCPTransport.HTTP,
        status=MCPServerStatus.DISCONNECTED,
        headers_encrypted="ciphertext-must-not-appear",
        oauth_encrypted="other-ciphertext-must-not-appear",
    )
    assert "ciphertext" not in repr(row)


async def test_oauth_discovery_validates_resource_and_issuer() -> None:
    from pydantic_settings import SettingsConfigDict

    from motoro import CoreSettings
    from motoro.config import configure, reset_for_testing
    from motoro.services.mcp_service import _discover_oauth

    class Settings(CoreSettings):
        model_config = SettingsConfigDict(extra="ignore")

    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "http://127.0.0.1:9000/mcp":
            return httpx.Response(
                401,
                headers={"WWW-Authenticate": ('Bearer resource_metadata="http://127.0.0.1:9000/resource-metadata"')},
            )
        if url == "http://127.0.0.1:9000/resource-metadata":
            return httpx.Response(
                200,
                json={
                    "resource": "http://127.0.0.1:9000/mcp",
                    "authorization_servers": ["http://127.0.0.1:9001/"],
                },
            )
        if url == "http://127.0.0.1:9001/.well-known/oauth-authorization-server":
            return httpx.Response(
                200,
                json={
                    "issuer": "http://127.0.0.1:9001/",
                    "authorization_endpoint": "http://127.0.0.1:9001/authorize",
                    "token_endpoint": "http://127.0.0.1:9001/token",
                    "registration_endpoint": "http://127.0.0.1:9001/register",
                    "code_challenge_methods_supported": ["S256"],
                },
            )
        return httpx.Response(404)

    reset_for_testing()
    configure(Settings(mcp_allow_private_urls=True))
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            resource, authorization, scope = await _discover_oauth("http://127.0.0.1:9000/mcp", client)
        assert str(resource.resource) == "http://127.0.0.1:9000/mcp"
        assert str(authorization.issuer) == "http://127.0.0.1:9001/"
        assert scope is None
    finally:
        reset_for_testing()


def test_oauth_pkce_and_state_are_random_and_state_index_is_one_way() -> None:
    import secrets

    from mcp.client.auth.oauth2 import PKCEParameters

    from motoro.mcp.oauth import state_hash

    first = PKCEParameters.generate()
    second = PKCEParameters.generate()
    assert len(first.code_verifier) == 128
    assert first.code_challenge != second.code_challenge
    state = secrets.token_urlsafe(32)
    digest = state_hash(state)
    assert len(digest) == 64
    assert state not in digest


async def test_registry_replaces_only_when_persisted_revision_advances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from motoro.mcp.client import MCPClient
    from motoro.mcp.registry import MCPServerRegistry

    async def connect(client: MCPClient) -> None:
        client._connected = True

    async def disconnect(client: MCPClient) -> None:
        client._connected = False

    monkeypatch.setattr(MCPClient, "connect", connect)
    monkeypatch.setattr(MCPClient, "disconnect", disconnect)
    registry = MCPServerRegistry()
    server_id = uuid.uuid4()
    first = await registry.ensure_registered(
        server_id=server_id,
        name="coherent",
        headers={"Authorization": "Bearer first-secret"},
        config_revision=1,
    )
    unchanged = await registry.ensure_registered(
        server_id=server_id,
        name="coherent",
        headers={"Authorization": "Bearer ignored-same-revision"},
        config_revision=1,
    )
    replaced = await registry.ensure_registered(
        server_id=server_id,
        name="coherent",
        headers={"Authorization": "Bearer replacement-secret"},
        config_revision=2,
    )
    cleared = await registry.ensure_registered(
        server_id=server_id,
        name="coherent",
        headers=None,
        config_revision=3,
    )

    assert unchanged is first
    assert replaced is not first
    assert not first.client.connected
    assert not replaced.client.connected
    assert replaced.config_revision == 2
    assert replaced.client._headers == {"Authorization": "Bearer replacement-secret"}
    assert cleared.client.connected
    assert cleared.client._headers == {}
    assert cleared.config_revision == 3

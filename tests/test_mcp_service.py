"""MCP server registration — persistence for the already-landed MCP client.

Three things worth pinning:

1. ``owner_id`` on ``MCPServerConfig`` follows the same severance as
   ``Agent``/``AgentRun``/``MemoryEntry``: opaque, nullable, no foreign key.
   ``source_plan_id`` (ARES's Plan Builder provenance link) is dropped
   entirely rather than made opaque — it names a product feature, not a fact
   about the server.
2. ``hydrate_registry`` is the point of persisting servers at all: a fresh,
   empty ``MCPServerRegistry`` reconnects from the table alone, with no code
   re-registering anything.
3. A real connect/list-tools/disconnect cycle against a live server completes
   within a bounded timeout. This is the regression guard for a genuine bug
   found while writing these tests: ``mcp==2.0.0`` (satisfies the previously
   unbounded ``mcp>=1.27.1`` floor) hangs ``MCPClient.connect()`` forever,
   because that method always installs a ``message_handler`` on
   ``ClientSession`` and 2.0.0's background message-reading task never lets
   the process exit. Verified directly: identical connect/list_tools call
   returns immediately under 1.27.1 (what ARES actually locks) and hangs
   indefinitely under 2.0.0. No existing test caught this because none
   exercised a live server before this file did — pyproject now caps
   ``mcp<2.0.0``; the wrapped tests below additionally guard with an explicit
   timeout so a future regression fails fast instead of hanging the suite.

Database-backed tests are skipped unless ``AGENTIC_TEST_DATABASE_URL`` is set.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
from pydantic_settings import SettingsConfigDict

from agentic_core import CoreSettings

DB_URL = os.environ.get("AGENTIC_TEST_DATABASE_URL", "")
needs_db = pytest.mark.skipif(not DB_URL, reason="AGENTIC_TEST_DATABASE_URL is not set")

_FIXTURE_SERVER = Path(__file__).parent / "fixtures" / "echo_mcp_server.py"
_ECHO_COMMAND = f"{sys.executable} {_FIXTURE_SERVER}"

# Every call that touches a live subprocess is bounded, so a regression like the
# one this file exists to catch fails the test instead of hanging the suite.
_LIVE_TIMEOUT = 15


class _Settings(CoreSettings):
    model_config = SettingsConfigDict(env_prefix="AGENTIC_TEST_", extra="ignore")


@pytest.fixture(scope="module", autouse=True)
def _configure() -> None:
    from agentic_core.config import configure, reset_for_testing

    if not DB_URL:
        return
    reset_for_testing()
    configure(_Settings(database_url=DB_URL, encryption_key="9Ka2Wb6GS2vfw9aBZiR_MtRNJtftxuIzl6YoZTU-fCA="))


@pytest.fixture(autouse=True)
async def _schema() -> Any:
    """Fresh schema per test, and a fresh engine after it (own event loop each test)."""
    if not DB_URL:
        yield
        return
    from agentic_core.models.database import dispose_engine
    from agentic_core.runner import init_schema
    from agentic_core.services.encryption import reset_for_testing as reset_encryption

    reset_encryption()
    await init_schema(drop_first=True)
    yield
    await dispose_engine()


async def _with_timeout(coro: Any) -> Any:
    return await asyncio.wait_for(coro, timeout=_LIVE_TIMEOUT)


# --------------------------------------------------------------------------- #
#  Model shape — same severance as Agent/AgentRun/MemoryEntry                  #
# --------------------------------------------------------------------------- #


def test_mcp_server_config_owner_id_is_opaque() -> None:
    from agentic_core.models.mcp_server import MCPServerConfig

    columns = {c.name for c in MCPServerConfig.__table__.columns}
    assert "owner_id" in columns
    assert "created_by_id" not in columns
    # source_plan_id is ARES's Plan Builder provenance link — a product
    # feature, not a fact about the server. Dropped, not made opaque.
    assert "source_plan_id" not in columns

    owner_col = MCPServerConfig.__table__.columns["owner_id"]
    assert owner_col.nullable is True
    assert not owner_col.foreign_keys


# --------------------------------------------------------------------------- #
#  Security modules — self-contained, no ares/agentic_core coupling            #
# --------------------------------------------------------------------------- #


def test_command_allowlist_accepts_known_executables() -> None:
    from agentic_core.security.mcp_command_allowlist import validate_stdio_command

    validate_stdio_command("python server.py")
    validate_stdio_command("npx -y some-mcp-server")


def test_command_allowlist_rejects_unknown_executable() -> None:
    from agentic_core.security.mcp_command_allowlist import MCPCommandError, validate_stdio_command

    with pytest.raises(MCPCommandError, match="not in the allowed list"):
        validate_stdio_command("bash -c 'echo hi'")


def test_command_allowlist_rejects_shell_metacharacters() -> None:
    from agentic_core.security.mcp_command_allowlist import MCPCommandError, validate_stdio_command

    with pytest.raises(MCPCommandError, match="shell meta-character"):
        validate_stdio_command("python server.py; rm -rf /")


def test_ssrf_guard_rejects_private_ip() -> None:
    from agentic_core.security.ssrf_guard import SSRFError, validate_outbound_url

    with pytest.raises(SSRFError, match="private/reserved"):
        validate_outbound_url("http://169.254.169.254/", resolve_dns=False)


def test_ssrf_guard_allows_private_ip_when_opted_in() -> None:
    from agentic_core.security.ssrf_guard import validate_outbound_url

    validate_outbound_url("http://192.168.1.5/", resolve_dns=False, allow_private=True)


def test_ssrf_guard_rejects_file_scheme() -> None:
    from agentic_core.security.ssrf_guard import SSRFError, validate_outbound_url

    with pytest.raises(SSRFError, match="not permitted"):
        validate_outbound_url("file:///etc/passwd")


# --------------------------------------------------------------------------- #
#  Encryption — server-side key, not per-user                                  #
# --------------------------------------------------------------------------- #


_TEST_ENCRYPTION_KEY = "9Ka2Wb6GS2vfw9aBZiR_MtRNJtftxuIzl6YoZTU-fCA="


def _db_kwargs() -> dict[str, str]:
    """Preserve the module's DB URL across a reconfigure, when there is one."""
    return {"database_url": DB_URL} if DB_URL else {}


def _restore_module_settings() -> None:
    """Reinstall exactly what the module-scoped ``_configure`` fixture set up.

    The module fixture runs once, before the first test — a mid-module
    ``configure()`` call (these two tests install a different
    ``encryption_key`` to test the two branches) installs a whole new settings
    instance, not a patch, so anything that resets state must put the module's
    baseline back rather than leaving settings unconfigured for whatever runs
    next.
    """
    from agentic_core.config import configure, reset_for_testing

    reset_for_testing()
    configure(_Settings(encryption_key=_TEST_ENCRYPTION_KEY, **_db_kwargs()))


def test_encryption_round_trips() -> None:
    from agentic_core.config import configure, reset_for_testing
    from agentic_core.services.encryption import decrypt, encrypt
    from agentic_core.services.encryption import reset_for_testing as reset_encryption

    reset_for_testing()
    reset_encryption()
    configure(_Settings(encryption_key=_TEST_ENCRYPTION_KEY, **_db_kwargs()))
    try:
        token = encrypt("a secret header value")
        assert token != "a secret header value"
        assert decrypt(token) == "a secret header value"
    finally:
        reset_encryption()
        _restore_module_settings()


def test_encryption_without_a_key_raises() -> None:
    from agentic_core.config import configure, reset_for_testing
    from agentic_core.services.encryption import encrypt
    from agentic_core.services.encryption import reset_for_testing as reset_encryption

    reset_for_testing()
    reset_encryption()
    configure(_Settings(encryption_key="", **_db_kwargs()))
    try:
        with pytest.raises(RuntimeError, match="encryption_key is not set"):
            encrypt("anything")
    finally:
        reset_encryption()
        _restore_module_settings()


# --------------------------------------------------------------------------- #
#  mcp_service — register, read, mutate, against a real stdio server           #
# --------------------------------------------------------------------------- #


@needs_db
async def test_register_server_connects_and_persists() -> None:
    from agentic_core.mcp.registry import MCPServerRegistry
    from agentic_core.services.mcp_service import register_server

    registry = MCPServerRegistry()
    config = await _with_timeout(
        register_server(
            name=f"echo-{uuid.uuid4().hex[:8]}", transport="stdio", command=_ECHO_COMMAND, registry=registry
        )
    )
    try:
        assert config.status.value == "connected"
        assert config.capabilities is not None
        assert {t["name"] for t in config.capabilities["tools"]} >= {"echo", "reset_session"}
        assert config.owner_id is None
    finally:
        await registry.disconnect_all()


def test_build_run_meta_includes_owner_id() -> None:
    from agentic_core.engine.context import RunContext
    from agentic_core.mcp.adapters import META_KEY_OWNER_ID, META_KEY_RUN_ID, META_KEY_WORKSPACE_ID, _build_run_meta
    from agentic_core.schemas.agent import ModelConfig

    owner = uuid.uuid4()
    run = uuid.uuid4()
    context = RunContext(
        agent_goal="g",
        system_prompt="s",
        model_config=ModelConfig(),
        user_input="u",
        run_id=run,
        owner_id=owner,
        workspace_id="exp1/cellA",
    )
    meta = _build_run_meta(context)
    assert meta == {
        META_KEY_WORKSPACE_ID: "exp1/cellA",
        META_KEY_RUN_ID: str(run),
        META_KEY_OWNER_ID: str(owner),
    }

    # No ambient identity at all -> None, not an empty dict (unchanged wire call).
    bare_context = RunContext(agent_goal="g", system_prompt="s", model_config=ModelConfig(), user_input="u")
    assert _build_run_meta(bare_context) is None


@needs_db
async def test_call_tool_delivers_meta_verbatim() -> None:
    """The exact wire-level guarantee _build_run_meta/execute_step rely on:
    whatever dict is passed as ``meta=`` to ``MCPClient.call_tool`` is what a
    real tool reads back from ``ctx.request_context.meta`` — not just what the
    sender intended to build. A downstream consumer's own copy of a meta key
    (e.g. asaree_workspace_core's META_KEY_WORKSPACE_ID) can silently drift
    from this without either side raising an error, which is exactly what
    happened before this test existed."""
    from agentic_core.mcp.registry import MCPServerRegistry
    from agentic_core.services.mcp_service import register_server

    registry = MCPServerRegistry()
    name = f"echo-{uuid.uuid4().hex[:8]}"
    await _with_timeout(register_server(name=name, transport="stdio", command=_ECHO_COMMAND, registry=registry))
    try:
        client = registry.servers[name].client
        sent_meta = {"agentic_core.workspace_id": "exp1/cellA", "agentic_core.owner_id": str(uuid.uuid4())}
        result = await _with_timeout(client.call_tool("echo_meta", {}, meta=sent_meta))
        assert not result.is_error
        import json

        received_meta = json.loads(result.content)
        for key, value in sent_meta.items():
            assert received_meta.get(key) == value, f"{key} did not arrive intact: {received_meta}"
    finally:
        await registry.disconnect_all()


@needs_db
async def test_register_server_rejects_disallowed_command() -> None:
    from agentic_core.mcp.registry import MCPServerRegistry
    from agentic_core.security.mcp_command_allowlist import MCPCommandError
    from agentic_core.services.mcp_service import register_server

    registry = MCPServerRegistry()
    with pytest.raises(MCPCommandError):
        await register_server(name="bad", transport="stdio", command="bash -c evil", registry=registry)
    # Rejected before anything was spawned or persisted.
    assert registry.servers == {}


@needs_db
async def test_register_server_rejects_private_url_by_default() -> None:
    from agentic_core.mcp.registry import MCPServerRegistry
    from agentic_core.security.ssrf_guard import SSRFError
    from agentic_core.services.mcp_service import register_server

    registry = MCPServerRegistry()
    with pytest.raises(SSRFError):
        await register_server(name="internal", transport="http", url="http://169.254.169.254/mcp", registry=registry)


@needs_db
async def test_get_server_and_get_server_by_name() -> None:
    from agentic_core.mcp.registry import MCPServerRegistry
    from agentic_core.services.mcp_service import get_server, get_server_by_name, register_server

    registry = MCPServerRegistry()
    name = f"echo-{uuid.uuid4().hex[:8]}"
    config = await _with_timeout(
        register_server(name=name, transport="stdio", command=_ECHO_COMMAND, registry=registry)
    )
    try:
        assert (await get_server(config.id)).id == config.id
        assert (await get_server_by_name(name)).id == config.id
        assert await get_server(uuid.uuid4()) is None
    finally:
        await registry.disconnect_all()


@needs_db
async def test_list_servers_filters_by_owner() -> None:
    from agentic_core.mcp.registry import MCPServerRegistry
    from agentic_core.services.mcp_service import list_servers, register_server

    registry = MCPServerRegistry()
    owner = uuid.uuid4()
    mine = await _with_timeout(
        register_server(
            name=f"echo-mine-{uuid.uuid4().hex[:8]}",
            transport="stdio",
            command=_ECHO_COMMAND,
            owner_id=owner,
            registry=registry,
        )
    )
    other = await _with_timeout(
        register_server(
            name=f"echo-other-{uuid.uuid4().hex[:8]}", transport="stdio", command=_ECHO_COMMAND, registry=registry
        )
    )
    try:
        all_servers = await list_servers()
        assert {s.id for s in all_servers} == {mine.id, other.id}

        owned = await list_servers(owner_id=owner)
        assert [s.id for s in owned] == [mine.id]
    finally:
        await registry.disconnect_all()


@needs_db
async def test_register_server_is_system_and_list_servers_includes_it_for_any_owner() -> None:
    from agentic_core.mcp.registry import MCPServerRegistry
    from agentic_core.services.mcp_service import list_servers, register_server

    registry = MCPServerRegistry()
    system_config = await _with_timeout(
        register_server(
            name=f"echo-system-{uuid.uuid4().hex[:8]}",
            transport="stdio",
            command=_ECHO_COMMAND,
            is_system=True,
            registry=registry,
        )
    )
    try:
        assert system_config.is_system is True
        assert system_config.owner_id is None

        # A system server shows up for ANY owner's filtered list, not just its own.
        someone_elses_view = await list_servers(owner_id=uuid.uuid4())
        assert system_config.id in {s.id for s in someone_elses_view}
    finally:
        await registry.disconnect_all()


@needs_db
async def test_delete_server_unregisters_and_removes_row() -> None:
    from agentic_core.mcp.registry import MCPServerRegistry
    from agentic_core.services.mcp_service import delete_server, get_server, register_server

    registry = MCPServerRegistry()
    config = await _with_timeout(
        register_server(
            name=f"echo-{uuid.uuid4().hex[:8]}", transport="stdio", command=_ECHO_COMMAND, registry=registry
        )
    )
    assert await delete_server(config.id, registry=registry) is True
    assert await get_server(config.id) is None
    assert registry.servers == {}
    assert await delete_server(config.id, registry=registry) is False


@needs_db
async def test_refresh_server_rediscovers_tools() -> None:
    from agentic_core.mcp.registry import MCPServerRegistry
    from agentic_core.services.mcp_service import refresh_server, register_server

    registry = MCPServerRegistry()
    config = await _with_timeout(
        register_server(
            name=f"echo-{uuid.uuid4().hex[:8]}", transport="stdio", command=_ECHO_COMMAND, registry=registry
        )
    )
    try:
        refreshed = await _with_timeout(refresh_server(config.id, registry=registry))
        assert refreshed.status.value == "connected"
        assert {t["name"] for t in refreshed.capabilities["tools"]} >= {"echo", "reset_session"}
    finally:
        await registry.disconnect_all()


@needs_db
async def test_reconnect_server_after_manual_unregister() -> None:
    """Simulates a fresh process: the registry has forgotten the server, but the DB hasn't."""
    from agentic_core.mcp.registry import MCPServerRegistry
    from agentic_core.services.mcp_service import reconnect_server, register_server

    registry = MCPServerRegistry()
    config = await _with_timeout(
        register_server(
            name=f"echo-{uuid.uuid4().hex[:8]}", transport="stdio", command=_ECHO_COMMAND, registry=registry
        )
    )
    await registry.unregister(config.name)
    assert registry.servers == {}

    try:
        reconnected = await _with_timeout(reconnect_server(config.id, registry=registry))
        assert reconnected.status.value == "connected"
        assert config.name in registry.servers
    finally:
        await registry.disconnect_all()


@needs_db
async def test_update_server_reconnects_with_new_settings() -> None:
    from agentic_core.mcp.registry import MCPServerRegistry
    from agentic_core.services.mcp_service import register_server, update_server

    registry = MCPServerRegistry()
    config = await _with_timeout(
        register_server(
            name=f"echo-{uuid.uuid4().hex[:8]}", transport="stdio", command=_ECHO_COMMAND, registry=registry
        )
    )
    try:
        new_name = f"{config.name}-renamed"
        updated = await _with_timeout(update_server(config.id, name=new_name, registry=registry))
        assert updated.name == new_name
        assert updated.status.value == "connected"
        assert new_name in registry.servers
    finally:
        await registry.disconnect_all()


@needs_db
async def test_call_server_tool_invokes_and_returns_content() -> None:
    from agentic_core.mcp.registry import MCPServerRegistry
    from agentic_core.services.mcp_service import call_server_tool, register_server

    registry = MCPServerRegistry()
    config = await _with_timeout(
        register_server(
            name=f"echo-{uuid.uuid4().hex[:8]}", transport="stdio", command=_ECHO_COMMAND, registry=registry
        )
    )
    try:
        is_error, content = await _with_timeout(call_server_tool(config.id, "echo", {"text": "hi"}, registry=registry))
        assert is_error is False
        assert content == "echo: hi"
    finally:
        await registry.disconnect_all()


@needs_db
async def test_call_server_tool_raises_when_not_connected() -> None:
    from agentic_core.mcp.registry import MCPServerRegistry
    from agentic_core.services.mcp_service import call_server_tool, register_server

    registry = MCPServerRegistry()
    config = await _with_timeout(
        register_server(
            name=f"echo-{uuid.uuid4().hex[:8]}", transport="stdio", command=_ECHO_COMMAND, registry=registry
        )
    )
    await registry.disconnect_all()  # server is now gone from the live registry
    with pytest.raises(RuntimeError, match="is not connected"):
        await call_server_tool(config.id, "echo", {}, registry=registry)


@needs_db
async def test_reset_server_session_parses_json_payload() -> None:
    from agentic_core.mcp.registry import MCPServerRegistry
    from agentic_core.services.mcp_service import register_server, reset_server_session

    registry = MCPServerRegistry()
    config = await _with_timeout(
        register_server(
            name=f"echo-{uuid.uuid4().hex[:8]}", transport="stdio", command=_ECHO_COMMAND, registry=registry
        )
    )
    try:
        payload = await _with_timeout(reset_server_session(config.id, registry=registry))
        assert payload == {"cleared": {"echo": 1}, "total": 1}
    finally:
        await registry.disconnect_all()


# --------------------------------------------------------------------------- #
#  hydrate_registry — the point of persisting anything at all                  #
# --------------------------------------------------------------------------- #


@needs_db
async def test_hydrate_registry_reconnects_from_the_table_alone() -> None:
    """A fresh, empty registry — a new process — rebuilds live connections from the DB."""
    from agentic_core.mcp.registry import MCPServerRegistry
    from agentic_core.services.mcp_service import hydrate_registry, register_server

    original_registry = MCPServerRegistry()
    config = await _with_timeout(
        register_server(
            name=f"echo-{uuid.uuid4().hex[:8]}", transport="stdio", command=_ECHO_COMMAND, registry=original_registry
        )
    )
    # Do NOT touch original_registry again — simulate a fresh process by only
    # ever using a brand-new, empty registry from here on.
    fresh_registry = MCPServerRegistry()
    assert fresh_registry.servers == {}

    try:
        failed = await _with_timeout(hydrate_registry(registry=fresh_registry))
        assert failed == []
        assert config.name in fresh_registry.servers
        assert fresh_registry.get(config.name).client.connected
    finally:
        # anyio's cancel-scope stack is per *task*, not per registry: both
        # clients' stdio transports were entered in this same test's task, so
        # they must be exited in exact reverse order regardless of which
        # registry owns which — fresh_registry's client connected second (via
        # hydrate_registry above), so it must be disconnected first.
        await fresh_registry.disconnect_all()
        await original_registry.disconnect_all()


@needs_db
async def test_hydrate_registry_is_a_no_op_for_already_live_servers() -> None:
    from agentic_core.mcp.registry import MCPServerRegistry
    from agentic_core.services.mcp_service import hydrate_registry, register_server

    registry = MCPServerRegistry()
    config = await _with_timeout(
        register_server(
            name=f"echo-{uuid.uuid4().hex[:8]}", transport="stdio", command=_ECHO_COMMAND, registry=registry
        )
    )
    entry_before = registry.get(config.name)
    try:
        failed = await _with_timeout(hydrate_registry(registry=registry))
        assert failed == []
        # Same entry object — untouched, not reconnected.
        assert registry.get(config.name) is entry_before
    finally:
        await registry.disconnect_all()


# --------------------------------------------------------------------------- #
#  The regression guard itself                                                 #
# --------------------------------------------------------------------------- #


@needs_db
async def test_live_connect_completes_within_a_bounded_timeout() -> None:
    """See the module docstring: mcp==2.0.0 hangs this forever. This must return."""
    from agentic_core.mcp.client import MCPClient, TransportType

    client = MCPClient(name="regression-guard", transport=TransportType.STDIO, command=_ECHO_COMMAND)
    await _with_timeout(client.connect())
    try:
        assert client.connected
        assert {t.name for t in client.tools} >= {"echo", "reset_session"}
    finally:
        await _with_timeout(client.disconnect())

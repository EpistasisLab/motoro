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

Database-backed tests are skipped unless ``MOTORO_TEST_DATABASE_URL`` is set.
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

from motoro import CoreSettings

DB_URL = os.environ.get("MOTORO_TEST_DATABASE_URL", "")
needs_db = pytest.mark.skipif(not DB_URL, reason="MOTORO_TEST_DATABASE_URL is not set")

_FIXTURE_SERVER = Path(__file__).parent / "fixtures" / "echo_mcp_server.py"
_ECHO_COMMAND = f"{sys.executable} {_FIXTURE_SERVER}"
# Kept in step with the fixture's own INSTRUCTIONS rather than duplicated, so
# editing the fixture's blurb can't silently stop the assertions meaning
# anything.
sys.path.insert(0, str(_FIXTURE_SERVER.parent))
from echo_mcp_server import INSTRUCTIONS as _ECHO_INSTRUCTIONS  # noqa: E402

# Every call that touches a live subprocess is bounded, so a regression like the
# one this file exists to catch fails the test instead of hanging the suite.
_LIVE_TIMEOUT = 15


class _Settings(CoreSettings):
    model_config = SettingsConfigDict(env_prefix="MOTORO_TEST_", extra="ignore")


@pytest.fixture(scope="module", autouse=True)
def _configure() -> None:
    from motoro.config import configure, reset_for_testing

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
    from motoro.models.database import dispose_engine
    from motoro.runner import init_schema
    from motoro.services.encryption import reset_for_testing as reset_encryption

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
    from motoro.models.mcp_server import MCPServerConfig

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
#  Security modules — self-contained, no ares/motoro coupling            #
# --------------------------------------------------------------------------- #


def test_command_allowlist_accepts_known_executables() -> None:
    from motoro.security.mcp_command_allowlist import validate_stdio_command

    validate_stdio_command("python server.py")
    validate_stdio_command("npx -y some-mcp-server")


def test_command_allowlist_rejects_unknown_executable() -> None:
    from motoro.security.mcp_command_allowlist import MCPCommandError, validate_stdio_command

    with pytest.raises(MCPCommandError, match="not in the allowed list"):
        validate_stdio_command("bash -c 'echo hi'")


def test_command_allowlist_rejects_shell_metacharacters() -> None:
    from motoro.security.mcp_command_allowlist import MCPCommandError, validate_stdio_command

    with pytest.raises(MCPCommandError, match="shell meta-character"):
        validate_stdio_command("python server.py; rm -rf /")


def test_ssrf_guard_rejects_private_ip() -> None:
    from motoro.security.ssrf_guard import SSRFError, validate_outbound_url

    with pytest.raises(SSRFError, match="private/reserved"):
        validate_outbound_url("http://169.254.169.254/", resolve_dns=False)


def test_ssrf_guard_allows_private_ip_when_opted_in() -> None:
    from motoro.security.ssrf_guard import validate_outbound_url

    validate_outbound_url("http://192.168.1.5/", resolve_dns=False, allow_private=True)


def test_ssrf_guard_rejects_file_scheme() -> None:
    from motoro.security.ssrf_guard import SSRFError, validate_outbound_url

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
    from motoro.config import configure, reset_for_testing

    reset_for_testing()
    configure(_Settings(encryption_key=_TEST_ENCRYPTION_KEY, **_db_kwargs()))


def test_encryption_round_trips() -> None:
    from motoro.config import configure, reset_for_testing
    from motoro.services.encryption import decrypt, encrypt
    from motoro.services.encryption import reset_for_testing as reset_encryption

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
    from motoro.config import configure, reset_for_testing
    from motoro.services.encryption import encrypt
    from motoro.services.encryption import reset_for_testing as reset_encryption

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
    from motoro.mcp.registry import MCPServerRegistry
    from motoro.services.mcp_service import register_server

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
        # The server's own `instructions`, captured from the initialize
        # handshake and stored alongside the tools -- a server-level
        # description has nowhere else to live, there being no column for one.
        assert config.capabilities["instructions"] == _ECHO_INSTRUCTIONS
        assert config.owner_id is None
    finally:
        await registry.disconnect_all()


def test_build_run_meta_includes_owner_id() -> None:
    from motoro.engine.context import RunContext
    from motoro.mcp.adapters import META_KEY_OWNER_ID, META_KEY_RUN_ID, META_KEY_WORKSPACE_ID, _build_run_meta
    from motoro.schemas.agent import ModelConfig

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


def test_build_run_meta_includes_agent_name_and_model_only_together() -> None:
    from motoro.engine.context import RunContext
    from motoro.mcp.adapters import META_KEY_AGENT_NAME, META_KEY_MODEL, _build_run_meta
    from motoro.schemas.agent import ModelConfig

    with_name = RunContext(
        agent_goal="g",
        system_prompt="s",
        model_config=ModelConfig(model="claude-opus-5"),
        user_input="u",
        agent_name="SF-DC",
    )
    assert _build_run_meta(with_name) == {META_KEY_AGENT_NAME: "SF-DC", META_KEY_MODEL: "claude-opus-5"}

    # model always has a non-empty default; without agent_name it must NOT
    # appear alone, or every single run would carry a meta dict regardless of
    # whether anything else is set.
    without_name = RunContext(agent_goal="g", system_prompt="s", model_config=ModelConfig(), user_input="u")
    assert _build_run_meta(without_name) is None


def test_build_run_meta_namespaces_caller_ambient_values() -> None:
    from motoro.engine.context import RunContext
    from motoro.mcp.adapters import META_AMBIENT_PREFIX, META_KEY_WORKSPACE_ID, _build_run_meta
    from motoro.schemas.agent import ModelConfig

    context = RunContext(
        agent_goal="g",
        system_prompt="s",
        model_config=ModelConfig(),
        user_input="u",
        workspace_id="exp1/cellA",
        ambient_meta={"dataset_names": ["spinal"], "skip_me": None, "": "unnamed"},
    )
    assert _build_run_meta(context) == {
        f"{META_AMBIENT_PREFIX}dataset_names": ["spinal"],
        META_KEY_WORKSPACE_ID: "exp1/cellA",
    }

    # A product key can never shadow one of core's: the prefix puts it in a
    # different namespace even when the bare name collides exactly.
    shadowing = RunContext(
        agent_goal="g",
        system_prompt="s",
        model_config=ModelConfig(),
        user_input="u",
        workspace_id="real",
        ambient_meta={"workspace_id": "spoofed"},
    )
    meta = _build_run_meta(shadowing)
    assert meta is not None
    assert meta[META_KEY_WORKSPACE_ID] == "real"
    assert meta[f"{META_AMBIENT_PREFIX}workspace_id"] == "spoofed"

    # Ambient values alone still count as identity worth sending.
    ambient_only = RunContext(
        agent_goal="g",
        system_prompt="s",
        model_config=ModelConfig(),
        user_input="u",
        ambient_meta={"dataset_names": ["a"]},
    )
    assert _build_run_meta(ambient_only) == {f"{META_AMBIENT_PREFIX}dataset_names": ["a"]}


def test_run_context_snapshot_round_trips_ambient_meta() -> None:
    """A paused run must resume against the same ambient references it started
    with — an id dropped by the snapshot would silently un-bind every tool call
    after the resume."""
    from motoro.engine.context import RunContext
    from motoro.schemas.agent import ModelConfig

    context = RunContext(
        agent_goal="g",
        system_prompt="s",
        model_config=ModelConfig(),
        user_input="u",
        ambient_meta={"dataset_names": ["spinal", "demographics"]},
    )
    restored = RunContext.from_snapshot(context.to_snapshot())
    assert restored.ambient_meta == {"dataset_names": ["spinal", "demographics"]}

    # Older snapshots predate the field entirely.
    legacy = dict(context.to_snapshot())
    legacy.pop("ambient_meta")
    assert RunContext.from_snapshot(legacy).ambient_meta == {}


@needs_db
async def test_call_tool_delivers_meta_verbatim() -> None:
    """The exact wire-level guarantee _build_run_meta/execute_step rely on:
    whatever dict is passed as ``meta=`` to ``MCPClient.call_tool`` is what a
    real tool reads back from ``ctx.request_context.meta`` — not just what the
    sender intended to build. A downstream consumer's own copy of a meta key
    (e.g. asaree_workspace_core's META_KEY_WORKSPACE_ID) can silently drift
    from this without either side raising an error, which is exactly what
    happened before this test existed."""
    from motoro.mcp.registry import MCPServerRegistry
    from motoro.services.mcp_service import register_server

    registry = MCPServerRegistry()
    name = f"echo-{uuid.uuid4().hex[:8]}"
    await _with_timeout(register_server(name=name, transport="stdio", command=_ECHO_COMMAND, registry=registry))
    try:
        client = registry.servers[name].client
        sent_meta = {"motoro.workspace_id": "exp1/cellA", "motoro.owner_id": str(uuid.uuid4())}
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
    from motoro.mcp.registry import MCPServerRegistry
    from motoro.security.mcp_command_allowlist import MCPCommandError
    from motoro.services.mcp_service import register_server

    registry = MCPServerRegistry()
    with pytest.raises(MCPCommandError):
        await register_server(name="bad", transport="stdio", command="bash -c evil", registry=registry)
    # Rejected before anything was spawned or persisted.
    assert registry.servers == {}


@needs_db
async def test_register_server_rejects_private_url_by_default() -> None:
    from motoro.mcp.registry import MCPServerRegistry
    from motoro.security.ssrf_guard import SSRFError
    from motoro.services.mcp_service import register_server

    registry = MCPServerRegistry()
    with pytest.raises(SSRFError):
        await register_server(name="internal", transport="http", url="http://169.254.169.254/mcp", registry=registry)


@needs_db
async def test_get_server_and_get_server_by_name() -> None:
    from motoro.mcp.registry import MCPServerRegistry
    from motoro.services.mcp_service import get_server, get_server_by_name, register_server

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
    from motoro.mcp.registry import MCPServerRegistry
    from motoro.services.mcp_service import list_servers, register_server

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
    from motoro.mcp.registry import MCPServerRegistry
    from motoro.services.mcp_service import list_servers, register_server

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
    from motoro.mcp.registry import MCPServerRegistry
    from motoro.services.mcp_service import delete_server, get_server, register_server

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
    from motoro.mcp.registry import MCPServerRegistry
    from motoro.services.mcp_service import refresh_server, register_server

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
        # list_tools never repeats `instructions`, so a refresh has to carry
        # forward what connect() captured rather than dropping the key.
        assert refreshed.capabilities["instructions"] == _ECHO_INSTRUCTIONS
    finally:
        await registry.disconnect_all()


@needs_db
async def test_reconnect_server_after_manual_unregister() -> None:
    """Simulates a fresh process: the registry has forgotten the server, but the DB hasn't."""
    from motoro.mcp.registry import MCPServerRegistry
    from motoro.services.mcp_service import reconnect_server, register_server

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
    from motoro.mcp.registry import MCPServerRegistry
    from motoro.services.mcp_service import register_server, update_server

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
    from motoro.mcp.registry import MCPServerRegistry
    from motoro.services.mcp_service import call_server_tool, register_server

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
    from motoro.mcp.registry import MCPServerRegistry
    from motoro.services.mcp_service import call_server_tool, register_server

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
    from motoro.mcp.registry import MCPServerRegistry
    from motoro.services.mcp_service import register_server, reset_server_session

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
    from motoro.mcp.registry import MCPServerRegistry
    from motoro.services.mcp_service import hydrate_registry, register_server

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
    from motoro.mcp.registry import MCPServerRegistry
    from motoro.services.mcp_service import hydrate_registry, register_server

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
    from motoro.mcp.client import MCPClient, TransportType

    client = MCPClient(name="regression-guard", transport=TransportType.STDIO, command=_ECHO_COMMAND)
    await _with_timeout(client.connect())
    try:
        assert client.connected
        assert {t.name for t in client.tools} >= {"echo", "reset_session"}
        assert client.instructions == _ECHO_INSTRUCTIONS
    finally:
        await _with_timeout(client.disconnect())
    # Cleared on disconnect alongside the tool list: neither survives the
    # session that produced it.
    assert client.instructions == ""


# --------------------------------------------------------------------------- #
#  Task affinity — connect and disconnect from different tasks                 #
# --------------------------------------------------------------------------- #
#
# MCPClient used to stash the half-entered stdio_client/ClientSession context
# managers on self for a later caller to __aexit__. Both wrap an anyio task
# group, and anyio requires a cancel scope to be exited by the task that
# entered it -- so the class only worked if connect() and disconnect() ran in
# the same task. When they didn't, anyio raised "Attempted to exit cancel scope
# in a different task than it was entered in" *and* cancelled the scope's
# owning task, which in production was a live agent run. The connection now
# lives in its own task (MCPClient._own_connection); these pin that down.


@needs_db
async def test_disconnect_from_a_different_task_than_connect() -> None:
    """The narrowest statement of the bug: connect here, disconnect over there.

    Asserting ``not client.connected`` afterwards is *not* enough to catch the
    regression -- a teardown failure is swallowed into a warning and the flag is
    cleared in a ``finally`` either way, so the old task-affine implementation
    passes that assertion while logging ``mcp.server.disconnect_failed``. That
    log line is the observable difference, so it is what gets asserted on.
    """
    import structlog

    from motoro.mcp.client import MCPClient, TransportType

    client = MCPClient(name="cross-task", transport=TransportType.STDIO, command=_ECHO_COMMAND)

    # Connect inside a task of its own, so the task that entered the transport
    # is definitively gone by the time we disconnect it from the main task.
    await _with_timeout(asyncio.create_task(client.connect()))
    assert client.connected

    with structlog.testing.capture_logs() as logs:
        await _with_timeout(asyncio.create_task(client.disconnect()))

    assert not client.connected
    failures = [e for e in logs if e.get("event") == "mcp.server.disconnect_failed"]
    assert not failures, f"teardown crossed a task boundary: {failures}"


@needs_db
async def test_reregistering_a_name_does_not_cancel_the_task_using_it() -> None:
    """The production failure, reduced: re-registering a live server used to
    cancel whichever task had connected it -- collateral damage from anyio
    delivering the cancel-scope cancellation to the scope's owning task."""
    from motoro.mcp.registry import MCPServerRegistry, TransportType

    registry = MCPServerRegistry()
    name = f"echo-{uuid.uuid4().hex[:8]}"
    victim_survived = asyncio.Event()

    async def victim() -> None:
        # Connects the server, then stays alive -- exactly what a protocol run
        # does between hydrating the registry and finishing its graph walk.
        await registry.register(name=name, transport=TransportType.STDIO, command=_ECHO_COMMAND)
        await asyncio.sleep(0.5)
        victim_survived.set()

    try:
        task = asyncio.create_task(victim())
        # Let it get all the way connected before we pull the rug.
        while registry.get(name) is None or not registry.get(name).client.connected:
            await asyncio.sleep(0.05)

        # Re-register the same name from *this* task. This is what a second
        # concurrent hydrate_registry did.
        await _with_timeout(registry.register(name=name, transport=TransportType.STDIO, command=_ECHO_COMMAND))

        await _with_timeout(task)
        assert not task.cancelled()
        assert victim_survived.is_set(), "re-registering cancelled the task that had connected the server"
        assert registry.get(name).client.connected
    finally:
        await _with_timeout(registry.disconnect_all())


@needs_db
async def test_concurrent_ensure_registered_connects_exactly_once() -> None:
    """ensure_registered's check happens under the registry lock, so N racing
    callers produce one connection rather than N spawn/teardown cycles."""
    from motoro.mcp.registry import MCPServerRegistry, TransportType

    registry = MCPServerRegistry()
    name = f"echo-{uuid.uuid4().hex[:8]}"

    try:
        entries = await _with_timeout(
            asyncio.gather(
                *(
                    registry.ensure_registered(name=name, transport=TransportType.STDIO, command=_ECHO_COMMAND)
                    for _ in range(6)
                )
            )
        )
        # Same ServerEntry object every time means only the first caller
        # actually registered; the rest short-circuited on a live connection.
        assert all(e is entries[0] for e in entries)
        assert entries[0].client.connected
        assert {t.name for t in entries[0].client.tools} >= {"echo"}
    finally:
        await _with_timeout(registry.disconnect_all())


@needs_db
async def test_ensure_registered_retries_a_dead_entry() -> None:
    """A slot that exists but failed to connect is retried, not left dead."""
    from motoro.mcp.registry import MCPServerRegistry, TransportType

    registry = MCPServerRegistry()
    name = f"echo-{uuid.uuid4().hex[:8]}"

    try:
        # A command that cannot spawn leaves an entry with connected=False.
        dead = await _with_timeout(
            registry.ensure_registered(
                name=name, transport=TransportType.STDIO, command="definitely-not-a-real-binary-xyz"
            )
        )
        assert not dead.client.connected

        live = await _with_timeout(
            registry.ensure_registered(name=name, transport=TransportType.STDIO, command=_ECHO_COMMAND)
        )
        assert live.client.connected
    finally:
        await _with_timeout(registry.disconnect_all())


@needs_db
async def test_disconnect_all_is_concurrent_and_clean() -> None:
    """disconnect_all had to be sequential+reverse-ordered to satisfy anyio's
    same-task/LIFO rule. Owner tasks removed both constraints -- so every
    transport must now close cleanly even when they all close at once, which
    is what the absence of a failure log asserts (see the cross-task test for
    why the flags alone would not catch it)."""
    import structlog

    from motoro.mcp.registry import MCPServerRegistry, TransportType

    registry = MCPServerRegistry()
    names = [f"echo-{uuid.uuid4().hex[:8]}" for _ in range(4)]
    for name in names:
        await _with_timeout(registry.register(name=name, transport=TransportType.STDIO, command=_ECHO_COMMAND))
    assert all(registry.get(n).client.connected for n in names)

    with structlog.testing.capture_logs() as logs:
        await _with_timeout(registry.disconnect_all())

    assert registry.servers == {}
    failures = [e for e in logs if e.get("event") in {"mcp.server.disconnect_failed", "mcp.server.disconnect_error"}]
    assert not failures, f"concurrent disconnect_all did not close cleanly: {failures}"

"""MCP client wrapper — connects to MCP servers and invokes tools."""

from __future__ import annotations

import asyncio
import contextlib
import enum
import os
import shlex
import shutil
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog
from mcp import ClientSession
from mcp import types as mcp_types
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from opentelemetry import trace

from motoro.config import settings
from motoro.observability.tracing import get_tracer
from motoro.services.retry import retry_with_backoff

log = structlog.get_logger()
_tracer = get_tracer("mcp")

# How long to wait for a connection's owner task to finish unwinding before
# cancelling it outright. Bounds worker/API shutdown against a wedged MCP
# subprocess that will not close its pipes.
_DISCONNECT_TIMEOUT_SECONDS = 10.0

# Default allowlist of safe environment variables for MCP subprocesses.
# These are standard POSIX variables that do not contain secrets.
MCP_DEFAULT_ALLOWED_ENV_VARS: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "SHELL",
        "TERM",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "ARES_API_URL",
        # Shared service secret passed to ARES's own MCP subprocesses so they can
        # authenticate their outbound API calls. NOT a user credential.
        "ARES_INTERNAL_MCP_API_KEY",
        # Bundle root for core's own OKF server (motoro.mcp_servers.okf) —
        # a directory path, not a secret, so it belongs in the default set
        # rather than requiring every deployment to configure the allowlist
        # just to spawn a first-party bundled server.
        "AGENTIC_OKF_BUNDLE_DIR",
    }
)


def _get_allowed_env_vars() -> frozenset[str]:
    """Return the full set of allowed env var names (default + user-configured)."""
    allowed = set(MCP_DEFAULT_ALLOWED_ENV_VARS)
    extra = settings.mcp_allowed_env_vars.strip()
    if extra:
        for name in extra.split(","):
            name = name.strip()
            if name:
                allowed.add(name)
    return frozenset(allowed)


def build_subprocess_env(
    *,
    allowed_env_vars: frozenset[str] | None = None,
    server_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a filtered environment dict for an MCP subprocess.

    Only variables in the allowlist (default + user-configured via
    ``ARES_MCP_ALLOWED_ENV_VARS``) are copied from the host environment.

    MCP server configs may declare specific env vars they need via
    ``server_env``. These are validated against the allowlist; any
    variable not in the allowlist is rejected and logged.

    Environment isolation prevents secrets (API keys, JWT secrets, DB
    credentials) from leaking to untrusted MCP server subprocesses.
    """
    if allowed_env_vars is None:
        allowed_env_vars = _get_allowed_env_vars()

    env: dict[str, str] = {}

    # Copy allowed vars from host environment
    for key, value in os.environ.items():
        if key in allowed_env_vars:
            env[key] = value

    # Merge server-declared env vars (validated against allowlist)
    if server_env:
        for key, value in server_env.items():
            if key in allowed_env_vars:
                env[key] = value
            else:
                log.warning(
                    "mcp.env.rejected",
                    var=key,
                    reason="not in allowlist",
                    component="mcp",
                )

    return env


class TransportType(enum.StrEnum):
    """MCP client transport type."""

    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"


@dataclass
class ToolInfo:
    """Discovered tool from an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result from invoking an MCP tool."""

    content: str
    is_error: bool = False


class MCPClient:
    """Wrapper around the MCP Python SDK for connecting to a single server.

    Supports both stdio (local subprocess) and HTTP (remote server) transports.
    """

    def __init__(
        self,
        name: str,
        transport: TransportType = TransportType.STDIO,
        command: str | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        server_env: dict[str, str] | None = None,
        on_tools_changed: Callable[[MCPClient], Awaitable[None]] | None = None,
    ) -> None:
        self.name = name
        self._transport = transport
        self._command = command
        self._url = url
        self._headers = headers or {}
        self._server_env = server_env
        self._tools: list[ToolInfo] = []
        # The server's own ``instructions`` from the initialize handshake --
        # its description of itself, as distinct from any per-tool
        # description. Captured once at connect because that is the only
        # place the protocol offers it: ``list_tools`` never repeats it, so
        # ``refresh_tools`` leaves this alone.
        self._instructions: str = ""
        self._session: ClientSession | None = None
        self._read: Any = None
        self._write: Any = None
        self._cm: Any = None
        self._session_cm: Any = None
        self._http_client: httpx.AsyncClient | None = None
        self._connected = False
        # Issue #732 — TTL cache for is_alive to avoid full list_tools per health poll.
        self._is_alive_ttl_seconds = 30.0
        self._is_alive_last_checked: float = 0.0
        self._is_alive_last_result: bool = False
        # Issue #723: callback fired whenever the in-memory tool cache changes
        # (initial connect, refresh_tools, transparent reconnect, or a
        # ``notifications/tools/list_changed`` push from the server). Service
        # code can plug a DB persistence step in here so the in-memory list
        # never drifts from ``MCPServerConfig.capabilities``.
        self._on_tools_changed = on_tools_changed
        # Issue #719: serialize reconnect attempts so two concurrent
        # ``call_tool`` calls that both observe a broken pipe don't kick off
        # parallel reconnects.
        self._reconnect_lock = asyncio.Lock()
        # The connection-owner task and its shutdown signal. See
        # ``_own_connection`` for why the transport is opened and closed
        # inside a task of its own rather than by whoever calls connect().
        self._owner_task: asyncio.Task[None] | None = None
        self._shutdown: asyncio.Event | None = None
        self._log = log.bind(server=name, transport=transport, component="mcp")

    def set_on_tools_changed(self, callback: Callable[[MCPClient], Awaitable[None]] | None) -> None:
        """Install or replace the tool-cache-change callback (Issue #723)."""
        self._on_tools_changed = callback

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def tools(self) -> list[ToolInfo]:
        return list(self._tools)

    @property
    def instructions(self) -> str:
        """What the server says it is, from the initialize handshake.

        Empty when the server sends none -- it is an optional field, and most
        servers omit it. Callers should treat "" as "this server has no
        description" rather than as an error.
        """
        return self._instructions

    async def connect(self) -> None:
        """Connect to the MCP server, initialize, and discover tools.

        Retries transient connection failures with exponential backoff.

        The actual transport is opened by a dedicated owner task
        (:meth:`_own_connection`), not by the caller -- see that method for
        why. This call blocks until the connection is live and tools are
        discovered, or raises whatever the connect failed with, so it is a
        drop-in for the previous inline implementation.
        """
        # A stale owner task from a previous connect would otherwise leak its
        # subprocess. No-op when there is none, which is the common path.
        await self._stop_owner_task()

        ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        # Handed to the owner task by value, and *also* kept on self for
        # _stop_owner_task to signal. The task must not read it back off self:
        # _stop_owner_task clears the attribute as soon as it has signalled,
        # which can happen while the task is still inside _open().
        shutdown = asyncio.Event()
        self._shutdown = shutdown
        self._owner_task = asyncio.create_task(
            self._own_connection(ready, shutdown),
            name=f"mcp-connection:{self.name}",
        )
        try:
            await ready
        except BaseException:
            # Either the connect failed inside the owner task (which is
            # already unwinding) or *we* were cancelled while waiting. Both
            # need the task joined before we propagate, so a cancelled connect
            # can't leave an orphaned subprocess behind.
            await self._stop_owner_task()
            raise

    async def _own_connection(self, ready: asyncio.Future[None], shutdown: asyncio.Event) -> None:
        """Own the transport for its entire lifetime, in one task.

        ``stdio_client``/``streamable_http_client``/``sse_client`` and
        ``ClientSession`` are all async context managers wrapping an anyio
        task group, and anyio requires a task group's cancel scope to be
        exited **by the task that entered it**. The client used to store the
        half-entered context managers on ``self`` for some later, arbitrary
        caller to exit -- which made the whole class task-affine: connect and
        disconnect had to happen in the same task or anyio raised
        ``RuntimeError: Attempted to exit cancel scope in a different task
        than it was entered in``, *and* cancelled the scope's owning task as
        collateral damage (a live agent run, in practice).

        Three call paths violated that: ``MCPServerRegistry.register``
        re-registering an existing name, ``disconnect_all``, and
        ``_attempt_reconnect``. Rather than make each one careful, the
        connection now lives in a task of its own: this one opens it, parks
        on ``_shutdown``, and closes it on the way out. Enter and exit are
        therefore always the same task no matter who calls connect() or
        disconnect(), and the class is task-agnostic.
        """
        try:
            await self._open()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            # Half-opened state is cleaned up by _open's own per-transport
            # handlers (Issues #711/#715); nothing further to close here.
            self._connected = False
            self._session = None
            return

        if not ready.done():
            ready.set_result(None)

        try:
            await shutdown.wait()
        finally:
            await self._close()

    async def _open(self) -> None:
        """Open the transport, initialize the session, and discover tools.

        Always runs inside the owner task -- see :meth:`_own_connection`.
        """

        async def _do_connect() -> None:
            if self._transport == TransportType.STDIO:
                await self._connect_stdio()
            elif self._transport == TransportType.HTTP:
                await self._connect_http()
            elif self._transport == TransportType.SSE:
                await self._connect_sse()
            else:
                raise ValueError(f"Unsupported transport: {self._transport}")

        await retry_with_backoff(
            _do_connect,
            max_retries=3,
            base_delay=1.0,
            retryable_exceptions=(ConnectionError, OSError, TimeoutError),
        )

        # Initialize session with a message_handler so the SDK delivers
        # ``notifications/tools/list_changed`` to us (Issue #721).
        self._session_cm = ClientSession(self._read, self._write, message_handler=self._handle_session_message)
        self._session = await self._session_cm.__aenter__()
        init = await self._session.initialize()
        self._instructions = (getattr(init, "instructions", None) or "").strip()

        result = await self._session.list_tools()
        self._tools = [
            ToolInfo(
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema if isinstance(t.inputSchema, dict) else {},
            )
            for t in result.tools
        ]
        self._connected = True
        self._log.info("mcp.server.connected", tools=len(self._tools))
        await self._fire_tools_changed()

    async def _handle_session_message(self, message: Any) -> None:
        """Receive notifications from the MCP server (Issue #721).

        When the server pushes ``notifications/tools/list_changed`` we
        re-discover tools so the cache stays consistent with the server's
        actual capabilities.
        """
        try:
            notif = getattr(message, "root", message)
            if isinstance(notif, mcp_types.ToolListChangedNotification):
                self._log.info("mcp.server.tools_list_changed")
                try:
                    await self.refresh_tools()
                except Exception:
                    self._log.warning("mcp.server.refresh_after_list_changed_failed", exc_info=True)
        except Exception:
            self._log.warning("mcp.session.message_handler_error", exc_info=True)

    async def _fire_tools_changed(self) -> None:
        """Invoke the ``on_tools_changed`` callback if installed."""
        if self._on_tools_changed is None:
            return
        try:
            await self._on_tools_changed(self)
        except Exception:
            self._log.warning("mcp.on_tools_changed.callback_failed", exc_info=True)

    async def _connect_stdio(self) -> None:
        """Establish stdio transport connection.

        Environment isolation: only variables in the allowlist are passed
        to the subprocess. See ``build_subprocess_env`` for details.

        If ``__aenter__`` fails or times out (Issue #711), the half-entered
        context manager is force-exited so the spawned subprocess is reaped
        rather than leaked as an orphan PID.
        """
        if not self._command:
            raise ValueError("stdio transport requires a command")
        parts = shlex.split(self._command)
        # Issue #734 — surface a clear error if the binary isn't on PATH (or absolute
        # path doesn't exist / isn't executable), instead of an opaque BrokenPipeError
        # from the subprocess pipe collapsing.
        resolved = self._resolve_stdio_executable(parts[0])
        if resolved is None:
            raise FileNotFoundError(
                f"MCP stdio executable '{parts[0]}' not found for server '{self.name}'. "
                "Ensure the binary is installed and on PATH."
            )
        env = build_subprocess_env(server_env=self._server_env)
        params = StdioServerParameters(command=parts[0], args=parts[1:], env=env)
        self._cm = stdio_client(params)
        try:
            self._read, self._write = await asyncio.wait_for(
                self._cm.__aenter__(), timeout=settings.tool_timeout_seconds
            )
        except TimeoutError:
            self._log.error(
                "mcp.stdio.connect_timeout",
                command=self._command,
                timeout_seconds=settings.tool_timeout_seconds,
            )
            await self._cleanup_failed_stdio_cm()
            raise TimeoutError(
                f"MCP stdio connection to '{self.name}' timed out after {settings.tool_timeout_seconds}s"
            ) from None
        except BaseException:
            self._log.error(
                "mcp.stdio.connect_failed",
                command=self._command,
                exc_info=True,
            )
            await self._cleanup_failed_stdio_cm()
            raise

    async def _cleanup_failed_stdio_cm(self) -> None:
        """Reap a half-entered stdio context manager (Issue #711).

        Best-effort: swallows exceptions but never re-raises — the caller is
        already propagating the original connect failure.
        """
        cm = self._cm
        self._cm = None
        if cm is None:
            return
        try:
            await cm.__aexit__(None, None, None)
        except BaseException:
            self._log.warning("mcp.stdio.cleanup_failed", exc_info=True)

    async def _connect_http(self) -> None:
        """Establish HTTP transport connection.

        If the underlying streamable HTTP context manager fails to enter
        (Issue #715), the just-created ``httpx.AsyncClient`` is closed so
        it is not leaked along with the failed connect.
        """
        if not self._url:
            raise ValueError("HTTP transport requires a url")
        self._http_client = httpx.AsyncClient(headers=self._headers, timeout=httpx.Timeout(30.0, connect=10.0))
        self._cm = streamable_http_client(self._url, http_client=self._http_client)
        try:
            streams = await asyncio.wait_for(self._cm.__aenter__(), timeout=30)
        except TimeoutError:
            self._log.error(
                "mcp.http.connect_timeout",
                url=self._url,
                timeout_seconds=30,
            )
            await self._cleanup_failed_http_cm()
            raise TimeoutError(f"MCP HTTP connection to '{self.name}' ({self._url}) timed out after 30s") from None
        except BaseException:
            self._log.error(
                "mcp.http.connect_failed",
                url=self._url,
                exc_info=True,
            )
            await self._cleanup_failed_http_cm()
            raise
        self._read, self._write = streams[0], streams[1]

    async def _cleanup_failed_http_cm(self) -> None:
        """Tear down a half-entered HTTP transport (Issue #715).

        Closes the httpx client and force-exits the context manager. Always
        best-effort; never raises.
        """
        cm = self._cm
        self._cm = None
        if cm is not None:
            try:
                await cm.__aexit__(None, None, None)
            except BaseException:
                self._log.warning("mcp.http.cm_cleanup_failed", exc_info=True)

        client = self._http_client
        self._http_client = None
        if client is not None:
            try:
                await client.aclose()
            except BaseException:
                self._log.warning("mcp.http.client_cleanup_failed", exc_info=True)

    async def _connect_sse(self) -> None:
        """Establish SSE transport connection.

        Uses the older MCP SSE transport (GET /sse for events, POST /messages/
        for requests). Compatible with most Node.js MCP servers and any server
        using mcp SDK < 1.1 or explicitly configured for SSE transport.
        """
        if not self._url:
            raise ValueError("SSE transport requires a url")
        self._cm = sse_client(
            self._url,
            headers=self._headers or None,
            timeout=10.0,
            sse_read_timeout=300.0,
        )
        try:
            streams = await asyncio.wait_for(self._cm.__aenter__(), timeout=30)
        except TimeoutError:
            self._log.error(
                "mcp.sse.connect_timeout",
                url=self._url,
                timeout_seconds=30,
            )
            await self._cleanup_failed_http_cm()
            raise TimeoutError(f"MCP SSE connection to '{self.name}' ({self._url}) timed out after 30s") from None
        except BaseException:
            self._log.error(
                "mcp.sse.connect_failed",
                url=self._url,
                exc_info=True,
            )
            await self._cleanup_failed_http_cm()
            raise
        self._read, self._write = streams[0], streams[1]

    async def refresh_tools(self) -> list[ToolInfo]:
        """Re-discover tools from the connected server. Updates internal tool list.

        Fires the ``on_tools_changed`` callback so persistence layers can
        write the fresh schema to the database (Issue #723).
        """
        if not self._session or not self._connected:
            raise RuntimeError(f"MCPClient '{self.name}' is not connected")

        result = await self._session.list_tools()
        self._tools = [
            ToolInfo(
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema if isinstance(t.inputSchema, dict) else {},
            )
            for t in result.tools
        ]
        self._log.info("mcp.server.tools_refreshed", tools=len(self._tools))
        await self._fire_tools_changed()
        return list(self._tools)

    @staticmethod
    def _resolve_stdio_executable(executable: str) -> str | None:
        """Resolve a stdio executable to an actual on-disk file. Issue #734.

        - Absolute / explicit paths must exist *and* be executable.
        - Bare names are looked up via ``shutil.which`` against ``PATH``.
        Returns the resolved path, or ``None`` if the binary cannot be found.
        """
        if os.path.sep in executable or executable.startswith("."):
            if os.path.isfile(executable) and os.access(executable, os.X_OK):
                return executable
            return None
        return shutil.which(executable)

    async def is_alive(self) -> bool:
        """Check if the server connection is still responsive.

        Issue #732 — the result is cached for ``_is_alive_ttl_seconds`` to keep
        polling cheap; the actual ``list_tools`` round-trip happens at most once
        per TTL window.
        """
        if not self._session or not self._connected:
            return False

        now = time.monotonic()
        if self._is_alive_last_checked and now - self._is_alive_last_checked < self._is_alive_ttl_seconds:
            return self._is_alive_last_result

        try:
            await self._session.list_tools()
            self._is_alive_last_result = True
        except Exception:
            self._connected = False
            self._is_alive_last_result = False
        self._is_alive_last_checked = now
        return self._is_alive_last_result

    @staticmethod
    def _is_broken_transport_error(exc: BaseException) -> bool:
        """Heuristic: does ``exc`` indicate a dead transport that may recover after reconnect?

        Covers explicit stream-closed errors from anyio, broken pipes, and
        generic connection-reset/connection-aborted from the OS. Excludes
        validation errors and timeouts (which signal a live but slow server).
        """
        if isinstance(exc, BrokenPipeError | ConnectionResetError | ConnectionAbortedError):
            return True
        name = type(exc).__name__
        # anyio.ClosedResourceError / EndOfStream — match by class name to
        # avoid an import dependency on anyio internals.
        return name in {"ClosedResourceError", "EndOfStream", "BrokenResourceError"}

    async def _attempt_reconnect(self) -> bool:
        """Tear down the dead session and reconnect once (Issue #719).

        Returns ``True`` if reconnect succeeded. Serialized via
        ``self._reconnect_lock`` so concurrent failed ``call_tool`` calls
        share a single reconnect attempt.
        """
        async with self._reconnect_lock:
            if self._connected:
                # Another caller raced us and already reconnected.
                return True

            # Tear the dead connection down through its owner task rather than
            # __aexit__-ing the context managers from whatever task happened to
            # hit the broken pipe -- see _own_connection. connect() would do
            # this for us anyway; it is spelled out here so a failure to close
            # is logged as a reconnect step rather than as a connect step.
            await self._stop_owner_task()

            try:
                await self.connect()
            except Exception:
                self._log.warning("mcp.reconnect.failed", exc_info=True)
                return False
            self._log.info("mcp.reconnect.succeeded")
            return True

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Invoke a tool on the connected MCP server.

        ``meta`` is sent out-of-band as the MCP request ``_meta`` (issue #1455):
        it carries ambient run identity (e.g. workspace_id) that must never live
        in the model-controlled ``arguments`` namespace. Passed through to the
        SDK's ``ClientSession.call_tool(..., meta=...)`` on both the initial call
        and the post-reconnect retry.

        On broken-transport errors the client transparently reconnects once
        and retries the call (Issue #719). Validation, timeout, and tool-level
        errors are not retried.
        """
        if not self._session or not self._connected:
            raise RuntimeError(f"MCPClient '{self.name}' is not connected")

        start_ms = int(time.monotonic() * 1000)

        with _tracer.start_as_current_span(
            "mcp.call_tool",
            attributes={"tool": tool_name, "server": self.name},
        ) as span:
            try:
                result = await self._session.call_tool(tool_name, arguments, meta=meta)
            except Exception as exc:
                if self._is_broken_transport_error(exc):
                    self._log.warning(
                        "mcp.tool.transport_broken_reconnecting",
                        tool=tool_name,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    self._connected = False
                    reconnected = await self._attempt_reconnect()
                    if reconnected and self._session is not None:
                        try:
                            result = await self._session.call_tool(tool_name, arguments, meta=meta)
                        except Exception as exc2:
                            latency_ms = int(time.monotonic() * 1000) - start_ms
                            span.record_exception(exc2)
                            span.set_status(trace.StatusCode.ERROR, description=str(exc2))
                            self._log.warning(
                                "mcp.tool.failed_after_reconnect",
                                tool=tool_name,
                                latency_ms=latency_ms,
                                error=str(exc2),
                            )
                            raise
                    else:
                        latency_ms = int(time.monotonic() * 1000) - start_ms
                        span.record_exception(exc)
                        span.set_status(trace.StatusCode.ERROR, description=str(exc))
                        raise
                else:
                    latency_ms = int(time.monotonic() * 1000) - start_ms
                    span.record_exception(exc)
                    span.set_status(trace.StatusCode.ERROR, description=str(exc))
                    self._log.warning(
                        "mcp.tool.failed",
                        tool=tool_name,
                        latency_ms=latency_ms,
                        error=str(exc),
                    )
                    raise

            latency_ms = int(time.monotonic() * 1000) - start_ms
            text_parts = [c.text for c in result.content if hasattr(c, "text") and isinstance(c.text, str)]
            content = "\n".join(text_parts) if text_parts else str(result.content)
            is_error = bool(result.isError)

            # OTel semantic convention: success/failure is indicated by span status,
            # not a boolean attribute (#983).  latency is captured by the span itself.
            if is_error:
                span.set_status(
                    trace.StatusCode.ERROR,
                    description="MCP tool returned isError=True",
                )

            if is_error:
                self._log.warning(
                    "mcp.tool.error_result",
                    tool=tool_name,
                    latency_ms=latency_ms,
                    content=content[:200],
                )
            else:
                self._log.debug(
                    "mcp.tool.executed",
                    tool=tool_name,
                    latency_ms=latency_ms,
                    success=True,
                )

            return ToolResult(content=content, is_error=is_error)

    async def disconnect(self) -> None:
        """Disconnect from the MCP server.

        Safe to call from any task, and safe to call when never connected or
        already disconnected -- the teardown itself happens in the owner task
        (:meth:`_own_connection`); this just signals it and waits.
        """
        await self._stop_owner_task()

    async def _stop_owner_task(self) -> None:
        """Signal the owner task to close the connection, and join it.

        Bounded by ``_DISCONNECT_TIMEOUT_SECONDS`` so one wedged subprocess
        cannot hang process shutdown; on overrun the owner task is cancelled,
        which still unwinds the transport in the task that opened it.
        """
        task = self._owner_task
        if task is None:
            return
        self._owner_task = None

        if self._shutdown is not None:
            self._shutdown.set()
        self._shutdown = None

        if task.done():
            # Surface a connection that died on its own rather than swallowing it.
            exc = None if task.cancelled() else task.exception()
            if exc is not None:
                self._log.warning(
                    "mcp.connection.owner_task_failed",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            return

        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=_DISCONNECT_TIMEOUT_SECONDS)
        except TimeoutError:
            self._log.warning(
                "mcp.connection.close_timeout",
                timeout_seconds=_DISCONNECT_TIMEOUT_SECONDS,
            )
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        except asyncio.CancelledError:
            # We were cancelled while waiting; the owner task still needs to
            # finish unwinding, so leave it to its own shielded completion.
            task.cancel()
            raise
        except Exception as exc:
            self._log.warning(
                "mcp.connection.owner_task_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )

    async def _close(self) -> None:
        """Tear the transport down. Always runs inside the owner task.

        Issue #737 — when teardown raises, the exception type and message are
        surfaced as a structured warning instead of silently swallowed.
        """
        reason = "explicit_disconnect"
        error_type: str | None = None
        error_msg: str | None = None
        try:
            if self._session_cm:
                await self._session_cm.__aexit__(None, None, None)
            if self._cm:
                await self._cm.__aexit__(None, None, None)
            if self._http_client:
                await self._http_client.aclose()
        except Exception as exc:
            reason = "error_during_disconnect"
            error_type = type(exc).__name__
            error_msg = str(exc)
            self._log.warning(
                "mcp.server.disconnect_failed",
                error_type=error_type,
                error=error_msg,
            )
        finally:
            self._session = None
            self._session_cm = None
            self._cm = None
            self._http_client = None
            self._read = None
            self._write = None
            self._connected = False
            self._tools = []
            self._instructions = ""
            self._is_alive_last_checked = 0.0
            self._is_alive_last_result = False
            self._log.info(
                "mcp.server.disconnected",
                reason=reason,
                error_type=error_type,
                error=error_msg,
            )

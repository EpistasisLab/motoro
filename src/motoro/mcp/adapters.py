"""Adapters between MCP tools and the agent engine."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from typing import Any

import structlog
from jsonschema import Draft7Validator, ValidationError

from motoro.config import settings
from motoro.engine.context import RunContext
from motoro.mcp.client import MCPClient, ToolInfo
from motoro.mcp.registry import MCPServerRegistry
from motoro.observability.metrics import record_tool_call
from motoro.schemas.llm import LLMCallRecord, PlanStep, ToolCallRecord
from motoro.services.credential_scrubber import redact_tool_args
from motoro.services.retry import retry_with_backoff

TRANSIENT_ERRORS = (ConnectionError, OSError, TimeoutError)
# Issue #742 — retry_with_backoff is used for transient-error retries only;
# timeouts are never retried (the tool may have completed server-side).
_RETRYABLE_FOR_BACKOFF = (ConnectionError, OSError)

# MCP request ``_meta`` keys for ambient run identity (issue #1455). Namespaced
# under ``motoro.`` so they never collide with SDK/transport meta (e.g.
# ``progressToken``). Servers read these from ``ctx.request_context.meta`` to
# resolve the current workspace instead of trusting model-supplied arguments.
META_KEY_WORKSPACE_ID = "motoro.workspace_id"
META_KEY_RUN_ID = "motoro.run_id"
# The run's owner (opaque, product-defined — same attribution tag as
# Agent.owner_id/AgentRun.owner_id). Lets a tool scope its own data access to
# the acting user without the model ever being able to pass a different one
# in arguments — e.g. a shared workspace-management tool checking a dataset's
# owner_id against this before reading it, rather than trusting a bare name.
META_KEY_OWNER_ID = "motoro.owner_id"
# The acting agent's own name and model — raw facts a server can format into
# whatever actor/attribution convention it needs (e.g. a spec-defined
# "<agent_name>/<model>" string), rather than core baking in one format.
META_KEY_AGENT_NAME = "motoro.agent_name"
META_KEY_MODEL = "motoro.model"

log = structlog.get_logger()


def _build_run_meta(context: RunContext) -> dict[str, Any] | None:
    """Build the MCP request ``_meta`` payload from a run's ambient identity.

    Returns ``None`` when the run carries no identity to inject (keeping the
    wire call identical to the pre-#1455 behaviour), otherwise a dict with only
    the keys that are actually present.
    """
    meta: dict[str, Any] = {}
    if context.workspace_id:
        meta[META_KEY_WORKSPACE_ID] = context.workspace_id
    if context.run_id is not None:
        meta[META_KEY_RUN_ID] = str(context.run_id)
    if context.owner_id is not None:
        meta[META_KEY_OWNER_ID] = str(context.owner_id)
    if context.agent_name:
        # model is nested under agent_name rather than its own independent
        # check: ModelConfig.model always has a non-empty default, so keying
        # its inclusion on its own truthiness would put a model key on every
        # single run, defeating the "no identity -> None" contract above.
        # Gating on agent_name also matches the one real consumer's need — an
        # actor string like "<agent_name>/<model>" needs both or neither.
        meta[META_KEY_AGENT_NAME] = context.agent_name
        if context.model_config.model:
            meta[META_KEY_MODEL] = context.model_config.model
    return meta or None


class ToolExecutionError(Exception):
    """Structured error from MCP tool execution with error classification."""

    def __init__(
        self,
        tool: str,
        server: str,
        error_type: str,
        message: str,
        tool_record: ToolCallRecord | None = None,
    ) -> None:
        self.tool = tool
        self.server = server
        self.error_type = error_type
        self.tool_record = tool_record
        super().__init__(message)


class MCPToolExecutor:
    """Step executor that invokes MCP tools instead of calling the LLM."""

    def __init__(self, registry: MCPServerRegistry) -> None:
        self._registry = registry

    def can_handle(self, step: PlanStep, context: RunContext | None = None) -> bool:
        """Check if this step specifies a tool that we can execute.

        When *context* is supplied the tool must also be in the run's allow-list
        (``context.available_tools``) — a tool that resolves in the global
        registry but was not granted to this run does not count as handleable
        (issue #1454). Callers that want the allow-list enforced *and recorded*
        as a tool error should instead route to :meth:`execute_step`, which
        rejects unauthorized tools with a :class:`ToolExecutionError`.
        """
        if not step.tool_name:
            return False
        resolved = self._resolve_tool(step.tool_name)
        if resolved is None:
            return False
        if context is None:
            return True
        server_name, bare_tool, _client = resolved
        return _tool_in_allowlist(server_name, bare_tool, context.available_tools)

    async def execute_step(
        self, step: PlanStep, context: RunContext
    ) -> tuple[str, LLMCallRecord | None, ToolCallRecord | None]:
        """Execute a step by calling the MCP tool.

        Validates arguments against the tool's input schema before execution.
        """
        if not step.tool_name:
            raise ValueError("Step has no tool_name")

        server_name, tool_name, client, tool_info = self._resolve_tool_with_info(step.tool_name)
        arguments = dict(step.tool_args) if step.tool_args else {}
        # Redacted copy used in ToolCallRecord persistence and logs — never the live dict.
        safe_args = redact_tool_args(arguments)

        step_log = log.bind(
            tool=tool_name,
            server=server_name,
            component="mcp_executor",
        )

        # Enforce the run's tool allow-list at the dispatch boundary (issue
        # #1454). ``context.available_tools`` is the set built from the run's
        # ``server_names`` + ``tool_names`` filters; the model is only *told*
        # about these, but nothing stopped it from naming a tool that resolves
        # elsewhere in the global registry. Reject anything whose resolved
        # (server, tool) identity is not in that set — fail-closed, so an empty
        # allow-list permits no tools. Not retried: re-invoking would reject
        # again.
        if not _tool_in_allowlist(server_name, tool_name, context.available_tools):
            error_msg = f"Tool '{step.tool_name}' is not in this run's allowed tool set"
            tool_record = ToolCallRecord(
                server=server_name,
                tool=tool_name,
                arguments=safe_args,
                result=f"Not authorized: {error_msg}",
                latency_ms=0,
                success=False,
                error_type="not_authorized",
            )
            step_log.warning(
                "mcp.tool.not_authorized",
                requested=step.tool_name,
                allowed=[str(t.get("name") or t.get("tool_name") or "") for t in context.available_tools],
            )
            record_tool_call(server_name, tool_name, success=False, latency_seconds=0.0)
            raise ToolExecutionError(
                tool=tool_name,
                server=server_name,
                error_type="not_authorized",
                message=error_msg,
                tool_record=tool_record,
            )

        # Validate arguments against input schema
        validation_error = _validate_tool_args(tool_name, arguments, tool_info)
        if validation_error:
            tool_record = ToolCallRecord(
                server=server_name,
                tool=tool_name,
                arguments=safe_args,
                result=f"Validation error: {validation_error}",
                latency_ms=0,
                success=False,
                error_type="validation",
            )
            step_log.warning("mcp.tool.validation_failed", error=validation_error)
            record_tool_call(server_name, tool_name, success=False, latency_seconds=0.0)
            raise ToolExecutionError(
                tool=tool_name,
                server=server_name,
                error_type="validation",
                message=validation_error,
                tool_record=tool_record,
            )

        # Inject the run's ambient identity into the request _meta channel (issue
        # #1455) so context-dependent tools resolve workspace/run without the
        # model ever threading those ids through ``arguments``.
        run_meta = _build_run_meta(context)

        timeout = settings.tool_timeout_seconds
        last_error: Exception | None = None
        latency_ms = 0
        start_ms = int(time.monotonic() * 1000)

        # Issue #742 — TimeoutError is a subclass of OSError, so we wrap it to
        # prevent ``retry_with_backoff`` from treating it as a transient error
        # and retrying. Timeouts must bubble out untouched on the first attempt.
        class _NonRetryableTimeoutError(Exception):
            pass

        async def _invoke() -> Any:
            try:
                return await asyncio.wait_for(
                    client.call_tool(tool_name, arguments, meta=run_meta),
                    timeout=timeout,
                )
            except TimeoutError as exc:
                raise _NonRetryableTimeoutError() from exc

        # Issue #742 — replace fixed 2s sleep with jittered exponential backoff
        # via ``retry_with_backoff``. Timeouts still terminate immediately (they
        # are not retried because the underlying tool may have completed).
        try:
            try:
                result = await retry_with_backoff(
                    _invoke,
                    max_retries=1,
                    base_delay=1.0,
                    retryable_exceptions=_RETRYABLE_FOR_BACKOFF,
                )
            except _NonRetryableTimeoutError:
                latency_ms = int(time.monotonic() * 1000) - start_ms
                error_msg = f"Timeout after {timeout}s"
                step_log.warning("mcp.tool.timeout", timeout_s=timeout, latency_ms=latency_ms)
                tool_record = ToolCallRecord(
                    server=server_name,
                    tool=tool_name,
                    arguments=safe_args,
                    result=error_msg,
                    latency_ms=latency_ms,
                    success=False,
                    error_type="timeout",
                )
                record_tool_call(server_name, tool_name, success=False, latency_seconds=latency_ms / 1000)
                raise ToolExecutionError(
                    tool=tool_name,
                    server=server_name,
                    error_type="timeout",
                    message=error_msg,
                    tool_record=tool_record,
                ) from None
        except ToolExecutionError:
            raise
        except Exception as e:
            last_error = e
            latency_ms = int(time.monotonic() * 1000) - start_ms
        else:
            latency_ms = int(time.monotonic() * 1000) - start_ms
            # MCP marks ``isError`` only when the tool *raises*. A tool that
            # catches its own failure and returns a ``{"error": ...}`` payload
            # (a common MCP server convention) otherwise looks like a success — which
            # let failed feature-engineering/data-cleaning calls be recorded as
            # ``success=true`` and inflate the tool-success metric. Treat a
            # returned error object as the failure it is, so metrics and the
            # failure-classifying patterns see it, while still returning the
            # content so the agent can read the message and retry.
            reported_error = _tool_reported_error(result.content)
            success = not result.is_error and reported_error is None
            if result.is_error:
                error_type: str | None = "server_error"
            elif reported_error is not None:
                error_type = "tool_reported"
            else:
                error_type = None
            tool_record = ToolCallRecord(
                server=server_name,
                tool=tool_name,
                arguments=safe_args,
                result=result.content,
                latency_ms=latency_ms,
                success=success,
                error_type=error_type,
            )
            record_tool_call(
                server_name,
                tool_name,
                success=success,
                latency_seconds=latency_ms / 1000,
            )
            if success:
                step_log.info("mcp.tool.executed", latency_ms=latency_ms, success=True)
            else:
                step_log.warning(
                    "mcp.tool.reported_error",
                    latency_ms=latency_ms,
                    error_type=error_type,
                    content=result.content[:200],
                )
            return result.content, None, tool_record

        # All attempts exhausted
        error_msg = f"{type(last_error).__name__}: {last_error}" if last_error else "Unknown error"
        is_connection = isinstance(last_error, TRANSIENT_ERRORS)
        error_type = "connection" if is_connection else "server_error"
        tool_record = ToolCallRecord(
            server=server_name,
            tool=tool_name,
            arguments=arguments,
            result=f"Error: {error_msg}",
            latency_ms=latency_ms,
            success=False,
            error_type=error_type,
        )
        step_log.error(
            "mcp.tool.failed",
            error_type=error_type,
            error=error_msg,
            latency_ms=latency_ms,
        )
        record_tool_call(server_name, tool_name, success=False, latency_seconds=latency_ms / 1000)
        raise ToolExecutionError(
            tool=tool_name,
            server=server_name,
            error_type=error_type,
            message=error_msg,
            tool_record=tool_record,
        ) from last_error

    def _resolve_tool(self, tool_name: str) -> tuple[str, str, MCPClient] | None:
        """Resolve a tool name. Returns None if not found."""
        try:
            server, bare, client, _ = self._resolve_tool_with_info(tool_name)
            return server, bare, client
        except ValueError:
            return None

    def _resolve_tool_with_info(self, tool_name: str) -> tuple[str, str, MCPClient, ToolInfo | None]:
        """Resolve a tool name to (server_name, tool_name, client, tool_info).

        Issue #746 — delegates the actual lookup to the registry, which keeps an
        index instead of scanning every server's tool list.
        """
        resolved = self._registry.lookup_tool(tool_name)
        if resolved is None:
            raise ValueError(f"Tool '{tool_name}' not found in any connected MCP server")
        server_name, bare_tool, client, tool_info = resolved
        log.debug(
            "mcp.tool.resolved",
            tool=tool_name,
            server=server_name,
            method="namespaced" if "." in tool_name else "bare",
            component="mcp_executor",
        )
        return server_name, bare_tool, client, tool_info


def _tool_in_allowlist(
    server_name: str,
    bare_tool: str,
    available_tools: list[dict[str, Any]],
) -> bool:
    """Return True if the resolved ``(server_name, bare_tool)`` is granted to this run.

    ``available_tools`` is the run's allow-list — the filtered descriptor list
    from ``run_service._gather_tools`` (issue #1454). Entries from
    :meth:`MCPServerRegistry.get_all_tools` carry ``server`` + ``tool_name`` plus
    a namespaced ``name`` (``server.tool``); matching against the *resolved*
    identity makes namespaced vs bare requests behave identically, since both
    resolve to the same ``(server, tool)`` pair before we get here.
    """
    full_name = f"{server_name}.{bare_tool}"
    for tool in available_tools:
        t_server = tool.get("server")
        t_bare = str(tool.get("tool_name") or "")
        t_full = str(tool.get("name") or "")
        # Precise match on the resolved server + bare tool name.
        if t_server == server_name and t_bare == bare_tool:
            return True
        # Fall back to name-only descriptors (no structured server/tool_name),
        # accepting either the namespaced or the bare form.
        if t_full and t_full in (full_name, bare_tool):
            return True
    return False


def _tool_reported_error(content: str) -> str | None:
    """Return the message when *content* is a tool-reported error, else ``None``.

    Detects the ares-sklearn convention of returning ``{"error": "..."}`` as an
    ordinary (non-``isError``) payload — a tool that caught its own failure. Only
    a JSON object with a truthy top-level ``error`` counts; anything else (plain
    text, a list, an object without ``error``, or an empty/false ``error``) is
    treated as a normal result so unrelated payloads are never misread as failures.
    """
    if not content:
        return None
    stripped = content.strip()
    if not stripped.startswith("{"):
        return None
    try:
        parsed = json.loads(stripped)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    err = parsed.get("error")
    if isinstance(err, str):
        return err if err.strip() else None
    if err in (None, "", False, [], {}):
        return None
    return str(err)


def _normalize_schema_for_validation(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *schema* with additionalProperties: false injected.

    Prevents unknown/extra keys from passing validation silently.
    """
    has_object_shape = schema.get("type") == "object" or "properties" in schema
    if has_object_shape and "additionalProperties" not in schema:
        return {**schema, "additionalProperties": False}
    return schema


def _validate_tool_args(
    tool_name: str,
    args: dict[str, Any],
    tool_info: ToolInfo | None,
) -> str | None:
    """Validate tool arguments against the tool's input schema.

    Returns None if valid, or an error message string if invalid.
    Skips validation if no schema is available.
    """
    if tool_info is None:
        return None

    schema = tool_info.input_schema
    if not schema or not isinstance(schema, dict):
        return None

    effective_schema = _normalize_schema_for_validation(schema)

    try:
        Draft7Validator(effective_schema).validate(args)
    except ValidationError as e:
        field_path = ".".join(str(p) for p in e.absolute_path) if e.absolute_path else "(root)"
        if e.validator == "additionalProperties":
            return f"Unexpected field(s) in tool arguments: {e.message}"
        if e.validator == "required":
            return f"Missing required field: {e.message}"
        if e.validator == "type":
            return f"Invalid type for '{field_path}': {e.message}"
        return f"Validation error at '{field_path}': {e.message}"

    return None


# OpenAI tool names must match ``^[a-zA-Z0-9_-]{1,64}$``. Anything outside that
# alphabet (dots in MCP namespacing, slashes, etc.) has to be stripped — which
# created silent collisions (``foo.bar`` and ``foo_bar`` both → ``foo_bar``).
# Issue #772 — when the original name doesn't survive sanitisation unchanged,
# we append a short stable hash so distinct MCP tool names always map to
# distinct OpenAI tool names.
_OPENAI_TOOL_NAME_OK = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_OPENAI_TOOL_NAME_MAX = 64
_OPENAI_TOOL_NAME_HASH_LEN = 8


def _sanitize_openai_tool_name(bare_name: str) -> str:
    """Return an OpenAI-compatible tool name for *bare_name*. Issue #772."""
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", bare_name)
    if sanitized == bare_name and _OPENAI_TOOL_NAME_OK.match(bare_name):
        return bare_name[:_OPENAI_TOOL_NAME_MAX]

    digest = hashlib.sha256(bare_name.encode("utf-8")).hexdigest()[:_OPENAI_TOOL_NAME_HASH_LEN]
    # Reserve room for ``_<digest>`` suffix while keeping under the 64-char cap.
    prefix_budget = _OPENAI_TOOL_NAME_MAX - (_OPENAI_TOOL_NAME_HASH_LEN + 1)
    return f"{sanitized[:prefix_budget]}_{digest}"


def tools_to_openai_format(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert MCP tool descriptors to OpenAI/Anthropic native tool format."""
    result = []
    for tool in tools:
        bare_name = str(tool.get("tool_name") or tool.get("name", "unknown"))
        safe_name = _sanitize_openai_tool_name(bare_name)
        description = str(tool.get("description", ""))
        schema = tool.get("input_schema") or {}
        if not isinstance(schema, dict):
            schema = {}
        result.append(
            {
                "type": "function",
                "function": {
                    "name": safe_name,
                    "description": description,
                    "parameters": schema,
                },
            }
        )
    return result


def build_openai_tool_name_map(tools: list[dict[str, Any]]) -> dict[str, str]:
    """Map sanitized OpenAI tool names back to their original MCP names.

    :func:`tools_to_openai_format` rewrites names to satisfy the provider's
    ``^[a-zA-Z0-9_-]{1,64}$`` constraint, so the name the model calls back with
    is not necessarily resolvable by :class:`MCPToolExecutor`. Callers that bind
    native tool schemas must translate through this map before dispatch.
    """
    mapping: dict[str, str] = {}
    for tool in tools:
        bare_name = str(tool.get("tool_name") or tool.get("name", "unknown"))
        mapping[_sanitize_openai_tool_name(bare_name)] = bare_name
    return mapping


def get_tools_for_context(registry: MCPServerRegistry) -> list[dict[str, Any]]:
    """Get tool descriptions formatted for the Sense phase context."""
    return registry.get_all_tools()


def format_tool_for_prompt(tool: dict[str, Any]) -> str:
    """Format a single tool as a concise prompt string with parameter info."""
    name = str(tool.get("name", "unknown"))
    description = str(tool.get("description", ""))
    schema = tool.get("input_schema")

    params_str = _format_params_from_schema(schema)
    line = f"- {name}({params_str}): {description}"

    if len(line) > 250:
        line = line[:247] + "..."
    return line


def format_tools_for_prompt(tools: list[dict[str, Any]]) -> str:
    """Format all tools as a prompt section with parameter info."""
    if not tools:
        return ""
    lines = [format_tool_for_prompt(t) for t in tools]
    return "Available tools:\n" + "\n".join(lines)


def _format_params_from_schema(schema: Any) -> str:
    """Extract parameter signatures from a JSON Schema object."""
    if not isinstance(schema, dict):
        return ""

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return ""

    required_set = set(schema.get("required", []))
    parts: list[str] = []

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue

        type_str = _json_type_to_str(param_schema)
        optional = param_name not in required_set
        marker = "?" if optional else ""

        default = param_schema.get("default")
        if default is not None and optional:
            default_repr = repr(default)
            if len(default_repr) <= 30:
                parts.append(f"{param_name}{marker}: {type_str} = {default_repr}")
                continue

        parts.append(f"{param_name}{marker}: {type_str}")

    return ", ".join(parts)


def _json_type_to_str(schema: dict[str, Any]) -> str:
    """Convert a JSON Schema type to a concise type string."""
    t = schema.get("type", "any")
    if t == "array":
        items = schema.get("items", {})
        item_type = items.get("type", "any") if isinstance(items, dict) else "any"
        return f"{item_type}[]"
    if t == "object":
        return "object"
    if isinstance(t, list):
        return " | ".join(str(x) for x in t)
    return str(t)

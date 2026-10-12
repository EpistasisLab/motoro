"""Names from multiple granted MCP servers must remain distinct through dispatch."""

import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from motoro.engine.act import ActPhase
from motoro.engine.context import RunContext
from motoro.engine.patterns.builtin.reason_act import _issued_calls
from motoro.engine.runtime import _validate_plan_tools
from motoro.mcp.adapters import (
    MCPToolExecutor,
    _sanitize_openai_tool_name,
    _tool_in_allowlist,
    build_openai_tool_name_map,
    format_tools_for_prompt,
    tool_names_for_run,
    tools_to_openai_format,
)
from motoro.mcp.client import MCPClient, ToolInfo, ToolResult
from motoro.mcp.registry import MCPServerRegistry
from motoro.schemas.agent import ModelConfig
from motoro.schemas.llm import LLMToolCall, PlanOutput, PlanStep, ToolCompletion
from tests.stub_llm import StubLLM, call_record


def catalog() -> list[dict[str, Any]]:
    return [
        {"server_id": str(uuid.uuid4()), "server": server, "tool_name": "ping", "name": f"{server}.ping"}
        for server in ["server-a", "server-b"]
    ]


def test_same_named_tools_have_unique_provider_names_and_round_trip() -> None:
    tools = catalog()
    names = [schema["function"]["name"] for schema in tools_to_openai_format(tools)]
    assert len(set(names)) == 2
    mapping = build_openai_tool_name_map(tools)
    assert [mapping[name] for name in names] == ["server-a.ping", "server-b.ping"]


def test_legacy_namespaced_descriptor_passes_resolved_allowlist() -> None:
    tool = catalog()[0]
    tool["tool_name"] = tool["name"]
    assert _tool_in_allowlist(uuid.UUID(tool["server_id"]), tool["server"], "ping", [tool])


@pytest.fixture
async def connected(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Exercise real discovery and lookup; replace only the MCP transport."""
    calls: list[tuple[str, str]] = []

    async def connect(client: MCPClient) -> None:
        client._connected = True
        client._tools = [ToolInfo(name="ping", description=f"Ping {client.name}")]

    async def invoke(client: MCPClient, tool_name: str, arguments: Any, **_kwargs: Any) -> ToolResult:
        calls.append((client.name, tool_name))
        return ToolResult(content=f"pong from {client.name}")

    monkeypatch.setattr(MCPClient, "connect", connect)
    monkeypatch.setattr(MCPClient, "call_tool", invoke)
    monkeypatch.setattr("motoro.mcp.adapters.settings", SimpleNamespace(tool_timeout_seconds=1))
    registry = MCPServerRegistry()
    for tool in catalog():
        await registry.register(server_id=uuid.UUID(tool["server_id"]), name=tool["server"], is_system=True)
    tools = registry.get_all_tools()
    return MCPToolExecutor(registry), tools, calls


def context(tools: list[dict[str, Any]]) -> RunContext:
    return RunContext(
        agent_goal="check both servers",
        system_prompt="Use the available tools",
        model_config=ModelConfig(model="gpt-4o"),
        user_input="ping",
        available_tools=tools,
    )


@pytest.mark.parametrize("legacy", [False, True])
async def test_provider_names_dispatch_to_each_granted_server(connected: Any, legacy: bool) -> None:
    executor, tools, calls = connected
    if legacy:
        for tool in tools:
            tool["tool_name"] = tool["name"]
    run_context = context(tools)
    mapping = build_openai_tool_name_map(tools)
    for schema, tool in zip(tools_to_openai_format(tools), tools, strict=True):
        provider_name = schema["function"]["name"]
        step = PlanStep(action="ping", description="Check server", tool_name=mapping[provider_name])
        assert executor.can_handle(step, run_context)
        result, _llm, record = await executor.execute_step(step, run_context)
        assert result == f"pong from {tool['server']}"
        assert record.server == tool["server"]
        assert record.tool == "ping"
        assert record.success
    assert calls == [("server-a", "ping"), ("server-b", "ping")]


async def test_one_granted_server_keeps_bare_name_and_rejects_other_server(connected: Any) -> None:
    executor, tools, calls = connected
    run_context = context(tools[:1])
    assert [schema["function"]["name"] for schema in tools_to_openai_format(tools[:1])] == ["ping"]
    step = PlanStep(action="ping", description="Check server", tool_name="ping")
    assert executor.can_handle(step, run_context)
    await executor.execute_step(step, run_context)
    denied = step.model_copy(update={"tool_name": "server-b.ping"})
    assert not executor.can_handle(denied, run_context)
    with pytest.raises(ValueError, match="not found in this run"):
        await executor.execute_step(denied, run_context)
    assert calls == [("server-a", "ping")]


async def test_native_act_calls_both_provider_functions(connected: Any) -> None:
    executor, tools, calls = connected
    stub = StubLLM()
    schemas = tools_to_openai_format(tools)
    stub.complete_with_tools = AsyncMock(
        return_value=ToolCompletion(
            tool_calls=[LLMToolCall(name=schema["function"]["name"]) for schema in schemas], record=call_record()
        )
    )
    outcome = await ActPhase(stub, mcp_executor=executor)._execute_single_step(
        PlanStep(action="ping", description="Check both servers"), context(tools)
    )
    assert calls == [("server-a", "ping"), ("server-b", "ping")]
    assert [record.server for _text, record in outcome.tool_results] == ["server-a", "server-b"]
    assert stub.complete_with_tools.call_args.kwargs["tools"] == schemas


async def test_text_selection_gets_namespaced_choices_and_dispatches(connected: Any) -> None:
    executor, tools, calls = connected
    stub = StubLLM()
    stub.complete_with_tools = AsyncMock(side_effect=RuntimeError("native tools unavailable"))
    stub.select_tool = AsyncMock(return_value=(True, "server-b.ping", {}, call_record()))
    outcome = await ActPhase(stub, mcp_executor=executor)._execute_single_step(
        PlanStep(action="ping", description="Check server B"), context(tools)
    )
    assert stub.select_tool.call_args.kwargs["tool_names"] == ["server-a.ping", "server-b.ping"]
    messages = stub.select_tool.call_args.kwargs["messages"]
    prompt = "\n".join(message["content"] for message in messages)
    assert "server-a.ping" in prompt and "server-b.ping" in prompt
    assert calls == [("server-b", "ping")]
    assert outcome.tool_results[0][1].server == "server-b"


async def test_planned_namespaced_step_executes_without_reselection(connected: Any) -> None:
    executor, tools, calls = connected
    stub = StubLLM()
    outcome = await ActPhase(stub, mcp_executor=executor)._execute_single_step(
        PlanStep(action="ping", description="Check server B", tool_name="server-b.ping"), context(tools)
    )
    assert calls == [("server-b", "ping")]
    assert outcome.tool_results[0][1].server == "server-b"
    assert stub.calls == []


def test_react_issued_calls_keep_the_namespaced_dispatch_identity() -> None:
    tools = catalog()
    schemas = tools_to_openai_format(tools)
    calls = [LLMToolCall(id=str(index), name=schema["function"]["name"]) for index, schema in enumerate(schemas)]
    issued = _issued_calls(calls, build_openai_tool_name_map(tools), step_count=1)
    assert [call.tool_name for call in issued] == ["server-a.ping", "server-b.ping"]
    assert [call.provider_name for call in issued] == [call.name for call in calls]


def test_plan_validation_retains_namespaced_tools_and_clears_ambiguous_bare_name() -> None:
    plan = PlanOutput(
        steps=[
            PlanStep(action="ping", description="Check server", tool_name=name, tool_args={})
            for name in ["server-a.ping", "server-b.ping", "ping"]
        ]
    )
    _validate_plan_tools(plan, catalog())
    assert [step.tool_name for step in plan.steps] == ["server-a.ping", "server-b.ping", None]
    assert plan.steps[-1].tool_args is None


def test_plan_validation_still_accepts_noncolliding_bare_name() -> None:
    plan = PlanOutput(steps=[PlanStep(action="ping", description="Check server", tool_name="ping")])
    _validate_plan_tools(plan, catalog()[:1])
    assert plan.steps[0].tool_name == "ping"


def test_mixed_catalog_uses_one_rule_and_keeps_noncolliding_names() -> None:
    tools = catalog()
    tools[1]["tool_name"] = tools[1]["name"]
    tools.append({**tools[0], "tool_name": "status", "name": "server-a.status"})
    assert tool_names_for_run(tools) == ["server-a.ping", "server-b.ping", "status"]
    names = [schema["function"]["name"] for schema in tools_to_openai_format(tools)]
    assert names[-1] == "status"
    assert all(0 < len(name) <= 64 and all(char.isalnum() or char in "_-" for char in name) for name in names)
    assert build_openai_tool_name_map(tools) == build_openai_tool_name_map(list(reversed(tools)))
    prompt = format_tools_for_prompt(tools)
    assert "server-a.ping" in prompt and "server-b.ping" in prompt


@pytest.mark.parametrize("field,value", [("server_id", str(uuid.uuid4())), ("server", "wrong"), ("tool_name", "wrong")])
def test_legacy_name_does_not_bypass_identity_checks(field: str, value: str) -> None:
    tool = catalog()[0]
    server_id = uuid.UUID(tool["server_id"])
    server = tool["server"]
    tool["tool_name"] = tool["name"]
    tool[field] = value
    assert not _tool_in_allowlist(server_id, server, "ping", [tool])


def test_sanitization_still_distinguishes_punctuation_and_long_names() -> None:
    tools = [
        {"tool_name": "a.b"},
        {"tool_name": "a_b"},
        {"tool_name": "x" * 100},
        {"tool_name": "x" * 99 + "y"},
    ]
    names = [schema["function"]["name"] for schema in tools_to_openai_format(tools)]
    assert len(set(names)) == len(tools)
    assert names[0] == _sanitize_openai_tool_name("a.b")
    assert [build_openai_tool_name_map(tools)[name] for name in names] == [tool["tool_name"] for tool in tools]

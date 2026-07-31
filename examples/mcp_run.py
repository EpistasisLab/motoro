#!/usr/bin/env python3
"""RUNTIME — a run that calls a real MCP tool.

Two things this ties together, both landed separately: ``services.mcp_service``
(register a server once, persist it) and the Act phase's existing MCP adapter
(call a tool during a run). Nothing here is core-internal — a product does
exactly this: register its servers as a deploy-adjacent step, hydrate the
registry at process start, and pass the aggregated tool list into
``execute_run``.

``examples/mcp_server.py`` is the server: one tool, ``get_secret_code``, that
returns a fact the model cannot know or guess. A correct code in the run's
output is unambiguous proof the tool actually ran — a calculator or a weather
lookup can't rule out the model just guessing plausibly.

Usage::

    set -a && . ./.env && set +a
    python examples/mcp_run.py
    python examples/mcp_run.py --input "What is the secret code for bravo?" --trace
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from settings import Settings

from agentic_core import configure
from agentic_core.mcp.registry import MCPServerRegistry
from agentic_core.runner import create_agent, create_run, execute_run, get_agent_by_name, get_run_steps
from agentic_core.schemas.agent import LLMProvider, ModelConfig
from agentic_core.services.mcp_service import get_server_by_name, hydrate_registry, register_server

_AGENT_NAME = "example-mcp-tools"
_SERVER_NAME = "example-tools"
_SERVER_COMMAND = f"{sys.executable} {Path(__file__).parent / 'mcp_server.py'}"


async def _provision_agent() -> Any:
    """Idempotent by name, same as provision.py — reason_act is the pattern that calls tools."""
    existing = await get_agent_by_name(_AGENT_NAME)
    if existing is not None:
        return existing
    return await create_agent(
        name=_AGENT_NAME,
        goal="Answer the user's question, using tools when the answer requires one.",
        model_config=ModelConfig(provider=LLMProvider.AZURE_FOUNDRY, model="claude-sonnet-5"),
        pattern_config={"execution_pattern": "reason_act"},
    )


async def _hydrated_registry() -> MCPServerRegistry:
    """Register the example server once, then reconnect from the table every time.

    Mirrors the real shape: ``register_server`` is a one-time setup step (like
    ``provision.py``); every run after that calls ``hydrate_registry()`` against
    a registry that starts empty in this process, the same as a worker or a
    restarted API would.
    """
    registry = MCPServerRegistry()
    # Explicit registry= on every call — the default is the process-global
    # singleton (agentic_core.mcp.registry.get_registry()), and this example
    # manages its own connection lifecycle end to end, so nothing should touch
    # that singleton or leave a second, undisconnected client behind it.
    if await get_server_by_name(_SERVER_NAME) is None:
        await register_server(name=_SERVER_NAME, transport="stdio", command=_SERVER_COMMAND, registry=registry)

    failed = await hydrate_registry(registry=registry)
    if failed:
        print(f"!! failed to reconnect: {failed}", file=sys.stderr)
    return registry


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="What is the secret code for alpha?")
    ap.add_argument("--trace", action="store_true", help="print the persisted steps, including the tool call")
    args = ap.parse_args()

    configure(Settings())
    agent = await _provision_agent()
    print(f"agent: {agent.id}  ({agent.name})")

    registry = await _hydrated_registry()
    try:
        print(f"tools available: {[t['name'] for t in registry.get_all_tools()]}\n")

        run = await create_run(agent_id=agent.id, user_input=args.input)
        result = await execute_run(run_id=run.id, registry=registry, available_tools=registry.get_all_tools())

        print(f"status: {result.status}")
        if result.error:
            print(f"error:  {result.error}", file=sys.stderr)
            return 1
        print(f"\n{result.output}")

        if args.trace:
            print("\npersisted trace:")
            for step in await get_run_steps(run.id):
                extra = f"  tool_call={step.tool_call}" if step.tool_call else ""
                print(f"  {step.sequence}  {step.phase.value}{extra}")
        return 0
    finally:
        # A long-lived process (a worker, an API) would keep this registry live
        # across many runs and disconnect it once, at shutdown — not per run.
        await registry.disconnect_all()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

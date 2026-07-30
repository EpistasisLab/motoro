#!/usr/bin/env python3
"""Create an agent, start a run, execute it — the whole core in one script.

This is the "tiny second consumer" the extraction plan calls for. ARES cannot
validate core's public surface, because core's code came *from* ARES and will
always satisfy it. A separate script that only ever touches documented entry
points is the cheapest thing that fails loudly when core secretly needs a product.

Usage::

    cp .env.example .env && docker compose up -d
    set -a && . ./.env && set +a
    export ANTHROPIC_API_KEY='sk-ant-...'
    python examples/run_agent.py --pattern reason_act

    # or, without spending anything:
    python examples/run_agent.py --dry-run

Requires `docker compose up -d` (Postgres, and Redis for working memory). No MCP
server is needed — an agent with no tools needs none.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from pydantic_settings import SettingsConfigDict

from agentic_core import CoreSettings, configure
from agentic_core.schemas.agent import LLMProvider, ModelConfig


class Settings(CoreSettings):
    """Stands in for a product's settings class, prefix and all."""

    model_config = SettingsConfigDict(env_prefix="AGENTIC_", env_file=".env", extra="ignore")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pattern",
        default="reason_act",
        choices=["reason_act", "single_agent_baseline"],
        help="execution pattern for the agent (default: reason_act)",
    )
    ap.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "azure_foundry"],
        help="anthropic reads ANTHROPIC_API_KEY; azure_foundry reads "
        "ANTHROPIC_FOUNDRY_API_KEY + ANTHROPIC_FOUNDRY_RESOURCE",
    )
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--goal", default="Answer the user's question accurately and concisely.")
    ap.add_argument("--input", default="What is 17 * 23? Show your reasoning briefly.")
    ap.add_argument("--reset", action="store_true", help="roll the schema back to base and re-migrate first")
    ap.add_argument("--dry-run", action="store_true", help="set everything up but do not call the LLM")
    args = ap.parse_args()

    # 1. Install settings before anything reads them.
    configure(Settings())

    # 2. Apply core's schema the way a product would: run core's migration chain.
    #    (runner.init_schema / create_all also works, but is for tests — it leaves
    #    the schema unversioned.)
    from agentic_core.migrations import current_revision, downgrade, upgrade_async
    from agentic_core.runner import create_agent, create_run, execute_run

    if args.reset and await current_revision() is not None:
        await asyncio.to_thread(downgrade, None, "base")
    await upgrade_async()
    print(f"schema at revision {await current_revision()}")

    # 3. Confirm the pattern registry sees exactly what we migrated.
    from agentic_core.engine.patterns.registry import PluginRegistry

    PluginRegistry.discover(raise_on_error=True)
    print(f"patterns available: {sorted(PluginRegistry.all())}")
    if args.pattern not in PluginRegistry.all():
        print(f"!! {args.pattern} is not registered", file=sys.stderr)
        return 1

    # The credential rides on the ModelConfig, so no resolver hook is needed and
    # core never has to know where secrets live. This is the whole of "setting an
    # LLM provider" — no users, no settings table, no encryption.
    if args.provider == "azure_foundry":
        api_key = os.environ.get("ANTHROPIC_FOUNDRY_API_KEY", "")
        resource = os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE", "")
        api_base = resource if resource.startswith("http") else f"https://{resource}.services.ai.azure.com"
        provider, key_var = LLMProvider.AZURE_FOUNDRY, "ANTHROPIC_FOUNDRY_API_KEY"
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        api_base = None
        provider, key_var = LLMProvider.ANTHROPIC, "ANTHROPIC_API_KEY"

    if not api_key and not args.dry_run:
        print(f"!! {key_var} is not set (use --dry-run to skip the LLM call)", file=sys.stderr)
        return 1

    model_config = ModelConfig(
        provider=provider,
        model=args.model,
        api_key=api_key or None,
        api_base=api_base,
        max_tokens=2048,
    )
    print(f"provider:      {provider.value}  model={args.model}")

    from agentic_core.models.database import system_session

    async with system_session(reason="examples/run_agent.py") as db:
        agent = await create_agent(
            db,
            name=f"example-{args.pattern}",
            goal=args.goal,
            description="Example agent for the agentic-core smoke test.",
            model_config=model_config,
            pattern_config={"execution_pattern": args.pattern},
        )
        print(f"agent created: {agent.id}  pattern={args.pattern}")

        run = await create_run(db, agent_id=agent.id, user_input=args.input)
        print(f"run created:   {run.id}  status={run.status.value}")

        if args.dry_run:
            print("\n--dry-run: stopping before the LLM call. Everything above is real.")
            return 0

        print("\nexecuting...\n")
        result = await execute_run(db, run_id=run.id)

        print(f"status:      {result.status}")
        print(f"steps:       {len(result.steps)}")
        print(f"tokens:      {result.total_prompt_tokens} prompt / {result.total_completion_tokens} completion")
        print(f"cost:        ${result.total_cost:.6f}")
        if result.error:
            print(f"error:       {result.error}")
        print(f"\noutput:\n{result.output}")

        # Prove the steps were persisted, not just returned.
        from sqlalchemy import func, select

        from agentic_core.models.run import RunStep

        n = (await db.execute(select(func.count()).select_from(RunStep).where(RunStep.run_id == run.id))).scalar()
        print(f"\nRunStep rows persisted: {n}")

    return 0 if result.error is None else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

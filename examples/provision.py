#!/usr/bin/env python3
"""ONE-TIME SETUP — provision an agent. Run once, keep the id.

An agent is a durable, reusable resource: you create it once and start many runs
against it. This is the shape a product wants — an admin action, a seed script, a
CI task — not something on the request path. Creating an agent per run
accumulates junk and, because names are unique per installation over live rows,
fails outright the second time.

The schema is **not** provisioned here. That is a separate deploy step:

    python -m agentic_core.migrations upgrade --url "$AGENTIC_DATABASE_URL"

Usage::

    set -a && . ./.env && set +a
    python examples/provision.py                       # prints the agent id
    python examples/provision.py --pattern single_agent_baseline
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from settings import Settings  # the product's settings class — see settings.py

from agentic_core import configure
from agentic_core.schemas.agent import LLMProvider, ModelConfig


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="agent name (default: example-<pattern>)")
    ap.add_argument("--pattern", default="reason_act", choices=["reason_act", "single_agent_baseline"])
    ap.add_argument("--provider", default="azure_foundry", choices=["anthropic", "azure_foundry"])
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--goal", default="Answer the user's question accurately and concisely.")
    args = ap.parse_args()

    configure(Settings())

    from agentic_core.migrations import current_revision
    from agentic_core.models.database import system_session
    from agentic_core.runner import create_agent

    # Verify, don't migrate. A product that silently migrates on startup races
    # its own replicas; a product that assumes the schema is there fails with an
    # unreadable SQL error. Checking is the middle ground.
    if await current_revision() is None:
        print(
            "!! schema not provisioned. Run the deploy step first:\n"
            '   python -m agentic_core.migrations upgrade --url "$AGENTIC_DATABASE_URL"',
            file=sys.stderr,
        )
        return 1

    name = args.name or f"example-{args.pattern}"
    provider = LLMProvider(args.provider)

    # No api_key here on purpose: it is `exclude=True`, so it would not survive
    # being persisted with the agent. The credential is resolved at call time
    # from settings — see agentic_core.services.credentials.
    model_config = ModelConfig(provider=provider, model=args.model, max_tokens=2048)

    async with system_session(reason="examples/provision.py") as db:
        # Idempotent by name. Agent names are unique per installation over live
        # rows, so a second invocation would otherwise fail on the constraint —
        # and a seed script that cannot be re-run safely is a seed script people
        # avoid running. Report the existing agent instead.
        from sqlalchemy import func, select

        from agentic_core.models.agent import Agent

        existing = (
            await db.execute(
                select(Agent).where(func.lower(Agent.name) == name.lower(), Agent.deleted_at.is_(None))
            )
        ).scalar_one_or_none()
        if existing is not None:
            print(f"agent already provisioned: {existing.id}")
            print(f"  name: {existing.name}")
            print(f"\nStart runs with:\n  python examples/run.py --agent-id {existing.id} --input '...'")
            return 0

        agent = await create_agent(
            db,
            name=name,
            goal=args.goal,
            description="Example agent for the agentic-core smoke test.",
            model_config=model_config,
            pattern_config={"execution_pattern": args.pattern},
        )
        print(f"agent provisioned: {agent.id}")
        print(f"  name:    {agent.name}")
        print(f"  pattern: {args.pattern}")
        print(f"  model:   {provider.value} / {args.model}")
        print(f"\nStore that id. Start runs with:\n  python examples/run.py --agent-id {agent.id} --input '...'")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

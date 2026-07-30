#!/usr/bin/env python3
"""ONE-TIME SETUP — provision an agent. Run once, keep the id.

An agent is a durable, reusable resource: create it once, start many runs against
it. That is an admin action, a seed script, a CI task — never the request path.

Note what is *not* here: no session, no core models, no schema. Core owns its own
database. The schema is applied separately, once, as a deploy step::

    python -m agentic_core.migrations upgrade --url "$AGENTIC_DATABASE_URL"

Usage::

    set -a && . ./.env && set +a
    python examples/provision.py
    python examples/provision.py --pattern single_agent_baseline
"""

from __future__ import annotations

import argparse
import asyncio

from settings import Settings

from agentic_core import configure
from agentic_core.runner import create_agent, get_agent_by_name
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

    name = args.name or f"example-{args.pattern}"

    # Idempotent by name. Agent names are unique per installation over live rows,
    # so a second invocation would otherwise fail the constraint — and a seed
    # script that cannot be re-run safely is one people avoid running.
    existing = await get_agent_by_name(name)
    if existing is not None:
        print(f"agent already provisioned: {existing.id}  ({existing.name})")
        print(f"\nStart runs with:\n  python examples/run.py --agent-id {existing.id} --input '...'")
        return 0

    provider = LLMProvider(args.provider)
    # No api_key here on purpose: it is `exclude=True`, so it would not survive
    # being persisted with the agent. Credentials resolve at call time from
    # settings — see agentic_core.services.credentials.
    model_config = ModelConfig(provider=provider, model=args.model, max_tokens=2048)

    agent = await create_agent(
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

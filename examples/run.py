#!/usr/bin/env python3
"""RUNTIME — start a run against an existing agent and execute it.

Note what is *not* here: no session, no core models, no schema. Core owns its own
database and manages its own connections; a product configures the URL once and
then only calls functions. A product's own tables live in a separate database it
manages itself.

Usage::

    set -a && . ./.env && set +a
    python examples/run.py --agent-name example-reason_act --input "What is 17 * 23?"
    python examples/run.py --agent-id <id>
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

from settings import Settings

from motoro import configure
from motoro.runner import create_run, execute_run, get_agent, get_agent_by_name, get_run_steps


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-id", default=None, help="agent UUID from provision.py")
    ap.add_argument("--agent-name", default=None, help="look the agent up by name instead")
    ap.add_argument("--input", default="What is 17 * 23? Show your reasoning briefly.")
    ap.add_argument("--trace", action="store_true", help="print the persisted SRPA steps")
    args = ap.parse_args()

    if not args.agent_id and not args.agent_name:
        ap.error("pass --agent-id (from provision.py) or --agent-name")

    # Once per process. In a web app this is the startup / lifespan hook.
    configure(Settings())

    agent = await get_agent(uuid.UUID(args.agent_id)) if args.agent_id else await get_agent_by_name(args.agent_name)
    if agent is None:
        print("!! agent not found — run examples/provision.py first", file=sys.stderr)
        return 1
    print(f"agent: {agent.id}  ({agent.name})")

    # Two calls, not one, so a product can enqueue rather than block: create the
    # run in the request, return its id, let a worker execute it. This CLI does
    # both inline — the one thing a real product would change.
    run = await create_run(agent_id=agent.id, user_input=args.input)
    print(f"run:   {run.id}  status={run.status.value}\n")

    result = await execute_run(run_id=run.id)

    print(f"status: {result.status}")
    print(f"steps:  {len(result.steps)}")
    print(f"tokens: {result.total_prompt_tokens} prompt / {result.total_completion_tokens} completion")
    print(f"cost:   ${result.total_cost:.6f}")
    if result.error:
        print(f"error:  {result.error}", file=sys.stderr)
        return 1
    print(f"\n{result.output}")

    if args.trace:
        print("\npersisted trace:")
        for step in await get_run_steps(run.id):
            print(f"  {step.sequence}  {step.phase.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

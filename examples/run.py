#!/usr/bin/env python3
"""RUNTIME — start a run against an existing agent and execute it.

This is the whole of what a product does per request. Everything else — applying
the schema, provisioning the agent — happened earlier, once. There are no
migrations here and no agent creation here, deliberately:

* Migrations are a deploy step (``python -m agentic_core.migrations upgrade``).
  Running them per process races replicas and turns a failed deploy into a
  crash-looping app.
* Agents are durable resources provisioned once (``examples/provision.py``).

``create_run`` and ``execute_run`` are separate calls so a product can **enqueue**
the execution instead of blocking on it. A web app creates the run in the request,
returns the id, and lets a worker call ``execute_run``. This script does both
inline because it is a CLI, and that is the one thing here a real product would
change.

Usage::

    set -a && . ./.env && set +a
    python -m agentic_core.migrations upgrade --url "$AGENTIC_DATABASE_URL"   # once
    python examples/provision.py                                             # once
    python examples/run.py --agent-id <id> --input "What is 17 * 23?"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

from settings import Settings

from agentic_core import configure


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-id", default=None, help="agent UUID from provision.py")
    ap.add_argument("--agent-name", default=None, help="look the agent up by name instead")
    ap.add_argument("--input", default="What is 17 * 23? Show your reasoning briefly.")
    args = ap.parse_args()

    if not args.agent_id and not args.agent_name:
        ap.error("pass --agent-id (from provision.py) or --agent-name")

    # 1. Install settings. Must happen before anything reads one.
    configure(Settings())

    from sqlalchemy import func, select

    from agentic_core.migrations import current_revision
    from agentic_core.models.agent import Agent
    from agentic_core.models.database import system_session
    from agentic_core.runner import create_run, execute_run

    # 2. Verify the schema — do not migrate it.
    if await current_revision() is None:
        print(
            "!! schema not provisioned. Deploy step:\n"
            '   python -m agentic_core.migrations upgrade --url "$AGENTIC_DATABASE_URL"',
            file=sys.stderr,
        )
        return 1

    async with system_session(reason="examples/run.py") as db:
        # 3. Look up the pre-provisioned agent.
        if args.agent_id:
            stmt = select(Agent).where(Agent.id == uuid.UUID(args.agent_id))
        else:
            stmt = select(Agent).where(
                func.lower(Agent.name) == args.agent_name.lower(), Agent.deleted_at.is_(None)
            )
        agent = (await db.execute(stmt)).scalar_one_or_none()
        if agent is None:
            print("!! agent not found — run examples/provision.py first", file=sys.stderr)
            return 1
        print(f"agent: {agent.id}  ({agent.name})")

        # 4. Create the run. A web app would return here and let a worker
        #    execute it; the two calls are separate precisely so it can.
        run = await create_run(db, agent_id=agent.id, user_input=args.input)
        print(f"run:   {run.id}  status={run.status.value}\n")

        # 5. Execute. Core assembles the loop, resolves the credential from
        #    settings, applies the agent's pattern, and records the outcome.
        result = await execute_run(db, run_id=run.id)

        print(f"status: {result.status}")
        print(f"steps:  {len(result.steps)}")
        print(f"tokens: {result.total_prompt_tokens} prompt / {result.total_completion_tokens} completion")
        print(f"cost:   ${result.total_cost:.6f}")
        if result.error:
            print(f"error:  {result.error}", file=sys.stderr)
            return 1
        print(f"\n{result.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

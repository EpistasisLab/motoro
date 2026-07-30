#!/usr/bin/env python3
"""RUNTIME — start a run against an existing agent and execute it.

Two things this file is trying to make obvious, because they are easy to get
wrong from the outside:

**The product owns the database connection.** Core never opens a session on its
own — every public entry point takes ``db: AsyncSession`` as its first argument.
Where that session comes from is entirely yours: a FastAPI ``Depends(get_db)``,
your own ``async_sessionmaker``, a session your web framework already manages.
``system_session`` used below is just the convenience core offers for contexts
with *no* request to scope to — a CLI like this one, a worker, a cron job. A web
app would not use it.

Core does own the *engine* by default (`get_engine()` reads
``CoreSettings.database_url``), so that core and product models — which share one
``Base`` and one database — share one connection pool rather than opening two.
A product is free to ignore it and hand in sessions from its own engine.

**Schema checks are a startup concern, not a per-request one.** The
``current_revision()`` call below sits in the startup section on purpose: it runs
once, at process start, to refuse to serve against a schema that is behind. A
feature developer never writes this — it lives in the app's startup path once, and
nobody imports it again. It is emphatically not something to do per request.

Usage::

    set -a && . ./.env && set +a
    python examples/run.py --agent-id <id> --input "What is 17 * 23?"
    python examples/run.py --agent-name example-reason_act
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from typing import TYPE_CHECKING

from settings import Settings

from agentic_core import configure

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from agentic_core.models.agent import Agent


async def verify_schema() -> bool:
    """Startup guard: is core's schema present and current?

    Called **once per process**, not per request. A product that would rather
    crash on the first query can skip this; the point of having it is that a
    schema-drift failure reads as "run the deploy step" instead of as an
    unexplained SQL error deep in a request.
    """
    from agentic_core.migrations import current_revision

    if await current_revision() is None:
        print(
            "!! core's schema is not provisioned. This is a deploy step, run once:\n"
            '   python -m agentic_core.migrations upgrade --url "$AGENTIC_DATABASE_URL"',
            file=sys.stderr,
        )
        return False
    return True


async def find_agent(db: AsyncSession, *, agent_id: str | None, name: str | None) -> Agent | None:
    """Look up the pre-provisioned agent. Ordinary product code over core's models."""
    from sqlalchemy import func, select

    from agentic_core.models.agent import Agent

    if agent_id:
        stmt = select(Agent).where(Agent.id == uuid.UUID(agent_id))
    else:
        assert name is not None
        stmt = select(Agent).where(func.lower(Agent.name) == name.lower(), Agent.deleted_at.is_(None))
    return (await db.execute(stmt)).scalar_one_or_none()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-id", default=None, help="agent UUID from provision.py")
    ap.add_argument("--agent-name", default=None, help="look the agent up by name instead")
    ap.add_argument("--input", default="What is 17 * 23? Show your reasoning briefly.")
    args = ap.parse_args()

    if not args.agent_id and not args.agent_name:
        ap.error("pass --agent-id (from provision.py) or --agent-name")

    # ── STARTUP — once per process ───────────────────────────────────────────
    # In a web app this is the lifespan / startup hook, and none of it is on the
    # request path.
    configure(Settings())
    if not await verify_schema():
        return 1

    from agentic_core.models.database import system_session
    from agentic_core.runner import create_run, execute_run

    # ── PER REQUEST — the only part a product runs per call ──────────────────
    # The session is the product's to provide. `system_session` is core's helper
    # for request-less contexts (CLI, worker); a FastAPI app would instead take
    # `db: AsyncSession = Depends(get_db)` and pass that straight through.
    async with system_session(reason="examples/run.py") as db:
        agent = await find_agent(db, agent_id=args.agent_id, name=args.agent_name)
        if agent is None:
            print("!! agent not found — run examples/provision.py first", file=sys.stderr)
            return 1
        print(f"agent: {agent.id}  ({agent.name})")

        # Two calls, not one, so a product can enqueue rather than block: create
        # the run in the request, return its id, let a worker execute it. This
        # CLI does both inline — the one thing a real product would change.
        run = await create_run(db, agent_id=agent.id, user_input=args.input)
        print(f"run:   {run.id}  status={run.status.value}\n")

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

#!/usr/bin/env python3
"""RUNTIME — a run that remembers, across processes.

Episodic memory is opt-in per agent (``memory_config={"episodic_memory_enabled":
True}``) and, unlike working memory, is not scoped to one run: it lives in
Postgres, so a fact stated in one invocation of this script is still there the
next time you run it — no state carried in this process, no Redis involved.

Note what is *not* here beyond the usual: no session, no core models, no schema —
see ``run.py``'s docstring for that half of the story. The one addition is
``MemoryService``, constructed once and passed to every ``execute_run`` call so
storage (after a run) and recall (at the start of the next one) share the same
embedding cache.

Usage::

    set -a && . ./.env && set +a

    # First call: nothing to recall yet, so it just answers and stores a summary.
    python examples/memory_run.py --input "My favourite programming language is Rust."

    # Second call, same or a later process: the Sense phase recalls the first
    # run's episodic summary and folds it into this run's Reason phase.
    python examples/memory_run.py --input "What did I say my favourite language was?"

    # No --input at all runs both turns back to back, so the effect is visible
    # in one invocation.
    python examples/memory_run.py --trace
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from settings import Settings

from agentic_core import configure
from agentic_core.runner import create_agent, create_run, execute_run, get_agent_by_name
from agentic_core.schemas.agent import LLMProvider, ModelConfig
from agentic_core.services.llm_service import LLMService
from agentic_core.services.memory_service import MemoryService

_AGENT_NAME = "example-episodic-memory"

_DEMO_TURNS = [
    "My favourite programming language is Rust.",
    "What did I just tell you my favourite programming language was?",
]


async def _provision() -> Any:
    """Idempotent by name, same as provision.py — the one difference is memory_config."""
    existing = await get_agent_by_name(_AGENT_NAME)
    if existing is not None:
        return existing
    return await create_agent(
        name=_AGENT_NAME,
        goal="Answer the user's question, remembering what they have told you before.",
        model_config=ModelConfig(provider=LLMProvider.AZURE_FOUNDRY, model="claude-sonnet-5"),
        pattern_config={"execution_pattern": "single_agent_baseline"},
        # Off by default (see agentic_core.schemas.agent.MemoryConfig) — a
        # product opts an agent in explicitly, same field ARES used.
        memory_config={"episodic_memory_enabled": True},
    )


async def _run_turn(agent_id: uuid.UUID, user_input: str, memory_service: MemoryService, *, trace: bool) -> None:
    run = await create_run(agent_id=agent_id, user_input=user_input)
    result = await execute_run(run_id=run.id, memory_service=memory_service)

    print(f"> {user_input}")
    if result.error:
        print(f"  error: {result.error}")
        return
    print(f"  {result.output}\n")

    if trace:
        # None on the very first run of an agent's life — nothing exists yet
        # to recall. Present but 0 means recall ran and found nothing
        # relevant; a positive count means Sense actually folded prior
        # episodes into this run's Reason phase.
        recalled = result.run_metadata.get("memory_recalled_count")
        print(f"  [memory_recalled_count={recalled}]\n")


async def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="one turn; omit to run the two-turn demo below")
    ap.add_argument("--trace", action="store_true", help="print memory_recalled_count after each turn")
    args = ap.parse_args()

    configure(Settings())
    agent = await _provision()
    print(f"agent: {agent.id}  ({agent.name})\n")

    # One MemoryService for the whole process: it owns the embedding cache, not
    # any per-run state, so sharing it across turns is free and expected.
    memory_service = MemoryService(llm_service=LLMService())

    turns = [args.input] if args.input else _DEMO_TURNS
    for turn in turns:
        await _run_turn(agent.id, turn, memory_service, trace=args.trace)

    count = await memory_service.episodic.count(agent.id)
    print(f"episodic memories stored for this agent: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

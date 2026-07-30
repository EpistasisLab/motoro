"""The composition root: create an agent, create a run, execute it.

`AgentRuntime` executes a loop but does not assemble itself — it expects a caller
to have built the phase objects, opened a session, created the `AgentRun` row, and
decided which pattern wraps it. In ARES that assembly lives in `run_service` plus
an arq task, tangled up with per-user credential resolution, experiment
membership, batches, and SSE publishing. None of that belongs to a runtime, so
core has its own slim version here.

This is the piece a product replaces the body of its own `execute_run_task` with::

    from agentic_core.runner import create_agent, create_run, execute_run

    agent = await create_agent(db, name="researcher", goal="...", model_config=cfg)
    run = await create_run(db, agent_id=agent.id, user_input="...")
    result = await execute_run(db, run_id=run.id)

Deliberately absent, because they are product concerns: authentication, ownership
enforcement, run cancellation plumbing, cost budgets, score extraction, event
streaming. Each has a seam here — `owner_id`, `cancel_event`, `publish_event` —
but core neither populates nor interprets them.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from agentic_core.engine.patterns.orchestrator import DEFAULT_EXECUTION_PATTERN, PatternOrchestrator
from agentic_core.engine.runtime import AgentConfig, AgentRuntime
from agentic_core.models.agent import Agent
from agentic_core.models.run import AgentRun, RunStatus
from agentic_core.schemas.agent import ModelConfig

if TYPE_CHECKING:
    import asyncio

    from sqlalchemy.ext.asyncio import AsyncSession

    from agentic_core.engine.phase import Phase
    from agentic_core.engine.ports import MemoryServicePort
    from agentic_core.engine.runtime import AgentRunResult
    from agentic_core.mcp.registry import MCPServerRegistry


async def init_schema(*, drop_first: bool = False) -> None:
    """Create core's tables directly from the ORM metadata.

    There is no migration chain yet: this issues ``CREATE TABLE`` for whatever is
    registered on ``Base.metadata``, which for core today is five tables and no
    pgvector extension. Adequate for development and tests, and honest about
    being so — nothing is version-tracked, so the first schema change means
    either ``drop_first`` or stamping an Alembic baseline.

    Importing a product's models before calling this creates those too, since
    they share core's ``Base``.
    """
    # Importing for the metadata side effect: a table absent from Base.metadata
    # is a table this will not create.
    import agentic_core.models.agent  # noqa: F401  PLC0415
    import agentic_core.models.pattern  # noqa: F401  PLC0415
    import agentic_core.models.pricing  # noqa: F401  PLC0415
    import agentic_core.models.run  # noqa: F401  PLC0415
    from agentic_core.models.base import Base
    from agentic_core.models.database import get_engine

    engine = get_engine()
    async with engine.begin() as conn:
        if drop_first:
            await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)


def build_phases(
    llm: Any,
    *,
    registry: MCPServerRegistry | None = None,
    memory_service: MemoryServicePort | None = None,
) -> dict[str, Phase]:
    """Build the four SRPA phase objects.

    Exposed rather than inlined because a product substituting one phase is a
    reasonable thing to want, and doing it here beats reimplementing the dict.
    """
    from agentic_core.engine.act import ActPhase
    from agentic_core.engine.plan import PlanPhase
    from agentic_core.engine.reason import ReasonPhase
    from agentic_core.engine.sense import SensePhase
    from agentic_core.mcp.adapters import MCPToolExecutor

    mcp_executor = MCPToolExecutor(registry) if registry else None
    return {
        "sense": SensePhase(memory_service=memory_service),
        "reason": ReasonPhase(llm_service=llm),
        "plan": PlanPhase(llm_service=llm),
        "act": ActPhase(llm_service=llm, mcp_executor=mcp_executor),
    }


async def create_agent(
    db: AsyncSession,
    *,
    name: str,
    goal: str,
    model_config: ModelConfig | None = None,
    description: str = "",
    system_prompt: str = "",
    pattern_config: dict[str, Any] | None = None,
    tool_config: dict[str, Any] | None = None,
    memory_config: dict[str, Any] | None = None,
    owner_id: uuid.UUID | None = None,
) -> Agent:
    """Persist an agent.

    *pattern_config* selects the execution pattern, e.g.
    ``{"execution_pattern": "single_agent_baseline"}``. ``None`` means the
    orchestrator's default, which is ``reason_act``.

    *owner_id* is stored verbatim and never interpreted — core has no users.
    """
    agent = Agent(
        name=name,
        goal=goal,
        description=description,
        system_prompt=system_prompt or f"You are {name}. {description}".strip(),
        model_config_data=(model_config or ModelConfig()).model_dump(mode="json"),
        tool_config_data=tool_config or {},
        memory_config_data=memory_config or {},
        pattern_config=pattern_config,
        owner_id=owner_id,
    )
    db.add(agent)
    await db.commit()
    await db.refresh(agent)
    return agent


async def create_run(
    db: AsyncSession,
    *,
    agent_id: uuid.UUID,
    user_input: str,
    pattern_overrides: dict[str, Any] | None = None,
    owner_id: uuid.UUID | None = None,
) -> AgentRun:
    """Create a pending run. Execution is a separate call, so a product is free
    to enqueue it rather than run it inline."""
    run = AgentRun(
        agent_id=agent_id,
        input=user_input,
        status=RunStatus.PENDING,
        pattern_overrides=pattern_overrides,
        owner_id=owner_id,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    return run


async def execute_run(
    db: AsyncSession,
    *,
    run_id: uuid.UUID,
    registry: MCPServerRegistry | None = None,
    memory_service: MemoryServicePort | None = None,
    available_tools: list[dict[str, Any]] | None = None,
    cancel_event: asyncio.Event | None = None,
    pause_event: asyncio.Event | None = None,
    publish_event: Any = None,
    principal_id: uuid.UUID | None = None,
    llm_service: Any = None,
) -> AgentRunResult:
    """Execute a run to completion and record the outcome on the run row.

    Assembles the loop, wraps it in the pattern the agent selected, runs it, then
    writes status, output, token counts and cost back to the ``AgentRun``.

    The execution pattern is resolved as: the run's ``pattern_overrides``, else
    the agent's ``pattern_config``, else ``reason_act`` — except that a model
    which cannot accept function schemas falls back to ``single_agent_baseline``,
    since ReAct on such a model could only fail.

    *llm_service* substitutes the LLM bridge. It exists so the loop can be
    exercised without a provider — four methods (``complete``, ``complete_text``,
    ``complete_with_tools``, ``select_tool``) plus a ``principal_id`` property are
    the whole surface the phases use.
    """
    from agentic_core.memory.working import WorkingMemoryConfig
    from agentic_core.services.llm_service import LLMService, model_supports_tool_calling

    run = (
        await db.execute(
            select(AgentRun).options(selectinload(AgentRun.agent)).where(AgentRun.id == run_id)
        )
    ).scalar_one()
    agent = run.agent

    model_config = ModelConfig(**(agent.model_config_data or {}))
    config = AgentConfig(
        agent_id=agent.id,
        goal=agent.goal,
        system_prompt=agent.system_prompt or f"You are {agent.name}. {agent.description}",
        model_config=model_config,
        memory_config_data=agent.memory_config_data or {},
    )

    llm = llm_service or LLMService(principal_id=principal_id or run.owner_id)
    runtime = AgentRuntime(
        config=config,
        phases=build_phases(llm, registry=registry, memory_service=memory_service),
        db=db,
        memory_service=memory_service,
        working_memory_config=WorkingMemoryConfig(),
        llm_service=llm,
        cancel_event=cancel_event,
        pause_event=pause_event,
        publish_event=publish_event,
    )

    default_pattern = (
        DEFAULT_EXECUTION_PATTERN if model_supports_tool_calling(model_config) else "single_agent_baseline"
    )
    orchestrator = PatternOrchestrator.from_pattern_config(
        runtime,
        run.pattern_overrides or agent.pattern_config,
        default_execution_pattern=default_pattern,
    )

    result = await orchestrator.run(
        run_id=run.id,
        user_input=run.input,
        available_tools=available_tools,
    )

    run.status = RunStatus(result.status)
    run.output = result.output
    run.error = result.error
    run.token_usage = {
        "prompt_tokens": result.total_prompt_tokens,
        "completion_tokens": result.total_completion_tokens,
    }
    run.cost_estimate = result.total_cost

    if run.status == RunStatus.PAUSED:
        # Keep the snapshot so the run can be resumed; a paused run is not over.
        run.state_snapshot = result.context_snapshot
    else:
        run.completed_at = datetime.now(tz=UTC)
        run.state_snapshot = None

    await db.commit()
    await db.refresh(run)
    return result

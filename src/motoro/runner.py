"""Core's public API. Core owns its database; a product never touches it.

Every function here manages its own session against core's own database. A product
does not open a session, does not import core's models, and does not know core's
schema — it calls these functions and gets plain objects back.

**Two databases.** Core's tables live in core's database (``CoreSettings.database_url``);
a product's tables live in the product's own, on its own engine and its own
migration chain. Neither reaches into the other. That is what makes core's schema
genuinely core's — it can change without coordinating a product migration, and no
product query can read a core table it was not meant to.

Consequences worth knowing, because they are not free:

* **No cross-database foreign keys.** A product table that references a run or an
  agent stores an opaque ``UUID`` and enforces the relationship itself. Postgres
  cannot check it. (In ARES this affects 5 columns across 3 tables: plan records,
  discovery records, and experiment runs.)
* **No cross-database joins.** "List the runs for my experiment" becomes: query
  the product database for the run ids, then :func:`get_runs` for those ids.
* **No cross-database transaction.** Creating a product row and a core run is two
  commits, so a partial failure is possible — make the product side idempotent or
  reconcilable rather than assuming atomicity.

Because the product cannot query core's tables, core has to expose a read API for
anything a product legitimately needs. That is the trade: a narrower blast radius
in exchange for core owning a service surface rather than a schema.

    from motoro.runner import create_agent, create_run, execute_run

    agent = await create_agent(name="researcher", goal="...", model_config=cfg)
    run = await create_run(agent_id=agent.id, user_input="...")
    result = await execute_run(run_id=run.id)

Deliberately absent, because they are product concerns: authentication, ownership
enforcement, cost budgets, score extraction, event streaming. Seams exist —
``owner_id``, ``cancel_event``, ``publish_event`` — but core neither populates nor
interprets them.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from motoro.models.agent import Agent
from motoro.models.run import AgentRun, RunStatus, RunStep
from motoro.schemas.agent import ModelConfig

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Sequence
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession

    from motoro.engine.phase import Phase
    from motoro.engine.ports import MemoryServicePort
    from motoro.engine.runtime import AgentRunResult
    from motoro.mcp.registry import MCPServerRegistry


# --------------------------------------------------------------------------- #
#  Schema                                                                      #
# --------------------------------------------------------------------------- #


async def init_schema(*, drop_first: bool = False) -> None:
    """Create core's tables directly from the ORM metadata. **Tests only.**

    The production path is core's migration chain — see
    ``python -m motoro.migrations upgrade``, run as a deploy step. This
    issues ``CREATE TABLE`` with no version tracking, which is fine for a test
    database and wrong for anything you intend to migrate later. A test asserts
    both paths produce an identical schema.
    """
    from sqlalchemy import text

    import motoro.models.agent  # noqa: F401  PLC0415
    import motoro.models.mcp_server  # noqa: F401  PLC0415
    import motoro.models.memory  # noqa: F401  PLC0415
    import motoro.models.pattern  # noqa: F401  PLC0415
    import motoro.models.pricing  # noqa: F401  PLC0415
    import motoro.models.run  # noqa: F401  PLC0415
    import motoro.models.skill  # noqa: F401  PLC0415
    from motoro.models.base import Base
    from motoro.models.database import get_engine

    engine = get_engine()
    async with engine.begin() as conn:
        if drop_first:
            await conn.run_sync(Base.metadata.drop_all)
        # Base.metadata.create_all issues CREATE TABLE only — it does not know
        # about the pgvector extension memory_entries.embedding depends on, so
        # this mirrors the migration chain's own `CREATE EXTENSION` step.
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


# --------------------------------------------------------------------------- #
#  Internals                                                                   #
# --------------------------------------------------------------------------- #


def _session(reason: str) -> AbstractAsyncContextManager[AsyncSession]:
    """A session against core's own database.

    Private on purpose: sessions are core's to manage. A product that finds
    itself wanting one is reaching past the API, and the right fix is a function
    here rather than a session handed out.
    """
    from motoro.models.database import system_session

    return system_session(reason=f"motoro.runner: {reason}")


def build_phases(
    llm: Any,
    *,
    registry: MCPServerRegistry | None = None,
    memory_service: MemoryServicePort | None = None,
) -> dict[str, Phase]:
    """Build the four SRPA phase objects.

    Exposed because substituting one phase is a reasonable thing to want, and
    doing it here beats reimplementing the dict.
    """
    from motoro.engine.act import ActPhase
    from motoro.engine.plan import PlanPhase
    from motoro.engine.reason import ReasonPhase
    from motoro.engine.sense import SensePhase
    from motoro.mcp.adapters import MCPToolExecutor

    mcp_executor = MCPToolExecutor(registry) if registry else None
    return {
        "sense": SensePhase(memory_service=memory_service),
        "reason": ReasonPhase(llm_service=llm),
        "plan": PlanPhase(llm_service=llm),
        "act": ActPhase(llm_service=llm, mcp_executor=mcp_executor),
    }


def _require_valid_pattern_config(pattern_config: dict[str, Any] | None) -> None:
    """Raise if *pattern_config* would not run, checked against the registry.

    Deliberately not a database lookup. ARES validates against the
    ``architectural_patterns`` table, which means an unseeded table rejects every
    agent — validation that depends on a data migration having run. The registry
    is loaded in this process and cannot be out of date with the code.
    """
    from motoro.engine.patterns.catalog import PatternConfigError, validate_pattern_config

    result = validate_pattern_config(pattern_config)
    if not result.valid:
        messages = [f"{e.field}: {e.message}" for e in result.errors]
        raise PatternConfigError("Invalid pattern_config: " + "; ".join(messages), messages)


# --------------------------------------------------------------------------- #
#  Agents                                                                      #
# --------------------------------------------------------------------------- #


async def create_agent(
    *,
    name: str,
    goal: str,
    model_config: ModelConfig | None = None,
    description: str = "",
    system_prompt: str = "",
    pattern_config: dict[str, Any] | None = None,
    tool_config: dict[str, Any] | None = None,
    memory_config: dict[str, Any] | None = None,
    skill_config: dict[str, Any] | None = None,
    output_contract: dict[str, Any] | None = None,
    budget_limit_usd: float | None = None,
    max_run_duration_seconds: int | None = None,
    owner_id: uuid.UUID | None = None,
) -> Agent:
    """Persist an agent. Names are unique per owner over live rows (case-insensitive).

    *pattern_config* selects the execution pattern, e.g.
    ``{"execution_pattern": "single_agent_baseline"}``. ``None`` means the
    orchestrator's default, ``reason_act``. It is validated against the pattern
    registry and raises :class:`~motoro.engine.patterns.catalog.
    PatternConfigError` if it names an unregistered pattern, misconfigures one, or
    leaves a dependency unsatisfiable — so a typo fails here rather than after a
    run has been created and a model billed.

    *skill_config* attaches registered Agent Skills, as
    ``{"skill_ids": [...]}`` — see :mod:`motoro.services.skill_service`. The ids
    are not validated here: a skill can be deleted long after an agent
    references it, so resolution is deliberately tolerant at run time (a missing
    skill is skipped and logged) rather than strict at create time, which would
    only move the same failure somewhere less useful.

    *output_contract*, given, makes :func:`execute_run` run one extraction pass
    per completed run coercing its free-text output into the contracted fields
    (see :mod:`motoro.services.output_contract`) — exposed on the run's
    ``output`` as an envelope's ``payload``.

    *owner_id* is stored verbatim and never interpreted — core has no users, and
    with a separate product database it could not have a foreign key to one.

    Do not put a credential on *model_config*: ``api_key`` is ``exclude=True``, so
    it would not survive being persisted here and rebuilt at run time. Credentials
    resolve per call — see :mod:`motoro.services.credentials`.
    """
    _require_valid_pattern_config(pattern_config)
    agent = Agent(
        name=name,
        goal=goal,
        description=description,
        system_prompt=system_prompt or f"You are {name}. {description}".strip(),
        model_config_data=(model_config or ModelConfig()).model_dump(mode="json"),
        tool_config_data=tool_config or {},
        memory_config_data=memory_config or {},
        pattern_config=pattern_config,
        skill_config=skill_config,
        output_contract=output_contract,
        budget_limit_usd=budget_limit_usd,
        max_run_duration_seconds=max_run_duration_seconds,
        owner_id=owner_id,
    )
    async with _session("create_agent") as db:
        db.add(agent)
        await db.commit()
        await db.refresh(agent)
    return agent


async def get_agent(agent_id: uuid.UUID) -> Agent | None:
    """Fetch an agent by id, or ``None``."""
    async with _session("get_agent") as db:
        return (
            await db.execute(select(Agent).where(Agent.id == agent_id, Agent.deleted_at.is_(None)))
        ).scalar_one_or_none()


async def get_agent_by_name(name: str, *, owner_id: uuid.UUID | None = None) -> Agent | None:
    """Fetch a live agent by name, case-insensitively, or ``None``.

    Names are unique per owner (``uq_agents_owner_name_active``), not per
    installation, so *owner_id* should be passed whenever the caller means
    "does *this owner* already have an agent named X" — e.g. a conflict
    pre-check before create. Omitting it matches any owner's row, for
    call sites that only need a single representative agent by name."""
    stmt = select(Agent).where(func.lower(Agent.name) == name.lower(), Agent.deleted_at.is_(None))
    if owner_id is not None:
        stmt = stmt.where(Agent.owner_id == owner_id)
    async with _session("get_agent_by_name") as db:
        return (await db.execute(stmt)).scalar_one_or_none()


async def list_agents(*, owner_id: uuid.UUID | None = None, limit: int = 100) -> Sequence[Agent]:
    """List live agents, newest first. *owner_id* filters on the opaque tag."""
    stmt = select(Agent).where(Agent.deleted_at.is_(None))
    if owner_id is not None:
        stmt = stmt.where(Agent.owner_id == owner_id)
    stmt = stmt.order_by(Agent.created_at.desc()).limit(limit)
    async with _session("list_agents") as db:
        return (await db.execute(stmt)).scalars().all()


async def update_agent(
    agent_id: uuid.UUID,
    *,
    name: str | None = None,
    goal: str | None = None,
    description: str | None = None,
    system_prompt: str | None = None,
    model_config: ModelConfig | None = None,
    pattern_config: dict[str, Any] | None = None,
    tool_config: dict[str, Any] | None = None,
    memory_config: dict[str, Any] | None = None,
    skill_config: dict[str, Any] | None = None,
    output_contract: dict[str, Any] | None = None,
    budget_limit_usd: float | None = None,
    max_run_duration_seconds: int | None = None,
) -> Agent | None:
    """Update a live agent's fields. ``None`` means "leave unchanged" for every
    parameter here, the same convention ``services.mcp_service.update_server``
    uses — there is no way to explicitly clear a field back to empty through
    this function, only to leave it as it was.

    *pattern_config*, given, is validated the same way :func:`create_agent`
    validates it — a typo here still fails before a run is created and a model
    billed, not after.

    Returns ``None`` if *agent_id* does not name a live (non-deleted) agent.
    """
    if pattern_config is not None:
        _require_valid_pattern_config(pattern_config)

    async with _session("update_agent") as db:
        agent = (
            await db.execute(select(Agent).where(Agent.id == agent_id, Agent.deleted_at.is_(None)))
        ).scalar_one_or_none()
        if agent is None:
            return None

        if name is not None:
            agent.name = name
        if goal is not None:
            agent.goal = goal
        if description is not None:
            agent.description = description
        if system_prompt is not None:
            agent.system_prompt = system_prompt
        if model_config is not None:
            agent.model_config_data = model_config.model_dump(mode="json")
        if pattern_config is not None:
            agent.pattern_config = pattern_config
        if tool_config is not None:
            agent.tool_config_data = tool_config
        if memory_config is not None:
            agent.memory_config_data = memory_config
        if skill_config is not None:
            # ``{"skill_ids": []}`` is how a caller detaches every skill — the
            # None-means-unchanged convention above leaves no other way to say
            # "none", and an empty list is unambiguous.
            agent.skill_config = skill_config
        if output_contract is not None:
            agent.output_contract = output_contract
        if budget_limit_usd is not None:
            agent.budget_limit_usd = budget_limit_usd
        if max_run_duration_seconds is not None:
            agent.max_run_duration_seconds = max_run_duration_seconds

        await db.commit()
        await db.refresh(agent)
        return agent


# --------------------------------------------------------------------------- #
#  Runs                                                                        #
# --------------------------------------------------------------------------- #


async def create_run(
    *,
    agent_id: uuid.UUID,
    user_input: str,
    pattern_overrides: dict[str, Any] | None = None,
    owner_id: uuid.UUID | None = None,
    metadata: dict[str, Any] | None = None,
    model_config_overrides: dict[str, Any] | None = None,
) -> AgentRun:
    """Create a pending run.

    Separate from :func:`execute_run` so a product can enqueue the execution
    rather than block on it: create the run in a request, return the id, let a
    worker execute it.

    *metadata* seeds ``run_metadata`` before the run ever starts. Two keys the
    engine itself reads back out, both feeding the same ambient ``_meta``
    channel that spares a tool from arguments a model could omit or corrupt:

    - ``workspace_id`` — lifted into ``RunContext.workspace_id`` (see
      :func:`execute_run`) and sent by ``mcp.adapters`` as ``motoro.workspace_id``.
    - ``ambient_meta`` — a dict of the caller's *own* references (a dataset
      name, an artifact path), lifted into ``RunContext.ambient_meta`` and sent
      key-by-key under ``motoro.ambient.``. Use this rather than asking for a
      new first-class field: core stays out of your vocabulary, and you do not
      need a core release to bind a new kind of id.

    Everything else here is carried, not interpreted.

    *model_config_overrides* is a partial dict shallow-merged onto the agent's
    own ``model_config_data`` at execute time (see :func:`execute_run`) — e.g.
    ``{"model": "claude-opus-5", "effort": "xhigh"}`` to vary the model/effort
    per run without touching the agent's stored configuration. The column this
    writes (``AgentRun.model_config_overrides``) already existed; nothing
    previously set it or read it back.
    """
    run = AgentRun(
        agent_id=agent_id,
        input=user_input,
        status=RunStatus.PENDING,
        pattern_overrides=pattern_overrides,
        owner_id=owner_id,
        run_metadata=metadata,
        model_config_overrides=model_config_overrides,
    )
    async with _session("create_run") as db:
        db.add(run)
        await db.commit()
        await db.refresh(run)
    return run


async def get_run(run_id: uuid.UUID) -> AgentRun | None:
    """Fetch a run by id, or ``None``."""
    async with _session("get_run") as db:
        return (await db.execute(select(AgentRun).where(AgentRun.id == run_id))).scalar_one_or_none()


async def get_runs(run_ids: Sequence[uuid.UUID]) -> Sequence[AgentRun]:
    """Fetch many runs by id.

    This is the replacement for a join. A product that keeps its own table of run
    ids — an experiment's members, a batch's contents — collects the ids from its
    own database and passes them here, rather than joining across databases,
    which is not possible.
    """
    if not run_ids:
        return []
    async with _session("get_runs") as db:
        return (await db.execute(select(AgentRun).where(AgentRun.id.in_(list(run_ids))))).scalars().all()


async def list_runs(
    *,
    agent_id: uuid.UUID | None = None,
    status: RunStatus | None = None,
    owner_id: uuid.UUID | None = None,
    metadata: dict[str, Any] | None = None,
    limit: int = 100,
) -> Sequence[AgentRun]:
    """List runs, newest first, optionally filtered.

    ``metadata``, if given, requires every key/value pair to match exactly
    against ``run_metadata`` (``run_metadata->>key = value``, cast to text —
    ``run_metadata`` is a plain, product-opaque JSON blob core never
    interprets, so this is a generic equality filter, not anything
    schema-aware). This is the server-side alternative to a product pulling
    *every* run for an agent and filtering client-side in Python: that
    approach silently truncates once the agent has more runs than ``limit``
    (a product filtering its own runs by e.g. an experiment id it stamped
    into metadata has no way to know it happened, since nothing errors —
    the result set is just quietly incomplete). Filtering here means
    ``limit`` only has to be large enough for what actually matches, not for
    the agent's entire history.
    """
    stmt = select(AgentRun)
    if agent_id is not None:
        stmt = stmt.where(AgentRun.agent_id == agent_id)
    if status is not None:
        stmt = stmt.where(AgentRun.status == status)
    if owner_id is not None:
        stmt = stmt.where(AgentRun.owner_id == owner_id)
    if metadata:
        for key, value in metadata.items():
            stmt = stmt.where(AgentRun.run_metadata[key].astext == str(value))
    stmt = stmt.order_by(AgentRun.created_at.desc()).limit(limit)
    async with _session("list_runs") as db:
        return (await db.execute(stmt)).scalars().all()


async def get_run_steps(run_id: uuid.UUID) -> Sequence[RunStep]:
    """The ordered SRPA steps of a run — the execution trace."""
    async with _session("get_run_steps") as db:
        return (
            (await db.execute(select(RunStep).where(RunStep.run_id == run_id).order_by(RunStep.sequence)))
            .scalars()
            .all()
        )


# --------------------------------------------------------------------------- #
#  Execution                                                                   #
# --------------------------------------------------------------------------- #


async def execute_run(
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
    writes status, output, token counts and cost back to the run. Holds one
    session against core's database for the duration — the runtime writes a
    ``RunStep`` per phase as it goes.

    The execution pattern is: the run's ``pattern_overrides``, else the agent's
    ``pattern_config``, else ``reason_act`` — except that a model which cannot
    accept function schemas falls back to ``single_agent_baseline``, since ReAct
    on such a model could only fail.

    *llm_service* substitutes the LLM bridge, so the loop can be exercised without
    a provider. Four methods (``complete``, ``complete_text``,
    ``complete_with_tools``, ``select_tool``) plus a ``principal_id`` property are
    the whole surface the phases use.

    ``run.run_metadata`` — whatever :func:`create_run` was given as *metadata* —
    is passed to the orchestrator, which lifts the ``workspace_id`` and
    ``ambient_meta`` keys out of it onto every MCP tool call's ambient ``_meta``
    (see :func:`create_run`). Not otherwise interpreted here;
    it is not written back afterward, so it still holds exactly what was seeded
    at creation once this returns.

    ``run.model_config_overrides`` is shallow-merged onto ``agent.model_config_data``
    before the effective :class:`ModelConfig` is built — a per-run key wins over
    the agent's own stored value for that key; anything the run doesn't override
    still comes from the agent.
    """
    from motoro.engine.patterns.orchestrator import DEFAULT_EXECUTION_PATTERN, PatternOrchestrator
    from motoro.engine.runtime import AgentConfig, AgentRuntime
    from motoro.memory.working import WorkingMemoryConfig
    from motoro.services.llm_service import LLMService, model_supports_tool_calling
    from motoro.services.skill_service import resolve_skills

    async with _session("execute_run") as db:
        run = (
            await db.execute(select(AgentRun).options(selectinload(AgentRun.agent)).where(AgentRun.id == run_id))
        ).scalar_one()
        agent = run.agent

        effective_model_config_data = {**(agent.model_config_data or {}), **(run.model_config_overrides or {})}
        model_config = ModelConfig(**effective_model_config_data)
        # Resolved here, once, rather than inside the engine: the engine never
        # reaches for a table, and a run then carries the skill *text* it
        # started with, so editing a skill mid-run cannot change the
        # instructions the agent is halfway through following.
        skills = await resolve_skills(agent.skill_config, owner_id=agent.owner_id, db=db)
        config = AgentConfig(
            agent_id=agent.id,
            name=agent.name,
            goal=agent.goal,
            system_prompt=agent.system_prompt or f"You are {agent.name}. {agent.description}",
            model_config=model_config,
            memory_config_data=agent.memory_config_data or {},
            skills=skills,
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
            run_metadata=run.run_metadata,
        )

        from motoro.services.output_contract import finalize_output

        run.status = RunStatus(result.status)
        run.output = await finalize_output(
            llm=llm, agent=agent, model_config=model_config, result=result, principal_id=run.owner_id
        )
        result.output = run.output  # keep the returned result consistent with the persisted row
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
        return result


_TERMINAL_RUN_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED})


async def fail_run(run_id: uuid.UUID, *, error: str) -> AgentRun | None:
    """Force a non-terminal run to FAILED from outside :func:`execute_run`.

    The only way a run's status otherwise changes is :func:`execute_run`'s own
    commit, written by whichever process is holding the loop. That leaves no
    path to close out a run whose process died mid-flight (killed worker,
    crashed host) — a product that wants to detect and fail such a run (e.g. a
    stale-run sweep keyed on ``AgentRun.last_heartbeat_at``) needs a way in from
    outside. This is that way in.

    No-op, returning the run unchanged, if it is already terminal (COMPLETED,
    FAILED or CANCELLED) — so a detector racing against a slow-but-live
    :func:`execute_run` commit can't clobber a status it already wrote. PENDING,
    RUNNING, PAUSED and AWAITING_HUMAN are all still forceable; the caller
    decides which of those it considers stale.
    """
    async with _session("fail_run") as db:
        run = (await db.execute(select(AgentRun).where(AgentRun.id == run_id))).scalar_one_or_none()
        if run is None or run.status in _TERMINAL_RUN_STATUSES:
            return run
        run.status = RunStatus.FAILED
        run.error = error
        run.completed_at = datetime.now(tz=UTC)
        run.state_snapshot = None
        await db.commit()
        await db.refresh(run)
        return run


__all__ = [
    "build_phases",
    "create_agent",
    "create_run",
    "execute_run",
    "fail_run",
    "get_agent",
    "get_agent_by_name",
    "get_run",
    "get_run_steps",
    "get_runs",
    "init_schema",
    "list_agents",
    "list_runs",
    "update_agent",
]

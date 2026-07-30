"""Agent runtime — orchestrates the Sense → Reason → Plan → Act loop."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog
from opentelemetry import trace
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_core.engine.context import RunContext
from agentic_core.engine.phase import Phase, PhaseResult
from agentic_core.models.run import RunStatus, RunStep, StepPhase
from agentic_core.observability.metrics import record_error, record_phase, record_run
from agentic_core.observability.tracing import get_tracer
from agentic_core.schemas.agent import ModelConfig
from agentic_core.schemas.llm import ActOutput, PlanOutput

if TYPE_CHECKING:
    from agentic_core.engine.ports import MemoryServicePort
    from agentic_core.memory.working import WorkingMemoryConfig, WorkingMemoryManager
    from agentic_core.services.llm_service import LLMService

log = structlog.get_logger()
_tracer = get_tracer("runtime")
logger = logging.getLogger(__name__)

# Type alias for the optional event publisher callback
EventPublisher = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


@dataclass
class AgentRunResult:
    """Result of a complete agent run."""

    status: str
    output: str
    steps: list[RunStep] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0.0
    error: str | None = None
    context_snapshot: dict[str, Any] | None = None
    run_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Agent configuration extracted from the Agent model."""

    agent_id: uuid.UUID
    goal: str
    system_prompt: str
    model_config: ModelConfig
    max_iterations: int = 10
    memory_config_data: dict[str, Any] = field(default_factory=dict)


class AgentRuntime:
    """Orchestrates the agentic loop for a single run.

    Instantiated per run — not shared across runs.
    """

    def __init__(
        self,
        config: AgentConfig,
        phases: dict[str, Phase],
        db: AsyncSession,
        memory_service: MemoryServicePort | None = None,
        working_memory_config: WorkingMemoryConfig | None = None,
        llm_service: LLMService | None = None,
        cancel_event: asyncio.Event | None = None,
        pause_event: asyncio.Event | None = None,
        publish_event: EventPublisher | None = None,
    ) -> None:
        self._config = config
        self._phases = phases
        self._db = db
        self._memory_service = memory_service
        self._working_memory_config = working_memory_config
        self._llm_service = llm_service
        self._steps: list[RunStep] = []
        self._step_sequence = 0
        self._step_sequence_lock = asyncio.Lock()
        self._cancel_event = cancel_event
        self._pause_event = pause_event
        self._publish_event = publish_event

    async def next_step_sequence(self) -> int:
        """Atomically allocate the next step sequence number.

        Plugins must use this helper rather than reading/incrementing
        ``self._step_sequence`` directly — concurrent post-hooks (hitl_review,
        timeout_guard, reason_act, etc.) would otherwise race and produce
        duplicate sequence values on RunStep rows.
        """
        async with self._step_sequence_lock:
            seq = self._step_sequence
            self._step_sequence += 1
            return seq

    def record_pattern_step(self, step: RunStep) -> None:
        """Append a pattern-generated step to the runtime's tracking list.

        Plugins that build a ``RunStep`` and add it to the DB session via
        ``self._db.add(step)`` should call this so the step is also returned in
        the final ``AgentRunResult.steps``. The runtime itself records its own
        steps via ``_create_step``.
        """
        self._steps.append(step)

    @property
    def _episodic_memory_enabled(self) -> bool:
        cfg = self._config.memory_config_data or {}
        return bool(cfg.get("episodic_memory_enabled", False))

    def _is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancel_event is not None and self._cancel_event.is_set()

    def _is_paused(self) -> bool:
        """Check if pause has been requested."""
        return self._pause_event is not None and self._pause_event.is_set()

    async def _check_resource_limits(self, context: RunContext) -> str | None:
        """Check whether this agent has exceeded the limits set on it.

        Returns an error message if limits are exceeded, None otherwise.

        Core enforces nothing by default: the check is a hook a product installs
        with ``engine.ports.set_resource_limit_checker``, and with none installed
        this returns None. That is not a behaviour change — the ARES original
        wrapped the whole call in a try/except that also returned None whenever
        the agent was not a child or the DB was unavailable.
        """
        from agentic_core.engine.ports import get_resource_limit_checker

        check_resource_limits = get_resource_limit_checker()
        if check_resource_limits is None:
            return None
        try:
            total_tokens = context.total_prompt_tokens + context.total_completion_tokens
            tool_calls = sum(
                1
                for output in context.phase_outputs.values()
                if isinstance(output, ActOutput) and output.results
                for _r in output.results
            )

            result = await check_resource_limits(
                self._db,
                self._config.agent_id,
                token_usage=total_tokens,
                tool_calls=tool_calls,
            )

            if result.get("exceeded"):
                exceeded = [k for k, v in result.items() if k != "exceeded" and v is True]
                return f"Resource limits exceeded: {', '.join(exceeded)}"
        except Exception:
            logger.warning(
                "Resource limit check failed for agent %s",
                self._config.agent_id,
                exc_info=True,
            )
        return None

    async def _publish(self, event: dict[str, Any]) -> None:
        """Publish an event if a publisher is configured.

        Issue #985: swallowed errors were logged only at WARNING, which meant
        that repeated SSE failures went unnoticed and the UI stream silently
        died.  We now log at ERROR so the problem is visible in the structured
        logs.  The error is still not re-raised — a broken event stream must
        never abort a run that is otherwise making progress.
        """
        if self._publish_event is not None:
            try:
                await self._publish_event(event)
            except Exception:
                log.error(
                    "runtime.publish_failed",
                    event_type=event.get("type", "unknown"),
                    run_id=event.get("run_id"),
                    exc_info=True,
                    component="runtime",
                )

    def _check_interrupt(self) -> RunStatus | None:
        """Check for cancel or pause signals. Returns status if interrupted."""
        if self._is_cancelled():
            return RunStatus.CANCELLED
        if self._is_paused():
            return RunStatus.PAUSED
        return None

    async def run(
        self,
        run_id: uuid.UUID,
        user_input: str,
        available_tools: list[dict[str, Any]] | None = None,
        resume_context: RunContext | None = None,
        resume_phase: str | None = None,
    ) -> AgentRunResult:
        """Execute the full agentic loop.

        Args:
            run_id: ID of the AgentRun record (already created by the caller).
            user_input: The user's query or goal input.
            available_tools: MCP tools to make available to the agent.
            resume_context: If resuming, the deserialized RunContext.
            resume_phase: If resuming, the phase to resume from (e.g. "plan").

        Returns:
            AgentRunResult with status, output, steps, and token usage.
        """
        if resume_context is not None:
            context = resume_context
        else:
            context = RunContext(
                agent_goal=self._config.goal,
                system_prompt=self._config.system_prompt,
                model_config=self._config.model_config,
                user_input=user_input,
                max_iterations=self._config.max_iterations,
                available_tools=available_tools or [],
                agent_id=self._config.agent_id,
                run_id=run_id,
                owner_id=self._llm_service.principal_id if self._llm_service else None,
            )
        context.memory_config_data = self._config.memory_config_data or {}

        # Inject Redis-backed working memory if configured
        wm: WorkingMemoryManager | None = None
        if self._working_memory_config is not None:
            wm = await self._create_working_memory(run_id)
            context.working_memory_manager = wm

        final_output = ""
        error_msg: str | None = None
        context_snapshot: dict[str, Any] | None = None
        status = RunStatus.COMPLETED
        actions_taken: list[str] = []
        run_start = asyncio.get_event_loop().time()

        # Phase execution order and mapping
        phase_sequence = [
            ("sense", StepPhase.SENSE),
            ("reason", StepPhase.REASON),
            ("plan", StepPhase.PLAN),
            ("act", StepPhase.ACT),
        ]

        # Determine starting phase index for resume
        start_phase_idx = 0
        if resume_phase is not None:
            phase_names = [p[0] for p in phase_sequence]
            if resume_phase in phase_names:
                start_phase_idx = phase_names.index(resume_phase)

        with _tracer.start_as_current_span(
            "agent_run",
            attributes={"run_id": str(run_id), "agent_id": str(self._config.agent_id)},
        ) as run_span:
            try:
                start_iteration = context.iteration
                for iteration in range(start_iteration, context.max_iterations):
                    context.iteration = iteration

                    # Determine which phases to run this iteration
                    first_phase = start_phase_idx if iteration == start_iteration else 0

                    for phase_idx in range(first_phase, len(phase_sequence)):
                        phase_name, step_phase = phase_sequence[phase_idx]

                        # Check for interrupt before each phase
                        interrupt = self._check_interrupt()
                        if interrupt is not None:
                            status = interrupt
                            final_output = ""
                            if interrupt == RunStatus.PAUSED:
                                context_snapshot = context.to_snapshot()
                            break

                        # Check resource limits BEFORE the next phase fires
                        # (issue #701) so an already-exceeded budget doesn't
                        # incur another LLM call before being detected.
                        limit_error = await self._check_resource_limits(context)
                        if limit_error is not None:
                            status = RunStatus.FAILED
                            error_msg = limit_error
                            final_output = final_output or ""
                            break

                        result = await self._execute_phase(phase_name, step_phase, run_id, context)

                        # Phase-specific post-processing
                        if phase_name == "plan":
                            plan_output = result.output
                            if isinstance(plan_output, PlanOutput) and plan_output.is_complete:
                                final_output = plan_output.completion_reason or "Goal already satisfied."
                                status = RunStatus.COMPLETED
                                break
                            if isinstance(plan_output, PlanOutput):
                                # Issue #769: validate tools on a *copy* of the
                                # PlanOutput so the already-persisted DB step
                                # row is not mutated after commit.  The validated
                                # copy is stored back into phase_outputs so Act
                                # uses the cleaned version.
                                validated_plan = plan_output.model_copy(deep=True)
                                _validate_plan_tools(validated_plan, context.available_tools)
                                context.phase_outputs["plan"] = validated_plan
                                actions_taken.extend(s.action for s in validated_plan.steps)

                        elif phase_name == "act":
                            act_output = result.output
                            if isinstance(act_output, ActOutput):
                                # Store act results in working memory if present
                                if wm is not None:
                                    from agentic_core.memory.working import WorkingMemoryEntryType

                                    await wm.add_entry(
                                        content=act_output.final_response,
                                        entry_type=WorkingMemoryEntryType.TOOL_RESULT,
                                    )
                                final_output = act_output.final_response
                                if not act_output.should_continue:
                                    break
                            else:
                                final_output = str(act_output)
                                break
                    else:
                        # All phases in this iteration completed — continue to next
                        continue
                    # Inner loop broke — propagate
                    break
                else:
                    # Max iterations reached
                    final_output = final_output or "Max iterations reached."

            except Exception as e:
                status = RunStatus.FAILED
                error_msg = f"{type(e).__name__}: {e}"
                final_output = ""
                run_span.record_exception(e)
                run_span.set_status(trace.StatusCode.ERROR, description=error_msg)
                record_error(type(e).__name__, "run")

            run_span.set_attribute("status", status.value)
            run_span.set_attribute("total_cost", context.total_cost)
            run_span.set_attribute("prompt_tokens", context.total_prompt_tokens)
            run_span.set_attribute("completion_tokens", context.total_completion_tokens)

        record_run(
            agent_id=str(self._config.agent_id),
            status=status.value,
            duration_seconds=asyncio.get_event_loop().time() - run_start,
        )

        # Store episodic memory summary after run completes
        if self._memory_service is not None and self._episodic_memory_enabled:
            await self._store_episode(
                run_id=run_id,
                user_input=user_input,
                final_output=final_output,
                status=status,
                actions_taken=actions_taken,
                context=context,
            )

        # Set TTL on working memory after run completes
        if wm is not None:
            try:
                await asyncio.wait_for(wm.cleanup(), timeout=10)
            except TimeoutError:
                logger.error("Working memory cleanup timed out after 10s for run %s", run_id)
            except Exception:
                logger.exception("Failed to clean up working memory for run %s", run_id)

        if context.iteration > 0:
            context.metadata["loop_iterations"] = context.iteration

        return AgentRunResult(
            status=status.value,
            output=final_output,
            steps=self._steps,
            total_prompt_tokens=context.total_prompt_tokens,
            total_completion_tokens=context.total_completion_tokens,
            total_cost=context.total_cost,
            error=error_msg,
            context_snapshot=context_snapshot,
            run_metadata={
                k: v for k, v in context.metadata.items() if k != "_runtime" and isinstance(v, (str, int, float, bool))
            },
        )

    async def _create_working_memory(self, run_id: uuid.UUID) -> WorkingMemoryManager | None:
        """Create a WorkingMemoryManager for this run.

        Returns None if Redis is unavailable (graceful degradation).
        """
        from agentic_core.memory.working import WorkingMemoryManager
        from agentic_core.models.redis import get_redis

        assert self._working_memory_config is not None
        redis_client = get_redis()
        try:
            await redis_client.ping()
        except Exception:
            logger.warning("Redis unavailable — running without working memory")
            return None
        return WorkingMemoryManager(
            run_id=run_id,
            config=self._working_memory_config,
            redis_client=redis_client,
            llm_service=self._llm_service,
            model_config=self._config.model_config,
        )

    async def _store_episode(
        self,
        run_id: uuid.UUID,
        user_input: str,
        final_output: str,
        status: RunStatus,
        actions_taken: list[str],
        context: RunContext,
    ) -> None:
        """Generate and store an episodic memory entry for this run."""
        assert self._memory_service is not None
        try:
            try:
                summary = await self._memory_service.episodic.generate_run_summary(
                    user_input=user_input,
                    final_output=final_output,
                    model_config=context.model_config,
                    status=status.value,
                    actions_taken=actions_taken or None,
                )
            except Exception:
                logger.warning(
                    "Failed to generate LLM summary for run %s; falling back to raw output for episodic memory",
                    run_id,
                    exc_info=True,
                )
                summary = f"Run {status.value}. Goal: {user_input[:200]}. Output: {final_output[:500]}"
            if context.owner_id is None:
                # Memory is owned data (M113 #1471); with no acting user there is
                # nobody to own the entry, so drop it rather than guess.
                logger.warning("Skipping episodic memory for run %s: no acting user on the run", run_id)
                return
            await self._memory_service.episodic.store_run_summary(
                run_id=run_id,
                agent_id=self._config.agent_id,
                summary=summary,
                metadata={
                    "status": status.value,
                    "prompt_tokens": context.total_prompt_tokens,
                    "completion_tokens": context.total_completion_tokens,
                },
                created_by_id=context.owner_id,
            )
        except Exception:
            logger.exception("Failed to store episodic memory for run %s", run_id)

    async def _execute_phase(
        self,
        phase_name: str,
        step_phase: StepPhase,
        run_id: uuid.UUID,
        context: RunContext,
    ) -> PhaseResult:
        """Execute a single phase, record it as a RunStep, and update context."""
        phase = self._phases[phase_name]
        started_at = datetime.now(tz=UTC)

        phase_log = log.bind(
            run_id=str(run_id),
            agent_id=str(self._config.agent_id),
            phase=phase_name,
            iteration=context.iteration,
            component="runtime",
        )

        with _tracer.start_as_current_span(
            f"phase.{phase_name}",
            attributes={
                "phase": phase_name,
                "iteration": context.iteration,
                "run_id": str(run_id),
                "agent_id": str(self._config.agent_id),
            },
        ) as phase_span:
            phase_log.debug("phase.started")

            try:
                result = await phase.execute(context)
            except Exception as e:
                completed_at = datetime.now(tz=UTC)
                phase_span.record_exception(e)
                phase_span.set_status(trace.StatusCode.ERROR, description=str(e))
                phase_log.error("phase.failed", error=f"{type(e).__name__}: {e}")
                record_error(type(e).__name__, phase_name)
                # Issue #798: use the phase's own input (output of the previous
                # phase) as input_data for the error step, not some other
                # phase's output.  Also: the `await self._db.commit()` inside an
                # except block can itself raise (e.g., connection lost), which
                # would mask the original exception.  Wrap in a nested
                # try/except so the original error always propagates.
                error_input = _safe_dump(context.phase_outputs.get(prev) if (prev := _prev_phase(phase_name)) else None)
                step = self._create_step(
                    run_id=run_id,
                    phase=step_phase,
                    input_data=error_input,
                    output_data={"error": f"{type(e).__name__}: {e}"},
                    started_at=started_at,
                    completed_at=completed_at,
                    iteration=context.iteration,
                )
                self._db.add(step)
                self._steps.append(step)
                try:
                    await self._db.commit()
                except Exception:
                    phase_log.error(
                        "phase.error_step_commit_failed",
                        original_error=f"{type(e).__name__}: {e}",
                        component="runtime",
                    )
                raise

            completed_at = datetime.now(tz=UTC)
            duration_s = (completed_at - started_at).total_seconds()

            # Track LLM usage
            if result.llm_call:
                await context.add_llm_usage(
                    result.llm_call.prompt_tokens,
                    result.llm_call.completion_tokens,
                    result.llm_call.cost_estimate,
                )
                phase_span.set_attribute("prompt_tokens", result.llm_call.prompt_tokens)
                phase_span.set_attribute("completion_tokens", result.llm_call.completion_tokens)
                phase_span.set_attribute("cost_usd", result.llm_call.cost_estimate)

            record_phase(str(self._config.agent_id), phase_name, duration_s)

            phase_log.debug(
                "phase.completed",
                duration_s=round(duration_s, 3),
                prompt_tokens=result.llm_call.prompt_tokens if result.llm_call else 0,
                completion_tokens=result.llm_call.completion_tokens if result.llm_call else 0,
            )

            # Record phase output in context
            context.record_phase_output(phase_name, result.output)

            # Extract tool_call data from ActOutput if present
            tool_call_data: dict[str, Any] | None = None
            if isinstance(result.output, ActOutput):
                tool_calls = [r.tool_call.model_dump() for r in result.output.results if r.tool_call is not None]
                if len(tool_calls) == 1:
                    tool_call_data = tool_calls[0]
                elif tool_calls:
                    tool_call_data = {
                        "calls": tool_calls,
                        "success": all(c.get("success", False) for c in tool_calls),
                    }

            # Build step input — include retry metadata if this is a re-execution
            step_input = _safe_dump(context.phase_outputs.get(prev) if (prev := _prev_phase(phase_name)) else None)
            retry_count = context.metadata.get("_retry_count")
            if retry_count is not None:
                step_input = step_input or {}
                step_input["event"] = "retry"
                step_input["retry_attempt"] = retry_count

            step = self._create_step(
                run_id=run_id,
                phase=step_phase,
                input_data=step_input,
                output_data=_safe_dump(result.output),
                llm_call=result.llm_call.model_dump() if result.llm_call else None,
                iteration=context.iteration,
                tool_call=tool_call_data,
                started_at=started_at,
                completed_at=completed_at,
            )
            self._db.add(step)
            await self._db.commit()
            self._steps.append(step)

            # Publish phase completion event
            await self._publish(
                {
                    "type": "phase_completed",
                    "run_id": str(run_id),
                    "phase": phase_name,
                    "iteration": context.iteration,
                }
            )

            return result

    def _create_step(
        self,
        run_id: uuid.UUID,
        phase: StepPhase,
        input_data: Any,
        output_data: Any,
        started_at: datetime,
        completed_at: datetime,
        llm_call: dict[str, Any] | None = None,
        tool_call: dict[str, Any] | None = None,
        iteration: int | None = None,
    ) -> RunStep:
        """Create a RunStep record."""
        step = RunStep(
            run_id=run_id,
            sequence=self._step_sequence,
            iteration=iteration,
            phase=phase,
            input=input_data,
            output=output_data,
            llm_call=llm_call,
            tool_call=tool_call,
            started_at=started_at,
            completed_at=completed_at,
        )
        self._step_sequence += 1
        return step


def _safe_dump(obj: Any) -> Any:
    """Safely convert a Pydantic model or other object to a dict for JSON storage."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def _prev_phase(phase_name: str) -> str | None:
    """Get the previous phase name in the loop order."""
    order = ["sense", "reason", "plan", "act"]
    idx = order.index(phase_name)
    return order[idx - 1] if idx > 0 else None


def _validate_plan_tools(plan: PlanOutput, available_tools: list[dict[str, Any]]) -> None:
    """Validate that PlanStep.tool_name values match available tools.

    Invalid tool names are cleared to None so the Act phase falls back to LLM.
    """
    if not plan.steps:
        return

    # Build set of valid names: both namespaced ("server.tool") and bare ("tool")
    valid_names: set[str] = set()
    for t in available_tools:
        full_name = str(t.get("name", ""))
        bare_name = str(t.get("tool_name", ""))
        if full_name:
            valid_names.add(full_name)
        if bare_name:
            valid_names.add(bare_name)

    if not valid_names:
        for step in plan.steps:
            if step.tool_name:
                log.warning(
                    "plan.tool_invalid",
                    tool=step.tool_name,
                    reason="no tools available",
                )
                step.tool_name = None
                step.tool_args = None
        return

    for step in plan.steps:
        if step.tool_name and step.tool_name not in valid_names:
            log.warning(
                "plan.tool_invalid",
                tool=step.tool_name,
                available=sorted(valid_names),
            )
            step.tool_name = None
            step.tool_args = None

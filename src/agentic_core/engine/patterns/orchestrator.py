"""Pattern orchestrator — wraps AgentRuntime to execute hooks at phase boundaries."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from typing import Any

import structlog
from pydantic import BaseModel
from sqlalchemy import text

from agentic_core.config import settings
from agentic_core.engine.context import RunContext
from agentic_core.engine.patterns import catalog, composition
from agentic_core.engine.patterns.base import HookAction, HookCallable, HookPoint, PatternPlugin
from agentic_core.engine.patterns.registry import PluginRegistry
from agentic_core.engine.runtime import AgentRunResult, AgentRuntime
from agentic_core.models.pattern import PatternCategory
from agentic_core.models.run import RunStatus, StepPhase
from agentic_core.observability.metrics import record_hook_duration
from agentic_core.schemas.llm import ActOutput, PlanOutput, ReasonOutput, SenseOutput

log = structlog.get_logger()
logger = logging.getLogger(__name__)

# What an agent gets when it configures no execution pattern at all (M112).
#
# The tool-calling loop, not the baseline. With tools it is the only one of the
# two that can act on what a tool returned — the baseline plans every step up
# front and cannot revise. Without tools it binds only the ``final_answer``
# terminator and finishes on its first turn, which is *one* LLM call against
# the baseline's three (reason, plan, act) — measured at ~4.6x cheaper on an
# identical toolless prompt. So the loop is the better default in both
# directions, not a capability traded against cost.
#
# The one thing it needs that the baseline does not is provider-side function
# calling; callers that can see the model guard for it (see the factory below).
DEFAULT_EXECUTION_PATTERN = "reason_act"

# Phase name ↔ StepPhase mapping
_PHASE_SEQUENCE: list[tuple[str, StepPhase]] = [
    ("sense", StepPhase.SENSE),
    ("reason", StepPhase.REASON),
    ("plan", StepPhase.PLAN),
    ("act", StepPhase.ACT),
]


async def _update_heartbeat(db: Any, run_id: uuid.UUID) -> None:
    """Update run heartbeat with retry on DB disconnect."""
    from agentic_core.worker.resilience import db_retry

    @db_retry
    async def _do() -> None:
        await db.execute(
            text("UPDATE agent_runs SET last_heartbeat_at = now() WHERE id = :run_id"),
            {"run_id": str(run_id)},
        )
        await db.commit()

    await _do()


def _resolve_abort(context: RunContext, current_output: str) -> tuple[RunStatus, str, str | None]:
    """Determine run status and output when a hook returns ABORT.

    A hook that sets ``final_output_override`` before returning ABORT intends
    a successful early completion (e.g. routing patterns that delegate and
    finish).  The ``partial_result`` fallback strategy produces a COMPLETED
    run with whatever output was generated before the abort.  All other
    strategies produce a FAILED run.
    """
    import json

    override = context.metadata.get("final_output_override")
    if override is not None:
        output = json.dumps(override) if isinstance(override, dict) else str(override)
        return RunStatus.COMPLETED, output, None

    error_msg = context.metadata.get("abort_error", "Aborted by hook")
    abort_error = str(error_msg)

    if "partial results" in abort_error.lower():
        partial = current_output
        if not partial:
            # Assemble output from completed phases
            act_out = context.phase_outputs.get("act")
            if act_out and hasattr(act_out, "final_response"):
                partial = act_out.final_response
            else:
                reason_out = context.phase_outputs.get("reason")
                if reason_out and hasattr(reason_out, "strategy"):
                    partial = f"[Partial — completed through Reason phase] {reason_out.strategy}"
                else:
                    sense_out = context.phase_outputs.get("sense")
                    if sense_out:
                        partial = "[Partial — only Sense phase completed]"
                    else:
                        partial = "[Partial — no phases completed before timeout]"
        return RunStatus.COMPLETED, partial, abort_error

    return RunStatus.FAILED, current_output, abort_error


_PHASE_HOOK_MAP: dict[str, tuple[HookPoint, HookPoint]] = {
    "sense": (HookPoint.PRE_SENSE, HookPoint.POST_SENSE),
    "reason": (HookPoint.PRE_REASON, HookPoint.POST_REASON),
    "plan": (HookPoint.PRE_PLAN, HookPoint.POST_PLAN),
    "act": (HookPoint.PRE_ACT, HookPoint.POST_ACT),
}

# Expected output type per phase — used by hook return-type validation
_PHASE_OUTPUT_TYPE: dict[str, type[BaseModel]] = {
    "sense": SenseOutput,
    "reason": ReasonOutput,
    "plan": PlanOutput,
    "act": ActOutput,
}


class PatternOrchestrator:
    """Adds pattern hook execution around the existing AgentRuntime phase loop.

    Composes with an ``AgentRuntime`` instance — delegates phase execution to
    the runtime's ``_execute_phase`` and adds hook invocation before/after each
    phase, loop-control hooks between iterations, and lifecycle callbacks.

    When no plugins are loaded it behaves identically to the plain runtime.
    """

    # Abort after this many consecutive errors from the same hook.
    _MAX_CONSECUTIVE_HOOK_ERRORS = 3

    def __init__(
        self,
        runtime: AgentRuntime,
        plugins: list[PatternPlugin] | None = None,
        *,
        fail_on_hook_error: bool = False,
        fail_on_type_mismatch: bool = False,
        hook_timeout: float = 300.0,
    ) -> None:
        self._runtime = runtime
        self._plugins = plugins or []
        self._fail_on_hook_error = fail_on_hook_error
        # #1047: when True, a hook that returns a wrong-type value (a BaseModel
        # that does not match the expected phase output, or any non-None /
        # non-HookAction object) aborts the run instead of being silently
        # dropped with a log line.
        self._fail_on_type_mismatch = fail_on_type_mismatch
        self._hook_timeout = hook_timeout

        # #1465: hooks belonging to an execution pattern are load-bearing. That
        # plugin owns the phase sequence — when its hook raises, the loop has
        # not half-run, it has not run, and continuing only defers the failure
        # to some downstream phase that reports a symptom instead of a cause.
        # Every other category stays advisory: a knowledge or quality hook that
        # dies should not take a working run down with it.
        self._load_bearing_slugs: set[str] = {p.slug for p in self._plugins if str(p.category) == "execution"}

        # Build consolidated hook pipeline: HookPoint → ordered list of (slug, callable)
        self._hooks: dict[HookPoint, list[tuple[str, HookCallable]]] = {}
        for plugin in self._plugins:
            for point, callables in plugin.get_hooks().items():
                self._hooks.setdefault(point, [])
                for cb in callables:
                    self._hooks[point].append((plugin.slug, cb))

        # Track consecutive errors per (slug, hook_point) to detect infinite
        # error loops and abort instead of silently retrying forever.
        self._consecutive_hook_errors: dict[tuple[str, HookPoint], int] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        run_id: uuid.UUID,
        user_input: str,
        available_tools: list[dict[str, Any]] | None = None,
        resume_context: RunContext | None = None,
        resume_phase: str | None = None,
        run_metadata: dict[str, Any] | None = None,
    ) -> AgentRunResult:
        """Execute the agentic loop with hook execution at phase boundaries.

        Signature-compatible with ``AgentRuntime.run`` so callers can
        substitute one for the other.
        """
        rt = self._runtime

        # --- Context setup (mirrors runtime logic) -------------------------
        if resume_context is not None:
            context = resume_context
        else:
            context = RunContext(
                agent_goal=rt._config.goal,
                system_prompt=rt._config.system_prompt,
                model_config=rt._config.model_config,
                user_input=user_input,
                max_iterations=rt._config.max_iterations,
                available_tools=available_tools or [],
                agent_id=rt._config.agent_id,
                run_id=run_id,
                owner_id=rt._llm_service.principal_id if rt._llm_service else None,
            )
            # The mirror above was missing this line: RunContext.memory_config_data
            # defaults to {} (engine/context.py), so without it every pattern-driven
            # run — which is every real run, since execute_run always wraps the
            # loop in a PatternOrchestrator — read episodic_memory_enabled as False
            # in Sense regardless of the agent's actual config. Storage still
            # worked (AgentRuntime._episodic_memory_enabled reads rt._config
            # directly, not context), so memory was written but never recalled.
            context.memory_config_data = rt._config.memory_config_data or {}

        # Working memory (returns None if Redis unavailable)
        wm = None
        if rt._working_memory_config is not None:
            wm = await rt._create_working_memory(run_id)
            context.working_memory_manager = wm

        # Seed context with run-level metadata from the API request
        if run_metadata:
            context.metadata.update(run_metadata)
            # Lift ambient workspace identity out of run_metadata into a first-class
            # field (issue #1455) so the MCP executor can inject it into request
            # _meta. Guarded so a resume (workspace_id already restored from the
            # snapshot) is not clobbered by a stale/absent metadata value.
            if context.workspace_id is None:
                ws = run_metadata.get("workspace_id")
                if isinstance(ws, str) and ws:
                    context.workspace_id = ws

        # Expose runtime to plugins that need LLM access (e.g., ReasonAct)
        context.metadata["_runtime"] = rt

        # --- Activate plugins -----------------------------------------------
        # Issue #809: track which plugins were successfully activated so that
        # on_deactivate and on_error are only called for those.
        activated_plugins: list[PatternPlugin] = []
        try:
            for plugin in self._plugins:
                await plugin.on_activate(context)
                activated_plugins.append(plugin)
        except Exception as activate_exc:
            _activate_err = f"{type(activate_exc).__name__}: {activate_exc}"
            context.metadata["error"] = _activate_err
            _activated_slugs = {p.slug for p in activated_plugins}
            for _slug, hook_fn in self._hooks.get(HookPoint.ON_ERROR, []):
                if _slug not in _activated_slugs:
                    continue
                with contextlib.suppress(Exception):
                    await hook_fn(context, None)
            for _p in activated_plugins:
                try:
                    await _p.on_deactivate(context)
                except Exception:
                    logger.exception("plugin.deactivate_error", extra={"slug": _p.slug})
            return AgentRunResult(
                status=RunStatus.FAILED.value,
                output="",
                steps=rt._steps,
                total_prompt_tokens=context.total_prompt_tokens,
                total_completion_tokens=context.total_completion_tokens,
                total_cost=context.total_cost,
                error=_activate_err,
                context_snapshot=None,
            )

        final_output = ""
        error_msg: str | None = None
        context_snapshot: dict[str, Any] | None = None
        status = RunStatus.COMPLETED
        actions_taken: list[str] = []
        run_start = asyncio.get_event_loop().time()

        start_phase_idx = 0
        if resume_phase is not None:
            phase_names = [p[0] for p in _PHASE_SEQUENCE]
            if resume_phase in phase_names:
                start_phase_idx = phase_names.index(resume_phase)

        try:
            start_iteration = context.iteration
            # Issue #976: if resumed at or past max_iterations, exit immediately.
            if start_iteration >= context.max_iterations:
                log.warning(
                    "orchestrator.resume_iteration_at_limit",
                    start_iteration=start_iteration,
                    max_iterations=context.max_iterations,
                    run_id=str(run_id),
                )
                final_output = "Max iterations reached on resume."
                status = RunStatus.COMPLETED
            else:
                for iteration in range(start_iteration, context.max_iterations):
                    context.iteration = iteration

                    # Consult loop_control hooks for custom phase ordering
                    phase_sequence = await self._get_phase_sequence(context)

                    # An empty phase sequence signals that the run is done
                    # (e.g., ReasonAct produced a final answer on the previous iteration).
                    # Issue #789: if this is the first iteration and there's no
                    # final_output_override, the empty sequence is suspicious — raise.
                    if not phase_sequence:
                        override = context.metadata.get("final_output_override")
                        if override is not None:
                            final_output = str(override)
                            break
                        if iteration == start_iteration:
                            raise RuntimeError(
                                "empty phase_sequence on the first iteration with no "
                                "final_output_override — loop_control hook misconfigured"
                            )
                        # Subsequent iterations: normal ReasonAct-style exit.
                        break

                    first_phase = start_phase_idx if iteration == start_iteration else 0
                    iteration_break = False
                    phases_this_iteration = {name for name, _ in phase_sequence}

                    for phase_idx in range(first_phase, len(phase_sequence)):
                        phase_name, step_phase = phase_sequence[phase_idx]
                        context.current_phase = phase_name

                        # --- Heartbeat ---
                        # Issue #747: log at WARNING so repeated failures are visible.
                        try:
                            await _update_heartbeat(rt._db, run_id)
                            context.metadata.pop("_heartbeat_failures", None)
                        except Exception:
                            failures = context.metadata.get("_heartbeat_failures", 0) + 1
                            context.metadata["_heartbeat_failures"] = failures
                            logger.warning(
                                "Heartbeat update failed for run %s (consecutive failures: %d)",
                                run_id,
                                failures,
                                exc_info=True,
                            )

                        # --- Interrupt check ---
                        interrupt = rt._check_interrupt()
                        if interrupt is not None:
                            status = interrupt
                            final_output = ""
                            if interrupt == RunStatus.PAUSED:
                                context_snapshot = context.to_snapshot()
                            iteration_break = True
                            break

                        # --- Pre-phase hooks ---
                        pre_hook, post_hook = _PHASE_HOOK_MAP.get(phase_name, (None, None))
                        if pre_hook is not None:
                            action = await self._run_hooks(pre_hook, context, None)
                            if action == HookAction.SKIP_PHASE:
                                continue
                            if action == HookAction.ABORT:
                                status, final_output, error_msg = _resolve_abort(context, final_output)
                                iteration_break = True
                                break
                            if action == HookAction.PAUSE:
                                status = RunStatus.PAUSED
                                context_snapshot = context.to_snapshot()
                                iteration_break = True
                                break

                        # --- Execute the phase via the underlying runtime -------
                        result = await rt._execute_phase(phase_name, step_phase, run_id, context)

                        # --- Post-phase hooks ---
                        if post_hook is not None:
                            action = await self._run_hooks(post_hook, context, result.output)
                            if action == HookAction.RETRY_PHASE:
                                retry_count = context.metadata.get("_retry_count", 0) + 1
                                context.metadata["_retry_count"] = retry_count
                                context.metadata["_retry_phase"] = phase_name
                                result = await rt._execute_phase(phase_name, step_phase, run_id, context)
                                context.metadata.pop("_retry_count", None)
                                context.metadata.pop("_retry_phase", None)
                            elif action == HookAction.ABORT:
                                status, final_output, error_msg = _resolve_abort(context, final_output)
                                iteration_break = True
                                break
                            elif action == HookAction.PAUSE:
                                status = RunStatus.PAUSED
                                context_snapshot = context.to_snapshot()
                                iteration_break = True
                                break

                        # Use the (possibly hook-modified) phase output for post-processing
                        effective_output = context.phase_outputs.get(phase_name, result.output)

                        # --- Phase-specific post-processing (from runtime) ------
                        if phase_name == "plan":
                            plan_output = effective_output
                            if isinstance(plan_output, PlanOutput) and plan_output.is_complete:
                                final_output = plan_output.completion_reason or "Goal already satisfied."
                                status = RunStatus.COMPLETED
                                iteration_break = True
                                break
                            if isinstance(plan_output, PlanOutput):
                                from agentic_core.engine.runtime import _validate_plan_tools

                                _validate_plan_tools(plan_output, context.available_tools)
                                # Issue #993: extend from effective_output (post-hook).
                                actions_taken.extend(s.action for s in plan_output.steps)

                        elif phase_name == "act":
                            act_output = effective_output
                            if isinstance(act_output, ActOutput):
                                # A pattern that replaces the plan phase injects
                                # its PlanOutput from a hook instead (reason_act
                                # does, and it is the default), so the plan
                                # branch above never ran and never recorded what
                                # the agent set out to do. Read the injected plan
                                # here so episodic summaries get an action list
                                # either way. Bare AgentRuntime always runs the
                                # plan phase, so it has no equivalent gap.
                                if "plan" not in phases_this_iteration:
                                    injected_plan = context.phase_outputs.get("plan")
                                    if isinstance(injected_plan, PlanOutput):
                                        actions_taken.extend(s.action for s in injected_plan.steps)
                                if wm is not None:
                                    from agentic_core.memory.working import WorkingMemoryEntryType

                                    await wm.add_entry(
                                        content=act_output.final_response,
                                        entry_type=WorkingMemoryEntryType.TOOL_RESULT,
                                    )
                                final_output = act_output.final_response
                                if not act_output.should_continue:
                                    iteration_break = True
                                    break
                            else:
                                final_output = str(act_output)
                                iteration_break = True
                                break

                    if iteration_break:
                        break
                else:
                    # The loop exhausted its range. A pattern that concluded on
                    # the *last* permitted iteration set final_output_override
                    # but never got the extra iteration where the empty
                    # phase_sequence above would have read it — so honour it
                    # here too, or the declared answer is silently replaced by
                    # the previous iteration's raw tool output.
                    override = context.metadata.get("final_output_override")
                    final_output = str(override) if override is not None else final_output or "Max iterations reached."

            # --- On-completion hooks ----------------------------------------
            if status == RunStatus.COMPLETED:
                try:
                    await self._run_hooks(HookPoint.ON_COMPLETION, context, None)
                except Exception as comp_exc:
                    log.error(
                        "hook.on_completion_error",
                        run_id=str(run_id),
                        error=str(comp_exc),
                        agent_id=str(rt._config.agent_id),
                    )
        except Exception as e:
            status = RunStatus.FAILED
            error_msg = f"{type(e).__name__}: {e}"
            final_output = ""
            # Counted here as in ``AgentRuntime.run``; without it a failed run
            # on the orchestrator path was absent from the error metric.
            from agentic_core.observability.metrics import record_error

            record_error(type(e).__name__, "run")
            # --- On-error hooks ---
            try:
                context.metadata["error"] = error_msg
                await self._run_hooks(HookPoint.ON_ERROR, context, None)
            except Exception:
                logger.exception("on_error hook failed")

        # --- Deactivate plugins ---------------------------------------------
        # Issue #809: use activated_plugins, not self._plugins.
        for plugin in activated_plugins:
            try:
                await plugin.on_deactivate(context)
            except Exception:
                logger.exception("plugin.deactivate_error", extra={"slug": plugin.slug})

        # --- Episodic memory ------------------------------------------------
        from agentic_core.observability.metrics import record_run

        record_run(
            agent_id=str(rt._config.agent_id),
            status=status.value,
            duration_seconds=asyncio.get_event_loop().time() - run_start,
        )
        # Episodic memory is opt-in (``episodic_memory_enabled``, default False)
        # and ``_memory_service`` is never None on the production path — every
        # run gets one built in ``run_service``. So the service check alone is
        # not a gate: without the flag this wrote a summary, and paid for the
        # LLM call that generates it, for agents that had opted out. Sense
        # already gates *recall* on the same flag, which is why opted-out
        # agents accumulated rows nothing ever read back.
        if rt._memory_service is not None and rt._episodic_memory_enabled:
            await rt._store_episode(
                run_id=run_id,
                user_input=user_input,
                final_output=final_output,
                status=status,
                actions_taken=actions_taken,
                context=context,
            )
        if wm is not None:
            # Bounded, as in ``AgentRuntime.run``: cleanup is a Redis round trip
            # on the way out, and a hung one must not hold the run open.
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
            steps=rt._steps,
            total_prompt_tokens=context.total_prompt_tokens,
            total_completion_tokens=context.total_completion_tokens,
            total_cost=context.total_cost,
            error=error_msg,
            context_snapshot=context_snapshot,
            # Omitting this left ``run_metadata`` at its default {} on the path
            # production actually runs, so ``run_service`` never saw
            # ``memory_recalled_count`` and recorded no recall telemetry.
            run_metadata={
                k: v for k, v in context.metadata.items() if k != "_runtime" and isinstance(v, (str, int, float, bool))
            },
        )

    # ------------------------------------------------------------------
    # Hook execution helpers
    # ------------------------------------------------------------------

    async def _run_hooks(
        self,
        point: HookPoint,
        context: RunContext,
        phase_output: BaseModel | None,
    ) -> HookAction | None:
        """Execute all hooks registered at *point* in order.

        Returns the first ``HookAction`` returned by any hook, or ``None``
        if all hooks returned ``None`` or a modified output.  Modified
        outputs are written back into ``context.phase_outputs``.
        """
        hooks = self._hooks.get(point, [])
        for slug, hook_fn in hooks:
            error_key = (slug, point)
            start = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    hook_fn(context, phase_output),
                    timeout=self._hook_timeout,
                )
            except TimeoutError:
                log.warning("hook.timeout", slug=slug, hook_point=point)
                if self._fail_on_hook_error or slug in self._load_bearing_slugs:
                    context.metadata.setdefault(
                        "abort_error",
                        f"Hook '{slug}' at {point.value} timed out after {self._hook_timeout}s.",
                    )
                    return HookAction.ABORT
                count = self._consecutive_hook_errors.get(error_key, 0) + 1
                self._consecutive_hook_errors[error_key] = count
                if count >= self._MAX_CONSECUTIVE_HOOK_ERRORS:
                    log.error(
                        "hook.repeated_failure_abort",
                        slug=slug,
                        hook_point=point,
                        consecutive_errors=count,
                    )
                    context.metadata["abort_error"] = (
                        f"Hook '{slug}' at {point.value} failed {count} consecutive "
                        f"times (timeout). Aborting to prevent infinite loop."
                    )
                    return HookAction.ABORT
                continue
            except Exception as exc:
                log.exception("hook.error", slug=slug, hook_point=point)
                if self._fail_on_hook_error or slug in self._load_bearing_slugs:
                    # Report what actually went wrong. Without this the run
                    # fails several layers downstream on whatever state the
                    # hook never got to write — a provider 404 in
                    # ``reason_act.pre_act`` used to surface as "Act phase
                    # requires PlanOutput in context" (#1465).
                    context.metadata.setdefault("abort_error", f"Hook '{slug}' at {point.value} failed: {exc}")
                    return HookAction.ABORT
                count = self._consecutive_hook_errors.get(error_key, 0) + 1
                self._consecutive_hook_errors[error_key] = count
                if count >= self._MAX_CONSECUTIVE_HOOK_ERRORS:
                    log.error(
                        "hook.repeated_failure_abort",
                        slug=slug,
                        hook_point=point,
                        consecutive_errors=count,
                        error=str(exc),
                    )
                    context.metadata["abort_error"] = (
                        f"Hook '{slug}' at {point.value} failed {count} consecutive "
                        f"times: {exc}. Aborting to prevent infinite loop."
                    )
                    return HookAction.ABORT
                continue
            finally:
                record_hook_duration(slug, point.value, time.perf_counter() - start)

            # Hook succeeded — reset its error counter
            self._consecutive_hook_errors.pop(error_key, None)

            if isinstance(result, HookAction):
                return result
            if isinstance(result, BaseModel):
                # Validate that the returned model matches the expected phase type
                if context.current_phase:
                    expected_type = _PHASE_OUTPUT_TYPE.get(context.current_phase)
                    if expected_type is not None and not isinstance(result, expected_type):
                        log.warning(
                            "hook.phase_output_type_mismatch",
                            hook=f"{slug}@{point.value}",
                            phase=context.current_phase,
                            expected=expected_type.__name__,
                            actual=type(result).__name__,
                        )
                        # #1047: in strict mode a wrong-typed BaseModel return
                        # aborts the run instead of being silently dropped.
                        if self._fail_on_type_mismatch:
                            context.metadata["abort_error"] = (
                                f"Hook '{slug}' at {point.value} returned "
                                f"{type(result).__name__} but phase "
                                f"'{context.current_phase}' expects "
                                f"{expected_type.__name__}."
                            )
                            return HookAction.ABORT
                        # Best-effort mode: drop the wrong-typed result and
                        # keep the prior phase_output rather than corrupting
                        # context.phase_outputs with the mismatched model.
                        continue
                # Hook modified the phase output — update context and continue.
                # #1039: use update_phase_output (not record_phase_output) so a
                # chain of output-modifying hooks does not append a fresh
                # conversation_history entry per hook and silently evict older
                # phase summaries via FIFO trimming.
                phase_output = result
                if context.current_phase:
                    context.update_phase_output(context.current_phase, result)
            elif result is not None:
                log.warning(
                    "hook.unexpected_return_type",
                    hook=f"{slug}@{point.value}",
                    type=type(result).__name__,
                )
                # #1047: a hook that returns something other than None /
                # HookAction / BaseModel is a programmer error.  Strict mode
                # surfaces it as an abort rather than letting the loop continue.
                if self._fail_on_type_mismatch:
                    context.metadata["abort_error"] = (
                        f"Hook '{slug}' at {point.value} returned "
                        f"{type(result).__name__}, expected None | HookAction | "
                        f"BaseModel."
                    )
                    return HookAction.ABORT
        return None

    async def _get_phase_sequence(self, context: RunContext) -> list[tuple[str, StepPhase]]:
        """Consult loop_control hooks to determine the phase sequence.

        If no loop_control hooks modify the sequence, the default
        Sense → Reason → Plan → Act order is used.
        """
        hooks = self._hooks.get(HookPoint.LOOP_CONTROL, [])
        if not hooks:
            return list(_PHASE_SEQUENCE)

        # loop_control hooks receive context and can set
        # context.metadata["phase_sequence"] to override the order.
        for slug, hook_fn in hooks:
            try:
                await asyncio.wait_for(hook_fn(context, None), timeout=self._hook_timeout)
            except Exception:
                log.exception("hook.loop_control_error", slug=slug)
                if self._fail_on_hook_error:
                    raise

        custom = context.metadata.get("phase_sequence")
        if custom is not None and isinstance(custom, list):
            # An explicitly empty list means "skip all phases this iteration"
            if not custom:
                return []
            # Validate and map phase names to StepPhase
            phase_map = dict(_PHASE_SEQUENCE)
            result = []
            for name in custom:
                if name in phase_map:
                    result.append((name, phase_map[name]))
            if result:
                return result

        return list(_PHASE_SEQUENCE)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pattern_config(
        cls,
        runtime: AgentRuntime,
        pattern_config: dict[str, Any] | None,
        *,
        fail_on_hook_error: bool = False,
        fail_on_type_mismatch: bool | None = None,
        hook_timeout: float | None = None,
        default_execution_pattern: str = DEFAULT_EXECUTION_PATTERN,
    ) -> PatternOrchestrator:
        """Build an orchestrator from a serialised ``PatternConfig`` dict.

        Looks up each active slug in the ``PluginRegistry``, instantiates
        and configures the plugins, then returns an orchestrator wrapping
        the given runtime.

        When *pattern_config* selects nothing, falls back to
        *default_execution_pattern*. Callers that know the run's model should
        pass ``single_agent_baseline`` when it cannot do function calling —
        see ``agentic_core.services.llm_service.model_supports_tool_calling``.
        """
        from agentic_core.schemas.pattern import PatternConfig

        PluginRegistry.discover()

        cfg = PatternConfig() if pattern_config is None else PatternConfig.model_validate(pattern_config)

        active_slugs = cfg.all_active_slugs()
        if not active_slugs:
            active_slugs = [default_execution_pattern]

        # Issue #1190: auto-resolve missing dependencies that can safely fill
        # an empty singleton-category slot (e.g. supervisor_architecture →
        # single_agent_baseline when no execution pattern is active). Fail fast on
        # any missing dep that cannot be safely auto-activated.
        auto_resolve_categories = {"execution", "routing", "coordination"}
        dep_graph: dict[str, list[str]] = {}
        category_map: dict[str, str] = {}
        for slug in active_slugs:
            plugin_cls = PluginRegistry.get(slug)
            if plugin_cls is None:
                continue
            deps = list(getattr(plugin_cls, "dependencies", []) or [])
            dep_graph[slug] = deps
            category_map[slug] = str(plugin_cls.category)
            for dep in deps:
                dep_cls = PluginRegistry.get(dep)
                if dep_cls is not None:
                    category_map[dep] = str(dep_cls.category)
        active_set = set(active_slugs)
        # ``category_map`` is not optional in practice: without it,
        # collect_missing_dependencies cannot apply its same-category singleton
        # skip, and a pattern that depends on another in its own singleton
        # category is reported missing — then rejected, because the slot it would
        # need is the one its dependent already occupies. ``reason_act ->
        # single_agent_baseline`` is the case named in that helper's own
        # docstring. ARES omits it too (orchestrator.py:767); the bug is dormant
        # there only because no plugin declares ``dependencies``, so the graph is
        # always empty and this whole block is unreachable.
        missing_deps = composition.collect_missing_dependencies(active_set, dep_graph, category_map)
        auto_added: list[str] = []
        unresolved: list[tuple[str, str]] = []
        for slug in active_slugs:
            for dep in dep_graph.get(slug, []):
                if dep in active_set or dep in auto_added:
                    continue
                if dep not in missing_deps:
                    continue
                dep_cat = category_map.get(dep)
                slot_occupied = any(category_map.get(s) == dep_cat for s in list(active_set) + auto_added)
                if dep_cat in auto_resolve_categories and not slot_occupied:
                    log.warning(
                        "pattern.orchestrator.auto_resolved_dep",
                        slug=dep,
                        required_by=slug,
                        category=dep_cat,
                    )
                    auto_added.append(dep)
                else:
                    unresolved.append((slug, dep))
        if unresolved:
            detail = "; ".join(f"'{s}' requires '{d}'" for s, d in unresolved)
            raise ValueError(f"Missing pattern dependencies: {detail}")
        active_slugs = list(active_slugs) + auto_added

        plugins: list[PatternPlugin] = []
        for slug in active_slugs:
            plugin_cls = PluginRegistry.get(slug)
            if plugin_cls is None:
                log.warning("pattern.orchestrator.unknown_slug", slug=slug)
                continue
            instance = plugin_cls()
            # Fill in the plugin's own ``configuration_schema`` defaults beneath
            # whatever the caller supplied. ARES reads these from the
            # architectural_patterns table, so an unseeded row silently fell back
            # to the ``params.get(key, literal)`` fallbacks inside configure() —
            # a second copy of every default, free to drift from the schema the
            # UI renders. Read off the plugin class, the two cannot disagree.
            params = catalog.merge_schema_defaults(plugin_cls, cfg.pattern_params.get(slug, {}))
            # Issue #1044: validate params at runtime, not just at agent
            # create/update.  A broken DB row (manual edit, stale migration,
            # older write) would otherwise silently configure the plugin
            # with the wrong types.  Mirror the API behaviour: refuse to
            # run rather than continue with bad config.
            errors = instance.validate_config(params)
            if errors:
                detail = "; ".join(errors)
                raise ValueError(f"Invalid pattern_params for '{slug}' at runtime: {detail}")
            instance.configure(params)
            plugins.append(instance)

        effective_timeout = hook_timeout if hook_timeout is not None else float(settings.hook_timeout_seconds)
        for plugin in plugins:
            # Coordination patterns (multi-agent supervisors) need long
            # timeouts because workers run their own SRPA loops inside.
            if plugin.category == PatternCategory.COORDINATION:
                effective_timeout = max(effective_timeout, 3600.0)
            # Patterns that fan out to multiple LLM calls inside one hook
            # (e.g., Tree-of-Thought) declare their own minimum.
            if plugin.recommended_hook_timeout is not None:
                effective_timeout = max(effective_timeout, plugin.recommended_hook_timeout)

        effective_fail_on_type_mismatch = (
            fail_on_type_mismatch
            if fail_on_type_mismatch is not None
            else bool(getattr(settings, "fail_on_hook_type_mismatch", False))
        )

        return cls(
            runtime,
            plugins,
            fail_on_hook_error=fail_on_hook_error,
            fail_on_type_mismatch=effective_fail_on_type_mismatch,
            hook_timeout=effective_timeout,
        )

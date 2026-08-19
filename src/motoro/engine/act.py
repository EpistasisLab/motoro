"""Act phase — executes plan steps and produces results."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

import structlog

from motoro.engine.context import RunContext
from motoro.engine.phase import PhaseResult
from motoro.schemas.llm import (
    ActOutput,
    LLMCallRecord,
    PlanOutput,
    PlanStep,
    SenseOutput,
    StepResult,
    ToolCallRecord,
)
from motoro.services.llm_errors import LLMBudgetExceededError
from motoro.services.llm_service import LLMService

if TYPE_CHECKING:
    from motoro.mcp.adapters import ToolExecutionError

log = structlog.get_logger()


def _as_tool_execution_error(e: Exception) -> ToolExecutionError | None:
    """Check if an exception is a ToolExecutionError (avoids top-level import)."""
    from motoro.mcp.adapters import ToolExecutionError

    return e if isinstance(e, ToolExecutionError) else None


def _accumulate_llm(total: LLMCallRecord, record: LLMCallRecord | None) -> LLMCallRecord:
    """Fold one call record into the running total (issue #765 pricing rules).

    If any accumulated step used ``override`` pricing the aggregate stays
    ``override``; otherwise the most recent step's ``pricing_source`` wins.
    ``record`` of ``None`` (a tool-only step) leaves the total unchanged.
    """
    if record is None:
        return total
    if total.model:
        either_override = total.pricing_source == "override" or record.pricing_source == "override"
        source = "override" if either_override else record.pricing_source
    else:
        source = record.pricing_source
    return LLMCallRecord(
        model=record.model,
        prompt_tokens=total.prompt_tokens + record.prompt_tokens,
        completion_tokens=total.completion_tokens + record.completion_tokens,
        latency_ms=total.latency_ms + record.latency_ms,
        cost_estimate=total.cost_estimate + record.cost_estimate,
        pricing_source=source,
    )


ACT_SYSTEM_PROMPT = """\
You are the Act module of an AI agent. Execute the given action step.
If tool functions are provided, call them using the tool API — do NOT write \
<tool_call> or <tool_use> blocks as text. Use the function calling interface directly.
"""

# Synthetic action name for the closing turn appended after a tool-ending plan.
FINAL_SYNTHESIS_ACTION = "synthesize_final_response"

# Patterns that run their own iteration loop compose the final answer themselves
# on the terminating turn. Setting this in ``context.metadata`` stops Act from
# adding a closing synthesis call to *every* iteration of such a loop.
SUPPRESS_FINAL_SYNTHESIS_KEY = "suppress_final_synthesis"

FINAL_SYNTHESIS_PROMPT = """\
You have finished executing your planned steps; their results are above. The \
plan ended on a tool call, so you have NOT yet written your final answer. \
Compose it now, following your original instructions and required output format \
exactly. Do NOT call any tools — return only the final answer as text.
"""


@dataclass(slots=True)
class _StepOutcome:
    """Outcome of executing one plan step.

    A native tool-calling turn can request several tool calls at once, so
    ``tool_results`` holds one ``(result_text, record)`` entry per executed
    call. ``text`` is the assistant prose from the same completion — the
    model's reasoning for the calls it made — and is empty for a pure
    tool-call turn.
    """

    text: str
    llm_record: LLMCallRecord | None
    tool_results: list[tuple[str, ToolCallRecord]] = field(default_factory=list)


class StepExecutor(Protocol):
    """Protocol for step execution strategies."""

    async def execute_step(self, step: PlanStep, context: RunContext) -> tuple[str, LLMCallRecord | None]:
        """Execute a single plan step."""
        ...


class LLMStepExecutor:
    """Default step executor: generates a response via LLM for each step."""

    def __init__(self, llm_service: LLMService) -> None:
        self._llm = llm_service

    async def execute_step(self, step: PlanStep, context: RunContext) -> tuple[str, LLMCallRecord | None]:
        """Execute a step by asking the LLM to carry out the action."""
        sense_output = context.phase_outputs.get("sense")
        user_input = sense_output.user_input if isinstance(sense_output, SenseOutput) else context.user_input

        messages = [
            {"role": "system", "content": ACT_SYSTEM_PROMPT},
            {"role": "system", "content": f"Agent goal: {context.agent_goal}"},
            {
                "role": "user",
                "content": (
                    _fence_user_input(user_input)
                    + f"\n\nAction to execute: {step.action}\n"
                    + f"Description: {step.description}"
                ),
            },
        ]

        from motoro.config import settings

        try:
            text, record = await asyncio.wait_for(
                self._llm.complete_text(
                    config=context.model_config,
                    messages=messages,
                ),
                timeout=settings.llm_call_timeout_seconds,
            )
        except TimeoutError as exc:
            # Issue #818: re-raise with action and timeout context so callers
            # can classify this as a timeout error with full diagnostics.
            log.error(
                "act.step.llm_timeout",
                timeout_seconds=settings.llm_call_timeout_seconds,
                action=step.action,
                component="act",
            )
            raise TimeoutError(
                f"LLM call timed out after {settings.llm_call_timeout_seconds}s (action={step.action!r})"
            ) from exc
        return text, record


def _fence_user_input(user_input: str) -> str:
    """Wrap user input in prompt-injection fence. Issue #813."""
    from motoro.security.prompt_injection import fence  # noqa: PLC0415

    return f"Original request: {fence(user_input)}"


class ActPhase:
    """Output layer: executes plan steps and collects results."""

    def __init__(
        self,
        llm_service: LLMService,
        mcp_executor: object | None = None,
    ) -> None:
        self._llm = llm_service
        self._default_executor = LLMStepExecutor(llm_service)
        self._mcp_executor = mcp_executor

    @property
    def name(self) -> str:
        return "act"

    async def execute(self, context: RunContext) -> PhaseResult:
        """Execute all plan steps and collect results."""
        plan_output = context.phase_outputs.get("plan")
        if not isinstance(plan_output, PlanOutput):
            raise ValueError("Act phase requires PlanOutput in context")

        results: list[StepResult] = []
        total_llm = LLMCallRecord(model="")
        final_parts: list[str] = []

        for i, step in enumerate(plan_output.steps):
            step_log = log.bind(step_index=i, action=step.action, component="act")
            try:
                outcome = await self._execute_single_step(step, context, prior_results=results)
                executor_type = "mcp" if outcome.tool_results else "llm"
                step_log.debug(
                    "act.step.completed",
                    executor=executor_type,
                    tool_calls=len(outcome.tool_results),
                )

                # Prose that arrived alongside the tool calls is the model's own
                # reasoning for them — keep it in the transcript, ahead of the
                # results it motivated.
                if outcome.tool_results and outcome.text:
                    results.append(
                        StepResult(
                            step_index=i,
                            action=step.action,
                            result=outcome.text,
                            success=True,
                        )
                    )
                    final_parts.append(outcome.text)

                if outcome.tool_results:
                    # One StepResult per tool call — runtime.py folds several of
                    # these into the ``{"calls": [...]}`` telemetry shape.
                    for result_text, call_record in outcome.tool_results:
                        results.append(
                            StepResult(
                                step_index=i,
                                action=step.action,
                                result=result_text,
                                success=call_record.success,
                                tool_call=call_record,
                            )
                        )
                        final_parts.append(result_text)
                else:
                    results.append(
                        StepResult(
                            step_index=i,
                            action=step.action,
                            result=outcome.text,
                            success=True,
                        )
                    )
                    final_parts.append(outcome.text)

                # Issue #765: keep "override" sticky across steps and otherwise
                # take the latest step's pricing_source (see _accumulate_llm).
                total_llm = _accumulate_llm(total_llm, outcome.llm_record)

            except Exception as e:
                tool_err = _as_tool_execution_error(e)
                if tool_err is not None:
                    # Structured tool error — attempt LLM fallback.
                    # Issue #761: preserve tool_record from the original error
                    # so it is always attached to the StepResult, even when
                    # the fallback itself fails.
                    tool_record = tool_err.tool_record
                    step_log.warning(
                        "act.step.tool_error",
                        tool=tool_err.tool,
                        server=tool_err.server,
                        error_type=tool_err.error_type,
                        error=str(tool_err),
                    )
                    fallback_text = await self._llm_fallback(step, tool_err, context)
                    if fallback_text is not None:
                        results.append(
                            StepResult(
                                step_index=i,
                                action=step.action,
                                result=fallback_text,
                                success=True,
                                tool_call=tool_record,
                            )
                        )
                        final_parts.append(fallback_text)
                    else:
                        # Issue #761: tool_record from original error is always
                        # attached here — fallback failure doesn't lose it.
                        results.append(
                            StepResult(
                                step_index=i,
                                action=step.action,
                                result=f"Tool error ({tool_err.error_type}): {tool_err}",
                                success=False,
                                tool_call=tool_record,
                            )
                        )
                else:
                    step_log.error("act.step.error", error=f"{type(e).__name__}: {e}")
                    results.append(
                        StepResult(
                            step_index=i,
                            action=step.action,
                            result=f"Error: {type(e).__name__}: {e}",
                            success=False,
                        )
                    )

        # The plan sometimes ends on a tool call (e.g. persist an artifact), so
        # the last thing in final_parts is a raw tool-result dump and the
        # agent's required final answer/report was never authored. Force one
        # closing LLM turn to compose it, grounded in the agent's own system
        # prompt and the accumulated step results. Degrades silently: on any
        # error we keep the tool-dump response rather than failing the run.
        if results and results[-1].tool_call is not None and not context.metadata.get(SUPPRESS_FINAL_SYNTHESIS_KEY):
            try:
                text, record = await self._synthesize_final_response(context, results)
                results.append(
                    StepResult(
                        step_index=len(results),
                        action=FINAL_SYNTHESIS_ACTION,
                        result=text,
                        success=True,
                    )
                )
                final_parts.append(text)
                total_llm = _accumulate_llm(total_llm, record)
            except Exception as e:  # never fail the run on the closing turn
                log.warning(
                    "act.synthesis.failed",
                    error=f"{type(e).__name__}: {e}",
                    component="act",
                )

        final_response = "\n\n".join(final_parts) if final_parts else "No results produced."

        output = ActOutput(
            results=results,
            final_response=final_response,
            should_continue=False,
        )

        return PhaseResult(
            output=output,
            llm_call=total_llm if total_llm.model else None,
        )

    async def _synthesize_final_response(
        self,
        context: RunContext,
        results: list[StepResult],
    ) -> tuple[str, LLMCallRecord | None]:
        """Run one closing LLM turn so the agent composes its final answer.

        Used when the plan's last executed step was a tool call, which would
        otherwise leave the raw tool-result dump as ``final_response`` and the
        agent's required report unwritten. Grounded in the agent's own system
        prompt plus a bounded summary of every executed step's result.
        """
        from motoro.config import settings

        sense_output = context.phase_outputs.get("sense")
        user_input = sense_output.user_input if isinstance(sense_output, SenseOutput) else context.user_input

        cap = settings.act_prior_result_max_chars
        parts: list[str] = []
        for r in results:
            status = "succeeded" if r.success else "failed"
            result_text = r.result or ""
            if cap and len(result_text) > cap:
                # Head-truncate: ids/keys a synthesis needs sit near the start.
                result_text = (
                    result_text[:cap]
                    + f"\n…[truncated {len(result_text) - cap} chars; "
                    + "full result persisted via the tool's artifact/dataset id above]"
                )
            parts.append(f"Step '{r.action}' [{status}]: {result_text}")
        steps_context = "\n".join(parts)

        messages = [
            {"role": "system", "content": context.system_prompt},
            {"role": "system", "content": f"Agent goal: {context.agent_goal}"},
            {
                "role": "user",
                "content": (
                    _fence_user_input(user_input)
                    + "\n\nExecuted step results:\n"
                    + steps_context
                    + "\n\n"
                    + FINAL_SYNTHESIS_PROMPT
                ),
            },
        ]
        text, record = await asyncio.wait_for(
            self._llm.complete_text(config=context.model_config, messages=messages),
            timeout=settings.llm_call_timeout_seconds,
        )
        return text, record

    async def _execute_single_step(
        self,
        step: PlanStep,
        context: RunContext,
        prior_results: list[StepResult] | None = None,
    ) -> _StepOutcome:
        """Execute a single step via native tool calling.

        Tool schemas are bound to the model, so the model that decides on a
        tool is the model that names it — there is no second selection call to
        diverge from the first. A turn may request several calls; all of them
        run.

        Error handling differs by cardinality. When exactly one call was
        requested and it fails, the error propagates so :meth:`execute` can run
        its LLM fallback (the historical single-tool behaviour). When several
        were requested, a failing call is recorded and the remaining calls
        still run — substituting generated text for one call out of a set would
        corrupt the set.
        """
        from motoro.mcp.adapters import (
            MCPToolExecutor,
            build_openai_tool_name_map,
            tools_to_openai_format,
        )
        from motoro.schemas.llm import PlanStep as _PlanStep
        from motoro.services.credential_scrubber import redact_tool_args

        mcp = self._mcp_executor if isinstance(self._mcp_executor, MCPToolExecutor) else None

        # A caller that already decided the tool owns that decision — execute it
        # rather than asking a second model to choose again. Re-deciding here is
        # what let an execution pattern's chosen tool diverge from the tool that
        # actually ran, and made the pattern's own trajectory record fictional.
        if step.tool_name and mcp is not None and mcp.can_handle(step):
            log.debug("act.step.using_planned_tool", tool=step.tool_name, component="act")
            result_text, _unused, tool_record = await mcp.execute_step(step, context)
            return _StepOutcome(
                text="",
                llm_record=None,
                tool_results=[(result_text, tool_record or ToolCallRecord(tool=step.tool_name))],
            )

        tool_schemas = tools_to_openai_format(context.available_tools) if context.available_tools else []
        if mcp is None or not tool_schemas:
            # No tools available — plain LLM text generation.
            text, plain_record = await self._default_executor.execute_step(step, context)
            return _StepOutcome(text=text, llm_record=plain_record)

        try:
            completion = await self._llm.complete_with_tools(
                config=context.model_config,
                messages=self._build_step_messages(step, context, prior_results),
                tools=tool_schemas,
            )
        except LLMBudgetExceededError:
            raise
        except Exception as exc:
            # There is no provider capability gate for native tool use yet, so a
            # model that rejects a ``tools`` payload must still be able to act.
            log.warning(
                "act.step.native_tools_failed",
                error=f"{type(exc).__name__}: {exc}",
                action=step.action,
                component="act",
            )
            return await self._select_tool_fallback(step, context, prior_results)

        if not completion.tool_calls:
            log.debug("act.step.using_llm", action=step.action, component="act")
            if completion.text:
                return _StepOutcome(text=completion.text, llm_record=completion.record)
            # Degenerate turn: no call, no prose. Ask for text explicitly rather
            # than letting the step contribute nothing to the final response.
            text, text_record = await self._default_executor.execute_step(step, context)
            return _StepOutcome(text=text, llm_record=_accumulate_llm(completion.record, text_record))

        name_map = build_openai_tool_name_map(context.available_tools)
        single = len(completion.tool_calls) == 1
        tool_results: list[tuple[str, ToolCallRecord]] = []

        for call in completion.tool_calls:
            # Translate back through the sanitisation applied when the schemas
            # were built (issue #772) — the model echoes the sanitized name.
            tool_name = name_map.get(call.name, call.name)
            tool_step = _PlanStep(
                action=step.action,
                description=step.description,
                tool_name=tool_name,
                tool_args=dict(call.arguments),
            )

            if not mcp.can_handle(tool_step):
                log.warning("act.step.mcp_cannot_handle", tool=tool_name, component="act")
                if single:
                    # Preserve the historical single-tool path: fall through to
                    # plain text rather than failing the step.
                    text, text_record = await self._default_executor.execute_step(step, context)
                    return _StepOutcome(
                        text=text,
                        llm_record=_accumulate_llm(completion.record, text_record),
                    )
                tool_results.append(
                    (
                        f"Tool '{tool_name}' is not available to this run",
                        ToolCallRecord(
                            tool=tool_name,
                            arguments=redact_tool_args(dict(call.arguments)),
                            result=f"Tool '{tool_name}' is not available to this run",
                            success=False,
                            error_type="not_found",
                        ),
                    )
                )
                continue

            log.debug("act.step.using_mcp", tool=tool_name, action=step.action, component="act")
            try:
                result_text, _unused, tool_record = await mcp.execute_step(tool_step, context)
            except Exception as exc:
                tool_err = _as_tool_execution_error(exc)
                if single or tool_err is None:
                    raise
                log.warning(
                    "act.step.tool_error_in_batch",
                    tool=tool_err.tool,
                    error_type=tool_err.error_type,
                    component="act",
                )
                tool_results.append(
                    (
                        f"Tool error ({tool_err.error_type}): {tool_err}",
                        tool_err.tool_record
                        or ToolCallRecord(
                            tool=tool_name,
                            result=str(tool_err),
                            success=False,
                            error_type=tool_err.error_type,
                        ),
                    )
                )
                continue

            tool_results.append(
                (result_text, tool_record if tool_record is not None else ToolCallRecord(tool=tool_name))
            )

        return _StepOutcome(
            text=completion.text,
            llm_record=completion.record,
            tool_results=tool_results,
        )

    async def _select_tool_fallback(
        self,
        step: PlanStep,
        context: RunContext,
        prior_results: list[StepResult] | None = None,
    ) -> _StepOutcome:
        """Legacy string-named tool selection, used when native tool use fails.

        Asks the model to name a tool from a textual catalogue instead of
        binding schemas. One call per turn, and the selection is a separate
        completion from the one that reasoned about the step.
        """
        from motoro.mcp.adapters import MCPToolExecutor
        from motoro.schemas.llm import PlanStep as _PlanStep

        mcp = self._mcp_executor if isinstance(self._mcp_executor, MCPToolExecutor) else None
        tool_names = [
            str(t.get("tool_name") or t.get("name", ""))
            for t in context.available_tools
            if t.get("tool_name") or t.get("name")
        ]
        if mcp is None or not tool_names:
            text, plain_record = await self._default_executor.execute_step(step, context)
            return _StepOutcome(text=text, llm_record=plain_record)

        use_tool, tool_name, tool_args, llm_record = await self._llm.select_tool(
            config=context.model_config,
            messages=self._build_step_messages(step, context, prior_results, include_tool_catalogue=True),
            tool_names=tool_names,
        )
        if use_tool and tool_name:
            tool_step = _PlanStep(
                action=step.action,
                description=step.description,
                tool_name=tool_name,
                tool_args=tool_args or {},
            )
            if mcp.can_handle(tool_step):
                result_text, _unused, tool_record = await mcp.execute_step(tool_step, context)
                return _StepOutcome(
                    text="",
                    llm_record=llm_record,
                    tool_results=[
                        (result_text, tool_record if tool_record is not None else ToolCallRecord(tool=tool_name))
                    ],
                )
            log.warning("act.step.mcp_cannot_handle", tool=tool_name, component="act")

        text, text_record = await self._default_executor.execute_step(step, context)
        return _StepOutcome(text=text, llm_record=_accumulate_llm(llm_record or LLMCallRecord(model=""), text_record))

    def _build_step_messages(
        self,
        step: PlanStep,
        context: RunContext,
        prior_results: list[StepResult] | None = None,
        include_tool_catalogue: bool = False,
    ) -> list[dict[str, str]]:
        """Build messages for an Act step.

        ``include_tool_catalogue`` renders the tools as prose. That is only
        needed by :meth:`_select_tool_fallback`; on the native path the schemas
        are bound to the request, so repeating them in the prompt would waste
        tokens and let the two descriptions drift apart.
        """
        sense_output = context.phase_outputs.get("sense")
        user_input = sense_output.user_input if isinstance(sense_output, SenseOutput) else context.user_input
        tool_hint = ""
        if include_tool_catalogue and context.available_tools:
            from motoro.mcp.adapters import format_tools_for_prompt

            tool_hint = "\n\n" + format_tools_for_prompt(context.available_tools)

        now_utc = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        prior_context = ""
        if prior_results:
            from motoro.config import settings

            cap = settings.act_prior_result_max_chars
            parts = []
            for r in prior_results:
                status = "succeeded" if r.success else "failed"
                result_text = r.result or ""
                if cap and len(result_text) > cap:
                    # Head-truncate: ids/keys a later step needs sit near the start.
                    result_text = (
                        result_text[:cap]
                        + f"\n…[truncated {len(result_text) - cap} chars; "
                        + "full result persisted via the tool's artifact/dataset id above]"
                    )
                parts.append(f"Step '{r.action}' [{status}]: {result_text}")
            prior_context = "\n\nPrevious step results:\n" + "\n".join(parts)

        return [
            {"role": "system", "content": ACT_SYSTEM_PROMPT},
            {
                "role": "system",
                "content": (f"Current UTC time: {now_utc}\nAgent goal: {context.agent_goal}{tool_hint}"),
            },
            {
                "role": "user",
                "content": (
                    _fence_user_input(user_input)
                    + f"{prior_context}\n\nAction to execute: {step.action}\n"
                    + f"Description: {step.description}"
                ),
            },
        ]

    async def _llm_fallback(
        self,
        step: PlanStep,
        error: ToolExecutionError,
        context: RunContext,
    ) -> str | None:
        """Attempt to handle a failed tool step via LLM text generation."""
        try:
            log.info(
                "act.step.llm_fallback",
                tool=error.tool,
                error_type=error.error_type,
                action=step.action,
                component="act",
            )
            text, _ = await self._default_executor.execute_step(step, context)
            return text
        except Exception as fallback_err:
            log.warning(
                "act.step.fallback_failed",
                action=step.action,
                error=str(fallback_err),
                component="act",
            )
            return None

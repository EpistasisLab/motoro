"""ReasonAct pattern — a native tool-calling agent loop.

One completion per iteration, over an append-only message history:

  Sense → [ complete_with_tools → Act(all requested tools) → tool results ]* → final_answer

The reasoning and the actions come from the same completion, so the recorded
thought is genuinely the thought behind the call that ran. Termination is
declared by calling the bound ``final_answer`` tool. The standard Reason and
Plan phases are skipped (their hooks return ``SKIP_PHASE``); the loop's single
LLM call happens in the PRE_ACT hook, which injects a ``PlanOutput`` that Act
executes verbatim.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, ClassVar

import structlog
from pydantic import BaseModel

from motoro.engine.act import SUPPRESS_FINAL_SYNTHESIS_KEY
from motoro.engine.context import RunContext
from motoro.engine.patterns.base import HookAction, HookCallable, HookPoint, PatternPlugin
from motoro.engine.patterns.prompts.reason_act import (
    build_final_answer_tool,
    build_initial_messages,
    format_tool_result,
    resolve_final_answer_name,
    window_messages,
)
from motoro.engine.patterns.registry import PluginRegistry
from motoro.engine.skills import (
    build_load_skill_tool,
    build_read_skill_file_tool,
    render_skill_body,
    render_skill_file,
    render_skill_index,
    resolve_load_skill_name,
    resolve_read_skill_file_name,
    skill_file_paths,
)
from motoro.models.pattern import PatternCategory, PatternPhase
from motoro.models.run import RunStep, StepPhase
from motoro.schemas.llm import (
    ActOutput,
    PlanOutput,
    PlanStep,
    SenseOutput,
    StepResult,
)
from motoro.schemas.patterns.reason_act import (
    ReasonActAction,
    ReasonActObservation,
    ReasonActStep,
    ReasonActToolCall,
)

log = structlog.get_logger()

# Keys used in RunContext.metadata by this plugin
_KEY_MESSAGES = "reason_act_messages"
_KEY_PENDING_CALLS = "reason_act_pending_calls"
_KEY_TURN = "reason_act_turn"
_KEY_OBSERVATIONS = "reason_act_observations"
_KEY_FINAL_ANSWER = "reason_act_final_answer"
_KEY_STEP_COUNT = "reason_act_step_count"
_KEY_MAX_ITER_HIT = "reason_act_max_iterations_hit"
_KEY_TOOL_CALLS = "reason_act_tool_calls"
_KEY_TERMINATION = "reason_act_termination"
_KEY_STATE = "reason_act_state"
_KEY_SKILLS_OPENED = "reason_act_skills_opened"


@PluginRegistry.register
class ReasonActPlugin(PatternPlugin):
    """Interleaved Reason-Act-Observe execution pattern.

    Each iteration is one ``complete_with_tools`` call against the run's tools
    plus a ``final_answer`` terminator. Every call the model requests is
    executed — including several in one turn — and each result is appended to
    the history as a ``role: "tool"`` turn keyed by the call id the provider
    issued.
    """

    slug = "reason_act"
    category = PatternCategory.EXECUTION

    # This loop discloses skills progressively — an index in the prompt prefix,
    # bodies via a bound ``load_skill`` tool — so the orchestrator must not
    # inline them. See ``motoro.engine.skills``.
    consumes_skills: ClassVar[bool] = True

    display_name: ClassVar[str] = "ReAct"
    description: ClassVar[str] = (
        "Alternates between deliberation and execution in a tight loop: the agent formulates an "
        "explicit Thought, selects and performs an Action, inspects the Observation, and repeats "
        "until a conclusive answer emerges."
    )
    complexity_phase: ClassVar[PatternPhase] = PatternPhase.INTROSPECTIVE
    # ARES's seed row says `solo_agent_loop` — the pre-rename slug, which
    # migration 0009 had to chase through the dependency arrays in data. Declared
    # on the class, a rename cannot leave a stale slug behind in a table.
    dependencies: ClassVar[list[str]] = ["single_agent_baseline"]
    configuration_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "max_iterations": {
                "type": "integer",
                "default": 15,
                "minimum": 1,
                "maximum": 100,
                "description": "Maximum Thought-Action-Observation cycles.",
            },
            "include_scratchpad": {
                "type": "boolean",
                "default": True,
                "description": "Include previous Thought-Action-Observation history in each Reason call.",
            },
            "scratchpad_window": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "maximum": 50,
                "description": "Maximum recent triples to include when include_scratchpad is true.",
            },
            "observation_format": {
                "type": "string",
                "enum": ["raw", "summarized"],
                "default": "raw",
                "description": (
                    "How tool results are formatted: raw returns output directly, summarized uses LLM to condense."
                ),
            },
        },
    }

    # reason_act and tree_of_thought both replace the Reason phase (M76 #1042).
    # The EXECUTION-category singleton rule already prevents co-activation,
    # but declaring ``conflicts_with`` surfaces a clearer error message and
    # documents the incompatibility for the composition checker.
    conflicts_with: ClassVar[list[str]] = ["tree_of_thought"]
    # PRE_REASON and PRE_PLAN always return SKIP_PHASE.
    replaces_phases: ClassVar[list[str]] = ["reason", "plan"]

    def configure(self, params: dict[str, Any]) -> None:
        self.max_iterations: int = params.get("max_iterations", 15)
        self.include_scratchpad: bool = params.get("include_scratchpad", True)
        self.scratchpad_window: int = params.get("scratchpad_window", 10)
        self.observation_format: str = params.get("observation_format", "raw")

    _VALID_OBSERVATION_FORMATS: ClassVar[tuple[str, ...]] = ("raw", "summarized")

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        errors: list[str] = []

        for field in ("max_iterations", "scratchpad_window"):
            if field not in config:
                continue
            value = config[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                errors.append(f"{field} must be a positive integer (got {value!r})")

        if "include_scratchpad" in config and not isinstance(config["include_scratchpad"], bool):
            errors.append(f"include_scratchpad must be a boolean (got {config['include_scratchpad']!r})")

        if "observation_format" in config:
            value = config["observation_format"]
            if value not in self._VALID_OBSERVATION_FORMATS:
                errors.append(f"observation_format must be one of {self._VALID_OBSERVATION_FORMATS} (got {value!r})")

        return errors

    def get_hooks(self) -> dict[HookPoint, list[HookCallable]]:
        return {
            HookPoint.LOOP_CONTROL: [self._loop_control],
            HookPoint.PRE_REASON: [self._pre_reason],
            HookPoint.PRE_PLAN: [self._pre_plan],
            HookPoint.PRE_ACT: [self._pre_act],
            HookPoint.POST_ACT: [self._post_act],
        }

    async def on_activate(self, context: RunContext) -> None:
        context.metadata[_KEY_MESSAGES] = []
        context.metadata[_KEY_STEP_COUNT] = 0
        context.metadata[_KEY_TOOL_CALLS] = 0
        context.max_iterations = self.max_iterations
        # This loop composes its own final answer on the terminating turn, so
        # Act must not bolt a synthesis call onto the end of *every* iteration.
        context.metadata[SUPPRESS_FINAL_SYNTHESIS_KEY] = True

        if not self.include_scratchpad and context.available_tools:
            # With no scratchpad the window collapses to the system/user prefix,
            # so the model never sees what its calls returned and will keep
            # asking for the same one until the ceiling. That is a valid
            # ablation, but it is not a useful production setting, and the
            # params are described to the coordinator LLM that picks them.
            log.warning(
                "reason_act.no_scratchpad_with_tools",
                max_iterations=self.max_iterations,
                tool_count=len(context.available_tools),
                component="reason_act",
            )

    async def on_deactivate(self, context: RunContext) -> None:
        # Clear per-iteration scratch keys so the snapshot stays compact (#1040).
        # The message history and final answer are retained as part of the
        # durable run summary — the history *is* the trajectory.
        for key in (_KEY_TURN, _KEY_PENDING_CALLS):
            context.metadata.pop(key, None)
        context.metadata.pop(SUPPRESS_FINAL_SYNTHESIS_KEY, None)
        await self._persist_state(context)

    # ------------------------------------------------------------------
    # Loop telemetry
    # ------------------------------------------------------------------

    def _state(self, context: RunContext) -> dict[str, Any]:
        """Summarise how the loop ran, for anything analysing the trajectory.

        A null effect from this pattern reads very differently depending on
        whether the loop took one turn or was cut off at the ceiling, and
        neither is recoverable from the step list alone once a downstream
        consumer only has the run record.
        """
        return {
            "iterations": context.metadata.get(_KEY_STEP_COUNT, 0),
            "max_iterations": self.max_iterations,
            "max_iterations_hit": bool(context.metadata.get(_KEY_MAX_ITER_HIT, False)),
            "tool_calls": context.metadata.get(_KEY_TOOL_CALLS, 0),
            "terminated_by": context.metadata.get(_KEY_TERMINATION, "incomplete"),
        }

    async def _persist_state(self, context: RunContext) -> None:
        """Mirror the loop summary into ``agent_runs.pattern_overrides``.

        Best-effort, exactly as ``adaptive_retry`` does it: this is telemetry,
        and losing it must never fail a run that otherwise succeeded.
        """
        runtime = context.metadata.get("_runtime")
        run_id = context.run_id
        if runtime is None or run_id is None:
            return
        db = getattr(runtime, "_db", None)
        if db is None:
            return
        try:
            from sqlalchemy import select

            from motoro.models.run import AgentRun

            result = await db.execute(select(AgentRun).where(AgentRun.id == run_id))
            run = result.scalar_one_or_none()
            if run is None:
                return
            overrides = dict(run.pattern_overrides or {})
            overrides[_KEY_STATE] = self._state(context)
            run.pattern_overrides = overrides
            await db.commit()
        except Exception:
            log.debug("reason_act.persist_state_failed", exc_info=True)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    async def _loop_control(self, context: RunContext, _output: BaseModel | None) -> None:
        """Override phase sequence based on iteration and state."""
        if context.metadata.get(_KEY_FINAL_ANSWER) is not None:
            context.metadata["phase_sequence"] = []
            return

        if context.iteration == 0:
            context.metadata["phase_sequence"] = ["sense", "reason", "act"]
        else:
            context.metadata["phase_sequence"] = ["reason", "act"]

    async def _pre_reason(self, context: RunContext, _output: BaseModel | None) -> HookAction:
        """Skip the standard Reason phase — all reasoning happens in _pre_act."""
        return HookAction.SKIP_PHASE

    async def _pre_plan(self, context: RunContext, _output: BaseModel | None) -> HookAction:
        """Always skip Plan — ReasonAct replaces planning with per-step reasoning."""
        return HookAction.SKIP_PHASE

    async def _pre_act(self, context: RunContext, _output: BaseModel | None) -> BaseModel | HookAction | None:
        """Make the iteration's single LLM call, then route on what it returned.

        Tool schemas are bound to the request, so the model names tools the
        provider has validated against those schemas. Whatever it asks for is
        handed to Act as a plan; Act executes it rather than deciding again.
        """
        sense_output = context.phase_outputs.get("sense")
        if not isinstance(sense_output, SenseOutput):
            return HookAction.SKIP_PHASE

        runtime = context.metadata.get("_runtime")
        if runtime is None:
            log.warning("reason_act.no_runtime", component="reason_act")
            return HookAction.SKIP_PHASE

        llm_service = runtime._llm_service
        if llm_service is None:
            log.warning("reason_act.no_llm_service", component="reason_act")
            return HookAction.SKIP_PHASE

        from motoro.mcp.adapters import build_openai_tool_name_map, tools_to_openai_format

        skills = context.skills or []

        messages: list[dict[str, Any]] = context.metadata.get(_KEY_MESSAGES) or []
        if not messages:
            messages = build_initial_messages(
                sense_output=sense_output,
                agent_system_prompt=context.system_prompt,
                skill_index=render_skill_index(skills) if skills else "",
            )
            context.metadata[_KEY_MESSAGES] = messages

        tools = tools_to_openai_format(context.available_tools) if context.available_tools else []
        name_map = build_openai_tool_name_map(context.available_tools)
        bound_names = {str(t["function"]["name"]) for t in tools}
        terminator = resolve_final_answer_name(bound_names)
        tools.append(build_final_answer_tool(terminator))
        # Bound every turn, not just the first: the model may decide a skill
        # applies only once a tool result has told it what it is dealing with.
        skill_loader = resolve_load_skill_name(bound_names | {terminator}) if skills else ""
        if skill_loader:
            tools.append(build_load_skill_tool(skills, skill_loader))
        # Level 3, and only when some skill actually bundles something: the tool's
        # enum of readable paths would otherwise be empty, which some providers
        # reject outright and the rest turn into a dead affordance.
        file_reader = (
            resolve_read_skill_file_name(bound_names | {terminator, skill_loader})
            if skill_loader and skill_file_paths(skills)
            else ""
        )
        if file_reader:
            tools.append(build_read_skill_file_tool(skills, file_reader))

        # ``include_scratchpad: false`` means no memory of earlier turns, so the
        # window collapses to the stable system/user prefix.
        window = self.scratchpad_window if self.include_scratchpad else 0
        completion = await llm_service.complete_with_tools(
            config=context.model_config,
            messages=window_messages(messages, window),
            tools=tools,
        )

        llm_record = completion.record
        if llm_record:
            await context.add_llm_usage(
                llm_record.prompt_tokens,
                llm_record.completion_tokens,
                llm_record.cost_estimate,
            )

        step_count = context.metadata.get(_KEY_STEP_COUNT, 0) + 1
        context.metadata[_KEY_STEP_COUNT] = step_count

        # The history grows by appending, never by rebuilding, so the prefix the
        # provider already cached stays byte-identical across iterations.
        issued = _issued_calls(completion.tool_calls, name_map, step_count)
        messages.append(_assistant_turn(completion.text, issued))

        final_call = next((c for c in issued if c.tool_name == terminator), None)

        if final_call is not None:
            # A turn can ask to finish *and* ask for more tools. The answer
            # wins — a model that declared completion should not be made to
            # declare it twice, and looping instead risks it never doing so.
            # But the calls it also asked for will not run, so say which.
            dropped = [c.tool_name for c in issued if c is not final_call]
            if dropped:
                log.info(
                    "reason_act.calls_dropped_on_final_answer",
                    dropped_tools=dropped,
                    step=step_count,
                    component="reason_act",
                )
            answer = str(final_call.tool_args.get("answer") or "").strip() or completion.text
            turn = ReasonActStep(
                thought=completion.text,
                action=ReasonActAction.FINAL_ANSWER,
                final_answer=answer,
            )
            await self._record_turn(context, runtime, turn, llm_record)
            return await self._conclude(context, runtime, answer, "final_answer")

        if not issued:
            # No calls and no terminator: the model answered in prose. Treat the
            # prose as the answer rather than burning another iteration asking
            # it to say the same thing through a tool.
            answer = completion.text.strip()
            log.info("reason_act.implicit_final_answer", step=step_count, component="reason_act")
            turn = ReasonActStep(
                thought=completion.text,
                action=ReasonActAction.FINAL_ANSWER,
                final_answer=answer,
            )
            await self._record_turn(context, runtime, turn, llm_record)
            return await self._conclude(context, runtime, answer, "implicit_final_answer")

        # ``load_skill`` and ``read_skill_file`` are answered here, not dispatched:
        # there is no MCP server behind either, only the skill bodies and bundled
        # files already resolved onto the context. Intercepted by name out of
        # ``issued``, the same way the terminator above is.
        skill_calls = [c for c in issued if skill_loader and c.tool_name == skill_loader]
        file_calls = [c for c in issued if file_reader and c.tool_name == file_reader]
        dispatch = [c for c in issued if c not in skill_calls and c not in file_calls]

        turn = ReasonActStep(
            thought=completion.text,
            action=ReasonActAction.TOOL_CALL,
            tool_calls=issued,
        )
        # Counts real tool calls; opening a skill is a context operation, not an
        # action taken in the world, and is tallied separately.
        context.metadata[_KEY_TOOL_CALLS] = context.metadata.get(_KEY_TOOL_CALLS, 0) + len(dispatch)
        await self._record_turn(context, runtime, turn, llm_record)

        if skill_calls:
            opened: list[str] = list(context.metadata.get(_KEY_SKILLS_OPENED) or [])
            for call in skill_calls:
                requested = str(call.tool_args.get("name") or "")
                # Every issued call needs an answering turn or the provider
                # rejects the next request, so this is appended immediately —
                # before Act runs for whatever else the same turn asked for.
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": render_skill_body(skills, requested, file_tool_name=file_reader),
                    }
                )
                opened.append(requested)
                log.info("reason_act.skill_opened", skill=requested, step=step_count, component="reason_act")
            context.metadata[_KEY_SKILLS_OPENED] = opened
            context.metadata[_KEY_MESSAGES] = messages

        if file_calls:
            for call in file_calls:
                wanted = str(call.tool_args.get("path") or "")
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": render_skill_file(skills, wanted)}
                )
                log.info("reason_act.skill_file_read", path=wanted, step=step_count, component="reason_act")
            context.metadata[_KEY_MESSAGES] = messages

        if not dispatch:
            # The turn asked only for skills, which are already answered — there
            # is nothing for Act to execute. Skip it and let the next iteration
            # reason again with the instructions now in hand. ``_post_act`` will
            # not run, so the ceiling it normally records is recorded here.
            context.metadata.pop(_KEY_PENDING_CALLS, None)
            context.metadata.pop(_KEY_TURN, None)
            context.record_phase_output("plan", PlanOutput(steps=[], is_complete=False))
            if step_count >= self.max_iterations:
                context.metadata[_KEY_MAX_ITER_HIT] = True
                context.metadata[_KEY_TERMINATION] = "max_iterations"
            return HookAction.SKIP_PHASE

        context.metadata[_KEY_TURN] = turn.model_dump()
        context.metadata[_KEY_PENDING_CALLS] = [c.model_dump() for c in dispatch]
        context.record_phase_output(
            "plan",
            PlanOutput(
                steps=[
                    # Name the tool rather than the turn. This string is what
                    # Act copies onto every StepResult and what reaches the
                    # episodic run summary, where "ReasonAct step 3" said
                    # nothing about what the agent actually did. The turn number
                    # is already on the RunStep this hook records.
                    PlanStep(
                        action=f"Call {call.tool_name}",
                        description=completion.text or f"Call {call.tool_name}",
                        tool_name=call.tool_name,
                        tool_args=call.tool_args,
                    )
                    for call in dispatch
                ],
                is_complete=False,
            ),
        )
        return None  # Continue to Act

    async def _post_act(self, context: RunContext, output: BaseModel | None) -> BaseModel | HookAction | None:
        """Append one ``role: "tool"`` turn per executed call, then keep looping."""
        pending_raw = context.metadata.get(_KEY_PENDING_CALLS)
        if not pending_raw:
            return None

        pending = [ReasonActToolCall.model_validate(c) for c in pending_raw]
        messages: list[dict[str, Any]] = context.metadata.get(_KEY_MESSAGES) or []

        from motoro.config import settings

        # ``PlanStep`` i produced the ``StepResult``s carrying ``step_index`` i,
        # so results pair back to calls by index rather than by position — Act
        # may emit more than one StepResult for a step, or fall back to prose.
        grouped: dict[int, list[StepResult]] = {}
        if isinstance(output, ActOutput):
            for result in output.results:
                grouped.setdefault(result.step_index, []).append(result)

        observations: list[dict[str, Any]] = []
        for index, call in enumerate(pending):
            step_results = grouped.get(index, [])
            raw = "\n".join(r.result for r in step_results if r.result)
            success = all(r.success for r in step_results) if step_results else False
            text = format_tool_result(
                result=raw,
                success=success,
                observation_format=self.observation_format,
                max_chars=settings.act_prior_result_max_chars,
            )
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
            observations.append(
                ReasonActObservation(
                    tool_call_id=call.id,
                    tool_name=call.tool_name,
                    tool_args=call.tool_args,
                    result=text,
                    success=success,
                ).model_dump()
            )

        context.metadata[_KEY_MESSAGES] = messages
        context.metadata[_KEY_OBSERVATIONS] = observations
        context.metadata.pop(_KEY_PENDING_CALLS, None)
        context.metadata.pop(_KEY_TURN, None)

        step_count = context.metadata.get(_KEY_STEP_COUNT, 0)
        if step_count >= self.max_iterations:
            # The loop is about to be cut off by the iteration ceiling rather
            # than by the model deciding it was done. Downstream analysis needs
            # to be able to tell those two endings apart.
            context.metadata[_KEY_MAX_ITER_HIT] = True
            context.metadata[_KEY_TERMINATION] = "max_iterations"
            log.warning(
                "reason_act.max_iterations_reached",
                max_iterations=self.max_iterations,
                component="reason_act",
            )

        if isinstance(output, ActOutput):
            return ActOutput(
                results=output.results,
                final_response=output.final_response,
                should_continue=True,
            )

        return output

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _conclude(self, context: RunContext, runtime: Any, answer: str, termination: str) -> HookAction:
        """Terminate the loop with *answer* and skip the Act phase."""
        context.metadata[_KEY_FINAL_ANSWER] = answer
        context.metadata["final_output_override"] = answer
        context.metadata[_KEY_MAX_ITER_HIT] = False
        context.metadata[_KEY_TERMINATION] = termination

        context.record_phase_output("plan", PlanOutput(steps=[], is_complete=False))

        act_output = ActOutput(
            results=[
                StepResult(
                    step_index=0,
                    action="final_answer",
                    result=answer,
                    success=True,
                )
            ],
            final_response=answer,
            should_continue=False,
        )
        await _record_step(
            context,
            runtime,
            phase=StepPhase.ACT,
            input_data={"action": "final_answer", "pattern": "reason_act"},
            output_data=act_output.model_dump(),
            llm_record=None,
        )
        context.record_phase_output("act", act_output)
        return HookAction.SKIP_PHASE

    async def _record_turn(
        self,
        context: RunContext,
        runtime: Any,
        turn: ReasonActStep,
        llm_record: Any,
    ) -> None:
        """Persist the turn as a REASON RunStep.

        Emitted before Act runs so the recorded phase order is reason-then-act,
        which is what ``score_structural_adherence`` checks for this pattern.
        """
        if llm_record is None:
            return
        await _record_step(
            context,
            runtime,
            phase=StepPhase.REASON,
            input_data={"pattern": "reason_act", "step": context.metadata.get(_KEY_STEP_COUNT, 0)},
            output_data=turn.model_dump(),
            llm_record=llm_record,
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _issued_calls(
    tool_calls: list[Any],
    name_map: dict[str, str],
    step_count: int,
) -> list[ReasonActToolCall]:
    """Translate provider tool calls into dispatchable records.

    Names are mapped back through the sanitisation applied when the schemas
    were built (issue #772), and a missing call id is synthesised — providers
    match tool-result turns on that id, so it cannot be left blank.
    """
    issued: list[ReasonActToolCall] = []
    for index, call in enumerate(tool_calls):
        issued.append(
            ReasonActToolCall(
                id=call.id or f"reason_act_{step_count}_{index}",
                tool_name=name_map.get(call.name, call.name),
                tool_args=dict(call.arguments),
                provider_name=call.name,
            )
        )
    return issued


def _assistant_turn(text: str, calls: list[ReasonActToolCall]) -> dict[str, Any]:
    """Build the assistant message recording this turn's prose and calls.

    ``content`` is omitted rather than sent empty: some providers reject an
    empty text block, and a tool-only assistant turn is legal without it.
    """
    message: dict[str, Any] = {"role": "assistant"}
    if text:
        message["content"] = text
    if calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                # Replay the provider's own spelling: the bound schema uses the
                # sanitized name, and a replayed turn naming the resolved MCP
                # tool would not match it.
                "function": {
                    "name": call.provider_name or call.tool_name,
                    "arguments": json.dumps(call.tool_args, default=str),
                },
            }
            for call in calls
        ]
    elif not text:
        message["content"] = ""
    return message


async def _record_step(
    context: RunContext,
    runtime: Any,
    *,
    phase: StepPhase,
    input_data: Any,
    output_data: Any,
    llm_record: Any,
) -> None:
    """Record a RunStep via the runtime's DB session."""
    run_id = context.run_id
    if run_id is None or runtime._db is None:
        return

    now = datetime.now(tz=UTC)
    sequence = await runtime.next_step_sequence()
    step = RunStep(
        run_id=run_id,
        sequence=sequence,
        iteration=context.iteration,
        phase=phase,
        input=input_data,
        output=output_data,
        llm_call=llm_record.model_dump() if hasattr(llm_record, "model_dump") else llm_record,
        started_at=now,
        completed_at=now,
    )
    runtime._db.add(step)
    runtime.record_pattern_step(step)

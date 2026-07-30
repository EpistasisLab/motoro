"""Plan phase — converts strategy into ordered executable steps."""

from __future__ import annotations

from typing import Any

import structlog
from pydantic import ValidationError

from agentic_core.engine.context import RunContext
from agentic_core.engine.phase import PhaseResult
from agentic_core.mcp.adapters import format_tools_for_prompt
from agentic_core.schemas.llm import PlanOutput, PlanStep, ReasonOutput, SenseOutput
from agentic_core.services.llm_service import LLMService

log = structlog.get_logger()

PLAN_SYSTEM_PROMPT = """\
You are the Plan module of an AI agent. Your job is to convert a high-level strategy \
into an ordered list of concrete, executable steps.

CRITICAL RULES FOR TOOL USAGE:
- When a step needs to call a tool, you MUST set the tool_name field to the exact tool \
name from the available tools list and set tool_args to a JSON object with the required parameters.
- Do NOT describe tool calls in the description text. The tool_name and tool_args fields \
are what actually trigger tool execution. If tool_name is null, no tool will be called.
- Only reference tools that are listed as available. Use the exact tool name as shown.
- Each step must be a single, specific action.
- If no tool is needed for a step (e.g., composing text), set tool_name to null.
- If the goal is already satisfied and no action is needed, set is_complete=true.
- Steps should be ordered by execution dependency.
"""


def _build_plan_prompt(
    sense_output: SenseOutput,
    reason_output: ReasonOutput,
    hitl_resolution: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build the LLM message list for the Plan phase."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
    ]

    context_parts: list[str] = [
        f"Agent goal: {sense_output.agent_goal}",
        f"Strategy: {reason_output.strategy}",
        f"Requires tools: {reason_output.requires_tools}",
    ]

    if hitl_resolution and hitl_resolution.get("action") == "provide_input":
        human_input = hitl_resolution.get("input_text", "")
        if human_input:
            context_parts.append(f"HUMAN REVIEWER INPUT (incorporate this guidance into your plan): {human_input}")

    if sense_output.available_tools:
        tools_section = format_tools_for_prompt(sense_output.available_tools)
        if tools_section:
            context_parts.append(tools_section)
            context_parts.append(
                "EXAMPLE: To call a tool, set tool_name and tool_args on the step:\n"
                '  {"action": "get file", "tool_name": "get_file_contents", '
                '"tool_args": {"owner": "user", "repo": "myrepo", "path": "README.md"}}'
            )

    if reason_output.key_observations:
        context_parts.append("Key observations:\n" + "\n".join(f"- {o}" for o in reason_output.key_observations))

    messages.append({"role": "system", "content": "\n".join(context_parts)})
    from agentic_core.security.prompt_injection import fence  # noqa: PLC0415

    messages.append({"role": "user", "content": fence(sense_output.user_input)})

    return messages


class PlanPhase:
    """Tactical layer: breaks strategy into ordered executable steps."""

    def __init__(self, llm_service: LLMService) -> None:
        self._llm = llm_service

    @property
    def name(self) -> str:
        return "plan"

    async def execute(self, context: RunContext) -> PhaseResult:
        """Convert the strategy into an ordered step list.

        On Instructor validation failure (issue #683), falls back to a
        single-step ``PlanOutput`` with ``requires_tools=False`` rather than
        propagating the raw ``ValidationError`` to the caller.
        """
        sense_output = context.phase_outputs.get("sense")
        reason_output = context.phase_outputs.get("reason")

        if not isinstance(sense_output, SenseOutput):
            raise ValueError("Plan phase requires SenseOutput in context")
        if not isinstance(reason_output, ReasonOutput):
            raise ValueError("Plan phase requires ReasonOutput in context")

        hitl_resolution = context.metadata.get("hitl_resolution")
        messages = _build_plan_prompt(sense_output, reason_output, hitl_resolution)

        try:
            result, llm_record = await self._llm.complete(
                config=context.model_config,
                messages=messages,
                response_model=PlanOutput,
            )
        except ValidationError as exc:
            # Issue #683: structured-output validation exhausted all retries.
            # Produce a degraded single-step fallback so the run can complete.
            log.warning(
                "plan.validation_fallback",
                error=str(exc)[:500],
                component="plan",
            )
            result = PlanOutput(
                steps=[
                    PlanStep(
                        action="respond",
                        description=("Structured output validation failed. Providing best-effort response to user."),
                        expected_outcome="User receives a response",
                    )
                ],
                is_complete=False,
            )
            llm_record = None

        return PhaseResult(output=result, llm_call=llm_record)

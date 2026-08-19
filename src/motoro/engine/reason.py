"""Reason phase — interprets input and formulates a high-level strategy."""

from __future__ import annotations

import structlog
from pydantic import ValidationError

from motoro.engine.context import RunContext
from motoro.engine.phase import PhaseResult
from motoro.mcp.adapters import format_tools_for_prompt
from motoro.schemas.llm import ReasonOutput, SenseOutput
from motoro.services.llm_service import LLMService

log = structlog.get_logger()

REASON_SYSTEM_PROMPT = """\
You are the Reason module of an AI agent. Your job is to analyze the input and \
formulate a HIGH-LEVEL STRATEGY for achieving the agent's goal.

Rules:
- Analyze the user's input and the agent's goal
- Formulate a strategy — what approach should be taken?
- Do NOT generate a plan or list of steps (that is the Plan module's job)
- Do NOT take any actions or call any tools
- Note whether the strategy requires tool usage
- Show your reasoning trace

Confidence calibration (0.0 to 1.0):
  0.95-1.0 — Reserved for tautologies or identity lookups with zero ambiguity
  0.85-0.94 — Straightforward task, well-understood domain, no missing info
  0.70-0.84 — Likely correct approach but some uncertainty (ambiguous phrasing, \
multiple valid strategies, mild knowledge gaps)
  0.50-0.69 — Genuinely uncertain; the task is ambiguous, under-specified, or \
at the edge of your knowledge
  0.30-0.49 — Significant doubt; key information is missing or the domain is \
unfamiliar
  0.00-0.29 — Near-guessing; you lack the knowledge or context to form a \
reliable strategy

Calibration rule: most real-world requests should land between 0.6 and 0.85. \
If you find yourself above 0.90, verify that there is truly zero ambiguity \
before committing to that score.
"""


def _build_reason_prompt(sense_output: SenseOutput) -> list[dict[str, str]]:
    """Build the LLM message list for the Reason phase."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": REASON_SYSTEM_PROMPT},
    ]

    if sense_output.system_prompt:
        messages.append(
            {
                "role": "system",
                "content": f"Agent system prompt: {sense_output.system_prompt}",
            }
        )

    context_parts: list[str] = [f"Agent goal: {sense_output.agent_goal}"]

    if sense_output.available_tools:
        tools_section = format_tools_for_prompt(sense_output.available_tools)
        if tools_section:
            context_parts.append(tools_section)

    if sense_output.memories:
        memory_texts = [str(m.get("content", "")) for m in sense_output.memories]
        context_parts.append("Relevant memories:\n" + "\n".join(f"- {t}" for t in memory_texts))

    if sense_output.conversation_history:
        context_parts.append(f"Prior conversation: {len(sense_output.conversation_history)} exchanges")

    messages.append({"role": "system", "content": "\n".join(context_parts)})
    from motoro.security.prompt_injection import fence  # noqa: PLC0415

    messages.append({"role": "user", "content": fence(sense_output.user_input)})

    return messages


class ReasonPhase:
    """Cognitive core: interprets perception against goals and formulates strategy."""

    def __init__(self, llm_service: LLMService) -> None:
        self._llm = llm_service

    @property
    def name(self) -> str:
        return "reason"

    async def execute(self, context: RunContext) -> PhaseResult:
        """Analyze input and produce a strategy.

        On Instructor validation failure (issue #683), falls back to a
        degraded ``ReasonOutput`` with low confidence rather than surfacing
        the raw ``ValidationError`` to the caller.  This keeps the run alive
        through the Plan phase with an explicit low-confidence signal.
        """
        sense_output = context.phase_outputs.get("sense")
        if not isinstance(sense_output, SenseOutput):
            raise ValueError("Reason phase requires SenseOutput in context.phase_outputs['sense']")

        messages = _build_reason_prompt(sense_output)

        try:
            result, llm_record = await self._llm.complete(
                config=context.model_config,
                messages=messages,
                response_model=ReasonOutput,
            )
        except ValidationError as exc:
            # Issue #683: structured-output validation exhausted all retries.
            # Produce a degraded fallback instead of propagating the error.
            log.warning(
                "reason.validation_fallback",
                error=str(exc)[:500],
                component="reason",
            )
            result = ReasonOutput(
                strategy=(
                    "Unable to formulate a validated strategy due to LLM output "
                    "format issues. Proceeding with best-effort approach."
                ),
                confidence=0.1,
                key_observations=["Structured output validation failed"],
                requires_tools=False,
                reasoning_trace=f"Validation failure: {str(exc)[:200]}",
            )
            llm_record = None

        # Issue #827: clamp confidence to [0.0, 1.0] even if the validator
        # accepted a value from a subsequent round that slipped past ge/le.
        clamped = max(0.0, min(1.0, result.confidence))
        if clamped != result.confidence:
            log.warning(
                "reason.confidence_clamped",
                original=result.confidence,
                clamped=clamped,
                component="reason",
            )
            result = result.model_copy(update={"confidence": clamped})

        return PhaseResult(output=result, llm_call=llm_record)

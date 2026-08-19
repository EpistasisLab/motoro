"""A stand-in for the LLM bridge, so the loop can be tested without a provider.

The four methods below are the *entire* surface the SRPA phases use, and their
return shapes are not uniform — which is worth pinning here rather than
rediscovering:

===========================  ===================================================
``complete``                 ``(response_model_instance, LLMCallRecord)``
``complete_text``            ``(str, LLMCallRecord)``
``complete_with_tools``      a single ``ToolCompletion`` (carries ``.record``)
``select_tool``              ``(use_tool, tool_name, tool_args, LLMCallRecord)``
===========================  ===================================================

If a future slice widens that surface, the end-to-end test fails here first,
which is the point.
"""

from __future__ import annotations

import uuid
from typing import Any

from motoro.schemas.llm import (
    ActOutput,
    LLMCallRecord,
    PlanOutput,
    PlanStep,
    ReasonOutput,
    SenseOutput,
    ToolCompletion,
)

FINAL_ANSWER = "stubbed final answer"


def call_record() -> LLMCallRecord:
    """A plausible non-zero record, so token/cost assertions mean something."""
    return LLMCallRecord(
        model="stub-model",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=1,
        cost_estimate=0.0,
    )


def canned(model: type) -> Any:
    """A minimal *valid* instance of whichever phase-output model is requested."""
    if model is SenseOutput:
        return SenseOutput(user_input="stub input", agent_goal="stub goal", system_prompt="stub prompt")
    if model is ReasonOutput:
        return ReasonOutput(strategy="answer directly", confidence=0.9, requires_tools=False)
    if model is PlanOutput:
        return PlanOutput(
            steps=[PlanStep(action="respond", description="Answer the question directly.")],
            is_complete=True,
        )
    if model is ActOutput:
        return ActOutput(results=[], final_response=FINAL_ANSWER, should_continue=False)
    # output_contract builds one-off pydantic models at runtime (every field
    # optional, per its own contract), so there's nothing to enumerate here by
    # name — construct with defaults rather than raising for every future one.
    try:
        return model()
    except Exception:
        raise AssertionError(f"StubLLM has no canned value for {model.__name__}") from None


class StubLLM:
    """Records which methods were called, and answers with canned structured output."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def principal_id(self) -> uuid.UUID | None:
        return None

    async def complete(
        self,
        config: Any = None,
        messages: Any = None,
        response_model: type | None = None,
        max_retries: int = 3,
        principal_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> tuple[Any, LLMCallRecord]:
        assert response_model is not None
        self.calls.append(f"complete:{response_model.__name__}")
        return canned(response_model), call_record()

    async def complete_text(self, *args: Any, **kwargs: Any) -> tuple[str, LLMCallRecord]:
        self.calls.append("complete_text")
        return FINAL_ANSWER, call_record()

    async def complete_with_tools(self, *args: Any, **kwargs: Any) -> ToolCompletion:
        self.calls.append("complete_with_tools")
        return ToolCompletion(text=FINAL_ANSWER, tool_calls=[], record=call_record())

    async def select_tool(
        self, *args: Any, **kwargs: Any
    ) -> tuple[bool, str | None, dict[str, object] | None, LLMCallRecord]:
        self.calls.append("select_tool")
        return False, None, None, call_record()

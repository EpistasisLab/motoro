"""Pydantic schemas for the ReasonAct (Thought-Action-Observation) pattern.

These are *records* of what the loop did, not generation targets. The model
emits native tool calls, so the provider — not a response schema — enforces the
shape of an action. What is left for these models to do is give the trajectory a
stable, queryable form in ``RunStep.output`` and ``RunContext.metadata``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ReasonActAction(StrEnum):
    """How a ReasonAct turn resolved.

    There is no ``request_info`` member: a turn that needs clarification calls
    ``final_answer`` with the question as its answer, which is what the user
    sees either way.
    """

    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"


class ReasonActToolCall(BaseModel):
    """One native tool call issued in a ReasonAct turn."""

    id: str = Field(description="Call id, echoed back on the matching tool-result turn")
    tool_name: str = Field(description="Tool name after translation back from the provider-sanitized form")
    tool_args: dict[str, Any] = Field(default_factory=dict)
    provider_name: str = Field(
        default="",
        description=(
            "Name exactly as the provider returned it. Replaying the turn in the "
            "message history has to use this form, not the resolved MCP name, or "
            "it will not match the schema that was bound."
        ),
    )


class ReasonActStep(BaseModel):
    """One recorded turn of the ReasonAct loop.

    A turn is a single completion, so the thought and the actions it motivated
    come from the same context — ``tool_calls`` is what the model actually
    asked for, not a second model's reading of what it asked for.
    """

    thought: str = Field(default="", description="Assistant prose accompanying the calls")
    action: ReasonActAction = Field(description="Whether the turn issued tool calls or concluded")
    tool_calls: list[ReasonActToolCall] = Field(
        default_factory=list,
        description="Calls issued this turn; several may run in parallel",
    )
    final_answer: str | None = Field(
        default=None,
        description="The conclusive answer (set when action='final_answer')",
    )


class ReasonActObservation(BaseModel):
    """Formatted observation from a tool call result."""

    tool_call_id: str = Field(default="", description="Id of the call this observation answers")
    tool_name: str = Field(description="Tool that was called")
    tool_args: dict[str, Any] = Field(default_factory=dict, description="Arguments that were passed")
    result: str = Field(description="Tool execution result, as shown to the model")
    success: bool = Field(default=True, description="Whether the tool call succeeded")

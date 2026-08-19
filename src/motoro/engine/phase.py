"""Phase protocol — interface that all agentic loop phases implement."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from motoro.engine.context import RunContext
from motoro.schemas.llm import LLMCallRecord


class PhaseResult(BaseModel):
    """Result returned by a phase execution."""

    output: BaseModel
    llm_call: LLMCallRecord | None = None

    model_config = {"arbitrary_types_allowed": True}


class Phase(Protocol):
    """Protocol for agentic loop phases.

    Each phase (Sense, Reason, Plan, Act) implements this interface.
    The runtime calls execute() and records the result.
    """

    @property
    def name(self) -> str:
        """Phase name (sense, reason, plan, act)."""
        ...

    async def execute(self, context: RunContext) -> PhaseResult:
        """Execute this phase and return the result.

        Args:
            context: The run's working memory. Phases read from and write to it.

        Returns:
            PhaseResult with the phase output and optional LLM call record.
        """
        ...

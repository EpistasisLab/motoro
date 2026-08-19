"""Single Agent Baseline — the default pattern.

Wires user-configured ``max_iterations`` and ``stop_on_first_success`` onto
the run context so that the values configured through the UI / API have an
effect at runtime.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel

from motoro.engine.context import RunContext
from motoro.engine.patterns.base import HookCallable, HookPoint, PatternPlugin
from motoro.engine.patterns.registry import PluginRegistry
from motoro.models.pattern import PatternCategory, PatternPhase
from motoro.schemas.llm import ActOutput


@PluginRegistry.register
class SingleAgentBaselinePlugin(PatternPlugin):
    """Default Sense → Reason → Plan → Act loop with no additional reasoning.

    This is the fallback pattern used when no ``pattern_config`` is set, and
    is also selectable directly. It honours two configuration parameters:

    - ``max_iterations`` — Hard cap on SRPA loop iterations. Wired onto
      ``context.max_iterations`` in ``on_activate`` so the orchestrator's
      iteration ``range`` reflects the user-configured value.
    - ``stop_on_first_success`` — When ``True`` (default), the loop stops as
      soon as ``ActOutput.should_continue`` is ``False``; when ``False``, a
      ``LOOP_CONTROL`` hook overrides ``should_continue`` to keep the loop
      running until ``max_iterations`` is hit.
    """

    slug = "single_agent_baseline"
    category = PatternCategory.EXECUTION

    display_name: ClassVar[str] = "Single Agent Baseline"
    description: ClassVar[str] = (
        "One LLM-backed agent operates independently with a defined objective, access to tools, "
        "and an iterative Sense-Reason-Plan-Act cycle. This is the foundational pattern from "
        "which all others build."
    )
    complexity_phase: ClassVar[PatternPhase] = PatternPhase.BASIC
    # Note: ARES's catalog row also declares a `confidence_threshold` property.
    # It is omitted here because this plugin never reads it — the schema
    # describes the parameters `configure` actually honours, which is the point
    # of the plugin owning it.
    configuration_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "max_iterations": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "maximum": 100,
                "description": "Maximum agentic loop iterations",
            },
            "stop_on_first_success": {
                "type": "boolean",
                "default": True,
                "description": "Terminate the loop as soon as the agent reports success.",
            },
        },
    }

    def configure(self, params: dict[str, Any]) -> None:
        self.max_iterations: int = int(params.get("max_iterations", 10))
        self.stop_on_first_success: bool = bool(params.get("stop_on_first_success", True))

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        mi = config.get("max_iterations")
        if mi is not None and (not isinstance(mi, int) or mi < 1):
            errors.append("max_iterations must be a positive integer")
        sfs = config.get("stop_on_first_success")
        if sfs is not None and not isinstance(sfs, bool):
            errors.append("stop_on_first_success must be a boolean")
        return errors

    async def on_activate(self, context: RunContext) -> None:
        """Apply user-configured ``max_iterations`` to the context."""
        context.max_iterations = self.max_iterations

    def get_hooks(self) -> dict[HookPoint, list[HookCallable]]:
        return {HookPoint.LOOP_CONTROL: [self._loop_control]}

    async def _loop_control(self, context: RunContext, _phase_output: BaseModel | None) -> BaseModel | None:
        """Honour ``stop_on_first_success`` by forcing continuation when disabled.

        The orchestrator exits the loop when ``ActOutput.should_continue`` is
        ``False``. When the user opts out of stop-on-first-success, we
        rewrite the most-recent ``act`` output so the loop keeps running
        until ``context.max_iterations`` is reached.
        """
        if self.stop_on_first_success:
            return None

        act_output = context.phase_outputs.get("act")
        if isinstance(act_output, ActOutput) and not act_output.should_continue:
            context.record_phase_output(
                "act",
                ActOutput(
                    results=act_output.results,
                    final_response=act_output.final_response,
                    should_continue=True,
                ),
            )
        return None

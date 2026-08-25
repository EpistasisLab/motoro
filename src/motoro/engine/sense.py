"""Sense phase — retrieves memories, and forwards the context already resolved."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from motoro.engine.context import RunContext
from motoro.engine.phase import PhaseResult
from motoro.schemas.llm import SenseOutput

if TYPE_CHECKING:
    from motoro.engine.ports import MemoryServicePort

logger = logging.getLogger(__name__)


class SensePhase:
    """Perception layer. Does NOT call the LLM — this is a data gathering phase.

    Read :meth:`execute` before adding anything here: memories are the only
    field this phase actually *gathers*, and the reason is a contract, not an
    accident of how it grew.

    A run's inputs reach the engine by one of three routes, and only one of them
    is this phase:

    1. **Capability** — the model, the execution pattern, the tool set, the
       skills. Resolved before the engine starts and carried on
       :class:`~motoro.engine.context.RunContext` as configuration. These change
       what the agent *can do*; they are never prose in a prompt, so there is
       nothing for a perception step to collect.
    2. **Reference** — an id pointing at data that lives somewhere else: a
       workspace, a dataset, an artifact. Bound into every MCP tool call's
       ambient request ``_meta`` (``RunContext.workspace_id`` and
       ``RunContext.ambient_meta``) so a tool receives it out-of-band. The
       contents are loaded by the tool, on demand — never read into context up
       front, and never transcribed by the model.
    3. **Content** — text that genuinely belongs in the prompt, and is small
       enough to belong there: the user input, the goal, retrieved memories.
       This is the only route that passes through Sense, and memories are the
       only part of it Sense fetches rather than forwards.

    So the honest summary is: retrieve memories (episodic via the injected
    MemoryService, plus working memory when configured), and snapshot the
    already-resolved rest into a :class:`SenseOutput` for the phases downstream.

    A new kind of input almost always belongs to route 1 or 2 — that is where
    it stays out of the context window. Reaching for Sense usually means it was
    misclassified as route 3.
    """

    def __init__(
        self,
        memory_service: MemoryServicePort | None = None,
        memory_top_k: int = 5,
        memory_threshold: float | None = None,
    ) -> None:
        self._memory_service = memory_service
        self._memory_top_k = memory_top_k
        self._memory_threshold = memory_threshold

    @property
    def name(self) -> str:
        return "sense"

    async def execute(self, context: RunContext) -> PhaseResult:
        """Retrieve memories if configured, then snapshot the context as SenseOutput.

        The five non-memory fields of the output are copies of values
        ``RunContext`` already held on entry — this phase does not source them.
        Note that not every pattern reads them back: ReAct, for one, consumes
        only ``agent_goal``/``memories``/``user_input`` and re-reads the rest off
        the context directly, so memory retrieval is Sense's entire contribution
        to it.
        """
        memories = list(context.memories)

        if (
            self._memory_service is not None
            and context.agent_goal
            and context.memory_config_data.get("episodic_memory_enabled", False)
        ):
            memories = await self._retrieve_memories(context)
            context.memories = memories

        # Append working memory context if available
        if context.working_memory_manager is not None:
            try:
                wm_context = await context.working_memory_manager.get_context_string()
                if wm_context:
                    memories = [
                        *memories,
                        {"type": "working", "content": wm_context},
                    ]
            except Exception:
                logger.exception("Failed to retrieve working memory context")

        output = SenseOutput(
            user_input=context.user_input,
            agent_goal=context.agent_goal,
            system_prompt=context.system_prompt,
            conversation_history=context.conversation_history,
            available_tools=context.available_tools,
            memories=memories,
        )
        return PhaseResult(output=output, llm_call=None)

    async def _retrieve_memories(self, context: RunContext) -> list[dict[str, Any]]:
        """Retrieve relevant memories from the memory service.

        Skips the embedding call entirely if the agent has no stored memories,
        avoiding the cost of loading and running the embedding model.
        """
        assert self._memory_service is not None
        query = context.user_input or context.agent_goal
        if context.agent_id is None:
            return []
        try:
            total = await self._memory_service.count(context.agent_id)
            if total == 0:
                return []

            results = await asyncio.wait_for(
                self._memory_service.recall(
                    agent_id=context.agent_id,
                    query=query,
                    top_k=self._memory_top_k,
                    threshold=self._memory_threshold,
                ),
                timeout=30,
            )
        except TimeoutError:
            logger.error("Memory recall timed out after 30s during Sense phase")
            return []
        except Exception:
            logger.exception("Memory retrieval failed during Sense phase")
            return []

        context.metadata["memory_recalled_count"] = len(results)
        context.metadata["memory_injected_count"] = len(results)

        return [
            {
                "type": entry.type.value,
                "content": entry.content,
                "score": round(score, 4),
                "memory_id": str(entry.id),
            }
            for entry, score in results
        ]

"""Sense phase — gathers input, context, and relevant memories for the Reason phase."""

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
    """Perception layer: collects user input, agent context, tools, and memories.

    Does NOT call the LLM — this is a data gathering phase.
    When a MemoryService is injected, relevant episodic and semantic memories
    are retrieved and injected into the RunContext before the Reason phase.
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
        """Gather and structure all available input, injecting memories if configured."""
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

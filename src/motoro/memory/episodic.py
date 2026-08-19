"""Episodic memory store — per-run summaries for cross-run learning.

Each method opens and closes its own session, matching every other public entry
point in core — see ``memory.semantic``'s module docstring for why, and the cost
that trade carries.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select

from motoro.models.memory import MemoryEntry, MemoryType
from motoro.observability.tracing import get_tracer

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession

    from motoro.memory.embedding import EmbeddingService
    from motoro.schemas.agent import ModelConfig
    from motoro.services.llm_service import LLMService

logger = logging.getLogger(__name__)
_tracer = get_tracer("memory.episodic")

_SUMMARIZE_PROMPT = """\
Summarize this agent run in 3-5 sentences. Include:
- The user's goal
- The strategy used
- Key tools or actions taken
- The outcome
- Any lessons or notable observations

Be concise and factual."""


def _session(reason: str) -> AbstractAsyncContextManager[AsyncSession]:
    from motoro.models.database import system_session

    return system_session(reason=f"memory.episodic: {reason}")


class EpisodicMemoryStore:
    """Stores and retrieves run summaries for cross-run learning."""

    def __init__(self, llm_service: LLMService, embedding_service: EmbeddingService) -> None:
        self._llm = llm_service
        self._embed = embedding_service

    async def store_run_summary(
        self,
        run_id: uuid.UUID | None,
        agent_id: uuid.UUID,
        summary: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Persist a run summary as an episodic memory entry with embedding.

        Args:
            run_id: The AgentRun this summary belongs to, or ``None`` for an
                episodic entry recorded outside a real run. ``run_id`` is a real
                foreign key into ``agent_runs``, so a caller cannot substitute a
                placeholder UUID here the way ARES's ``MemoryService.remember``
                did (``uuid.UUID(int=0)``) — that value is not a row in
                ``agent_runs`` and would violate the constraint. ``None`` is the
                only legitimate way to say "no run".
            agent_id: The agent that executed the run.
            summary: Human-readable summary text.
            metadata: Optional extra metadata (e.g. success/failure, cost).

        Returns:
            The persisted MemoryEntry.
        """
        try:
            embedding: list[float] | None = await self._embed.embed(summary)
            embedding_status = "ok"
        except Exception:
            logger.warning(
                "Failed to generate embedding for episodic memory (run %s); "
                "storing without embedding — semantic search will not include this entry",
                run_id,
            )
            embedding = None
            embedding_status = "failed"
        entry = MemoryEntry(
            agent_id=agent_id,
            run_id=run_id,
            type=MemoryType.EPISODIC,
            content=summary,
            embedding=embedding,
            # Stamp the embedding model identity even when the remote embed
            # call failed, so backfill can rebuild later under the same model.
            embedding_model=self._embed.model,
            embedding_version=self._embed.version,
            embedding_dimensions=len(embedding) if embedding is not None else None,
            meta=metadata or {},
            # Record embedding outcome so a backfill job can retry rows whose
            # embedding generation failed.
            embedding_status=embedding_status,
        )
        async with _session("store_run_summary") as db:
            db.add(entry)
            await db.commit()
            await db.refresh(entry)
        return entry

    async def get_recent(
        self,
        agent_id: uuid.UUID,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Retrieve recent run summaries in reverse chronological order.

        Args:
            agent_id: Filter to this agent's episodes.
            limit: Maximum number of entries to return.

        Returns:
            List of MemoryEntry objects, newest first.
        """
        async with _session("get_recent") as db:
            result = await db.execute(
                select(MemoryEntry)
                .where(MemoryEntry.agent_id == agent_id)
                .where(MemoryEntry.type == MemoryType.EPISODIC)
                .order_by(MemoryEntry.created_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    async def count(self, agent_id: uuid.UUID) -> int:
        """Count episodic memory entries for an agent."""
        async with _session("count") as db:
            result = await db.execute(
                select(func.count())
                .select_from(MemoryEntry)
                .where(MemoryEntry.agent_id == agent_id)
                .where(MemoryEntry.type == MemoryType.EPISODIC)
            )
            return result.scalar_one()

    async def search_relevant(
        self,
        agent_id: uuid.UUID,
        query: str,
        top_k: int = 5,
        threshold: float | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """Semantic search within this agent's episodic memories.

        Args:
            agent_id: Filter to this agent's episodes.
            query: Query text to embed and compare.
            top_k: Maximum number of results.
            threshold: Cosine *distance* upper bound. Rows with
                ``cosine_distance > threshold`` are excluded. 0 = identical,
                2 = opposite.
            score_threshold: Cosine *similarity* lower bound. Rows whose
                returned ``score`` (``1 - distance``) is below this are
                excluded. Range: -1 to 1 (1 = identical). Mutually exclusive
                with ``threshold``.
            metadata_filter: Optional dict of key/value pairs that must all
                match inside the ``meta`` JSON column. For example,
                ``{"status": "success"}`` restricts results to successful
                runs. Each key is matched with a JSON path equality check
                (``meta->>'key' = 'value'``). Values are compared as strings.

        Returns:
            List of (MemoryEntry, score) tuples, most relevant first. Score is
            cosine similarity (1 = identical, -1 = opposite). Ties are broken
            by ``created_at DESC, id DESC`` so identical input is identical
            output.

        Raises:
            ValueError: If both ``threshold`` and ``score_threshold`` are set.
        """
        from sqlalchemy import type_coerce
        from sqlalchemy.dialects.postgresql import JSONB

        if threshold is not None and score_threshold is not None:
            raise ValueError(
                "threshold (distance) and score_threshold (similarity) are mutually exclusive; pass only one"
            )

        with _tracer.start_as_current_span(
            "memory.episodic.search_relevant",
            attributes={
                "memory.store": "episodic",
                "memory.top_k": top_k,
                "memory.query_length": len(query),
            },
        ) as span:
            query_embedding = await self._embed.embed(query)

            distance_expr = MemoryEntry.embedding.cosine_distance(query_embedding)

            stmt = (
                select(MemoryEntry, distance_expr.label("distance"))
                .where(MemoryEntry.agent_id == agent_id)
                .where(MemoryEntry.type == MemoryType.EPISODIC)
                # Filter by the current embedding model identity so we never
                # compare vectors produced by different embedding backends.
                # The model filter (indexed) also implicitly excludes
                # NULL-embedding rows because writers stamp the two columns
                # together.
                .where(MemoryEntry.embedding_model == self._embed.model)
                .where(MemoryEntry.embedding.is_not(None))
                .order_by(
                    distance_expr.asc(),
                    MemoryEntry.created_at.desc(),
                    MemoryEntry.id.desc(),
                )
                .limit(top_k)
            )

            if threshold is not None:
                stmt = stmt.where(distance_expr <= threshold)
            elif score_threshold is not None:
                stmt = stmt.where(distance_expr <= (1.0 - score_threshold))

            if metadata_filter:
                for key, value in metadata_filter.items():
                    stmt = stmt.where(type_coerce(MemoryEntry.meta, JSONB)[key].as_string() == str(value))

            async with _session("search_relevant") as db:
                result = await db.execute(stmt)
                rows = result.all()
            span.set_attribute("memory.results_returned", len(rows))
            return [(row[0], float(1.0 - row[1])) for row in rows]

    async def generate_run_summary(
        self,
        user_input: str,
        final_output: str,
        model_config: ModelConfig,
        status: str = "completed",
        actions_taken: list[str] | None = None,
    ) -> str:
        """Generate a run summary via LLM call.

        Args:
            user_input: The original user request.
            final_output: The agent's final response.
            model_config: LLM config for the summarization call.
            status: Run status (completed/failed).
            actions_taken: List of action descriptions from the run.

        Returns:
            A text summary of the run.
        """
        actions_str = ""
        if actions_taken:
            actions_str = "\n\nActions taken:\n" + "\n".join(f"- {a}" for a in actions_taken)

        messages = [
            {"role": "system", "content": _SUMMARIZE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User goal: {user_input}\nRun status: {status}{actions_str}\n\nFinal output: {final_output}"
                ),
            },
        ]
        text, _ = await self._llm.complete_text(model_config, messages)
        return text

"""Unified memory service — single interface over episodic and semantic backends.

Conforms to :class:`motoro.engine.ports.MemoryServicePort`, the four-method
surface the SRPA loop actually calls (``count``, ``recall``, and
``episodic.generate_run_summary``/``store_run_summary``). Everything else here —
``remember``, ``forget`` — is for a product that wants to write or manage memory
directly, outside a run.

Not pulled from ARES's equivalent: ``list_entries`` (takes a ``Viewer`` and scopes
by isolation, out of scope for this slice) and ``create_from_request`` (adapts an
API request schema; core has no HTTP layer, and this had no other caller in ARES
either). ``get_context`` is also dropped — it had zero callers in ARES itself.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from motoro.memory.embedding import EmbeddingService
from motoro.memory.episodic import EpisodicMemoryStore
from motoro.memory.semantic import SemanticMemoryStore
from motoro.models.memory import MemoryEntry, MemoryType

if TYPE_CHECKING:
    from motoro.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class MemoryService:
    """Unified interface over episodic and semantic memory backends.

    Abstracts the two persistent memory types behind a single API so callers
    don't need to know which backend handles each type.
    """

    def __init__(
        self,
        llm_service: LLMService,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self._embed = embedding_service or EmbeddingService()
        self.semantic = SemanticMemoryStore(self._embed)
        self.episodic = EpisodicMemoryStore(llm_service, self._embed)

    async def remember(
        self,
        agent_id: uuid.UUID,
        content: str,
        *,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        metadata: dict[str, Any] | None = None,
        run_id: uuid.UUID | None = None,
        expires_at: datetime | None = None,
    ) -> MemoryEntry:
        """Store a memory entry to the appropriate backend.

        Args:
            agent_id: The agent that owns this memory.
            content: Text content to store.
            memory_type: Whether to store as semantic or episodic.
            metadata: Optional metadata dict.
            run_id: Optional run that produced this memory.
            expires_at: Optional expiry timestamp.

        Returns:
            The created MemoryEntry.
        """
        meta = metadata or {}
        if memory_type == MemoryType.SEMANTIC:
            return await self.semantic.store(
                content=content,
                metadata=meta,
                agent_id=agent_id,
                run_id=run_id,
                expires_at=expires_at,
            )
        # run_id is a real foreign key into agent_runs (see models/memory.py), so
        # an omitted run_id must stay None rather than being papered over with a
        # placeholder — ARES's equivalent substituted uuid.UUID(int=0), which is
        # not a row in agent_runs and would violate the constraint the moment
        # this path is actually exercised outside a real run.
        return await self.episodic.store_run_summary(
            run_id=run_id,
            agent_id=agent_id,
            summary=content,
            metadata=meta,
        )

    async def recall(
        self,
        agent_id: uuid.UUID,
        query: str,
        types: list[MemoryType] | None = None,
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """Search across memory types ranked by cosine similarity.

        Falls back to recency-based retrieval when embedding search fails or
        returns no results (e.g. missing API key, NULL embeddings).

        Args:
            agent_id: The agent whose memories to search.
            query: The search query.
            types: Which memory types to include (None = all types).
            top_k: Maximum results per type.
            threshold: If set, exclude entries with cosine distance > threshold.

        Returns:
            Combined and deduplicated list of (MemoryEntry, score) sorted by score.
        """
        effective_types = types or list(MemoryType)
        results: list[tuple[MemoryEntry, float]] = []

        if MemoryType.SEMANTIC in effective_types:
            try:
                # recall always operates within a specific agent's scope. Use
                # scope='agent' to ensure we never leak rows from other agents
                # (or accidentally return only global rows).
                sem = await self.semantic.search(
                    query,
                    top_k=top_k,
                    agent_id=agent_id,
                    threshold=threshold,
                    scope="agent",
                )
                results.extend(sem)
            except Exception:
                logger.warning("Semantic memory search failed for agent %s, skipping", agent_id, exc_info=True)

        if MemoryType.EPISODIC in effective_types:
            try:
                epi = await self.episodic.search_relevant(agent_id, query, top_k=top_k, threshold=threshold)
                results.extend(epi)
            except Exception:
                logger.warning("Episodic embedding search failed for agent %s, skipping", agent_id, exc_info=True)

        # Fallback: when embedding search returned nothing, use recency.
        if not results and MemoryType.EPISODIC in effective_types:
            try:
                recent = await self.episodic.get_recent(agent_id, limit=top_k)
                results.extend((entry, 0.0) for entry in recent)
            except Exception:
                logger.warning("Recency-based memory fallback also failed for agent %s", agent_id, exc_info=True)

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def count(self, agent_id: uuid.UUID) -> int:
        """Count total memory entries for an agent (both types). Fast DB query, no embedding."""
        episodic = await self.episodic.count(agent_id)
        semantic = await self.semantic.count(agent_id)
        return episodic + semantic

    async def forget(self, entry_id: uuid.UUID) -> bool:
        """Delete a memory entry by ID (works for both types).

        Returns:
            True if deleted, False if not found.
        """
        return await self.semantic.delete(entry_id)

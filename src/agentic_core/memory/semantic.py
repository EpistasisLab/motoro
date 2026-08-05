"""Semantic memory store — embedding-based storage and retrieval via pgvector.

Each method opens and closes its own session, like every other public entry
point in core (see ``runner.py``'s module docstring) — there is no ``db``
parameter here, and never was one in this store's constructor. The cost: a
caller that needs both a semantic and an episodic lookup for the same query
(``MemoryService.recall`` does) pays for two round trips rather than sharing
one transaction. That is the same trade the rest of core already made in
exchange for never handing a session to a caller.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy import func, select

from agentic_core.models.memory import MemoryEntry, MemoryType
from agentic_core.observability.tracing import get_tracer

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession

    from agentic_core.memory.embedding import EmbeddingService

_tracer = get_tracer("memory.semantic")

# Explicit scope for semantic search so callers cannot accidentally leak rows
# across agents by passing agent_id=None.
#   - "agent"  -> only rows owned by the supplied agent_id (agent_id MUST be set)
#   - "global" -> only rows where agent_id IS NULL (cross-agent shared knowledge)
#   - "any"    -> no agent filter at all (admin / cross-agent retrieval)
SearchScope = Literal["agent", "global", "any"]


def _session(reason: str) -> AbstractAsyncContextManager[AsyncSession]:
    from agentic_core.models.database import system_session

    return system_session(reason=f"memory.semantic: {reason}")


class SemanticMemoryStore:
    """Stores and retrieves knowledge using cosine similarity over pgvector embeddings."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self._embed = embedding_service

    async def store(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
        agent_id: uuid.UUID | None = None,
        run_id: uuid.UUID | None = None,
        expires_at: datetime | None = None,
    ) -> MemoryEntry:
        """Embed content and persist as a semantic memory entry.

        Args:
            content: Text to embed and store.
            metadata: Arbitrary key/value metadata to attach.
            agent_id: Agent that owns this memory (None = global).
            run_id: Run that produced this memory (optional).
            expires_at: Optional expiry timestamp.

        Returns:
            The persisted MemoryEntry.
        """
        embedding = await self._embed.embed(content)
        entry = MemoryEntry(
            agent_id=agent_id,
            run_id=run_id,
            type=MemoryType.SEMANTIC,
            content=content,
            embedding=embedding,
            # Stamp the embedding model identity so retrieval can filter by
            # "rows produced by the same model" and never mix vector spaces.
            embedding_model=self._embed.model,
            embedding_version=self._embed.version,
            embedding_dimensions=len(embedding) if embedding is not None else None,
            meta=metadata or {},
            expires_at=expires_at,
            # The semantic store always succeeds (exceptions propagate), so the
            # status is always "ok" — unlike episodic, which has a fallback.
            embedding_status="ok",
        )
        async with _session("store") as db:
            db.add(entry)
            await db.commit()
            await db.refresh(entry)
        return entry

    async def search(
        self,
        query: str,
        top_k: int = 5,
        agent_id: uuid.UUID | None = None,
        threshold: float | None = None,
        score_threshold: float | None = None,
        scope: SearchScope = "agent",
        mmr_lambda: float | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """Search for semantically similar entries using cosine distance.

        Args:
            query: The search query to embed and compare.
            top_k: Maximum number of results to return.
            agent_id: The agent whose memories to search. Required when
                ``scope='agent'``; ignored when ``scope='global'`` or
                ``scope='any'``.
            threshold: Cosine *distance* upper bound. If set, only rows with
                ``cosine_distance(row, query) <= threshold`` are returned.
                Range: 0 (identical) to 2 (opposite). Use this when you already
                think in pgvector's distance space.
            score_threshold: Cosine *similarity* lower bound. If set, only rows
                whose returned ``score`` (``1 - distance``) is
                ``>= score_threshold`` are returned. Range: -1 to 1, with 1
                being identical. Prefer this when you think in similarity
                terms (matching how the returned ``score`` is shaped).

                ``threshold`` and ``score_threshold`` are mutually exclusive —
                passing both raises ``ValueError``.
            scope: Explicit agent scoping:
                - ``"agent"``  (default): only rows where ``agent_id`` matches
                  the supplied agent_id. ``agent_id`` MUST be non-None.
                - ``"global"``: only rows where ``agent_id IS NULL`` (shared
                  cross-agent knowledge, e.g. an advisor corpus).
                - ``"any"``: no agent filter (admin / cross-agent retrieval).
                  Use sparingly; this can leak data across agents.
            mmr_lambda: If set, apply Maximal Marginal Relevance re-ranking to
                improve result diversity. The DB retrieves ``top_k * 4``
                candidates (oversampling) and MMR selects ``top_k`` with the
                best balance of relevance and novelty. ``mmr_lambda=1.0`` is
                pure relevance (identical to no MMR); ``mmr_lambda=0.0`` is
                pure diversity. Typical value: 0.5-0.7. When ``None``
                (default) MMR is not applied.

        Returns:
            List of (MemoryEntry, score) tuples ordered by relevance descending
            (highest score first). Score is cosine similarity
            (``1 - cosine_distance``). Ties on distance are broken
            deterministically by ``created_at DESC, id DESC`` so identical
            inputs always produce the same row order.

        Raises:
            ValueError: If ``scope='agent'`` but ``agent_id`` is None, or if
                both ``threshold`` and ``score_threshold`` are supplied.
        """
        if scope == "agent" and agent_id is None:
            raise ValueError(
                "scope='agent' requires a non-None agent_id; "
                "use scope='global' for shared rows or scope='any' for cross-agent retrieval"
            )
        if threshold is not None and score_threshold is not None:
            raise ValueError(
                "threshold (distance) and score_threshold (similarity) are mutually exclusive; pass only one"
            )

        with _tracer.start_as_current_span(
            "memory.semantic.search",
            attributes={
                "memory.store": "semantic",
                "memory.top_k": top_k,
                "memory.scope": scope,
                "memory.query_length": len(query),
            },
        ) as span:
            query_embedding = await self._embed.embed(query)

            distance_expr = MemoryEntry.embedding.cosine_distance(query_embedding)

            stmt = (
                select(MemoryEntry, distance_expr.label("distance"))
                .where(MemoryEntry.type == MemoryType.SEMANTIC)
                # Filter by the current embedding model identity so we never
                # compare vectors generated by different models in the same
                # query. Because every row that has an embedding also has an
                # embedding_model (writes stamp them together), this filter
                # also excludes rows where embedding IS NULL, via the indexed
                # column rather than a separate IS NOT NULL scan.
                .where(MemoryEntry.embedding_model == self._embed.model)
                .where(MemoryEntry.embedding.is_not(None))
                .order_by(
                    distance_expr.asc(),
                    MemoryEntry.created_at.desc(),
                    MemoryEntry.id.desc(),
                )
            )

            if scope == "agent":
                stmt = stmt.where(MemoryEntry.agent_id == agent_id)
            elif scope == "global":
                stmt = stmt.where(MemoryEntry.agent_id.is_(None))
            # scope == "any": no agent_id filter

            if threshold is not None:
                stmt = stmt.where(distance_expr <= threshold)
            elif score_threshold is not None:
                stmt = stmt.where(distance_expr <= (1.0 - score_threshold))

            # When MMR is requested, fetch more candidates than top_k so the
            # MMR selection has room to improve diversity.
            fetch_k = top_k if mmr_lambda is None else top_k * 4
            stmt = stmt.limit(fetch_k)

            async with _session("search") as db:
                result = await db.execute(stmt)
                rows = result.all()

            pairs = [(row[0], float(1.0 - row[1])) for row in rows]

            if mmr_lambda is not None and len(pairs) > 1:
                pairs = await self._apply_mmr(pairs, query_embedding, top_k, mmr_lambda)

            span.set_attribute("memory.results_returned", len(pairs))
            return pairs

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two equal-length vectors."""
        import math

        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def _apply_mmr(
        self,
        pairs: list[tuple[MemoryEntry, float]],
        query_embedding: list[float],
        k: int,
        mmr_lambda: float,
    ) -> list[tuple[MemoryEntry, float]]:
        """Re-rank pairs using Maximal Marginal Relevance.

        Fetches embeddings for all candidate entries and selects the ``k``
        entries that best balance relevance to the query and novelty relative
        to already-selected entries.
        """
        texts = [entry.content for entry, _ in pairs]
        embeddings = await self._embed.embed_batch(texts)

        query_sims = [self._cosine_sim(e, query_embedding) for e in embeddings]

        selected: list[int] = []
        remaining = list(range(len(pairs)))

        for _ in range(min(k, len(pairs))):
            best_idx = -1
            best_score = -float("inf")

            for idx in remaining:
                relevance = query_sims[idx]
                redundancy = 0.0
                if selected:
                    redundancy = max(self._cosine_sim(embeddings[idx], embeddings[s]) for s in selected)
                mmr_score = mmr_lambda * relevance - (1.0 - mmr_lambda) * redundancy
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return [(pairs[i][0], query_sims[i]) for i in selected]

    async def delete(self, entry_id: uuid.UUID) -> bool:
        """Delete a memory entry by ID (works for either memory type).

        The in-process embedding cache is not invalidated: it is keyed by
        (model, tenant, agent, text), not by entry ID, so a deleted entry's
        text embedding may still reside in the cache. Subsequent search
        queries return no DB row for it either way, so a stale cache entry is
        harmless for correctness — it wastes memory at worst.

        Returns:
            True if the entry was found and deleted, False if not found.
        """
        async with _session("delete") as db:
            result = await db.execute(select(MemoryEntry).where(MemoryEntry.id == entry_id))
            entry = result.scalar_one_or_none()
            if entry is None:
                return False
            await db.delete(entry)
            await db.commit()
            return True

    async def count(self, agent_id: uuid.UUID | None = None) -> int:
        """Count semantic memory entries, optionally filtered by agent."""
        stmt = select(func.count()).select_from(MemoryEntry).where(MemoryEntry.type == MemoryType.SEMANTIC)
        if agent_id is not None:
            stmt = stmt.where(MemoryEntry.agent_id == agent_id)
        async with _session("count") as db:
            result = await db.execute(stmt)
            return result.scalar_one()

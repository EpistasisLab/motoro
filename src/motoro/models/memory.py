"""MemoryEntry ORM model — episodic run summaries and semantic knowledge.

No owner column. ARES's ``MemoryEntry.created_by_id`` (``NOT NULL FK -> users``)
exists for exactly one reason: ``security/isolation/registry.py`` scopes memory
rows to the viewer that created them. That is per-user isolation, out of scope for
slice 1 (see ``engine/patterns/base.py`` and the core README), and the one method
that reads it that way — listing memories by viewer — is not pulled either.

A memory's identity is ``agent_id`` (which agent it belongs to) and, for episodic
entries, ``run_id`` (which run produced it) — both real foreign keys, since
``agents``/``agent_runs`` are core's own tables in the same database. Nothing else
is needed to know what a memory is or who can read it; a product enforcing
per-user isolation does it the same way it would for any other agent-scoped data,
by filtering on the agent's own ``owner_id``.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from motoro.models.base import Base, generate_uuid


class MemoryType(enum.StrEnum):
    """Type of memory entry."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class MemoryEntry(Base):
    """A single memory entry — episodic (run summary) or semantic (knowledge)."""

    __tablename__ = "memory_entries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    agent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=True,
    )
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    type: Mapped[MemoryType] = mapped_column(
        Enum(
            MemoryType,
            name="memory_type",
            create_constraint=True,
            native_enum=True,
            values_callable=lambda obj: [e.value for e in obj],
        ),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)
    # Track the embedding model + dimensions per row so switching the global
    # model never silently mixes vector spaces in a search.
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    embedding_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    embedding_dimensions: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Whether the embedding was generated successfully, so a backfill job can
    # retry rows where remote embedding failed without a vector.
    embedding_status: Mapped[str | None] = mapped_column(String(16), nullable=True, default=None)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_memory_entries_agent_id", "agent_id"),
        Index("ix_memory_entries_type", "type"),
        Index("ix_memory_entries_run_id", "run_id"),
        Index("ix_memory_entries_embedding_model", "embedding_model"),
        Index("ix_memory_entries_meta_gin", "metadata", postgresql_using="gin"),
    )

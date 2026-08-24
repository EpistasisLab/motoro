"""Agent ORM model."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, DateTime, Index, Integer, Numeric, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from motoro.models.base import Base, generate_uuid

if TYPE_CHECKING:
    from motoro.models.run import AgentRun


class Agent(Base):
    """An AI agent with a goal, model configuration, and tool access."""

    __tablename__ = "agents"

    # Names are unique per owner, case-insensitively ("Researcher" and
    # "researcher" are one name to whoever typed it), and only over live rows,
    # so a soft-deleted agent releases its name. Two different owners (or two
    # NULL-owner rows -- core used standalone, where "owner" is meaningless)
    # may share a name; Postgres treats every NULL as distinct, so this index
    # does not actually enforce anything among NULL-owner rows, which is fine
    # since owner_id is a trusted opaque tag, never itself user input.
    __table_args__ = (
        Index(
            "uq_agents_owner_name_active",
            "owner_id",
            text("lower(name)"),
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # Opaque attribution tag: no ForeignKey and no relationship, because
    # ``users`` is a product table. A product supplies the constraint and the
    # NOT NULL in its own migration; core only carries the column so that its
    # own autogenerate does not propose dropping it.
    owner_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")
    model_config_data: Mapped[dict[str, Any]] = mapped_column("model_config", JSON, nullable=False, default=dict)
    tool_config_data: Mapped[dict[str, Any]] = mapped_column("tool_config", JSON, nullable=False, default=dict)
    memory_config_data: Mapped[dict[str, Any]] = mapped_column("memory_config", JSON, nullable=False, default=dict)
    pattern_config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True, default=None)
    # Which registered Skills this agent carries: {"skill_ids": [...]}. A
    # reference rather than an embedded copy — a skill is an independently
    # editable, reusable document (``models.skill.Skill``), so an agent points
    # at one the same way ``tool_config`` points at tools rather than inlining
    # their schemas. Resolved to bodies at run time by
    # ``services.skill_service.resolve_skills``.
    skill_config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True, default=None)
    # Optional output contract: a compact field-spec ({"name", "fields": [...]})
    # used to extract a structured payload into the run output envelope. None
    # means the run still gets the universal envelope, just no domain payload.
    output_contract: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, default=None)
    is_system: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false", default=False)
    budget_limit_usd: Mapped[float | None] = mapped_column(Numeric(precision=10, scale=6), nullable=True, default=None)
    max_run_duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    auto_eval_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true", default=True)
    auto_eval_model: Mapped[str | None] = mapped_column(String(100), nullable=True, default=None)

    # Relationships
    # M113 #1491: `runs`/`batches` are eager-loaded (`selectin`) but nothing in
    # the codebase actually reads `agent.runs`/`agent.batches` — any user who
    # can start a run on a shared `is_system` agent means every `get_agent()`
    # on it eagerly pulls *every user's* runs against that agent. `lazy="raise"`
    # makes an accidental future access fail loudly (at the ORM level, before
    # a serializer could turn it into a live cross-user leak) instead of
    # silently reintroducing this performance/isolation bomb.
    runs: Mapped[list["AgentRun"]] = relationship(back_populates="agent", cascade="all, delete-orphan", lazy="raise")

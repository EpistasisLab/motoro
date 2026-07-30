"""AgentRun and RunStep ORM models."""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, Numeric, Text, func
from sqlalchemy.dialects.postgresql import JSON, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agentic_core.models.base import Base, generate_uuid

if TYPE_CHECKING:
    from agentic_core.models.agent import Agent


class RunStatus(enum.StrEnum):
    """Status of an agent run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    AWAITING_HUMAN = "awaiting_human"


class StepPhase(enum.StrEnum):
    """Phase of the agentic loop."""

    SENSE = "sense"
    REASON = "reason"
    PLAN = "plan"
    ACT = "act"
    HITL = "hitl"


class AgentRun(Base):
    """A single execution of an agent's agentic loop."""

    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[RunStatus] = mapped_column(
        Enum(
            RunStatus,
            name="run_status",
            create_constraint=True,
            native_enum=True,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        default=RunStatus.PENDING,
    )
    input: Mapped[str] = mapped_column(Text, nullable=False)
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    token_usage: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    cost_estimate: Mapped[float | None] = mapped_column(Numeric(precision=10, scale=6), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    state_snapshot: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True, doc="Serialized RunContext for paused runs"
    )
    pattern_overrides: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True, doc="Per-run PatternConfig overrides merged with agent config"
    )
    model_config_overrides: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True, doc="Per-run partial ModelConfig override deep-merged onto the agent's model_config"
    )
    config_snapshot: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, doc="Resolved agent config snapshot captured at dispatch time"
    )
    run_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        "run_metadata",
        JSON,
        nullable=True,
    )
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, default=None)
    # Opaque attribution tag — see the note on ``Agent.owner_id``. No
    # ForeignKey and no relationship: ``users`` is a product table.
    owner_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    agent: Mapped["Agent"] = relationship(back_populates="runs", lazy="selectin")
    # A product that joins runs to its own tables (batches, workers, experiments)
    # declares that relationship on *its* side with ``backref``, so core's
    # mappers configure without it. See AGENTIC_CORE_BOUNDARY.md §7b.
    steps: Mapped[list["RunStep"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="RunStep.sequence",
    )

    __table_args__ = (
        Index("ix_agent_runs_agent_id", "agent_id"),
        Index("ix_agent_runs_status", "status"),
    )


class RunStep(Base):
    """A single phase execution within an agent run."""

    __tablename__ = "run_steps"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    iteration: Mapped[int | None] = mapped_column(Integer, nullable=True)
    phase: Mapped[StepPhase] = mapped_column(
        Enum(
            StepPhase,
            name="step_phase",
            create_constraint=True,
            native_enum=True,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
    )
    # NOTE: Python attribute 'input'/'output' map to DB columns 'input_data'/'output_data'.
    # The rename avoids shadowing Python builtins 'input'/'output' at the column level.
    # Schemas that serialize this model use populate_by_name=True to handle both names.
    # See issue #883 — this alias is intentional and consistent with RunSchema.
    input: Mapped[dict[str, Any] | None] = mapped_column("input_data", JSON, nullable=True)
    output: Mapped[dict[str, Any] | None] = mapped_column("output_data", JSON, nullable=True)
    llm_call: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    tool_call: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    run: Mapped["AgentRun"] = relationship(back_populates="steps", lazy="selectin")

    __table_args__ = (Index("ix_run_steps_run_id", "run_id"),)

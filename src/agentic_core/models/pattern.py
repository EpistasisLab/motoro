"""ArchitecturalPattern ORM model."""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import Boolean, DateTime, Enum, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from agentic_core.models.base import Base, generate_uuid


class PatternCategory(StrEnum):
    """Category of architectural pattern."""

    EXECUTION = "execution"
    SAFETY = "safety"
    COORDINATION = "coordination"
    KNOWLEDGE = "knowledge"
    QUALITY = "quality"
    ROUTING = "routing"
    RESOLUTION = "resolution"


class PatternPhase(StrEnum):
    """Complexity phase of an architectural pattern."""

    BASIC = "basic"
    DYNAMIC = "dynamic"
    INTROSPECTIVE = "introspective"
    SELF_CORRECTING = "self_correcting"
    MULTI_AGENT = "multi_agent"
    ADVANCED_MULTI_AGENT = "advanced_multi_agent"


class ArchitecturalPattern(Base):
    """Catalog entry for an architectural agent pattern."""

    __tablename__ = "architectural_patterns"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    slug: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[PatternCategory] = mapped_column(
        Enum(
            PatternCategory,
            name="pattern_category",
            create_constraint=True,
            native_enum=True,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        index=True,
    )
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    phase: Mapped[PatternPhase] = mapped_column(
        Enum(
            PatternPhase,
            name="pattern_phase",
            create_constraint=True,
            native_enum=True,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        index=True,
    )
    configuration_schema: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    requires_multi_agent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    dependencies: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    version: Mapped[str] = mapped_column(String(32), nullable=False, default="1.0.0")
    is_implemented: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

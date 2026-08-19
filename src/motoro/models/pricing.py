"""LLM pricing override model."""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from motoro.models.base import Base, TimestampMixin, generate_uuid


class LLMPricingOverride(Base, TimestampMixin):
    __tablename__ = "llm_pricing_overrides"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    model_name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    input_cost_per_mtok: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    output_cost_per_mtok: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    effective_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

"""Schemas for LLM pricing overrides."""

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field, field_validator


class PricingOverrideCreate(BaseModel):
    model_name: str = Field(
        min_length=1,
        max_length=255,
        description="LLM model identifier (e.g. 'claude-sonnet-4-20250514').",
    )
    input_cost_per_mtok: Decimal = Field(
        description="Cost per million input tokens (USD).",
    )
    output_cost_per_mtok: Decimal = Field(
        description="Cost per million output tokens (USD).",
    )
    effective_date: datetime | None = Field(
        default=None,
        description="When this pricing takes effect. Defaults to now.",
    )
    notes: str | None = Field(
        default=None,
        description="Optional notes (e.g. 'Azure enterprise agreement').",
    )

    @field_validator("input_cost_per_mtok", "output_cost_per_mtok")
    @classmethod
    def must_be_non_negative(cls, v: Decimal) -> Decimal:
        if v < 0:
            raise ValueError("Price must be non-negative")
        return v


class PricingOverrideResponse(BaseModel):
    id: uuid.UUID
    model_name: str
    input_cost_per_mtok: Decimal
    output_cost_per_mtok: Decimal
    effective_date: datetime
    notes: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PricingOverrideList(BaseModel):
    items: list[PricingOverrideResponse]
    total: int


class PricingModelListResponse(BaseModel):
    """Known model names from agent configs and LLM call history."""

    models: list[str]
    total: int

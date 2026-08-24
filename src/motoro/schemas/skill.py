"""Pydantic schemas for Agent Skills.

Core has no HTTP layer, so these exist for the same reason
:mod:`motoro.schemas.agent` does: a product putting routes over
:mod:`motoro.services.skill_service` should not have to re-describe the shape
of a skill, and the two descriptions should not be able to drift.

The format's own limits are enforced in the service
(``validate_skill_name`` / ``validate_skill_description``), not duplicated as
Pydantic constraints — a skill arriving as an uploaded file never passes
through these schemas at all, and one rule in one place beats two that agree
until they don't.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SkillCreate(BaseModel):
    """Create a skill from separated fields (rather than an uploaded file)."""

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    body: str = Field(default="")


class SkillUpdate(BaseModel):
    """Update a skill. All fields optional; ``None`` means "leave unchanged"."""

    name: str | None = Field(default=None, min_length=1)
    description: str | None = Field(default=None, min_length=1)
    body: str | None = None


class SkillResponse(BaseModel):
    """A stored skill."""

    id: uuid.UUID
    name: str
    description: str
    body: str
    is_system: bool = False
    source_filename: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SkillListResponse(BaseModel):
    """A list of skills."""

    items: list[SkillResponse]
    total: int

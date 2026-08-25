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

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    # Bundled level-3 paths only, never their contents: a list endpoint would
    # otherwise carry every byte of every bundle, and nothing showing a skill
    # needs the text — the agent reads it through ``read_skill_file``, and a
    # product wanting to display it can ask the service for that one file.
    files: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @field_validator("files", mode="before")
    @classmethod
    def _paths_only(cls, value: object) -> object:
        """Accept the ORM's ``list[SkillFile]`` as well as a plain list of paths."""
        if isinstance(value, list):
            return [item if isinstance(item, str) else getattr(item, "path", str(item)) for item in value]
        return value


class SkillListResponse(BaseModel):
    """A list of skills."""

    items: list[SkillResponse]
    total: int

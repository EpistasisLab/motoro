"""Agent Skill ORM model — one stored ``SKILL.md``.

The published Agent Skills format packages a skill as a *directory* whose only
required member is ``SKILL.md``: YAML frontmatter (``name``, ``description``)
plus a markdown body. The directory exists so a skill can ship bundled
resources — reference documents and executable scripts the agent reads or runs
on demand.

Core stores the file, not the directory, and that is a deliberate scope
decision rather than a simplification of the format:

- A skill with no bundled resources *is* exactly one ``SKILL.md``. Rendering
  ``<root>/<name>/SKILL.md`` from this row reconstitutes a spec-conformant
  skill directory whenever a real one is needed (see
  ``services.skill_service.render_skill_md``).
- The bundled-*script* half of the format presumes a filesystem and a shell.
  A Motoro agent's only way to act is an MCP tool call; it has no working
  directory to resolve a relative path against and no sandbox to execute one
  in. Storing a directory would therefore buy a level of the format the engine
  could not honour anyway.

``owner_id`` is the same opaque attribution tag as ``Agent.owner_id`` — no
foreign key and no relationship, because ``users`` is a product table.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Index, String, Text, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from motoro.models.base import Base, generate_uuid


class Skill(Base):
    """A single Agent Skill: frontmatter metadata plus its markdown body."""

    __tablename__ = "skills"

    # Same shape as ``uq_agents_owner_name_active``: unique per owner,
    # case-insensitively, over live rows only, so a soft-deleted skill releases
    # its name. Skill names are additionally lowercase-only by format rule (see
    # ``services.skill_service.validate_skill_name``), so the ``lower()`` here
    # is belt-and-braces against a row written past that validation.
    __table_args__ = (
        Index(
            "uq_skills_owner_name_active",
            "owner_id",
            text("lower(name)"),
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    # 64 is the format's own cap on a skill name, not an arbitrary column width.
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    # Load-bearing, not decorative: the description is the *only* thing an agent
    # sees for a skill it has not opened, so it is what decides whether the
    # skill is ever used. Capped at 1024 by the format.
    description: Mapped[str] = mapped_column(Text, nullable=False)
    # The markdown body below the frontmatter — the instructions themselves.
    body: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # Opaque attribution tag — see the module docstring and Agent.owner_id.
    owner_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)
    is_system: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false", default=False)
    # The filename the skill was uploaded as, kept only so a product can show
    # "from spinal-mri-qc.md" back to whoever uploaded it. Never used to resolve
    # anything.
    source_filename: Mapped[str | None] = mapped_column(String(255), nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, default=None)

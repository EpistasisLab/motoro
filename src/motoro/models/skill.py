"""Agent Skill ORM model — a stored skill *directory*.

The published Agent Skills format packages a skill as a directory whose only
required member is ``SKILL.md``: YAML frontmatter (``name``, ``description``)
plus a markdown body. The directory exists so a skill can ship bundled
resources — reference documents and executable scripts the agent reads or runs
on demand — which is what the format calls level 3.

Core stores that directory as two tables rather than as files on a disk:
:class:`Skill` is the ``SKILL.md`` (levels 1 and 2), and :class:`SkillFile` is
one row per bundled *text* file (level 3). Rows rather than a filesystem
because a skill has no other reason to need one — no storage root to
configure, no volume for a product's API and worker containers to agree on,
and replacing or removing a bundle is a transaction rather than a directory
tree to reconcile against the rows that describe it.

What core still does not support is the bundled-*script* half of the format,
and that one is not a storage decision. A Motoro agent's only way to act is an
MCP tool call: there is no shell, no working directory to resolve a relative
path against, and no sandbox. Reading a reference document into context is
something the engine can honour (``engine.skills``' ``read_skill_file``);
executing ``fill_form.py`` is not, so an upload carrying one is rejected at the
boundary rather than silently accepted and quietly ignored. A skill that needs
to execute code should ship as an MCP server, which is already first class
here.

Text, therefore, and not binary: level 3 for a Motoro agent means "read this
file into the context window", and an asset that cannot enter a context window
and cannot be executed either has no way to affect a run. See
``services.skill_service.validate_bundle_path``.

``owner_id`` is the same opaque attribution tag as ``Agent.owner_id`` — no
foreign key and no relationship, because ``users`` is a product table.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from motoro.models.base import Base, generate_uuid


class Skill(Base):
    """A single Agent Skill: frontmatter metadata, its markdown body, and any
    bundled level-3 files (:attr:`files`)."""

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

    # Eager by default because every reader wants them: resolving a skill for a
    # run needs the level-3 contents in the same breath as the body, and a
    # lazy load would fire outside the session the caller opened. The bundle is
    # capped at MAX_BUNDLE_FILES/MAX_BUNDLE_BYTES precisely so "load it all"
    # stays a defensible default.
    files: Mapped[list[SkillFile]] = relationship(
        back_populates="skill",
        cascade="all, delete-orphan",
        order_by="SkillFile.position",
        lazy="selectin",
    )


class SkillFile(Base):
    """One bundled level-3 file — ``FORMS.md``, ``references/schema.md``.

    A real row per file rather than a JSON blob on :class:`Skill` so a product
    can list a bundle's paths without dragging every byte of its contents
    along, and so the unique constraint below is the database's job rather than
    application code's.
    """

    __tablename__ = "skill_files"

    # Case-insensitive, matching the skill-name index above and for a related
    # reason: the model addresses these by typing the path into a tool call, and
    # a bundle holding both ``FORMS.md`` and ``forms.md`` makes the answer to
    # "which one did it mean" depend on row order.
    __table_args__ = (
        Index("uq_skill_files_skill_path", "skill_id", text("lower(path)"), unique=True),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    # Hard FK with ON DELETE CASCADE, unlike owner_id above: skills is core's
    # own table, so there is a real referent to point at.
    skill_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("skills.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Forward-slashed and relative to the skill directory, always — the same
    # string the SKILL.md body links to and the model passes to read_skill_file.
    # Never absolute, never containing "..": see validate_bundle_path.
    path: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # Upload order, preserved so the "bundled files" listing an agent sees is
    # stable between runs rather than whatever the database hands back.
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    skill: Mapped[Skill] = relationship(back_populates="files")

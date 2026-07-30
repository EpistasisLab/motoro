"""SQLAlchemy declarative base and shared model utilities.

Core owns ``Base``; a product's models subclass it, so ``Base.metadata`` spans
both packages and one Alembic run can see the whole schema.

**There is deliberately no ``OwnedMixin`` here.** The ARES original defined one
adding ``created_by_id`` / ``updated_by_id`` as ``ForeignKey("users.id")``, which
would make core's schema depend on a ``users`` table core does not own and does
not want to. Core manages agents, runs, and steps; it does not manage users.
That also matches the documented data model in ARES ``docs/ARCHITECTURE.md``,
whose core entities carry ``created_at``/``updated_at`` and no owner at all — the
ownership columns arrived later, with per-user isolation.

A product that wants ownership has two options, and should prefer the first:

1. **Opaque tag.** Core models expose a nullable, un-constrained
   ``owner_id: UUID | None``. The product adds the foreign key, the ``NOT NULL``,
   and any per-owner unique constraint in its *own* migration, and enforces the
   semantics in its own service layer. Core never learns that users exist.

   The column is declared in core rather than added purely by a product
   migration for a concrete reason: core's Alembic autogenerate diffs against
   ``Base.metadata``, and a column present in the database but absent from
   core's model is a column core proposes to *drop*.

2. **Product-owned join table.** Ownership lives entirely in a product table
   keyed by ``(resource_type, resource_id, owner_id)``. Cleaner separation, at
   the cost of a join on every scoped read.

Either way the rule is the one in ARES ``project_plan/AGENTIC_CORE_BOUNDARY.md``:
a core table may carry an opaque UUID referring to a product row, but never a
``ForeignKey`` or a ``relationship()`` to a product table.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class TimestampMixin:
    """Mixin that adds created_at and updated_at columns."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


def generate_uuid() -> uuid.UUID:
    """Generate a new UUID4."""
    return uuid.uuid4()

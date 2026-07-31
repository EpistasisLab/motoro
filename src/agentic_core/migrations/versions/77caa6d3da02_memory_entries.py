"""memory_entries: episodic and semantic memory

Revision ID: 77caa6d3da02
Revises: c58058956d24
Create Date: 2026-07-30 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "77caa6d3da02"
down_revision: str | None = "c58058956d24"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "memory_entries",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("agent_id", sa.UUID(), nullable=True),
        sa.Column("run_id", sa.UUID(), nullable=True),
        sa.Column(
            "type",
            sa.Enum("episodic", "semantic", name="memory_type", create_constraint=True),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("embedding_model", sa.String(length=255), nullable=True),
        sa.Column("embedding_version", sa.String(length=64), nullable=True),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=True),
        sa.Column("embedding_status", sa.String(length=16), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_entries_agent_id", "memory_entries", ["agent_id"], unique=False)
    op.create_index("ix_memory_entries_type", "memory_entries", ["type"], unique=False)
    op.create_index("ix_memory_entries_run_id", "memory_entries", ["run_id"], unique=False)
    op.create_index("ix_memory_entries_embedding_model", "memory_entries", ["embedding_model"], unique=False)
    op.create_index("ix_memory_entries_meta_gin", "memory_entries", ["metadata"], unique=False, postgresql_using="gin")


def downgrade() -> None:
    op.drop_index("ix_memory_entries_meta_gin", table_name="memory_entries")
    op.drop_index("ix_memory_entries_embedding_model", table_name="memory_entries")
    op.drop_index("ix_memory_entries_run_id", table_name="memory_entries")
    op.drop_index("ix_memory_entries_type", table_name="memory_entries")
    op.drop_index("ix_memory_entries_agent_id", table_name="memory_entries")
    op.drop_table("memory_entries")

    # Autogenerate emits CREATE TYPE for a native enum but never the matching
    # DROP TYPE — see c58058956d24's downgrade for the same fix and why it is
    # needed (dropping the table does not cascade to the type).
    op.execute("DROP TYPE IF EXISTS memory_type")
    # Left in place: CREATE EXTENSION IF NOT EXISTS vector. Other tables in
    # this database may already depend on the extension by the time this
    # downgrade runs (or will by the time an upgrade runs again), and
    # DROP EXTENSION would need CASCADE to remove it while columns still use
    # the vector type — which is exactly backwards for a downgrade.

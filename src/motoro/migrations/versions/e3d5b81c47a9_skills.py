"""skills: registered Agent Skills, and agents.skill_config

Revision ID: e3d5b81c47a9
Revises: b7e2a91c4f30
Create Date: 2026-08-24 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from motoro.migrations.guards import drop_column

revision: str = "e3d5b81c47a9"
down_revision: str | None = "b7e2a91c4f30"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "skills",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("owner_id", sa.UUID(), nullable=True),
        sa.Column("is_system", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("source_filename", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_index("ix_skills_owner_id", "skills", ["owner_id"], unique=False, if_not_exists=True)
    # Partial + expression index: names are unique per owner, case-insensitively,
    # over live rows only. Mirrors uq_agents_owner_name_active (f64307429723).
    op.create_index(
        "uq_skills_owner_name_active",
        "skills",
        ["owner_id", sa.text("lower(name)")],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
        if_not_exists=True,
    )
    op.add_column(
        "agents",
        sa.Column("skill_config", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        if_not_exists=True,
    )


def downgrade() -> None:
    drop_column("agents", "skill_config")
    op.drop_index("uq_skills_owner_name_active", table_name="skills", if_exists=True)
    op.drop_index("ix_skills_owner_id", table_name="skills", if_exists=True)
    op.drop_table("skills", if_exists=True)

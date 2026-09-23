"""preserve portable skill metadata and binary resources

Revision ID: d4b8e2a71c90
Revises: a1f7c2be40d9
Create Date: 2026-09-23 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "d4b8e2a71c90"
down_revision: str | None = "a1f7c2be40d9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "skills",
        sa.Column(
            "frontmatter",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        if_not_exists=True,
    )
    op.add_column(
        "skill_files",
        sa.Column("encoding", sa.String(length=16), server_default="utf-8", nullable=False),
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_column("skill_files", "encoding", if_exists=True)
    op.drop_column("skills", "frontmatter", if_exists=True)

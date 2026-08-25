"""skill_files: the bundled level-3 files of a skill directory

Revision ID: a1f7c2be40d9
Revises: e3d5b81c47a9
Create Date: 2026-08-24 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "a1f7c2be40d9"
down_revision: str | None = "e3d5b81c47a9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "skill_files",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("skill_id", sa.UUID(), nullable=False),
        sa.Column("path", sa.String(length=255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["skill_id"], ["skills.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_skill_files_skill_id", "skill_files", ["skill_id"], unique=False)
    # Case-insensitive per skill: the model addresses a bundled file by typing
    # its path into read_skill_file, so a bundle holding both FORMS.md and
    # forms.md would make the resolution depend on row order.
    op.create_index(
        "uq_skill_files_skill_path",
        "skill_files",
        ["skill_id", sa.text("lower(path)")],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_skill_files_skill_path", table_name="skill_files")
    op.drop_index("ix_skill_files_skill_id", table_name="skill_files")
    op.drop_table("skill_files")

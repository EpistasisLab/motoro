"""mcp_server_configs: persisted MCP server registration

Revision ID: ac5e1e11d56e
Revises: 77caa6d3da02
Create Date: 2026-07-31 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "ac5e1e11d56e"
down_revision: str | None = "77caa6d3da02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "mcp_server_configs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column(
            "transport",
            sa.Enum("stdio", "http", "sse", name="mcp_transport", create_constraint=True),
            nullable=False,
        ),
        sa.Column("command", sa.Text(), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("capabilities", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "status",
            sa.Enum("connected", "disconnected", "error", name="mcp_server_status", create_constraint=True),
            nullable=False,
        ),
        sa.Column("headers_encrypted", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("is_system", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("owner_id", sa.UUID(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_mcp_server_configs_owner_id", "mcp_server_configs", ["owner_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_mcp_server_configs_owner_id", table_name="mcp_server_configs")
    op.drop_table("mcp_server_configs")

    # See c58058956d24's downgrade for why: autogenerate emits CREATE TYPE for a
    # native enum but never the matching DROP TYPE.
    for enum_name in ("mcp_server_status", "mcp_transport"):
        op.execute(f"DROP TYPE IF EXISTS {enum_name}")

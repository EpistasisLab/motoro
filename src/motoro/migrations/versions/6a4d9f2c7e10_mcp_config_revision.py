"""add MCP config revision for cross-process registry coherence

Revision ID: 6a4d9f2c7e10
Revises: 2e6f4c8a91d3
Create Date: 2026-09-28 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "6a4d9f2c7e10"
down_revision: str | None = "2e6f4c8a91d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "mcp_server_configs",
        sa.Column("config_revision", sa.BigInteger(), server_default="1", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("mcp_server_configs", "config_revision")

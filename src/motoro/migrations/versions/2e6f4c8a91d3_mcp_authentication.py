"""persist encrypted MCP stdio and OAuth authentication material

Revision ID: 2e6f4c8a91d3
Revises: 8f2c1a6d9b40
Create Date: 2026-09-28 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "2e6f4c8a91d3"
down_revision: str | None = "8f2c1a6d9b40"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("mcp_server_configs", sa.Column("stdio_env_encrypted", sa.Text(), nullable=True))
    op.add_column("mcp_server_configs", sa.Column("oauth_encrypted", sa.Text(), nullable=True))
    op.add_column("mcp_server_configs", sa.Column("oauth_pending_encrypted", sa.Text(), nullable=True))
    op.add_column("mcp_server_configs", sa.Column("oauth_state_hash", sa.String(length=64), nullable=True))
    op.add_column(
        "mcp_server_configs",
        sa.Column("oauth_authorization_required", sa.Boolean(), server_default="false", nullable=False),
    )
    op.create_unique_constraint("uq_mcp_server_configs_oauth_state_hash", "mcp_server_configs", ["oauth_state_hash"])


def downgrade() -> None:
    op.drop_constraint("uq_mcp_server_configs_oauth_state_hash", "mcp_server_configs", type_="unique")
    op.drop_column("mcp_server_configs", "oauth_authorization_required")
    op.drop_column("mcp_server_configs", "oauth_state_hash")
    op.drop_column("mcp_server_configs", "oauth_pending_encrypted")
    op.drop_column("mcp_server_configs", "oauth_encrypted")
    op.drop_column("mcp_server_configs", "stdio_env_encrypted")

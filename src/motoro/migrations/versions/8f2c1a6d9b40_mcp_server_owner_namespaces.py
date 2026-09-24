"""scope MCP server names per owner and preserve ownerless uniqueness

Revision ID: 8f2c1a6d9b40
Revises: d4b8e2a71c90
Create Date: 2026-09-23 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "8f2c1a6d9b40"
down_revision: str | None = "d4b8e2a71c90"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("mcp_server_configs_name_key", "mcp_server_configs", type_="unique")
    op.create_index(
        "uq_mcp_server_configs_owner_name",
        "mcp_server_configs",
        ["owner_id", "name"],
        unique=True,
        postgresql_where=sa.text("owner_id IS NOT NULL"),
    )
    op.create_index(
        "uq_mcp_server_configs_ownerless_name",
        "mcp_server_configs",
        ["name"],
        unique=True,
        postgresql_where=sa.text("owner_id IS NULL"),
    )


def downgrade() -> None:
    # Restoring installation-global uniqueness after different owners have
    # legitimately reused a name would otherwise fail as an opaque index-build
    # error. Abort before changing the schema and explain the required cleanup.
    op.execute(
        sa.text(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM mcp_server_configs
                    GROUP BY name
                    HAVING count(*) > 1
                ) THEN
                    RAISE EXCEPTION
                        'cannot downgrade MCP server name scoping: duplicate names exist across owners';
                END IF;
            END
            $$
            """
        )
    )
    op.drop_index("uq_mcp_server_configs_ownerless_name", table_name="mcp_server_configs")
    op.drop_index("uq_mcp_server_configs_owner_name", table_name="mcp_server_configs")
    op.create_unique_constraint("mcp_server_configs_name_key", "mcp_server_configs", ["name"])

"""per-owner name uniqueness: scope agents.name to owner_id

Was unique per installation (case-insensitive, over live rows) -- a name is a
namespace shared by every user of the install, not by design so much as by
never having been revisited. Two different owners creating an agent with the
same, often repo-hardcoded, name (e.g. a shared use-case's fixed agent names)
collided with a raw conflict instead of each owner getting their own row.

owner_id is nullable (core's opaque attribution tag -- see agent.py's module
docs), so a NULL-owner row (core used without a product) is not covered by
this index: Postgres treats every NULL as distinct, so multiple NULL-owner
rows sharing a name would NOT collide at the DB level. That's acceptable here
-- NULL-owner rows come from core used standalone, never from arbitrary
per-request user input the way an owned row's name is.

mcp_server_configs.name stays globally unique for now -- its live connection
registry (motoro.mcp.registry.MCPServerRegistry) is keyed by name alone
with no owner dimension, so two owners' rows sharing a name would silently
fight over one underlying connection. Scoping that table's uniqueness needs
the registry (and hydrate_registry/call_server_tool's name-keyed lookups)
reworked first; out of scope here.

Revision ID: f64307429723
Revises: ac5e1e11d56e
Create Date: 2026-08-05 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f64307429723"
down_revision: str | None = "ac5e1e11d56e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("uq_agents_name_active", table_name="agents", postgresql_where=sa.text("deleted_at IS NULL"), if_exists=True)
    op.create_index(
        "uq_agents_owner_name_active",
        "agents",
        ["owner_id", sa.literal_column("lower(name)")],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("uq_agents_owner_name_active", table_name="agents", postgresql_where=sa.text("deleted_at IS NULL"), if_exists=True)
    op.create_index(
        "uq_agents_name_active",
        "agents",
        [sa.literal_column("lower(name)")],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
        if_not_exists=True,
    )
